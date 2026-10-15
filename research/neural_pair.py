"""Run 14: a neural next-number model on the original data (218-300 numbers per response).

Question (2026-10-15): with the original data only, can a better algorithm stop GPT-6.1 Sol responses from
being called GPT-6 Astra? On this data the two models are each misjudged about 30% of the time (run 7: whole
environment held out, Astra 0.716 / Sol 0.708), so the goal is a better separation of both, judged by the
worse model.

Every model tried so far (histogram LDA, n-gram counts, a log-linear next-number model and its Fisher score,
kernel embeddings, TF-IDF, fork evidence at recurring contexts) stops at about the same level. The family not
yet tried is a learned sequence model: a GRU predicts the next number from the whole history, conditioned on
the model label, and a response is scored by the likelihood ratio

    LLR = sum_t [ log p(x_t | x_<t, Sol) - log p(x_t | x_<t, Astra) ]

which is the Bayes decision if the two conditionals were known. The history is the network's job (shared
across labels), so a label difference anywhere in the response, not only at recurring opening contexts, can
add up.

Data: training rows of the original data (extended_panel). Leave one environment out over the 12 pair
environments (rows of that environment are removed for every label, so the network never sees them); the
outside panels are scored by the network trained on all training rows. Only reference rows are used for
training; early stopping on a random 10% of the reference rows (next-number log-loss).

Variants (one job each): H64 / H128 = GRU width, trained on all labels' references (the other models teach
the shared number dynamics); H64pair = width 64, Astra and Sol references only.

Reported (worse model first): per-label recall, all-combination k3, AUC for
  - the current method (v4, per-row decisions saved by run 11, data/v4_d_run11.json);
  - the neural LLR with threshold 0 and with a calibrated threshold (logistic on the other environments);
  - v4 and the LLR combined (logistic on the other environments' held-out scores; outside: all training);
  - where the LLR evidence sits: partial sums over positions [0,10), [10,40), [40,125), [125,end) and the
    first / second half, AUC of each.
Pre-specified bar: the target is worse-model k3 >= 0.95; a change is worth adopting only if the AUC gain over
v4 has an environment-bootstrap interval above 0 on the training protocol and the same sign outside.

Usage: python research/neural_pair.py H64 | H128 | H64pair
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.hard_pair import A, ENVS, OUT_ROWS, S, TR, boot_ci, label_metrics, p, seq, yb  # noqa: E402
from research.runner_pair import auc  # noqa: E402

VMAX = 356                  # values 0..356 are their own tokens
OTHER = VMAX + 1            # anything else
NOUT = VMAX + 2             # output classes
BOS = NOUT                  # input-only start token
MAXLEN = 400
WINDOWS = ((0, 10), (10, 40), (40, 125), (125, MAXLEN))
T0 = time.time()
RES = Path("results")
RES.mkdir(exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() or 1))


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def tok(v):
    return v if 0 <= v <= VMAX else OTHER


class LM(nn.Module):
    """GRU over the previous numbers; the label enters as an embedding at every step, in the initial state
    and in the output layer."""

    def __init__(self, n_labels, hidden, emb=32, lab=8):
        super().__init__()
        self.emb = nn.Embedding(NOUT + 1, emb)
        self.lab = nn.Embedding(n_labels, lab)
        self.h0 = nn.Linear(lab, hidden)
        self.gru = nn.GRU(emb + lab + 2, hidden, batch_first=True)
        self.out = nn.Linear(hidden + lab, NOUT)
        self.drop = nn.Dropout(0.1)

    def forward(self, xin, num, pos, c):
        le = self.lab(c)
        T = xin.shape[1]
        lt = le[:, None, :].expand(-1, T, -1)
        z = torch.cat([self.drop(self.emb(xin)), lt, num[..., None], pos[..., None]], -1)
        h, _ = self.gru(z, torch.tanh(self.h0(le))[None].contiguous())
        return self.out(torch.cat([self.drop(h), lt], -1))


ARR: dict = {}


def arrays(i):
    if i not in ARR:
        x = seq(i)[:MAXLEN]
        t = np.array([tok(v) for v in x], dtype=np.int64)
        prev = np.array([0] + x[:-1], dtype=np.float32)
        ARR[i] = (t, np.r_[BOS, t[:-1]].astype(np.int64), np.clip(prev, 0, 400) / 355.0,
                  np.arange(len(x), dtype=np.float32) / 300.0)
    return ARR[i]


def batch(idx, labels):
    T = max(len(arrays(i)[0]) for i in idx)
    n = len(idx)
    tgt, xin = np.zeros((n, T), np.int64), np.full((n, T), BOS, np.int64)
    num, pos, mask = np.zeros((n, T), np.float32), np.zeros((n, T), np.float32), np.zeros((n, T), np.float32)
    for k, i in enumerate(idx):
        t, a, u, q = arrays(i)
        L = len(t)
        tgt[k, :L], xin[k, :L], num[k, :L], pos[k, :L], mask[k, :L] = t, a, u, q, 1
    f = torch.from_numpy
    return f(tgt), f(xin), f(num), f(pos), f(mask), torch.as_tensor(np.asarray(labels, dtype=np.int64))


def nll_sum(model, idx, labels, bs=64):
    model.eval()
    tot, cnt = 0.0, 0.0
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            tgt, xin, num, pos, mask, c = batch(idx[s:s + bs], labels[s:s + bs])
            lp = torch.log_softmax(model(xin, num, pos, c), -1).gather(-1, tgt[..., None])[..., 0]
            tot -= float((lp * mask).sum())
            cnt += float(mask.sum())
    return tot / cnt


def train(rows, labels, n_labels, hidden, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    perm = rng.permutation(len(rows))
    nv = max(20, len(rows) // 10)
    va, tr = perm[:nv], perm[nv:]
    model = LM(n_labels, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    for ep in range(40):
        model.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), 32):
            b = order[s:s + 32]
            tgt, xin, num, pos, mask, c = batch(rows[b], labels[b])
            lp = torch.log_softmax(model(xin, num, pos, c), -1).gather(-1, tgt[..., None])[..., 0]
            loss = -(lp * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = nll_sum(model, rows[va], labels[va])
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            log(f"  epoch {ep:2d} val nll/number {v:.4f} (best {best:.4f})")
        if bad >= 4:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best


def position_llr(model, idx, lab_sol, lab_astra, bs=64):
    """Per-position log p(x_t | Sol) - log p(x_t | Astra) for every row."""
    out = {}
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            tgt, xin, num, pos, mask, _ = batch(b, [0] * len(b))
            lps = []
            for lab in (lab_sol, lab_astra):
                c = torch.full((len(b),), lab, dtype=torch.int64)
                lps.append(torch.log_softmax(model(xin, num, pos, c), -1).gather(-1, tgt[..., None])[..., 0])
            diff = ((lps[0] - lps[1]) * mask).numpy()
            for k, i in enumerate(b):
                out[int(i)] = diff[k, :len(arrays(i)[0])]
    return out


def stack(d_cols):
    """Logistic combination fitted on the other environments' held-out scores (outside: all training rows)."""
    Z = np.c_[tuple(d_cols)]
    out = np.full(len(p.rows), np.nan)
    for e, te in [(e, TR[p.env[TR] == e]) for e in ENVS] + [("outside", OUT_ROWS)]:
        ref = TR if e == "outside" else TR[p.env[TR] != e]
        mu, sd = Z[ref].mean(0), Z[ref].std(0) + 1e-9
        lr = LogisticRegression(C=1.0).fit((Z[ref] - mu) / sd, yb[ref])
        out[te] = lr.decision_function((Z[te] - mu) / sd)
    return out


def main(variant):
    hidden = int(variant[1:].replace("pair", ""))
    pair_only = variant.endswith("pair")
    assert set(np.flatnonzero(np.isin(p.y, [A, S]))) & set(p.train) == set(TR)
    base = TR if pair_only else p.train
    labs = np.unique(p.y[base])
    lab_of = {int(c): k for k, c in enumerate(labs)}
    log(f"variant {variant}: hidden {hidden}, training labels {len(labs)}, reference rows {len(base)}, "
        f"pair train {len(TR)} ({len(ENVS)} environments), outside {len(OUT_ROWS)}, threads {torch.get_num_threads()}")

    v4 = json.loads(Path("data/v4_d_run11.json").read_text())
    both = np.concatenate([TR, OUT_ROWS])
    assert set(map(int, v4)) == set(map(int, both))
    d_v4 = np.full(len(p.rows), np.nan)
    for k, v in v4.items():
        d_v4[int(k)] = v

    pos_llr: dict = {}
    folds = [(e, TR[p.env[TR] == e]) for e in ENVS] + [("outside", OUT_ROWS)]
    for f, (fname, te) in enumerate(folds):
        rows = base if fname == "outside" else base[p.env[base] != fname]
        labels = np.array([lab_of[int(c)] for c in p.y[rows]])
        model, best = train(rows, labels, len(labs), hidden, seed=f)
        pos_llr.update(position_llr(model, te, lab_of[S], lab_of[A]))
        llr_te = np.array([pos_llr[int(i)].sum() for i in te])
        ok_a = np.mean(llr_te[yb[te] == 0] <= 0) if (yb[te] == 0).any() else float("nan")
        ok_s = np.mean(llr_te[yb[te] == 1] > 0) if (yb[te] == 1).any() else float("nan")
        log(f"fold {fname}: {len(rows)} reference rows, val nll/number {best:.4f}; held-out recall A {ok_a:.3f} S {ok_s:.3f}")

    llr = np.full(len(p.rows), np.nan)
    win = {w: np.full(len(p.rows), np.nan) for w in WINDOWS}
    half = {h: np.full(len(p.rows), np.nan) for h in ("first half", "second half")}
    for i, v in pos_llr.items():
        llr[i] = v.sum()
        for a, b in WINDOWS:
            win[(a, b)][i] = v[a:b].sum()
        half["first half"][i], half["second half"][i] = v[:len(v) // 2].sum(), v[len(v) // 2:].sum()

    variants = {"v4 (run 11)": d_v4, "neural LLR, threshold 0": llr,
                "neural LLR, calibrated": stack([llr]), "v4 + neural (stacked)": stack([d_v4, llr])}
    out = {}
    for tag, idx in (("train (LOEO)", TR), ("outside", OUT_ROWS)):
        for name, d in variants.items():
            r = label_metrics(d, idx)
            out[f"{tag} {name}"] = r
            print(f"NEURAL {variant} {tag:12} {name:24} k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} worst {r['k1_worst']:.3f} | "
                  f"k3_all A {r['k3_all_astra']:.3f} S {r['k3_all_sol']:.3f} worst {r['k3_all_worst']:.3f} | auc {r['auc']:.3f}", flush=True)
        for name in ("neural LLR, threshold 0", "v4 + neural (stacked)"):
            d = variants[name]
            ci, pneg = boot_ci(lambda ni, d_=d: auc(d_[ni], yb[ni]) - auc(d_v4[ni], yb[ni]), idx)
            g = auc(d[idx], yb[idx]) - auc(d_v4[idx], yb[idx])
            out[f"{tag} {name} auc gain over v4"] = {"gain": round(g, 3), "ci": ci, "boot_share<=0": pneg}
            print(f"NEURAL {variant} {tag:12} {name:24} AUC gain over v4 {g:+.3f} {ci} (boot P<=0 {pneg})", flush=True)
        for c, lab in ((0, "Astra"), (1, "Sol")):
            q = idx[yb[idx] == c]
            out[f"{tag} corr(v4, LLR) {lab}"] = round(float(np.corrcoef(d_v4[q], llr[q])[0, 1]), 3)
        parts = {f"[{a},{b})": win[(a, b)] for a, b in WINDOWS} | half
        for name, v in parts.items():
            q = idx[~np.isnan(v[idx])]
            m = {lab: round(float(np.mean(v[q[yb[q] == c]])), 3) for c, lab in ((0, "Astra"), (1, "Sol"))}
            out[f"{tag} evidence {name}"] = {"auc": round(auc(v[q], yb[q]), 3), "mean": m}
            print(f"NEURAL {variant} {tag:12} evidence {name:12} auc {auc(v[q], yb[q]):.3f} mean LLR Astra {m['Astra']:+.3f} Sol {m['Sol']:+.3f}", flush=True)
        print(f"NEURAL {variant} {tag:12} corr(v4, LLR) Astra {out[f'{tag} corr(v4, LLR) Astra']} Sol {out[f'{tag} corr(v4, LLR) Sol']}", flush=True)
    (RES / f"neural_{variant}.json").write_text(json.dumps(
        {"summary": out, "rows": {str(i): {"env": p.env[i], "sol": int(yb[i]), "v4": round(float(d_v4[i]), 4),
                                            "llr": round(float(llr[i]), 4)} for i in both}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
    log("done")
