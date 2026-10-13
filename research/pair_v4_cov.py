"""The original method (deliverable v4 cluster model, gated_fingerprint.fit_cluster / local_logits) on
the original data, with one change only: the covariance of its two linear discriminants
(gated_fingerprint.LDA_COV). Protocols as pair_deliverable_check.py; every result is reported per
label (GPT-6 Astra, GPT-6.1 Sol), the headline is the worse label.

Usage: python research/pair_v4_cov.py <mode> <protocol>
  mode      auto            v4 as delivered (sklearn shrinkage="auto")
            weighted        Ledoit-Wolf in the family-weighted space, distribution and score LDA
            weighted_score  the same for the score LDA only
  protocol  lobo        leave one training batch out (3 folds)
            codex_env   10-11 batch only, leave one environment out
            env         all training batches, leave one environment out
            outside     fitted on all training batches, scored on the two outside panels
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.gated_fingerprint as G  # noqa: E402
from research.data import PAIR_ROLE, extended_panel  # noqa: E402

p = extended_panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
yb = (p.y == S).astype(int)
TRB = ("train_original", "train_supplemental", PAIR_ROLE)
T0 = time.time()


def rows(roles):
    return np.flatnonzero(np.isin(p.y, [A, S]) & np.isin(p.role, roles))


def groups(roles):
    return p.group_members[np.isin(p.group_role, roles) & np.isin(p.group_y, [A, S])]


def fit_decide(ref, te):
    t = time.time()
    cl = G.fit_cluster([p.rows[i] for i in ref], p.classes, [A, S])
    s = np.array([G.local_logits(cl, p.rows[i]["numbers"]) for i in te])
    print(f"[{time.time() - T0:6.0f}s] cluster on {len(ref)} refs ({time.time() - t:.0f}s), weights "
          f"{ {k: round(v, 3) for k, v in cl['combine'].items() if k.startswith('w_')} }", flush=True)
    return s[:, 1] - s[:, 0]  # Sol logit - Astra logit


def k3_all(d, idx, rng, n_max=20000):
    """Per label: share of correct same-label triples from three different environments
    (exhaustive when small, else n_max random triples)."""
    out = {}
    for c, lab in ((0, "astra"), (1, "sol")):
        r = idx[yb[idx] == c]
        envs = sorted(set(p.env[r]))
        if len(envs) < 3:
            out[lab] = None
            continue
        by = [d[r[p.env[r] == e]] for e in envs]
        tot = sum(len(by[a]) * len(by[b]) * len(by[e]) for a in range(len(by)) for b in range(a + 1, len(by)) for e in range(b + 1, len(by)))
        sgn = 1 if c else -1
        if tot <= n_max:
            cor = n = 0
            for a in range(len(by)):
                for b in range(a + 1, len(by)):
                    for e in range(b + 1, len(by)):
                        s = by[a][:, None, None] + by[b][None, :, None] + by[e][None, None, :]
                        cor += int(np.sum(sgn * s > 0))
                        n += s.size
            out[lab] = round(cor / n, 4)
        else:
            cor = 0
            for _ in range(n_max):
                a, b, e = rng.choice(len(by), 3, replace=False)
                cor += sgn * (rng.choice(by[a]) + rng.choice(by[b]) + rng.choice(by[e])) > 0
            out[lab] = round(cor / n_max, 4)
    return out


def per_label(d, idx, g):
    rng = np.random.default_rng(0)
    ra, rs = float(np.mean(d[idx][yb[idx] == 0] <= 0)), float(np.mean(d[idx][yb[idx] == 1] > 0))
    ok3 = (d[g].sum(1) > 0) == yb[g[:, 0]]
    k3a = float(np.mean(ok3[yb[g[:, 0]] == 0])) if (yb[g[:, 0]] == 0).any() else None
    k3s = float(np.mean(ok3[yb[g[:, 0]] == 1])) if (yb[g[:, 0]] == 1).any() else None
    ka = k3_all(d, idx, rng)
    return {"n": len(idx), "k1": round(float(np.mean((d[idx] > 0) == yb[idx])), 3),
            "recall_astra": round(ra, 3), "recall_sol": round(rs, 3), "k1_worst": round(min(ra, rs), 3),
            "native_k3": round(float(ok3.mean()), 3) if len(g) else None, "native_triples": len(g),
            "native_k3_astra": None if k3a is None else round(k3a, 3), "native_k3_sol": None if k3s is None else round(k3s, 3),
            "k3_all_astra": ka["astra"], "k3_all_sol": ka["sol"],
            "k3_all_worst": None if None in ka.values() else min(ka.values())}


def report(tag, d, parts):
    out = {}
    for name, (idx, g) in parts.items():
        r = per_label(d, idx, g)
        out[name] = r
        print(f"RESULT {tag:32} {name:14} k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} worst {r['k1_worst']:.3f} | "
              f"native k3 {r['native_k3']} (A {r['native_k3_astra']} S {r['native_k3_sol']}, {r['native_triples']} triples) | "
              f"k3_all A {r['k3_all_astra']} S {r['k3_all_sol']} worst {r['k3_all_worst']}", flush=True)
    return out


def main(mode, proto):
    if mode != "auto":
        G.LDA_COV = "weighted"
        G.LDA_COV_PARTS = ("dense", "score") if mode == "weighted" else ("score",)
    d = np.zeros(len(p.rows))
    tag = f"{mode} {proto}"
    if proto == "lobo":
        for b in TRB:
            te = rows((b,))
            d[te] = fit_decide(rows(tuple(x for x in TRB if x != b)), te)
        parts = {b[-12:]: (rows((b,)), groups((b,))) for b in TRB}
        parts["all"] = (rows(TRB), groups(TRB))
    elif proto == "codex_env":
        cx = rows((PAIR_ROLE,))
        for e in sorted(set(p.env[cx])):
            te = cx[p.env[cx] == e]
            d[te] = fit_decide(cx[p.env[cx] != e], te)
        parts = {"10-11": (cx, groups((PAIR_ROLE,)))}
    elif proto == "env":
        tr = rows(TRB)
        for e in sorted(set(p.env[tr])):
            te = tr[p.env[tr] == e]
            d[te] = fit_decide(tr[p.env[tr] != e], te)
        parts = {b[-12:]: (rows((b,)), groups((b,))) for b in TRB}
        parts["all"] = (tr, groups(TRB))
    elif proto == "outside":
        te = rows(("old_outside", "fresh_outside"))
        d[te] = fit_decide(rows(TRB), te)
        parts = {r: (rows((r,)), groups((r,))) for r in ("old_outside", "fresh_outside")}
    else:
        raise ValueError(proto)
    res = {"mode": mode, "protocol": proto, "results": report(tag, d, parts),
           "decisions": {str(i): round(float(d[i]), 4) for i in np.flatnonzero(np.isin(p.y, [A, S])) if d[i] != 0}}
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"pair_v4_cov_{mode}_{proto}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
