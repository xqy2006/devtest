"""Run 21 (2026-10-19): GPT-6.1 Sol across reasoning efforts, and the user's selective-decision target.

Data: the training batches of the pair (427 rows, all reasoning effort "low"; Astra's train_original rows have
no recorded effort), the outside panels (150 rows, low), and the new Sol references (47 rows: low / medium /
high / xhigh, one per effort in each of the 12 original environments with the same prompt across efforts,
query-28 medium missing; data/sol_multieffort_20261019).

Every fit is the deliverable's own fit_cluster (v5) on the fold's references only. Outer folds:
  loeo_low   12 folds: references = training rows of the other 11 environments; test = the held-out
             environment's training rows and its new Sol rows
  loeo_all   the same, with the new Sol rows of the other 11 environments added to the references
  lobo_low   3 folds: leave one training batch out (references = the other two batches)
  lobo_all   the same, plus all new Sol rows in the references
  full_low   references = all training rows; test = outside panels and all new Sol rows
  full_all   references = all training rows + new Sol rows; test = outside panels
Stored per test row: the local log-ratio d (Sol - Astra), its four weighted components (distribution, fork,
score, residual) and the fold's calibration (clip, shrink, Platt a_k and b_k for k = 1, 2, 3).

The user's rule (REPORT 19.21): three independent responses; attribution shown when the larger probability is
>= 0.75, otherwise "uncertain, submit 3 more" (earlier responses kept: 6, at most 9). Reported per true label:
first-try decision rate and accuracy of shown attributions, and after up to two retries the final correct /
wrong / still-uncertain shares and the mean number of tries. Probabilities: the deliverable's display (clip 2,
shrink 0.65) and plain Platt (no clip, no shrink) at decision thresholds 0.75 / 0.85 / 0.9 / 0.95.

Effort diagnostics: components by effort; effort detectability (the 195-dim vector, shrinkage LDA, leave one
environment out, AUC) for low vs each other effort and new-low vs old Sol; Spearman of d with reasoning tokens.

Usage: python research/effort_pair.py folds <i>/<n> | aggregate
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import RES, lda05, log  # noqa: E402
from research.hard_pair import A, IA, IS, OUT_ROWS, S, TR, p, yb  # noqa: E402
from research.runner_pair import auc  # noqa: E402

# every multi-effort collection of the pair (data/<model>_multieffort_<date>/<model>_multieffort_references.jsonl)
ME_ROWS = [json.loads(line) for f in sorted((ROOT / "data").glob("*multieffort*/*multieffort_references.jsonl"))
           for line in f.read_text().splitlines() if line]
assert all(r["source"] in (p.classes[A], p.classes[S]) for r in ME_ROWS)
ROWS = p.rows + ME_ROWS
N0 = len(p.rows)
ME = np.arange(N0, N0 + len(ME_ROWS))
ENV = np.concatenate([p.env, [r["condition_id"] for r in ME_ROWS]])
Y = np.concatenate([yb, [int(r["source"] == p.classes[S]) for r in ME_ROWS]])
BATCH = np.concatenate([p.role, [r["role"] for r in ME_ROWS]])
EFF = np.array([r.get("reasoning_effort") or "unrecorded" for r in ROWS])


def _tokens(r):
    v = r.get("reasoning_output_tokens", (r.get("usage") or {}).get("reasoning_output_tokens"))
    return float(v) if v is not None else np.nan


TOK = np.array([_tokens(r) for r in ROWS])
ENVS = sorted(set(ENV[TR]))
EFFORTS = ("low", "medium", "high", "xhigh")
assert set(ENV[ME]) <= set(ENVS) and (EFF[ME] != "unrecorded").all()


def fold_list():
    F = []
    for e in ENVS:
        te = np.concatenate([TR[ENV[TR] == e], ME[ENV[ME] == e]])
        F.append(("loeo_low", e, TR[ENV[TR] != e], te))
        F.append(("loeo_all", e, np.concatenate([TR[ENV[TR] != e], ME[ENV[ME] != e]]), te))
    for b in sorted(set(BATCH[TR])):
        F.append(("lobo_low", b, TR[BATCH[TR] != b], TR[BATCH[TR] == b]))
        F.append(("lobo_all", b, np.concatenate([TR[BATCH[TR] != b], ME]), TR[BATCH[TR] == b]))
    F.append(("full_low", "all training", TR, np.concatenate([OUT_ROWS, ME])))
    F.append(("full_all", "all training + new Sol", np.concatenate([TR, ME]), OUT_ROWS))
    return F


def run_fold(j):
    proto, fname, ref, te = fold_list()[j]
    t = time.time()
    cl = G.fit_cluster([ROWS[i] for i in ref], p.classes, [A, S])
    assert cl["members"] == [p.classes[k] for k in sorted([A, S])]
    w = cl["combine"]
    comp = np.zeros((len(te), 4))
    d = np.zeros(len(te))
    for k, i in enumerate(te):
        x = ROWS[i]["numbers"]
        parts = [w["w_distribution"] * G._dense_logits(cl, x), w["w_branch"] * G.branch_scores(cl["branch"], x, 2),
                 w["w_score"] * G._score_logits(cl["score"], x), w["w_resid"] * G._resid_logits(cl["resid"], x)]
        comp[k] = [v[IS] - v[IA] for v in parts]
        s = G.local_logits(cl, x)
        d[k] = s[IS] - s[IA]
    assert np.abs(comp.sum(1) - d).max() < 1e-6
    cal = w["calibration"]
    out = {"protocol": proto, "fold": str(fname), "test": te.tolist(), "d": d.tolist(), "comp": comp.tolist(),
           "clip": cal["clip"], "shrink": cal["shrink"],
           "a": {k: m["a"] for k, m in cal["maps"].items()},
           "b": {k: m["b"][IS] - m["b"][IA] for k, m in cal["maps"].items()},
           "weights": {k: v for k, v in w.items() if k.startswith("w_")}}
    (RES / f"effort_fold_{j:02d}.json").write_text(json.dumps(out, default=float))
    log(f"fold {j} {proto}/{fname}: refs {len(ref)} test {len(te)} ({time.time() - t:.0f}s); "
        f"weights {{{', '.join(f'{k}: {v:.3f}' for k, v in out['weights'].items())}}}; k3 map a {cal['maps']['3']['a']:.3f}")
    return j


def run_parallel(fn, n_folds, shard):
    """Folds j with j % n == i, one process each. ProcessPoolExecutor, so a worker that dies (e.g. out of memory)
    fails the job instead of hanging it (run 24: one shard of multiprocessing.Pool hung for 20 minutes)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    i, n = map(int, shard.split("/"))
    J = [j for j in range(n_folds) if j % n == i]
    procs = min(len(J), int(os.environ.get("PROCS", os.cpu_count() or 1)))
    log(f"shard {shard}: folds {J} on {procs} processes")
    with ProcessPoolExecutor(procs, mp_context=mp.get_context("fork")) as ex:
        for _ in ex.map(fn, J):
            pass


def run_folds(shard):
    run_parallel(run_fold, len(fold_list()), shard)


# ----------------------------------------------------------------------------- aggregate

THRESHOLDS = (0.75, 0.85, 0.9, 0.95)
DISPLAYS = {"v5 display (clip 2, x0.65)": (2.0, 0.65), "platt (no clip, x1)": (None, 1.0)}


def load(proto, prefix="effort_fold_"):
    """Per-row arrays (NaN outside the protocol's test rows): d, components, a_k, b_k."""
    n = len(ROWS)
    R = {"d": np.full(n, np.nan), "comp": np.full((n, 4), np.nan),
         **{f"a{k}": np.full(n, np.nan) for k in (1, 2, 3)}, **{f"b{k}": np.full(n, np.nan) for k in (1, 2, 3)}}
    for fp_ in sorted(RES.glob(f"{prefix}*.json")):
        r = json.loads(fp_.read_text())
        if r["protocol"] != proto:
            continue
        te = np.array(r["test"], dtype=int)
        R["d"][te], R["comp"][te] = r["d"], r["comp"]
        for k in (1, 2, 3):
            R[f"a{k}"][te], R[f"b{k}"][te] = r["a"][str(k)], r["b"][str(k)]
    return R


def cells(idx, c):
    return [idx[(ENV[idx] == e) & (Y[idx] == c)] for e in ENVS if ((ENV[idx] == e) & (Y[idx] == c)).any()]


def group_logits(R, idx, c, k, clip, shrink):
    """All same-label groups of k rows from k different environments (vectorized additive form)."""
    kk = min(k, 3)
    cl = cells(idx, c)
    z, b = [], []
    for cell in cl:
        d = R["d"][cell]
        d = np.clip(d, -clip, clip) if clip else d
        z.append(shrink * R[f"a{kk}"][cell] * d)
        b.append(shrink * R[f"b{kk}"][cell])
    out = []
    for comb in itertools.combinations(range(len(cl)), k):
        s = z[comb[0]] + b[comb[0]] / k
        for j in comb[1:]:
            s = s[..., None] + (z[j] + b[j] / k).reshape((1,) * s.ndim + (-1,))
        out.append(s.ravel())
    return np.concatenate(out) if out else np.zeros(0)


def selective(L, c, thr):
    t = np.log(thr / (1 - thr))
    right = L >= t if c == 1 else L <= -t
    wrong = L <= -t if c == 1 else L >= t
    dec = right | wrong
    return {"recall": float(np.mean(L > 0) if c == 1 else np.mean(L < 0)), "decide": float(dec.mean()),
            "acc_decided": float(right.sum() / max(dec.sum(), 1)), "wrong_shown": float(wrong.mean())}


def retry_sim(R, idx, c, clip, shrink, thr, rng, n=40000, tries=3):
    """Sequences of up to 3 tries x 3 responses from distinct environments (earlier responses kept)."""
    cl = cells(idx, c)
    tries = min(tries, len(cl) // 3)
    t = np.log(thr / (1 - thr))
    lens = np.array([len(x) for x in cl])
    M = np.zeros((len(cl), lens.max()), dtype=int)
    for i, x in enumerate(cl):
        M[i, :len(x)] = x
    perm = np.argsort(rng.random((n, len(cl))), axis=1)[:, :3 * tries]
    rows = M[perm, (rng.random(perm.shape) * lens[perm]).astype(int)]
    d = R["d"][rows]
    d = np.clip(d, -clip, clip) if clip else d
    state = np.zeros(n, int)  # 0 undecided, 1 right, -1 wrong
    when = np.full(n, tries)
    first = None
    for tr_ in range(1, tries + 1):
        m = slice(0, 3 * tr_)
        L = shrink * (R["a3"][rows[:, m]].mean(1) * d[:, m].sum(1) + R["b3"][rows[:, m]].mean(1))
        right = L >= t if c == 1 else L <= -t
        wrong = L <= -t if c == 1 else L >= t
        new = (state == 0) & (right | wrong)
        state[new & right], state[new & wrong] = 1, -1
        when[new] = tr_
        if tr_ == 1:
            first = (right | wrong, right)
    return {"first_decide": float(first[0].mean()), "first_acc": float(first[1].sum() / max(first[0].sum(), 1)),
            "final_right": float(np.mean(state == 1)), "final_wrong": float(np.mean(state == -1)),
            "undecided": float(np.mean(state == 0)), "mean_tries": float(when.mean()), "max_tries": tries}


def report_set(tag, R, idx, labels, rng):
    for c in labels:
        lab = "Sol" if c == 1 else "Astra"
        rows = idx[Y[idx] == c]
        if not len(rows) or np.isnan(R["d"][rows]).any():
            continue
        print(f"SET {tag:34} {lab:5} n {len(rows):3} envs {len(cells(idx, c)):2} | mean d {np.mean(R['d'][rows]):+.2f} "
              f"| comp D {np.mean(R['comp'][rows, 0]):+.2f} B {np.mean(R['comp'][rows, 1]):+.2f} "
              f"F {np.mean(R['comp'][rows, 2]):+.2f} N {np.mean(R['comp'][rows, 3]):+.2f}", flush=True)
        for disp, (clip, shrink) in DISPLAYS.items():
            L1, L3 = group_logits(R, idx, c, 1, clip, shrink), group_logits(R, idx, c, 3, clip, shrink)
            for thr in (THRESHOLDS if disp.startswith("platt") else (0.75,)):
                s1, s3 = selective(L1, c, thr), selective(L3, c, thr)
                rs = retry_sim(R, idx, c, clip, shrink, thr, rng)
                print(f"SEL {tag:34} {lab:5} {disp:27} thr {thr:.2f} | k1 rec {s1['recall']:.3f} dec {s1['decide']:.3f} "
                      f"acc {s1['acc_decided']:.3f} | k3 rec {s3['recall']:.3f} dec {s3['decide']:.3f} "
                      f"acc {s3['acc_decided']:.3f} wrong {s3['wrong_shown']:.3f} | retry<= {rs['max_tries'] - 1}: "
                      f"right {rs['final_right']:.3f} wrong {rs['final_wrong']:.3f} undec {rs['undecided']:.3f} "
                      f"tries {rs['mean_tries']:.2f}", flush=True)


def effort_diagnostics():
    """Effort detectability on the 195-dim vector (label-free PL weights on all pair rows)."""
    seqs = [list(map(int, ROWS[i]["numbers"])) for i in range(len(ROWS))]
    use = np.concatenate([TR, OUT_ROWS, ME])
    pi = G.fit_pl([seqs[i] for i in np.concatenate([TR, ME])])
    X = np.full((len(ROWS), 195), np.nan)
    X[use] = G.resid_vectors(pi, [seqs[i] for i in use])

    def loeo_auc(pos, neg):
        rows = np.concatenate([pos, neg])
        lab = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
        sc = np.zeros(len(rows))
        for e in sorted(set(ENV[rows])):
            te, tr = ENV[rows] == e, ENV[rows] != e
            if te.any() and len(set(lab[tr])) == 2:
                sc[te] = lda05(X[rows[tr]], lab[tr], X[rows[te]])
        return auc(sc, lab)
    me = {e: ME[(EFF[ME] == e) & (Y[ME] == 1)] for e in EFFORTS}
    old_sol, astra = TR[Y[TR] == 1], TR[Y[TR] == 0]
    pairs = [(f"Sol {e} vs Sol low (new, same prompts)", me[e], me["low"]) for e in EFFORTS[1:]]
    pairs += [("Sol high+xhigh vs Sol low+medium (new)", np.concatenate([me["high"], me["xhigh"]]),
               np.concatenate([me["low"], me["medium"]])),
              ("Sol low new vs Sol low training", me["low"], old_sol),
              ("Sol all new vs Sol training", ME[Y[ME] == 1], old_sol)]
    pairs += [(f"Sol {e} new vs Astra training", me[e], astra) for e in EFFORTS]
    pairs += [("Sol training vs Astra training (reference)", old_sol, astra)]
    for name, pos, neg in pairs:
        print(f"EFFDET {name:44} n {len(pos)} / {len(neg)} | LOEO AUC {loeo_auc(pos, neg):.3f}", flush=True)
    R = load("loeo_low")
    for tag, rows in [("training Astra", astra), ("training Sol", old_sol)] + [(f"new Sol {e}", me[e]) for e in EFFORTS] \
            + [("new Sol all", ME[Y[ME] == 1])]:
        ok = ~np.isnan(TOK[rows]) & ~np.isnan(R["d"][rows])
        if ok.sum() > 5:
            r = spearmanr(R["d"][rows][ok], TOK[rows][ok]).correlation
            print(f"TOKENS {tag:16} n {ok.sum():3} median tokens {np.median(TOK[rows][ok]):6.0f} "
                  f"| Spearman(d, reasoning tokens) {r:+.3f}", flush=True)


def aggregate():
    files = sorted(RES.glob("effort_fold_*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    rng = np.random.default_rng(0)
    for proto in ("loeo_low", "loeo_all", "lobo_low", "lobo_all", "full_low", "full_all"):
        R = load(proto)
        if proto.startswith("full"):
            report_set(f"{proto}: outside", R, OUT_ROWS, (0, 1), rng)
            for b in ("old_outside", "fresh_outside"):
                report_set(f"{proto}: {b}", R, OUT_ROWS[BATCH[OUT_ROWS] == b], (0, 1), rng)
        else:
            report_set(f"{proto}: training", R, TR, (0, 1), rng)
        if proto in ("loeo_low", "loeo_all", "full_low"):
            for e in EFFORTS:
                report_set(f"{proto}: new {e}", R, ME[EFF[ME] == e], (0, 1), rng)
            report_set(f"{proto}: new all efforts", R, ME, (0, 1), rng)
    effort_diagnostics()
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
