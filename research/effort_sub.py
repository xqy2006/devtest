"""Run 22b (2026-10-19): model x reasoning-effort subclasses.

Run 21: with low-effort references, GPT-6.1 Sol at high / xhigh effort lands on the GPT-6 Astra side (xhigh: every
triple misjudged), and adding the new Sol rows as plain Sol references did not fix xhigh (one Sol class cannot
hold four effort clusters), while within the 195-dim vector Sol xhigh and Astra (low) are well apart (AUC 0.92).
Here each (model, effort) with references is its own member of the deliverable's fit_cluster:
Astra@low (all Astra rows: low or unrecorded), Sol@low (training rows + new low), Sol@medium, Sol@high, Sol@xhigh.
A query's model probability sums its subclasses after the calibrated map. Run 22b (first version) used sklearn's
default LDA priors (reference proportions): the small subclasses then carried a per-response prior of about -3.6
that the Platt intercept had to undo, and clipping / summing over 6-9 responses broke that balance (everything
became Sol). Run 23: equal LDA priors (G.LDA_EQUAL_PRIORS), and the combination weights and the Platt maps maximize
the likelihood of the true model (sum over its subclasses; member_groups), with weight 1/2 per model split evenly
over its subclasses, so the map's output already has class prior 1/2.

Caveat: there are no Astra references above low effort, so Astra at medium / high / xhigh is untested here.

Protocols as run 21 (references always include the new Sol rows of the reference environments / batches):
loeo (12), lobo (3), full (training + new Sol -> outside panels).

Usage: python research/effort_sub.py folds <i>/<n> | aggregate
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import RES, log  # noqa: E402
from research.effort_pair import BATCH, EFF, EFFORTS, ENV, ENVS, ME, OUT_ROWS, ROWS, TR, Y, cells  # noqa: E402
from research.hard_pair import A, S, p  # noqa: E402

SUB = [f"{p.classes[A]}@low"] + [f"{p.classes[S]}@{e}" for e in EFFORTS]
SOLSUB = np.array([False, True, True, True, True])
GROUPS = SOLSUB.astype(int).tolist()
PREFIX = "sub_fold_"
DISPLAYS = {"v5 display (clip 2, x0.65)": (2.0, 0.65), "platt (no clip, x1)": (None, 1.0)}
THRESHOLDS = (0.75, 0.85, 0.9, 0.95)


def sub_rows(idx):
    out = []
    for i in idx:
        r = dict(ROWS[i])
        r["source"] = SUB[0] if Y[i] == 0 else f"{p.classes[S]}@{EFF[i]}"
        assert r["source"] in SUB
        out.append(r)
    return out


def fold_list():
    F = [("loeo", e, np.concatenate([TR[ENV[TR] != e], ME[ENV[ME] != e]]),
          np.concatenate([TR[ENV[TR] == e], ME[ENV[ME] == e]])) for e in ENVS]
    F += [("lobo", b, np.concatenate([TR[BATCH[TR] != b], ME]), TR[BATCH[TR] == b]) for b in sorted(set(BATCH[TR]))]
    F.append(("full", "outside", np.concatenate([TR, ME]), OUT_ROWS))
    return F


def run_fold(j):
    proto, fname, ref, te = fold_list()[j]
    t = time.time()
    G.LDA_EQUAL_PRIORS = True
    cl = G.fit_cluster(sub_rows(ref), SUB, list(range(len(SUB))), member_groups=GROUPS)
    assert cl["members"] == SUB
    s = np.array([G.local_logits(cl, ROWS[i]["numbers"]) for i in te])
    cal = cl["combine"]["calibration"]
    out = {"protocol": proto, "fold": str(fname), "test": te.tolist(), "s": s.tolist(), "clip": cal["clip"],
           "shrink": cal["shrink"], "a": {k: m["a"] for k, m in cal["maps"].items()},
           "b": {k: m["b"] for k, m in cal["maps"].items()},
           "weights": {k: v for k, v in cl["combine"].items() if k.startswith("w_")}}
    (RES / f"{PREFIX}{j:02d}.json").write_text(json.dumps(out, default=float))
    log(f"fold {j} {proto}/{fname}: refs {len(ref)} test {len(te)} ({time.time() - t:.0f}s); "
        f"weights {{{', '.join(f'{k}: {v:.3f}' for k, v in out['weights'].items())}}}; k3 map {cal['maps']['3']}")
    return j


def run_folds(shard):
    import multiprocessing as mp
    i, n = map(int, shard.split("/"))
    J = [j for j in range(len(fold_list())) if j % n == i]
    procs = min(len(J), int(os.environ.get("PROCS", os.cpu_count() or 1)))
    log(f"shard {shard}: folds {J} on {procs} processes")
    with mp.get_context("fork").Pool(procs) as pool:
        for _ in pool.imap_unordered(run_fold, J):
            pass


# ----------------------------------------------------------------------------- aggregate

def load(proto):
    n, K = len(ROWS), len(SUB)
    R = {"s": np.full((n, K), np.nan), **{f"a{k}": np.full(n, np.nan) for k in (1, 2, 3)},
         **{f"b{k}": np.full((n, K), np.nan) for k in (1, 2, 3)}}
    for fp_ in sorted(RES.glob(f"{PREFIX}*.json")):
        r = json.loads(fp_.read_text())
        if r["protocol"] != proto:
            continue
        te = np.array(r["test"], dtype=int)
        R["s"][te] = r["s"]
        for k in (1, 2, 3):
            R[f"a{k}"][te], R[f"b{k}"][te] = r["a"][str(k)], r["b"][str(k)]
    return R


def sol_logodds(R, rows, clip, shrink):
    """rows: (n, m) row indices scored together -> log p(Sol) / p(Astra) per group (map for min(m, 3))."""
    k = min(rows.shape[1], 3)
    c = R["s"][rows] - R["s"][rows].mean(-1, keepdims=True)
    if clip:
        c = np.clip(c, -clip / 2, clip / 2)
    loc = shrink * (R[f"a{k}"][rows].mean(1)[:, None] * c.sum(1) + R[f"b{k}"][rows].mean(1))
    return logsumexp(loc[:, SOLSUB], axis=1) - logsumexp(loc[:, ~SOLSUB], axis=1)


def sample_rows(idx, c, rng, n, m):
    cl = cells(idx, c)
    m = min(m, len(cl))
    lens = np.array([len(x) for x in cl])
    M = np.zeros((len(cl), lens.max()), dtype=int)
    for i, x in enumerate(cl):
        M[i, :len(x)] = x
    perm = np.argsort(rng.random((n, len(cl))), axis=1)[:, :m]
    return M[perm, (rng.random(perm.shape) * lens[perm]).astype(int)]


def report_set(tag, R, idx, labels, rng, n=40000):
    for c in labels:
        lab = "Sol" if c == 1 else "Astra"
        rows_c = idx[Y[idx] == c]
        if not len(rows_c) or np.isnan(R["s"][rows_c]).any():
            continue
        rows = sample_rows(idx, c, rng, n, 9)
        tries = rows.shape[1] // 3
        sign = 1 if c == 1 else -1
        for disp, (clip, shrink) in DISPLAYS.items():
            L1 = sign * sol_logodds(R, rows_c[:, None], clip, shrink)
            Lt = [sign * sol_logodds(R, rows[:, :3 * t], clip, shrink) for t in range(1, tries + 1)]
            for thr in (THRESHOLDS if disp.startswith("platt") else (0.75,)):
                th = np.log(thr / (1 - thr))
                state, when = np.zeros(n, int), np.full(n, tries)
                for t, L in enumerate(Lt, 1):
                    new = (state == 0) & (np.abs(L) >= th)
                    state[new & (L > 0)], state[new & (L < 0)] = 1, -1
                    when[new] = t
                d3 = np.abs(Lt[0]) >= th
                print(f"SEL {tag:34} {lab:5} {disp:27} thr {thr:.2f} | k1 rec {np.mean(L1 > 0):.3f} "
                      f"dec {np.mean(np.abs(L1) >= th):.3f} acc {np.sum(L1 >= th) / max(np.sum(np.abs(L1) >= th), 1):.3f} "
                      f"| k3 rec {np.mean(Lt[0] > 0):.3f} dec {d3.mean():.3f} acc {np.sum(Lt[0] >= th) / max(d3.sum(), 1):.3f} "
                      f"wrong {np.mean(Lt[0] <= -th):.3f} | retry<= {tries - 1}: right {np.mean(state == 1):.3f} "
                      f"wrong {np.mean(state == -1):.3f} undec {np.mean(state == 0):.3f} tries {when.mean():.2f}", flush=True)


def aggregate():
    files = sorted(RES.glob(f"{PREFIX}*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    rng = np.random.default_rng(0)
    R = load("loeo")
    report_set("sub loeo: training", R, TR, (0, 1), rng)
    for e in EFFORTS:
        report_set(f"sub loeo: new Sol {e}", R, ME[EFF[ME] == e], (1,), rng)
    report_set("sub loeo: new Sol all efforts", R, ME, (1,), rng)
    report_set("sub lobo: training", load("lobo"), TR, (0, 1), rng)
    R = load("full")
    report_set("sub full: outside", R, OUT_ROWS, (0, 1), rng)
    for b in ("old_outside", "fresh_outside"):
        report_set(f"sub full: {b}", R, OUT_ROWS[BATCH[OUT_ROWS] == b], (0, 1), rng)
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
