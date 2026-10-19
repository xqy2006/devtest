"""Run 34 (2026-10-19): model x effort subclasses as a mixture with a two-class Platt map.

Run 33 (effort_union.py) fitted a Platt intercept per subclass and summed the subclass probabilities per model;
with similar subclasses those intercepts are not identified (fitted values near -4, slopes 0.14-0.2), and the fork
evidence, split over five subclasses, got a negative weight. Subclass mode of the deliverable now:
  - fork evidence fit per model and shared by the model's subclasses (it was the evidence least moved by effort);
  - combination weights by the likelihood of the true model, each model's subclasses equally likely;
  - k responses of one query share one effort: per subclass the k responses' logits are summed, the model's
    log-likelihood is the log-mean-exp over its subclasses, and one two-class Platt map (a_k, b_k) of the
    resulting log-odds is fitted on inner leave-one-environment-out groups (G.mixture_logodds, _fit_mixture_map).
Library: all low-effort pair rows (621) + new Sol medium / high / xhigh (35). Protocols as run 33 (loeo 12, lobo 7
batches; holding out multieffort_20261019 removes all new Sol rows). Variants: subclass mean shrinkage 12 / 48.

Usage: python research/effort_mix.py folds <i>/<n> | aggregate
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.union_pair as UP  # noqa: E402  (extends effort_pair's rows; import before effort_sub)
import research.effort_sub as ES  # noqa: E402
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import RES, log  # noqa: E402
from research.effort_pair import cells, run_parallel  # noqa: E402

U, HI, ENV, BATCH, EFF, TR, Y, ENVS, EFFORTS = UP.U, UP.HI, UP.ENV, UP.BATCH, UP.EFF, UP.TR, UP.Y, UP.ENVS, UP.EFFORTS
ROWS = UP.ROWS
ALL = np.concatenate([U, HI])
SUB, GROUPS = ES.SUB, np.array(ES.GROUPS)
VARIANTS = {"lam12": 12.0, "lam48": 48.0}
PREFIX = "mix_fold_"
DISPLAYS = {"platt x0.65": 0.65, "platt (no clip, x1)": 1.0}
THRESHOLDS = (0.75, 0.85, 0.9, 0.95)


def fold_list():
    F = []
    for v in VARIANTS:
        F += [(f"loeo|{v}", e, ALL[ENV[ALL] != e], ALL[ENV[ALL] == e]) for e in ENVS]
        F += [(f"lobo|{v}", b, ALL[BATCH[ALL] != b], ALL[BATCH[ALL] == b]) for b in sorted(set(BATCH[U]))]
    return F


def run_fold(j):
    proto, fname, ref, te = fold_list()[j]
    t = time.time()
    G.SUB_MEAN_LAMBDA = VARIANTS[proto.split("|")[1]]
    cl = G.fit_cluster(ES.sub_rows(ref), SUB, list(range(len(SUB))), member_groups=GROUPS.tolist())
    assert cl["members"] == SUB
    s = np.array([G.local_logits(cl, ROWS[i]["numbers"]) for i in te])
    maps = cl["combine"]["calibration"]["maps"]
    out = {"protocol": proto, "fold": str(fname), "test": te.tolist(), "s": s.tolist(),
           "a": {k: m["a"] for k, m in maps.items()}, "b": {k: m["b"] for k, m in maps.items()},
           "weights": {k: v for k, v in cl["combine"].items() if k.startswith("w_")}}
    (RES / f"{PREFIX}{j:02d}.json").write_text(json.dumps(out, default=float))
    log(f"fold {j} {proto}/{fname}: refs {len(ref)} test {len(te)} ({time.time() - t:.0f}s); "
        f"weights {{{', '.join(f'{k}: {v:.3f}' for k, v in out['weights'].items())}}}; "
        f"maps {{{', '.join(f'{k}: ({m[chr(97)]:.3f}, {m[chr(98)]:+.3f})' for k, m in maps.items())}}}")
    return j


def run_folds(shard):
    run_parallel(run_fold, len(fold_list()), shard)


def load(proto):
    n, K = len(ROWS), len(SUB)
    R = {"s": np.full((n, K), np.nan), **{f"a{k}": np.full(n, np.nan) for k in (1, 2, 3)},
         **{f"b{k}": np.full(n, np.nan) for k in (1, 2, 3)}}
    for fp_ in sorted(RES.glob(f"{PREFIX}*.json")):
        r = json.loads(fp_.read_text())
        if r["protocol"] != proto:
            continue
        te = np.array(r["test"], dtype=int)
        R["s"][te] = r["s"]
        for k in (1, 2, 3):
            R[f"a{k}"][te], R[f"b{k}"][te] = r["a"][str(k)], r["b"][str(k)]
    return R


def logodds(R, rows, shrink):
    """rows (n, m) scored together -> calibrated log p(Sol) / p(Astra) (map for min(m, 3))."""
    k = min(rows.shape[1], 3)
    M = G.mixture_logodds(R["s"][rows].sum(1), GROUPS)
    return shrink * (R[f"a{k}"][rows].mean(1) * M + R[f"b{k}"][rows].mean(1))


def report_set(tag, R, idx, labels, rng, n=40000):
    for c in labels:
        lab = "Sol" if c == 1 else "Astra"
        rows_c = idx[Y[idx] == c]
        if not len(rows_c) or np.isnan(R["s"][rows_c]).any():
            continue
        cl = cells(idx, c)
        m = min(9, len(cl))
        lens = np.array([len(x) for x in cl])
        Mx = np.zeros((len(cl), lens.max()), dtype=int)
        for i, x in enumerate(cl):
            Mx[i, :len(x)] = x
        perm = np.argsort(rng.random((n, len(cl))), axis=1)[:, :m]
        rows = Mx[perm, (rng.random(perm.shape) * lens[perm]).astype(int)]
        tries = m // 3
        sign = 1 if c == 1 else -1
        for disp, shrink in DISPLAYS.items():
            L1 = sign * logodds(R, rows_c[:, None], shrink)
            Lt = [sign * logodds(R, rows[:, :3 * t], shrink) for t in range(1, tries + 1)]
            for thr in THRESHOLDS:
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
    inenv = U[np.isin(ENV[U], ENVS)]
    me = ALL[BATCH[ALL] == "multieffort_20261019"]
    for v in VARIANTS:
        R = load(f"loeo|{v}")
        report_set(f"mx loeo|{v}: union rows", R, inenv, (0, 1), rng)
        report_set(f"mx loeo|{v}: training rows", R, TR, (0, 1), rng)
        for e in EFFORTS:
            report_set(f"mx loeo|{v}: new Sol {e}", R, me[EFF[me] == e], (1,), rng)
        R = load(f"lobo|{v}")
        report_set(f"mx L|{v}: union rows", R, U, (0, 1), rng)
        for b in sorted(set(BATCH[U])):
            report_set(f"mx L|{v}: {b[:20]}", R, U[BATCH[U] == b], (0, 1), rng)
        for e in EFFORTS:
            report_set(f"mx L|{v}: Sol {e} unseen", R, me[EFF[me] == e], (1,), rng)
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
