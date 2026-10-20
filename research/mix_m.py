"""Run 39 (2026-10-20): the run 34 mixture library with m responses per try (reads run 34's per-row results).

Run 32: with the binary v5 library (low effort only) about 6 responses per try meet the REPORT 19.21 requirement
at low effort, but GPT-6.1 Sol at medium / high / xhigh is called Astra (run 21), and more responses only make that
surer. Run 34's mixture library (Astra@low, Sol@low / medium / high / xhigh) recognises Sol at every effort, at a
cost to low-effort Sol with 3 responses. Here: the same per-row results (mix_fold_*.json of run 34, downloaded into
results/), m = 3..6 responses per try, all of one query's responses at one effort, from m different environments;
shown when the larger probability >= the threshold, otherwise m more are added (earlier ones kept) while
environments last (at most 3 tries). Maps: the fold's k = 3 map for any number of responses, as the deliverable.

Usage: python research/mix_m.py   (results/mix_fold_*.json present)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.effort_mix as EM  # noqa: E402
from research.effort_pair import cells  # noqa: E402

U, ENV, BATCH, EFF, Y, ENVS, EFFORTS, ALL = EM.U, EM.ENV, EM.BATCH, EM.EFF, EM.Y, EM.ENVS, EM.EFFORTS, EM.ALL
DISPLAYS = {"platt x0.65": 0.65, "platt x1": 1.0}
THRESHOLDS = (0.75, 0.85, 0.9)


def sim(R, idx, c, m, shrink, thr, rng, n=40000):
    cl = cells(idx, c)
    tries = min(3, len(cl) // m)
    lens = np.array([len(x) for x in cl])
    M = np.zeros((len(cl), lens.max()), dtype=int)
    for i, x in enumerate(cl):
        M[i, :len(x)] = x
    perm = np.argsort(rng.random((n, len(cl))), axis=1)[:, :m * tries]
    rows = M[perm, (rng.random(perm.shape) * lens[perm]).astype(int)]
    th = np.log(thr / (1 - thr))
    sign = 1 if c == 1 else -1
    state, when, first = np.zeros(n, int), np.full(n, tries), None
    for t in range(1, tries + 1):
        L = sign * EM.logodds(R, rows[:, :m * t], shrink)
        new = (state == 0) & (np.abs(L) >= th)
        state[new & (L > 0)], state[new & (L < 0)] = 1, -1
        when[new] = t
        if t == 1:
            first = (np.mean(np.abs(L) >= th), np.sum(L >= th) / max(np.sum(np.abs(L) >= th), 1))
    return first + (np.mean(state == -1), np.mean(state == 0), tries)


def report(tag, R, idx, labels, rng):
    for m in (3, 4, 5, 6):
        for disp, shrink in DISPLAYS.items():
            for thr in THRESHOLDS:
                out = []
                for c in labels:
                    rows_c = idx[Y[idx] == c]
                    if not len(rows_c) or np.isnan(R["s"][rows_c]).any():
                        continue
                    o = sim(R, idx, c, m, shrink, thr, rng)
                    out.append(f"{'Sol' if c else 'Astra'}: first decide {o[0]:.3f} acc {o[1]:.3f} final wrong "
                               f"{o[2]:.3f} undecided {o[3]:.3f} (tries <= {o[4]})")
                print(f"MM {tag:30} m {m} {disp:11} thr {thr:.2f} | " + " | ".join(out), flush=True)


def main():
    files = sorted(EM.RES.glob(f"{EM.PREFIX}*.json"))
    assert len(files) == len(EM.fold_list()), f"{len(files)} fold files"
    rng = np.random.default_rng(0)
    inenv = U[np.isin(ENV[U], ENVS)]
    me = ALL[BATCH[ALL] == "multieffort_20261019"]
    for v in EM.VARIANTS:
        R = EM.load(f"loeo|{v}")
        report(f"loeo|{v}: union rows", R, inenv, (0, 1), rng)
        for e in EFFORTS[1:]:
            report(f"loeo|{v}: new Sol {e}", R, me[EFF[me] == e], (1,), rng)
        R = EM.load(f"lobo|{v}")
        report(f"lobo|{v}: union rows", R, U, (0, 1), rng)


if __name__ == "__main__":
    main()
