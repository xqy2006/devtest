"""Run 32 (2026-10-19): how many responses per try would the low-effort pair need for the REPORT 19.21 target?

Reads the per-row results of run 31 (union_pair.py: deliverable v5, references = all 621 low-effort pair rows,
leave one environment / one collection batch out; downloaded into results/). For m = 3..6 responses per try, from m
different environments, same model: the attribution is shown when the larger probability is >= the threshold,
otherwise m more responses are added (earlier ones kept) while environments last (12 in all). The k = 3 map of
each row's fold is used for any number of responses, as the deliverable does. Displays: the deliverable's (clip 2,
x0.65) and plain Platt; thresholds 0.75 / 0.85 / 0.9.

Usage: python research/kneeded.py   (results/union_fold_*.json present)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.union_pair as UP  # noqa: E402
from research.effort_pair import cells, load  # noqa: E402

DISPLAYS = {"v5 display": (2.0, 0.65), "platt": (None, 1.0)}


def sim(R, idx, c, m, clip, shrink, thr, rng, n=40000):
    cl = cells(idx, c)
    tries = len(cl) // m
    lens = np.array([len(x) for x in cl])
    M = np.zeros((len(cl), lens.max()), dtype=int)
    for i, x in enumerate(cl):
        M[i, :len(x)] = x
    perm = np.argsort(rng.random((n, len(cl))), axis=1)[:, :m * tries]
    rows = M[perm, (rng.random(perm.shape) * lens[perm]).astype(int)]
    d = R["d"][rows]
    d = np.clip(d, -clip, clip) if clip else d
    t = np.log(thr / (1 - thr))
    state, when, first = np.zeros(n, int), np.full(n, tries), None
    for k in range(1, tries + 1):
        s = slice(0, m * k)
        L = shrink * (R["a3"][rows[:, s]].mean(1) * d[:, s].sum(1) + R["b3"][rows[:, s]].mean(1))
        L = L if c == 1 else -L
        new = (state == 0) & (np.abs(L) >= t)
        state[new & (L > 0)], state[new & (L < 0)] = 1, -1
        when[new] = k
        if k == 1:
            first = (np.abs(L) >= t, L >= t)
    return first[0].mean(), first[1].sum() / max(first[0].sum(), 1), np.mean(state == -1), np.mean(state == 0), tries


def main():
    rng = np.random.default_rng(0)
    for proto in ("loeo", "lobo"):
        R = load(proto, UP.PREFIX)
        idx = UP.U[~np.isnan(R["d"][UP.U])]
        for m in (3, 4, 5, 6):
            for disp, (clip, shrink) in DISPLAYS.items():
                for thr in ((0.75,) if disp == "v5 display" else (0.75, 0.85, 0.9)):
                    out = [sim(R, idx, c, m, clip, shrink, thr, rng) for c in (0, 1)]
                    print(f"KNEED {proto} m {m} {disp:10} thr {thr:.2f} | " + " | ".join(
                        f"{lab}: first decide {o[0]:.3f} acc {o[1]:.3f} final wrong {o[2]:.3f} undecided {o[3]:.3f} "
                        f"(tries <= {o[4]})" for lab, o in zip(("Astra", "Sol"), out)), flush=True)


if __name__ == "__main__":
    main()
