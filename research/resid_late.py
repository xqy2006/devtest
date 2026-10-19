"""Run 27 (2026-10-19): late-weighted versions of the residual vector (low effort, quick check of one component).

The 195-dim residual vector averages its statistics with weights that decay from the start of the response
(time scales 16 / 64) or are flat. The N=1000 pilot carried most of its evidence late in the response, and a
response's end is where the shared preference has been used up. Candidates, all from the numbers only:
  late99   the 99 time statistics with weights decaying from the end (time scales 16 / 64; the flat part repeats)
  late96   the Plackett-Luce residuals accumulated with weights decaying from the end of the first-occurrence order
  quarters the flat time statistics of each quarter of the response (4 x 33)
Each set is scored alone by the same standardize + shrinkage LDA (0.5) as the deliverable's residual evidence,
on the low-effort pair rows: leave one environment out, leave one training batch out, all training -> outside.
Reported: AUC and k3 recall at threshold 0 (all same-label triples from three environments, worse label).

Usage: python research/resid_late.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import lda05, log  # noqa: E402
from research.effort_pair import BATCH, ENV, ENVS, OUT_ROWS, ROWS, TR, Y  # noqa: E402
from research.runner_pair import auc  # noqa: E402


def late96(x, pi):
    z = G._first_order(x)
    w = pi[z]
    P1, P2 = w[:, None] * G._PHI[z], w[:, None] * G._PHI[z] ** 2
    S0 = pi.sum() - (np.cumsum(w) - w)
    E = (G._PHI.T @ pi - (np.cumsum(P1, 0) - P1)) / S0[:, None]
    Var = np.maximum(((G._PHI ** 2).T @ pi - (np.cumsum(P2, 0) - P2)) / S0[:, None] - E ** 2, 0.0)
    r = G._PHI[z] - E
    back = np.arange(len(z))[::-1]
    out = []
    for tau in (16.0, 64.0):
        g = np.exp(-back / tau)
        out.append((g @ r) / np.sqrt(0.1 + (g ** 2) @ Var))
    return np.concatenate(out)


def late99(x):
    return G.time_stats(list(x)[::-1])[:66]


def quarters(x):
    x = list(x)
    q = len(x) // 4
    return np.concatenate([G.time_stats(x[i * q:(i + 1) * q] if i < 3 else x[3 * q:])[66:] for i in range(4)])


def k3_worse(score, idx):
    rec = []
    for c in (0, 1):
        cells = [score[idx[(ENV[idx] == e) & (Y[idx] == c)]] for e in ENVS]
        cells = [v for v in cells if len(v)]
        sgn = 1 if c else -1
        tot = hit = 0
        for a, b, d in itertools.combinations(range(len(cells)), 3):
            s = cells[a][:, None, None] + cells[b][None, :, None] + cells[d][None, None, :]
            hit += int(np.sum(sgn * s > 0))
            tot += s.size
        rec.append(hit / tot)
    return min(rec)


def main():
    use = np.concatenate([TR, OUT_ROWS])
    seqs = {i: list(map(int, ROWS[i]["numbers"])) for i in use}
    pi = G.fit_pl([seqs[i] for i in TR])
    base = {i: np.concatenate([G.pl_residual(seqs[i], pi), G.time_stats(seqs[i])]) for i in use}
    extra = {"late99": {i: late99(seqs[i]) for i in use}, "late96": {i: late96(seqs[i], pi) for i in use},
             "quarters": {i: quarters(seqs[i]) for i in use}}
    sets = {"base195": [], "late99": ["late99"], "late96": ["late96"], "late99+late96": ["late99", "late96"],
            "quarters": ["quarters"], "all": ["late99", "late96", "quarters"]}
    for name, parts in sets.items():
        for mode in ("with base", "alone"):
            if name == "base195" and mode == "alone":
                continue
            def vec(i, parts=parts, mode=mode):
                v = [extra[q][i] for q in parts]
                return np.concatenate(([base[i]] if mode == "with base" or not parts else []) + v)
            X = {i: vec(i) for i in use}
            res = {}
            for proto, folds in (("loeo", [(TR[ENV[TR] != e], TR[ENV[TR] == e]) for e in ENVS]),
                                 ("lobo", [(TR[BATCH[TR] != b], TR[BATCH[TR] == b]) for b in sorted(set(BATCH[TR]))]),
                                 ("outside", [(TR, OUT_ROWS)])):
                sc = np.full(len(ROWS), np.nan)
                for ref, te in folds:
                    sc[te] = lda05(np.array([X[i] for i in ref]), Y[ref], np.array([X[i] for i in te]))
                idx = OUT_ROWS if proto == "outside" else TR
                res[proto] = (auc(sc[idx], Y[idx]), k3_worse(sc, idx))
            print(f"LATE {name:15} {mode:9} dims {len(X[use[0]]):4} | " + " | ".join(
                f"{k} AUC {a:.3f} k3w {b:.3f}" for k, (a, b) in res.items()), flush=True)
    log("done")


if __name__ == "__main__":
    main()
