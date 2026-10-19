"""Run 36 (2026-10-19): would more low-effort references close the gap? A learning curve and its asymptote.

The REPORT 19.21 target needs a single-response d' of about 1.9 (AUC about 0.91) at three responses per try; the
low-effort pair is at AUC 0.80-0.82. Proxy for the deliverable: its residual evidence (the 195-dim vector,
standardize + shrinkage LDA 0.5; the component with the largest weight), on all 621 low-effort pair rows
(union_pair.U). Leave one environment out; the references of each fold are a random fraction f of the other
environments' rows (per model), f in 0.15 ... 1.0, 8 repeats. Fit d'(n) = d_inf * n / (n + n0) (n = references per
model) by least squares on the mean d' (d' = sqrt(2) * Phi^-1(AUC)), and report d_inf, its AUC, and the number
of references per model at which d' would reach 1.9 if the fit held.

Usage: python research/ref_ceiling.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.union_pair as UP  # noqa: E402
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import lda05, log  # noqa: E402
from research.runner_pair import auc  # noqa: E402

U, ENV, Y, ROWS, ENVS = UP.U, UP.ENV, UP.Y, UP.ROWS, UP.ENVS


def main():
    U_ = U[np.isin(ENV[U], ENVS)]
    seqs = [list(map(int, ROWS[i]["numbers"])) for i in U_]
    pi = G.fit_pl(seqs)
    X = G.resid_vectors(pi, seqs)
    env, y = ENV[U_], Y[U_]
    rng = np.random.default_rng(0)
    fracs = (0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
    res = []
    for f in fracs:
        aucs, ns = [], []
        for rep in range(8 if f < 1 else 1):
            sc = np.zeros(len(U_))
            nref = []
            for e in ENVS:
                te = env == e
                tr = np.flatnonzero(~te)
                keep = np.concatenate([rng.choice(tr[y[tr] == c], max(5, int(round(f * np.sum(y[tr] == c)))),
                                                  replace=False) for c in (0, 1)])
                nref.append(len(keep) / 2)
                sc[te] = lda05(X[keep], y[keep], X[te])
            aucs.append(auc(sc, y))
            ns.append(np.mean(nref))
        a = float(np.mean(aucs))
        res.append((float(np.mean(ns)), a, float(np.std(aucs))))
        print(f"CURVE frac {f:.2f} refs per model {np.mean(ns):6.1f} | LOEO AUC {a:.4f} (sd {np.std(aucs):.4f}) "
              f"| d' {np.sqrt(2) * norm.ppf(a):.3f}", flush=True)
    n = np.array([r[0] for r in res])
    d = np.sqrt(2) * norm.ppf(np.array([r[1] for r in res]))
    for name, fn, p0 in (("d_inf * n / (n + n0)", lambda n, di, n0: di * n / (n + n0), (1.5, 100.0)),
                         ("d_inf - c / sqrt(n)", lambda n, di, c: di - c / np.sqrt(n), (1.5, 5.0))):
        p, _ = curve_fit(fn, n, d, p0=p0, maxfev=20000)
        di = p[0]
        need = np.nan
        if di > 1.9:
            grid = np.logspace(1, 7, 2000)
            hit = grid[fn(grid, *p) >= 1.9]
            need = hit[0] if len(hit) else np.inf
        print(f"FIT {name:22} params {np.round(p, 3).tolist()} | d_inf {di:.3f} -> AUC {norm.cdf(di / np.sqrt(2)):.3f} "
              f"| refs per model for d' 1.9: {need}", flush=True)
    log("done")


if __name__ == "__main__":
    main()
