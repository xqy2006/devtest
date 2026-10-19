"""Run 26 (2026-10-19): where do the existing batches sit on GPT-6.1 Sol's reasoning-effort axis?

Astra's train_original rows (2026-09-05) carry no reasoning-effort field, all other pair rows say "low", and that
batch was the easiest one in every leave-one-batch-out run (k3 0.958 when held out). If it was collected at a higher
effort and GPT-6 Astra reacts to effort like GPT-6.1 Sol, its rows should sit further along Sol's effort axis than
Astra's low-effort batches.

Vectors: the 195-dim residual vector (label-free Plackett-Luce weights fit on all pair rows; standardized).
Effort axis: from the new Sol rows only, shrinkage LDA (0.5) of low vs high + xhigh, and the within-prompt mean
difference xhigh - low (paired by environment) whitened by the training rows' within-batch covariance (shrunk).
Model axis: shrinkage LDA (0.5) Astra vs Sol on the low-effort training rows.
Reported: mean projection (and SE) per model x batch / effort, scaled so that new Sol low = 0 and new Sol xhigh = 1;
cosine between the effort and model axes; leave-one-environment-out AUC of "train_original vs the other batches"
within each model (shrinkage LDA on the vector).

Usage: python research/effort_probe.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.covariance import shrunk_covariance  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import log  # noqa: E402
from research.effort_pair import BATCH, EFF, EFFORTS, ENV, ME, OUT_ROWS, ROWS, TR, Y  # noqa: E402
from research.runner_pair import auc  # noqa: E402


def main():
    use = np.concatenate([TR, OUT_ROWS, ME])
    seqs = {i: list(map(int, ROWS[i]["numbers"])) for i in use}
    pi = G.fit_pl([seqs[i] for i in np.concatenate([TR, ME])])
    X = np.full((len(ROWS), 195), np.nan)
    X[use] = G.resid_vectors(pi, [seqs[i] for i in use])
    mu, sd = X[TR].mean(0), X[TR].std(0) + 1e-9
    Z = (X - mu) / sd
    sol_me = ME[Y[ME] == 1]
    low, hx = sol_me[EFF[sol_me] == "low"], sol_me[np.isin(EFF[sol_me], ["high", "xhigh"])]
    xh = sol_me[EFF[sol_me] == "xhigh"]

    axes = {}
    m = LDA(solver="lsqr", shrinkage=0.5).fit(Z[np.concatenate([low, hx])],
                                              np.r_[np.zeros(len(low)), np.ones(len(hx))])
    axes["effort LDA (Sol low vs high+xhigh)"] = m.coef_[0]
    pairs = [(a, b) for a in low for b in xh if ENV[a] == ENV[b]]
    delta = np.mean([Z[b] - Z[a] for a, b in pairs], 0)
    R = np.vstack([Z[TR[(Y[TR] == c) & (BATCH[TR] == bt)]] - Z[TR[(Y[TR] == c) & (BATCH[TR] == bt)]].mean(0)
                   for c in (0, 1) for bt in sorted(set(BATCH[TR]))])
    C = shrunk_covariance(R.T @ R / len(R), 0.5)
    axes["effort paired xhigh - low (whitened)"] = np.linalg.solve(C, delta)
    mm = LDA(solver="lsqr", shrinkage=0.5).fit(Z[TR], Y[TR])
    axes["model LDA (Sol - Astra, training)"] = mm.coef_[0]
    names = list(axes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = axes[names[i]], axes[names[j]]
            log(f"cosine({names[i]}, {names[j]}) = {a @ b / np.linalg.norm(a) / np.linalg.norm(b):+.3f}")
    groups = []
    for c, name in ((0, "Astra"), (1, "Sol")):
        for bt in ("train_original", "train_supplemental", "train_codex_20261011", "old_outside", "fresh_outside"):
            rows = np.flatnonzero((Y == c) & (BATCH == bt))
            rows = rows[np.isin(rows, use)]
            if len(rows):
                groups.append((f"{name} {bt} (effort {','.join(sorted(set(EFF[rows])))})", rows))
    for e in EFFORTS:
        groups.append((f"new Sol {e}", sol_me[EFF[sol_me] == e]))
    for name, w in axes.items():
        s = Z[use] @ w
        proj = np.full(len(ROWS), np.nan)
        proj[use] = s
        lo, hi = np.mean(proj[low]), np.mean(proj[xh])
        log(f"axis: {name} (scaled: new Sol low = 0, new Sol xhigh = 1)")
        for g, rows in groups:
            v = (proj[rows] - lo) / (hi - lo)
            print(f"PROJ {name[:12]:12} {g:58} n {len(rows):3} mean {v.mean():+.3f} se {v.std(ddof=1) / np.sqrt(len(rows)):.3f}",
                  flush=True)
    for c, name in ((0, "Astra"), (1, "Sol")):
        rows = TR[Y[TR] == c]
        lab = (BATCH[rows] == "train_original").astype(int)
        sc = np.zeros(len(rows))
        for e in sorted(set(ENV[rows])):
            te, tr = ENV[rows] == e, ENV[rows] != e
            Ztr = Z[rows[tr]]
            sc[te] = LDA(solver="lsqr", shrinkage=0.5).fit(Ztr, lab[tr]).decision_function(Z[rows[te]])
        log(f"{name}: train_original vs other training batches, LOEO AUC {auc(sc, lab):.3f}")
    log("done")


if __name__ == "__main__":
    main()
