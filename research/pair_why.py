"""Why the method separates other models but not the confusable pair: every class pair compared.

Historical training batch only (train_original: 36 responses per class, 12 environments x 3), so all
136 class pairs have the same amount of data. For each pair:
  k1        single-response accuracy, leave-one-environment-out, dense LDA of the v2 local model
            (value355, ordered, digits, jump1, moments; standardize, env projection rank 2)
  k1_value  same with the value histogram (value355) alone: "which numbers the model likes"
  snr       first-order signal-to-noise: Hellinger distance between the two classes' pooled value
            distributions divided by the mean split-half distance within each class (random halves,
            same sizes). snr ~ 1 means the two pooled number preferences differ no more than two
            halves of the same model.
"""
from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import extended_panel  # noqa: E402

p = extended_panel()
z = np.load(Path(__file__).with_name("cache_all_seqfeatures.npz"))
FAMS = ["value355", "ordered", "digits", "jump1", "moments"]
X = np.hstack([z[f] for f in FAMS])
W = np.concatenate([np.full(z[f].shape[1], 1 / np.sqrt(z[f].shape[1])) for f in FAMS])
XV = z["value355"]
TR = np.flatnonzero(p.role == "train_original")
C = len(p.classes)
COUNTS = np.zeros((len(p.rows), 355))
for i in TR:
    np.add.at(COUNTS[i], np.asarray(p.rows[i]["numbers"]) - 1, 1)


def k1(cols_X, w, idx, y):
    d = np.zeros(len(idx))
    env = p.env[idx]
    for e in sorted(set(env)):
        tr, te = env != e, env == e
        mu, sd = cols_X[idx[tr]].mean(0), cols_X[idx[tr]].std(0)
        sd[sd < 1e-12] = 1
        Z = (cols_X[idx] - mu) / sd * w
        offs = []
        for c in (0, 1):
            m = tr & (y == c)
            for k in sorted(set(env[m])):
                offs.append(Z[m & (env == k)].mean(0) - Z[m].mean(0))
        _, _, vt = np.linalg.svd(np.array(offs), full_matrices=False)
        B = vt[:2]
        Z = Z - (Z @ B.T) @ B
        d[te] = LDA(solver="lsqr", shrinkage="auto").fit(Z[tr], y[tr]).decision_function(Z[te])
    return float(np.mean((d > 0) == y))


def hell(a, b):
    a, b = a / a.sum(), b / b.sum()
    return np.sqrt(0.5 * ((np.sqrt(a) - np.sqrt(b)) ** 2).sum())


rng = np.random.default_rng(0)


def split_half(rows_c, reps=50):
    out = []
    for _ in range(reps):
        perm = rng.permutation(rows_c)
        h = len(perm) // 2
        out.append(hell(COUNTS[perm[:h]].sum(0), COUNTS[perm[h:2 * h]].sum(0)))
    return float(np.mean(out))


if __name__ == "__main__":
    rows_c = {c: TR[p.y[TR] == c] for c in range(C)}
    noise = {c: split_half(rows_c[c]) for c in range(C)}
    res = []
    for a, b in itertools.combinations(range(C), 2):
        idx = np.concatenate([rows_c[a], rows_c[b]])
        y = (p.y[idx] == b).astype(int)
        # between-class distance with the same half sizes as the split-half reference
        h = len(rows_c[a]) // 2
        between = np.mean([hell(COUNTS[rng.permutation(rows_c[a])[:h]].sum(0),
                                COUNTS[rng.permutation(rows_c[b])[:h]].sum(0)) for _ in range(50)])
        res.append({"a": p.classes[a], "b": p.classes[b], "k1": k1(X, W, idx, y),
                    "k1_value": k1(XV, np.full(355, 1 / np.sqrt(355)), idx, y),
                    "snr": float(between / ((noise[a] + noise[b]) / 2))})
        print(f"{res[-1]['a']:26} {res[-1]['b']:26} k1 {res[-1]['k1']:.3f} value {res[-1]['k1_value']:.3f} snr {res[-1]['snr']:.2f}", flush=True)
    Path(__file__).with_name("out_pair_why.json").write_text(json.dumps(res, indent=1))
    res.sort(key=lambda r: r["k1"])
    print("\nhardest 10 pairs:")
    for r in res[:10]:
        print(f"  {r['a']:26} {r['b']:26} k1 {r['k1']:.3f} value {r['k1_value']:.3f} snr {r['snr']:.2f}")
    k = np.array([r["k1"] for r in res]); s = np.array([r["snr"] for r in res])
    print(f"\nall {len(res)} pairs: median k1 {np.median(k):.3f}, median snr {np.median(s):.2f}; pairs with k1<0.9: {(k<0.9).sum()}")
