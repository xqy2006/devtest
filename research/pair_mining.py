"""High-dimensional vector mining on the confusable pair (numbers only).

Vector bank per response: all dense statistic families (seqfeatures, seqfeatures2, seqfeatures3)
plus an opening-position vector (one-hot of the value at each of the first OPEN positions).
Pre-processing (fitted on reference rows only): standardize, family weight 1/sqrt(dim),
environment-nuisance projection (rank 2, within-class environment offsets).
Classifiers for many weak, correlated features (hyper-parameters by inner leave-one-environment-
out on the reference rows only):
  pam    nearest shrunken centroids (Tibshirani et al. 2002), threshold chosen by inner CV
  pls    PLS discriminant analysis, number of components chosen by inner CV
  rslda  random-subspace shrinkage-LDA ensemble (Ho 1998 / Skurichina & Duin 2002)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research.seqfeatures2 import SEQ2_FAMILIES  # noqa: E402
from research.seqfeatures3 import SEQ3_FAMILIES  # noqa: E402

p, yb = L.p, L.yb
OPEN = 12
for fams in (SEQ2_FAMILIES, SEQ3_FAMILIES):
    for f, fn in fams.items():
        if f not in L.FEAT:
            L.FEAT[f] = np.zeros((len(p.rows), len(fn(np.asarray(p.rows[L.pair[0]]["numbers"])))))
            for i in L.pair:
                L.FEAT[f][i] = fn(np.asarray(p.rows[i]["numbers"]))
op = np.zeros((len(p.rows), OPEN * 356))
for i in L.pair:
    x = p.rows[i]["numbers"][:OPEN]
    op[i, [t * 356 + v for t, v in enumerate(x)]] = 1.0
L.FEAT["opening"] = op
DENSE = [f for f in L.FEAT if f not in ("opening", "permshape")]


def prepare(families, ref, rank=2):
    X = np.hstack([L.FEAT[f] for f in families])
    w = np.concatenate([np.full(L.FEAT[f].shape[1], 1 / np.sqrt(L.FEAT[f].shape[1])) for f in families])
    mu, sd = X[ref].mean(0), X[ref].std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd * w
    B = L.nuisance_basis(Z[ref], ref, rank, 0)
    return Z - (Z @ B.T) @ B


def inner_folds(ref):
    env = p.env[ref]
    return [(ref[env != e], ref[env == e]) for e in sorted(set(env))]


class PAM:
    GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

    def __init__(self, families):
        self.families = families

    @staticmethod
    def _fit(Z, ref):
        y = yb[ref]
        m = Z[ref].mean(0)
        mc = [Z[ref][y == c].mean(0) for c in (0, 1)]
        resid = np.vstack([Z[ref][y == c] - mc[c] for c in (0, 1)])
        s = np.sqrt((resid ** 2).sum(0) / (len(ref) - 2))
        s0 = np.median(s[s > 1e-12]) if (s > 1e-12).any() else 1.0
        n = len(ref)
        mk = [np.sqrt(1 / (y == c).sum() - 1 / n) for c in (0, 1)]
        d = [(mc[c] - m) / (mk[c] * (s + s0)) for c in (0, 1)]
        return m, d, mk, s, s0

    @staticmethod
    def _decide(Z, idx, model, delta):
        m, d, mk, s, s0 = model
        cent = [m + mk[c] * (s + s0) * np.sign(d[c]) * np.maximum(np.abs(d[c]) - delta, 0) for c in (0, 1)]
        dist = [(((Z[idx] - cent[c]) / (s + s0)) ** 2).sum(1) for c in (0, 1)]
        return (dist[0] - dist[1]) / 2

    def fit(self, ref):
        self.Z = prepare(self.families, ref)
        acc = np.zeros(len(self.GRID))
        for tr, te in inner_folds(ref):
            Zi = prepare(self.families, tr)
            mod = self._fit(Zi, tr)
            for k, dl in enumerate(self.GRID):
                acc[k] += ((self._decide(Zi, te, mod, dl) > 0) == yb[te]).sum()
        self.delta = self.GRID[int(np.argmax(acc))]
        self.model = self._fit(self.Z, ref)
        return self

    def decision(self, idx):
        return self._decide(self.Z, idx, self.model, self.delta)


class PLS:
    GRID = (1, 2, 3, 5, 8)

    def __init__(self, families):
        self.families = families

    def fit(self, ref):
        acc = np.zeros(len(self.GRID))
        for tr, te in inner_folds(ref):
            Zi = prepare(self.families, tr)
            for k, nc in enumerate(self.GRID):
                m = PLSRegression(nc, scale=False).fit(Zi[tr], yb[tr] - 0.5)
                acc[k] += ((m.predict(Zi[te]).ravel() > 0) == yb[te]).sum()
        self.nc = self.GRID[int(np.argmax(acc))]
        self.Z = prepare(self.families, ref)
        self.m = PLSRegression(self.nc, scale=False).fit(self.Z[ref], yb[ref] - 0.5)
        return self

    def decision(self, idx):
        return self.m.predict(self.Z[idx]).ravel()


class RSLDA:
    def __init__(self, families, n_models=150, dims=60, seed=0):
        self.families, self.n, self.k, self.seed = families, n_models, dims, seed

    def fit(self, ref):
        self.Z = prepare(self.families, ref)
        rng = np.random.default_rng(self.seed)
        self.models = []
        for _ in range(self.n):
            cols = rng.choice(self.Z.shape[1], min(self.k, self.Z.shape[1]), replace=False)
            m = LDA(solver="lsqr", shrinkage="auto").fit(self.Z[ref][:, cols], yb[ref])
            sc = np.std(m.decision_function(self.Z[ref][:, cols])) + 1e-9
            self.models.append((cols, m, sc))
        return self

    def decision(self, idx):
        return np.mean([m.decision_function(self.Z[idx][:, c]) / s for c, m, s in self.models], axis=0)


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("bank=")] or ["pam", "pls", "rslda"]
    banks = {"dense": DENSE, "dense+opening": DENSE + ["opening"]}
    sel = [a[5:] for a in sys.argv[1:] if a.startswith("bank=")]
    if sel:
        banks = {k: v for k, v in banks.items() if k in sel}
    out = {}
    print({k: sum(L.FEAT[f].shape[1] for f in v) for k, v in banks.items()}, "dims", flush=True)
    for bank, fams in banks.items():
        for name in which:
            mk = {"pam": lambda: PAM(fams), "pls": lambda: PLS(fams), "rslda": lambda: RSLDA(fams)}[name]
            t = time.time()
            r = L.evaluate(mk)
            out[f"{name}[{bank}]"] = r
            print(L.line(f"{name}[{bank}]", r) + f" ({time.time()-t:.0f}s)", flush=True)
    Path(__file__).with_name(f"out_pair_mining_{'_'.join(which)}.json").write_text(json.dumps(out, indent=1))
