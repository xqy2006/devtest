"""Cheap optimisation round for the newest batch (10-11), using cached per-fold score vectors
(pair_fisher.py, shared model fitted on each fold's reference rows) and cached dense families.

Protocol: pooled leave-one-environment-out over all training batches (the protocol on which the
10-11 rows reach their best accuracy); results reported for the 10-11 rows and for all rows.
All variants use one weighted shrinkage LDA (equal weights reproduce sklearn's lsqr/auto LDA up to
the prior term), so that reference weighting can be tested.

Variants
  D, F, D+F       dense v3 families / score[val+dg+pb] / both (early fusion)
  recency g       references of older batches get weight g (newest batch 1): drift prior
  pcaR            project out the top R within-class principal components (response-level
                  over-dispersion) after the environment projection
  struct          value-score projected on structured bases (last digit, 36-band, mod 3/4/7)
  k3 len          triple decision = length-weighted sum (longer responses carry more evidence)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research import fisher_score as FS  # noqa: E402
from research.gated_fingerprint import LOCAL_FAMILIES  # noqa: E402

p, yb = L.p, L.yb
NPZ = np.load(Path(__file__).with_name("cache_pair_fisher.npz"))
POS = np.full(len(p.rows), -1)
POS[L.pair] = np.arange(len(L.pair))
TLEN = np.array([len(r["numbers"]) for r in p.rows], float)
NEW = L.PAIR_ROLE
vals = np.arange(1, 356)
PHI = np.hstack([np.eye(10)[vals % 10], np.eye(36)[(vals - 1) * 36 // 355],
                 np.eye(3)[vals % 3], np.eye(4)[vals % 4], np.eye(7)[vals % 7]])


def score_matrix(ref, groups):
    U = NPZ[str(hash(tuple(sorted(ref.tolist()))))]
    cols = [U[:, FS.OFF[g]:FS.OFF[g] + FS.GROUPS[g]] for g in groups if g != "struct"]
    if "struct" in groups:
        cols.append(U[:, FS.OFF["val"]:FS.OFF["val"] + 355] @ PHI)
    X = np.zeros((len(p.rows), sum(c.shape[1] for c in cols)))
    X[L.pair] = np.hstack(cols)
    w = np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols])
    return X, w


def wlda(Z, y, w):
    """Weighted binary LDA with Ledoit-Wolf shrinkage in standardized coordinates."""
    m = [np.average(Z[y == c], axis=0, weights=w[y == c]) for c in (0, 1)]
    R = Z - np.where(y[:, None] == 1, m[1], m[0])
    sd = np.sqrt(np.average(R ** 2, axis=0, weights=w)) + 1e-12
    Rs = R / sd * np.sqrt(w / w.mean())[:, None]
    S = ledoit_wolf(Rs, assume_centered=True)[0]
    coef = np.linalg.solve(S, (m[1] - m[0]) / sd) / sd
    return coef, -coef @ (m[0] + m[1]) / 2


class Model:
    def __init__(self, parts=("D",), recency=1.0, pca=0):
        self.parts, self.g, self.pca = parts, recency, pca

    def fit(self, ref):
        Xs, ws = [], []
        if "D" in self.parts:
            X, w = L.design(list(LOCAL_FAMILIES))
            Xs.append(X); ws.append(w)
        for key, groups in (("F", ("val", "dg", "pb")), ("S", ("struct", "dg", "pb")), ("FS", ("val", "struct", "dg", "pb"))):
            if key in self.parts:
                X, w = score_matrix(ref, groups)
                Xs.append(X); ws.append(w)
        self.X, self.w = np.hstack(Xs), np.concatenate(ws)
        rw = np.where(p.role[ref] == NEW, 1.0, self.g)
        self.mu = np.average(self.X[ref], axis=0, weights=rw)
        self.sd = np.sqrt(np.average((self.X[ref] - self.mu) ** 2, axis=0, weights=rw))
        self.sd[self.sd < 1e-12] = np.inf
        Z = (self.X[ref] - self.mu) / self.sd * self.w
        B = L.nuisance_basis(Z, ref, 2, 0)
        Z = Z - (Z @ B.T) @ B
        if self.pca:
            y = yb[ref]
            R = Z - np.where(y[:, None] == 1, Z[y == 1].mean(0), Z[y == 0].mean(0))
            _, _, vt = np.linalg.svd(R, full_matrices=False)
            B = np.vstack([B, vt[:self.pca]])
            q, _ = np.linalg.qr(B.T)
            B = q.T
            Z = (self.X[ref] - self.mu) / self.sd * self.w
            Z = Z - (Z @ B.T) @ B
        self.B = B
        self.coef, self.icpt = wlda(Z, yb[ref], rw)
        return self

    def decision(self, idx):
        Z = (self.X[idx] - self.mu) / self.sd * self.w
        Z = Z - (Z @ self.B.T) @ self.B
        return Z @ self.coef + self.icpt


def run(make):
    tr = L.rows(L.TRB)
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[tr])):
        te = tr[p.env[tr] == e]
        d[te] = make().fit(tr[p.env[tr] != e]).decision(te)
    G = L.groups(L.TRB)
    out = {}
    for name, g in (("new", G[p.role[G[:, 0]] == NEW]), ("all", G)):
        idx = g.ravel()
        out[name] = {"k1": round(float(np.mean((d[idx] > 0) == yb[idx])), 3),
                     "k3": round(float(np.mean((d[g].sum(1) > 0) == yb[g[:, 0]])), 3),
                     "k3len": round(float(np.mean(((d[g] * TLEN[g] / TLEN[g].mean()).sum(1) > 0) == yb[g[:, 0]])), 3)}
    return out


if __name__ == "__main__":
    runs = {
        "D": lambda: Model(("D",)),
        "F": lambda: Model(("F",)),
        "D+F": lambda: Model(("D", "F")),
        "S (struct val)": lambda: Model(("S",)),
        "F+struct": lambda: Model(("FS",)),
        "D+F+struct": lambda: Model(("D", "FS")),
    }
    for g in (0.5, 0.25, 0.1):
        runs[f"D+F recency {g}"] = (lambda g=g: Model(("D", "F"), recency=g))
    for r in (3, 10, 30):
        runs[f"D+F pca{r}"] = (lambda r=r: Model(("D", "F"), pca=r))
    res = {}
    for name, mk in runs.items():
        res[name] = run(mk)
        print(f"{name:22} new {res[name]['new']}  all {res[name]['all']}", flush=True)
    Path(__file__).with_name("out_pair_newbatch.json").write_text(json.dumps(res, indent=1))
