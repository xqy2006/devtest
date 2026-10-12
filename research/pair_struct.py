"""Structured projections of the value score (cached per-fold score vectors from pair_fisher.py).

pair_newbatch.py found that the 355-d value score projected on a small structured basis
(last digit, 36 bands, residues mod 3/4/7) generalizes better to the newest batch than the full
355-d value score. Here: which bases carry it, on every protocol (lab evaluate: LOBO, 10-11 env,
pooled env, outside panels as development only), with per-batch k1/k3 under LOBO.

A projection u_val @ PHI sums the per-value score over the values sharing a structural property,
i.e. it is the score of the shared model for the coarser hypothesis "the two labels differ only
through that property" -- a prior that the stable difference is smooth in the value, not
value-specific (value-specific preferences are where deployments drift).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research import fisher_score as FS  # noqa: E402

p, yb = L.p, L.yb
NPZ = np.load(Path(__file__).with_name("cache_pair_fisher.npz"))
v = np.arange(1, 356)


def onehot(k, n):
    return np.eye(n)[k]


def fourier(K):
    a = 2 * np.pi * np.outer(v - 1, np.arange(1, K + 1)) / 355
    return np.hstack([np.cos(a), np.sin(a)])


BASES = {
    "d10": onehot(v % 10, 10),
    "t10": onehot(v // 10 % 10, 10),
    "h4": onehot(v // 100, 4),
    "b6": onehot((v - 1) * 6 // 355, 6),
    "b12": onehot((v - 1) * 12 // 355, 12),
    "b36": onehot((v - 1) * 36 // 355, 36),
    "m3": onehot(v % 3, 3), "m4": onehot(v % 4, 4), "m7": onehot(v % 7, 7),
    "f8": fourier(8), "f24": fourier(24),
}
_OLD = np.hstack([BASES[k] for k in ("d10", "b36", "m3", "m4", "m7")])


def block(U, name):
    """One feature block of the 577 pair rows: a raw score group or a projected value score."""
    if name in FS.GROUPS:
        return U[:, FS.OFF[name]:FS.OFF[name] + FS.GROUPS[name]]
    val = U[:, FS.OFF["val"]:FS.OFF["val"] + 355]
    if name == "S60":
        return val @ _OLD
    return val @ BASES[name]


class Struct:
    def __init__(self, blocks, rank=2):
        self.blocks, self.rank = blocks, rank

    def fit(self, ref):
        U = NPZ[str(hash(tuple(sorted(ref.tolist()))))]
        cols = [block(U, b) for b in self.blocks]
        self.X = np.zeros((len(p.rows), sum(c.shape[1] for c in cols)))
        self.X[L.pair] = np.hstack(cols)
        self.w = np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols])
        self.mu, self.sd = self.X[ref].mean(0), self.X[ref].std(0)
        self.sd[self.sd < 1e-12] = np.inf
        Z = (self.X[ref] - self.mu) / self.sd * self.w
        self.B = L.nuisance_basis(Z, ref, self.rank, 0)
        Z = Z - (Z @ self.B.T) @ self.B
        self.lda = LDA(solver="lsqr", shrinkage="auto").fit(Z, yb[ref])
        return self

    def decision(self, idx):
        Z = (self.X[idx] - self.mu) / self.sd * self.w
        Z = Z - (Z @ self.B.T) @ self.B
        return self.lda.decision_function(Z)


def lobo_batches(make):
    d = np.zeros(len(p.rows))
    for b in L.TRB:
        te = L.rows((b,))
        d[te] = make().fit(L.rows(tuple(x for x in L.TRB if x != b))).decision(te)
    out = {}
    for b in L.TRB:
        idx, g = L.rows((b,)), L.groups((b,))
        out[b[-8:]] = [round(x, 3) for x in L.acc(d, idx, g)]
    return out


RUNS = {
    "val (full 355)": ("val", "dg", "pb"),
    "S60 (newbatch S)": ("S60", "dg", "pb"),
    "S60 only": ("S60",),
    "d10": ("d10", "dg", "pb"),
    "b36": ("b36", "dg", "pb"),
    "m3+m4+m7": ("m3", "m4", "m7", "dg", "pb"),
    "d10+t10+h4": ("d10", "t10", "h4", "dg", "pb"),
    "b6+b12+b36": ("b6", "b12", "b36", "dg", "pb"),
    "f8": ("f8", "dg", "pb"),
    "f24": ("f24", "dg", "pb"),
    "d10+b36 sep": ("d10", "b36", "dg", "pb"),
    "d10+t10+b12+mods": ("d10", "t10", "b12", "m3", "m4", "m7", "dg", "pb"),
    "S60+j1+j2": ("S60", "dg", "pb", "j1", "j2"),
    "S60+all other": ("S60", "dg", "pb", "j1", "j2", "bd", "jj", "cv"),
    "S60 no pb": ("S60", "dg"),
}

if __name__ == "__main__":
    out = {}
    for name, blocks in RUNS.items():
        mk = (lambda blocks=blocks: Struct(blocks))
        r = L.evaluate(mk)
        r["lobo_batch"] = lobo_batches(mk)
        out[name] = r
        print(L.line(name, r), "| LOBO k1/k3 by batch", r["lobo_batch"], flush=True)
        Path(__file__).with_name("out_pair_struct.json").write_text(json.dumps(out, indent=1))
