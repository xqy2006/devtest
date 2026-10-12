"""Robust standardization and bounded per-response evidence for the triple sum.

The score-LDA decisions are heavy-tailed (sd 4 on 09-05, 44-55 on 09-30 / 10-11; a few responses
reach |d| > 300) because a response can sit far outside the references on a feature with small
reference variance, and the linear discriminant extrapolates. In the triple sum such a response
dominates the other two. Two classical remedies, both fitted on references only:
  clip c   winsorize the standardized features at +-c (Huber-type) before the projection/LDA
  k3 tanh  bounded evidence per response: s * tanh(d / s), s = median |d| of the references'
           in-sample decisions (k3 vote = plain majority of signs, for comparison)
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
from research.gated_fingerprint import LOCAL_FAMILIES  # noqa: E402
from research.pair_struct import NPZ, block  # noqa: E402

p, yb = L.p, L.yb


class Robust:
    def __init__(self, score=("val", "dg", "pb"), dense=False, clip=None, bound=None):
        self.score, self.dense, self.clip, self.bound = score, dense, clip, bound

    def _z(self, idx):
        Z = (self.X[idx] - self.mu) / self.sd
        if self.clip:
            Z = np.clip(Z, -self.clip, self.clip)
        return Z * self.w

    def fit(self, ref):
        Xs, ws = [], []
        if self.dense:
            X, w = L.design(list(LOCAL_FAMILIES))
            Xs.append(X); ws.append(w)
        if self.score:
            U = NPZ[str(hash(tuple(sorted(ref.tolist()))))]
            cols = [block(U, b) for b in self.score]
            X = np.zeros((len(p.rows), sum(c.shape[1] for c in cols)))
            X[L.pair] = np.hstack(cols)
            Xs.append(X); ws.append(np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols]))
        self.X, self.w = np.hstack(Xs), np.concatenate(ws)
        self.mu, self.sd = self.X[ref].mean(0), self.X[ref].std(0)
        self.sd[self.sd < 1e-12] = np.inf
        Z = self._z(ref)
        self.B = L.nuisance_basis(Z, ref, 2, 0)
        Z = Z - (Z @ self.B.T) @ self.B
        self.lda = LDA(solver="lsqr", shrinkage="auto").fit(Z, yb[ref])
        self.s = np.median(np.abs(self.lda.decision_function(Z)))
        return self

    def decision(self, idx):
        Z = self._z(idx)
        Z = Z - (Z @ self.B.T) @ self.B
        d = self.lda.decision_function(Z)
        return self.s * np.tanh(d / self.s) if self.bound else d


def protocols(make):
    """Out-of-fold decisions for LOBO, 10-11 env and pooled env (outside panels: dev only)."""
    tr = L.rows(L.TRB)
    out = {}
    d = np.zeros(len(p.rows))
    for b in L.TRB:
        te = L.rows((b,))
        d[te] = make().fit(L.rows(tuple(x for x in L.TRB if x != b))).decision(te)
    out["lobo"] = d
    cx = L.rows((L.PAIR_ROLE,))
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[cx])):
        d[cx[p.env[cx] == e]] = make().fit(cx[p.env[cx] != e]).decision(cx[p.env[cx] == e])
    out["codex_env"] = d
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[tr])):
        d[tr[p.env[tr] == e]] = make().fit(tr[p.env[tr] != e]).decision(tr[p.env[tr] == e])
    out["env"] = d
    m = make().fit(tr)
    d = np.zeros(len(p.rows))
    for role in ("old_outside", "fresh_outside"):
        te = L.rows((role,))
        d[te] = m.decision(te)
    out["outside"] = d
    return out


def k1k3(d, roles, vote=False):
    idx, g = L.rows(roles), L.groups(roles)
    k1 = np.mean((d[idx] > 0) == yb[idx])
    agg = np.sign(d[g]).sum(1) if vote else d[g].sum(1)
    return round(float(k1), 3), round(float(np.mean((agg > 0) == yb[g[:, 0]])), 3)


def report(D, vote=False):
    r = {"lobo_all": k1k3(D["lobo"], L.TRB, vote)}
    for b in L.TRB:
        r["lobo_" + b[-8:]] = k1k3(D["lobo"], (b,), vote)
    r["codex_env"] = k1k3(D["codex_env"], (L.PAIR_ROLE,), vote)
    r["env_all"] = k1k3(D["env"], L.TRB, vote)
    r["env_20261011"] = k1k3(D["env"], (L.PAIR_ROLE,), vote)
    r["old(dev)"] = k1k3(D["outside"], ("old_outside",), vote)
    r["fresh(dev)"] = k1k3(D["outside"], ("fresh_outside",), vote)
    return r


RUNS = {
    "score": dict(),
    "score tanh": dict(bound=True),
    "score clip5": dict(clip=5),
    "score clip3": dict(clip=3),
    "score clip3 tanh": dict(clip=3, bound=True),
    "score clip2": dict(clip=2),
    "dense+score clip3": dict(dense=True, clip=3),
    "dense+score clip3 tanh": dict(dense=True, clip=3, bound=True),
}

if __name__ == "__main__":
    out = {}
    for name, kw in RUNS.items():
        D = protocols(lambda kw=kw: Robust(**kw))
        out[name] = report(D)
        print(f"{name:24}", out[name], flush=True)
        if name in ("score", "score clip3"):
            out[name + " vote"] = report(D, vote=True)
            print(f"{name + ' vote':24}", out[name + " vote"], flush=True)
        Path(__file__).with_name("out_pair_robust.json").write_text(json.dumps(out, indent=1))
