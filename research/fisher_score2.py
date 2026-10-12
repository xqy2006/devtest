"""Fisher-score vectors, round 2: a richer shared next-number model (numbers only).

Same principle as fisher_score.py (score of a shared conditional log-linear model = per-decision
innovation; for the value group it is the martingale residual of each value's "hazard" of being
chosen in a draw-without-much-replacement process). The shared model now also knows
  bd2   12-band of x_{t-2} -> 12-band of the candidate (145)
  vst   value x stage (t<50, 50-149, >=150): stage-specific preferences (1065)
  cvb   coverage state of the candidate: distance to the nearest used value (7) x unused fraction
        of its 36-band (5) x stage (3) (105; replaces cv)
so that more of the path-dependent part of each choice is explained and the residuals are less
noisy. Groups are generic: each is a table from a context index to the 355 candidate ids.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from research.fisher_score import BDT, C6T, DGT, JBT, JJT, NJ, PBT, V, _c6, _posb, _vals

_band = (_vals - 1) * 12 // V
_band36 = (_vals - 1) * 36 // V
_B36_SIZE = np.bincount(_band36, minlength=36)
_COV_EDGES = np.array([1, 2, 3, 5, 9, 17])
VST = np.arange(3)[:, None] * V + (_vals - 1)[None, :]
# context kinds: number of contexts and how a step maps to its context index
KINDS = {"const": 1, "prev": V + 1, "prev2": V + 1, "jj": 7 * (V + 1), "pos": 5, "stage": 3}
# group -> (size, kind, table[n_ctx, V]) ; "cvb" is per step (kind "step")
GROUP_DEFS = {
    "val": (V, "const", np.arange(V)[None, :]),
    "j1": (NJ + 1, "prev", JBT),
    "j2": (NJ + 1, "prev2", JBT),
    "dg": (101, "prev", DGT),
    "bd": (145, "prev", BDT),
    "bd2": (145, "prev2", BDT),
    "jj": (37, "jj", JJT.reshape(7 * (V + 1), V)),
    "pb": (60, "pos", PBT),
    "vst": (3 * V, "stage", VST),
    "cvb": (105, "step", None),
}


class Steps2:
    def __init__(self, seqs):
        rows, prev, prev2, kprev, pb, tb, cvb, tgt = [], [], [], [], [], [], [], []
        for r, x in enumerate(seqs):
            x = np.asarray(x, int)
            dist = np.full(V, 10 ** 6)
            unused = _B36_SIZE.astype(float).copy()
            for t, v in enumerate(x):
                st = 0 if t < 50 else 1 if t < 150 else 2
                rows.append(r)
                prev.append(x[t - 1] if t >= 1 else 0)
                prev2.append(x[t - 2] if t >= 2 else 0)
                kprev.append(int(_c6(x[t - 1] - x[t - 2])) if t >= 2 else 6)
                pb.append(_posb(t))
                tb.append(st)
                cb = np.searchsorted(_COV_EDGES, dist, side="right")
                fb = np.minimum((unused / _B36_SIZE * 5).astype(int), 4)[_band36]
                cvb.append((cb * 15 + fb * 3 + st).astype(np.uint8))
                tgt.append(v)
                if dist[v - 1] > 0:
                    unused[_band36[v - 1]] -= 1
                dist = np.minimum(dist, np.abs(_vals - v))
        self.row = np.array(rows)
        self.ctx = {"const": np.zeros(len(rows), int), "prev": np.array(prev), "prev2": np.array(prev2),
                    "jj": np.array(kprev) * (V + 1) + np.array(prev), "pos": np.array(pb), "stage": np.array(tb)}
        self.cvb = np.array(cvb, dtype=np.uint8)
        self.tgt = np.array(tgt)
        self.n_rows = len(seqs)
        self.T = np.bincount(self.row, minlength=self.n_rows)


class Base:
    def __init__(self, groups=tuple(GROUP_DEFS)):
        self.groups = list(groups)
        self.size = {g: GROUP_DEFS[g][0] for g in self.groups}
        off = np.cumsum([0] + [self.size[g] for g in self.groups])
        self.sl = {g: slice(off[k], off[k + 1]) for k, g in enumerate(self.groups)}
        self.dim = int(off[-1])

    def _ids(self, st, s):
        """Global candidate ids per group for steps s (list of arrays steps x V)."""
        out = []
        for g in self.groups:
            size, kind, tab = GROUP_DEFS[g]
            local = st.cvb[s].astype(np.int64) if kind == "step" else tab[st.ctx[kind][s]]
            out.append(self.sl[g].start + local)
        return out

    def _logits(self, st, s, tabs, theta):
        L = 0.0
        for kind, T in tabs.items():
            L = L + T[st.ctx[kind][s]]
        if "cvb" in self.groups:
            L = L + theta[self.sl["cvb"]][st.cvb[s]]
        return L

    def _tables(self, theta):
        tabs = {}
        for g in self.groups:
            size, kind, tab = GROUP_DEFS[g]
            if kind == "step":
                continue
            T = theta[self.sl[g]][tab]
            tabs[kind] = tabs.get(kind, 0) + T
        return tabs

    def fit(self, st, mask=None, lam=1.0, maxiter=80):
        sel = np.flatnonzero(np.ones(len(st.row), bool) if mask is None else mask[st.row])
        obs = np.zeros(self.dim)
        chunks = [sel[a:a + 3000] for a in range(0, len(sel), 3000)]
        for i in chunks:
            k = st.tgt[i] - 1
            for g in self._ids(st, i):
                obs += np.bincount(g[np.arange(len(i)), k], minlength=self.dim)
        kinds = sorted({GROUP_DEFS[g][1] for g in self.groups} - {"step"})
        onehots = [{kd: sparse.csr_matrix((np.ones(len(i)), (st.ctx[kd][i], np.arange(len(i)))),
                                          shape=(KINDS[kd], len(i))) for kd in kinds} for i in chunks]

        def f(theta):
            tabs = self._tables(theta)
            A = {kd: np.zeros((KINDS[kd], V)) for kd in kinds}
            gstep = np.zeros(self.size.get("cvb", 0))
            nll = 0.0
            for i, M in zip(chunks, onehots):
                Lg = self._logits(st, i, tabs, theta)
                m = Lg.max(1, keepdims=True)
                E = np.exp(Lg - m)
                Z = E.sum(1, keepdims=True)
                P = E / Z
                nll -= (Lg[np.arange(len(i)), st.tgt[i] - 1] - m[:, 0] - np.log(Z[:, 0])).sum()
                for kd in kinds:
                    A[kd] += M[kd] @ P
                if "cvb" in self.groups:
                    gstep += np.bincount(st.cvb[i].ravel(), weights=P.ravel(), minlength=len(gstep))
            exp = np.zeros(self.dim)
            for g in self.groups:
                size, kind, tab = GROUP_DEFS[g]
                exp[self.sl[g]] = gstep if kind == "step" else np.bincount(tab.ravel(), A[kind].ravel(), size)
            nll += 0.5 * lam * theta @ theta
            return nll, exp - obs + lam * theta

        self.theta = minimize(f, np.zeros(self.dim), jac=True, method="L-BFGS-B",
                              options={"maxiter": maxiter}).x
        return self

    def scores(self, st):
        """Per-response mean innovation (observed - expected feature counts), rows x dim."""
        U = np.zeros((st.n_rows, self.dim))
        tabs = self._tables(self.theta)
        for a in range(0, len(st.row), 3000):
            i = np.arange(a, min(len(st.row), a + 3000))
            rows = st.row[i]
            Lg = self._logits(st, i, tabs, self.theta)
            P = np.exp(Lg - Lg.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            k = st.tgt[i] - 1
            for g in self._ids(st, i):
                np.add.at(U, (rows, g[np.arange(len(i)), k]), 1.0)
                U -= np.bincount((rows[:, None] * self.dim + g).ravel(), weights=P.ravel(),
                                 minlength=st.n_rows * self.dim).reshape(st.n_rows, self.dim)
        return U / np.maximum(st.T, 1)[:, None]
