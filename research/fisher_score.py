"""Fisher-score vectors of a shared next-number model (numbers only).

Rationale. When two labels are the same network up to a small perturbation (quantization,
distillation), their next-number distributions are p_theta0 and p_theta0+delta with small delta.
The locally most powerful statistic for such a difference is the score of the shared model
(Rao score test; Fisher kernel, Jaakkola & Haussler 1999):

    u(x) = (1/T) sum_t [ f(x_t, h_t) - E_theta0[ f(., h_t) ] ]

i.e. what the response chose minus what the shared model expected it to choose *given that exact
history*. Raw histograms keep the path noise (which values were already used, where the previous
number was); the score removes it, keeping only the per-decision innovation.

Shared model: conditional log-linear (maximum-entropy) next-number model over the 355 candidates,
fitted by L-BFGS on reference rows only, with feature groups
  val   value (355)                          j1   signed jump from x_{t-1}, 37 bins (+none)
  j2    signed jump from x_{t-2} (+none)      dg   last-digit transition (100, +none)
  bd    12-band transition (144, +none)      jj   coarse previous jump x coarse jump (36, +none)
  pb    position bucket x 12-band (60)        cv   distance to nearest used value (7) x stage (3)
Score vectors then go through the usual pipeline (standardize on references, group weight
1/sqrt(dim), environment-offset projection, shrinkage LDA).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

V = 355
_MAG = np.array([1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48, 64, 85, 113, 150, 200, 265, 356])
NJ = 1 + 2 * (len(_MAG) - 1)  # 37


def _jbin(d):
    m = np.abs(d)
    b = np.searchsorted(_MAG, m, side="right")  # 1..18 for m >= 1
    return np.where(d == 0, 0, np.where(d > 0, b, b + len(_MAG) - 1))


def _c6(d):
    return np.digitize(d, [-120, -40, 0, 40, 120])  # 0..5 (0 counts as up-small)


_vals = np.arange(1, V + 1)
_band = (_vals - 1) * 12 // V
# row 0 of every table = "no previous number": a constant id, which cancels in the softmax
JBT = np.vstack([np.full(V, NJ), _jbin(_vals[None, :] - _vals[:, None])])
DGT = np.vstack([np.full(V, 100), (_vals[:, None] % 10) * 10 + _vals[None, :] % 10])
BDT = np.vstack([np.full(V, 144), _band[:, None] * 12 + _band[None, :]])
C6T = np.vstack([np.zeros(V, int), _c6(_vals[None, :] - _vals[:, None])])
PBT = np.arange(5)[:, None] * 12 + _band[None, :]
GROUPS = {"val": V, "j1": NJ + 1, "j2": NJ + 1, "dg": 101, "bd": 145, "jj": 37, "pb": 60, "cv": 21}
OFF = dict(zip(GROUPS, np.cumsum([0] + list(GROUPS.values()))[:-1]))
DIM = sum(GROUPS.values())
_COV_EDGES = np.array([1, 2, 3, 5, 9, 17])


def _posb(t):
    return np.digitize(t, [1, 5, 20, 100])


class Steps:
    """Per-step context of a set of responses (compact; candidate ids are built per chunk)."""

    def __init__(self, seqs):
        rows, prev, prev2, kprev, pb, tb, cov, tgt = [], [], [], [], [], [], [], []
        for r, x in enumerate(seqs):
            x = np.asarray(x, int)
            dist = np.full(V, 10 ** 6)
            for t, v in enumerate(x):
                rows.append(r)
                prev.append(x[t - 1] if t >= 1 else 0)
                prev2.append(x[t - 2] if t >= 2 else 0)
                kprev.append(int(_c6(x[t - 1] - x[t - 2])) if t >= 2 else 6)
                pb.append(_posb(t))
                tb.append(0 if t < 50 else 1 if t < 150 else 2)
                cov.append(np.searchsorted(_COV_EDGES, dist, side="right").astype(np.uint8))
                tgt.append(v)
                dist = np.minimum(dist, np.abs(_vals - v))
        self.row, self.prev, self.prev2 = np.array(rows), np.array(prev), np.array(prev2)
        self.kprev, self.pb, self.tb = np.array(kprev), np.array(pb), np.array(tb)
        self.tgt = np.array(tgt)
        # distance-to-used bin + 7 * stage, kept as uint8 (steps x 355)
        self.cv = np.array(cov, dtype=np.uint8) + (7 * self.tb[:, None]).astype(np.uint8)
        self.n_rows = len(seqs)
        self.T = np.bincount(self.row, minlength=self.n_rows)

    def ids(self, s):
        """Candidate feature ids (global, offset per group) for steps in slice/index s."""
        prev, prev2, kp = self.prev[s], self.prev2[s], self.kprev[s]
        jj = np.where(kp[:, None] == 6, 36, kp[:, None] * 6 + C6T[prev])
        return [
            np.broadcast_to(OFF["val"] + np.arange(V), (len(prev), V)),
            OFF["j1"] + JBT[prev],
            OFF["j2"] + JBT[prev2],
            OFF["dg"] + DGT[prev],
            OFF["bd"] + BDT[prev],
            OFF["jj"] + jj,
            OFF["pb"] + PBT[self.pb[s]],
            OFF["cv"] + self.cv[s].astype(np.int64),
        ]

    def target_ids(self, s):
        ids = self.ids(s)
        k = self.tgt[s] - 1
        r = np.arange(len(k))
        return [g[r, k] for g in ids]


def _chunks(n, size=3000):
    for a in range(0, n, size):
        yield slice(a, min(n, a + size))


JJT = np.stack([np.full((V + 1, V), 36) if k == 6 else k * 6 + C6T for k in range(7)])


def _onehot(idx, n):
    from scipy import sparse
    return sparse.csr_matrix((np.ones(len(idx)), (idx, np.arange(len(idx)))), shape=(n, len(idx)))


def fit_base(st: Steps, mask=None, lam=1.0, maxiter=60):
    """Shared conditional log-linear model by L-BFGS on the steps of the masked rows.
    Groups that depend on the context only through x_{t-1} (or x_{t-2}, previous jump class,
    position bucket) are evaluated as per-context tables; expected counts are aggregated per
    context with sparse one-hot products."""
    sel = np.flatnonzero(np.ones(len(st.row), bool) if mask is None else mask[st.row])
    obs = np.zeros(DIM)
    for s in _chunks(len(sel)):
        for g in st.target_ids(sel[s]):
            obs += np.bincount(g, minlength=DIM)
    parts = []
    for s in _chunks(len(sel)):
        i = sel[s]
        parts.append((i, _onehot(st.prev[i], V + 1), _onehot(st.prev2[i], V + 1),
                      _onehot(st.kprev[i] * (V + 1) + st.prev[i], 7 * (V + 1)), _onehot(st.pb[i], 5)))
    sl = {g: slice(OFF[g], OFF[g] + n) for g, n in GROUPS.items()}

    def f(theta):
        th = {g: theta[sl[g]] for g in GROUPS}
        TP = th["j1"][JBT] + th["dg"][DGT] + th["bd"][BDT]
        TJ2 = th["j2"][JBT]
        TJJ = th["jj"][JJT].reshape(7 * (V + 1), V)
        TPB = th["pb"][PBT]
        Ap, Ap2, Ajj, Apb = np.zeros((V + 1, V)), np.zeros((V + 1, V)), np.zeros((7 * (V + 1), V)), np.zeros((5, V))
        gcv = np.zeros(21)
        nll = 0.0
        for i, Mp, Mp2, Mjj, Mpb in parts:
            cv = st.cv[i]
            L = th["val"] + TP[st.prev[i]] + TJ2[st.prev2[i]] + TJJ[st.kprev[i] * (V + 1) + st.prev[i]] \
                + TPB[st.pb[i]] + th["cv"][cv]
            m = L.max(1, keepdims=True)
            E = np.exp(L - m)
            Z = E.sum(1, keepdims=True)
            P = E / Z
            nll -= (L[np.arange(len(i)), st.tgt[i] - 1] - m[:, 0] - np.log(Z[:, 0])).sum()
            Ap += Mp @ P
            Ap2 += Mp2 @ P
            Ajj += Mjj @ P
            Apb += Mpb @ P
            gcv += np.bincount(cv.ravel(), weights=P.ravel(), minlength=21)
        exp = np.concatenate([
            Apb.sum(0),
            np.bincount(JBT.ravel(), Ap.ravel(), GROUPS["j1"]),
            np.bincount(JBT.ravel(), Ap2.ravel(), GROUPS["j2"]),
            np.bincount(DGT.ravel(), Ap.ravel(), GROUPS["dg"]),
            np.bincount(BDT.ravel(), Ap.ravel(), GROUPS["bd"]),
            np.bincount(JJT.ravel(), Ajj.ravel(), GROUPS["jj"]),
            np.bincount(PBT.ravel(), Apb.ravel(), GROUPS["pb"]),
            gcv,
        ])
        nll += 0.5 * lam * theta @ theta
        return nll, exp - obs + lam * theta

    res = minimize(f, np.zeros(DIM), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x


def score_vectors(st: Steps, theta, raw=False):
    """Per-response mean innovation (observed minus expected feature counts) at theta.
    raw=True returns the observed counts only (the histogram analogue, for comparison)."""
    U = np.zeros((st.n_rows, DIM))
    for s in _chunks(len(st.row)):
        rows = st.row[s]
        for g in st.target_ids(s):
            np.add.at(U, (rows, g), 1.0)
        if raw:
            continue
        ids = st.ids(s)
        L = sum(theta[g] for g in ids)
        P = np.exp(L - L.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        for g in ids:
            # expected counts per (row, feature): bincount over a combined index
            U -= np.bincount((rows[:, None] * DIM + g).ravel(), weights=P.ravel(),
                             minlength=st.n_rows * DIM).reshape(st.n_rows, DIM)
    return U / np.maximum(st.T, 1)[:, None]


def group_weights():
    return np.concatenate([np.full(n, 1 / np.sqrt(n)) for n in GROUPS.values()])
