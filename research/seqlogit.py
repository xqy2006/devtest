"""Sequential conditional-logit model of a number stream.

For class c, P(x_t = v | x_<t) ∝ exp(theta_c[v] + phi_c[jump_bin(v - x_{t-1})] + rho_c * used_t(v)).
This is a Plackett-Luce-style without-replacement model with a jump kernel and a soft repeat
penalty. Class parameters are MAP-shrunk toward a pooled reference model, so a new class is
enrolled from its own reference responses alone (pooled model can stay frozen).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

V = 355
B = 24
NPAR = V + B + 1 + 1  # theta, phi (B real bins + 1 dummy start bin), rho
CHUNK = 3000


def seq_design(x):
    x = np.asarray(x, dtype=np.int64) - 1
    T = len(x)
    first = np.full(V, T + 1)
    for t in range(T - 1, -1, -1):
        first[x[t]] = t
    prev = np.r_[-1, x[:-1]]
    return {"x": x, "prev": prev, "first": first, "T": T}


def _chunks(designs):
    """Yield stacked (x, prev, used, jb) chunks over all steps of the given sequences."""
    buf = []
    size = 0
    for d in designs:
        buf.append(d)
        size += d["T"]
        if size >= CHUNK:
            yield _stack(buf)
            buf, size = [], 0
    if buf:
        yield _stack(buf)


def _stack(ds):
    xs, prevs, useds = [], [], []
    vals = np.arange(V)
    for d in ds:
        t = np.arange(d["T"])
        xs.append(d["x"])
        prevs.append(d["prev"])
        useds.append(t[:, None] > d["first"][None, :])
    x = np.concatenate(xs)
    prev = np.concatenate(prevs)
    used = np.concatenate(useds)
    jb = ((vals[None, :] - prev[:, None]) + (V - 1)) * B // (2 * V - 1)
    jb[prev < 0] = B  # dummy start bin: constant across candidates
    return x, prev, used, jb


def nll_grad(w, chunks):
    theta, phi, rho = w[:V], w[V:V + B + 1], w[-1]
    nll = 0.0
    g = np.zeros_like(w)
    n = 0
    for x, prev, used, jb in chunks:
        logits = theta[None, :] + phi[jb] + rho * used
        lse = logsumexp(logits, axis=1)
        rows = np.arange(len(x))
        nll -= float((logits[rows, x] - lse).sum())
        P = np.exp(logits - lse[:, None])
        # observed minus expected
        np.add.at(g[:V], x, -1.0)
        g[:V] += P.sum(0)
        gphi = np.bincount(jb.ravel(), weights=P.ravel(), minlength=B + 1)
        gphi -= np.bincount(jb[rows, x], minlength=B + 1)
        g[V:V + B + 1] += gphi
        g[-1] += float((P * used).sum() - used[rows, x].sum())
        n += len(x)
    return nll, g, n


def fit(designs, prior=None, lam=1.0, lam0=1e-3, maxiter=200):
    chunks = list(_chunks(designs))
    w0 = np.zeros(NPAR) if prior is None else prior.copy()
    center = np.zeros(NPAR) if prior is None else prior
    reg = lam0 if prior is None else lam

    def f(w):
        nll, g, n = nll_grad(w, chunks)
        d = w - center
        d[V + B] = 0.0  # dummy bin is unidentified; leave it unpenalized but it has zero gradient
        return nll / n + 0.5 * reg * float(d @ d), g / n + reg * d

    res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x


def loglik(w, design):
    total = 0.0
    for x, prev, used, jb in _chunks([design]):
        logits = w[:V][None, :] + w[V:V + B + 1][jb] + w[-1] * used
        lse = logsumexp(logits, axis=1)
        total += float((logits[np.arange(len(x)), x] - lse).sum())
    return total


class SeqLogit:
    """Harness method: pooled model + per-class shrunk models; score = per-class log-likelihood."""

    def __init__(self, lam=1.0, per_step=True):
        self.lam = lam
        self.per_step = per_step
        self._designs = {}

    def design(self, p, i):
        if i not in self._designs:
            self._designs[i] = seq_design(p.rows[i]["numbers"])
        return self._designs[i]

    def fit(self, p, train):
        pooled = fit([self.design(p, i) for i in train])
        classes = sorted(set(p.y[train]))
        W = {}
        for c in classes:
            W[c] = fit([self.design(p, i) for i in train if p.y[i] == c], prior=pooled, lam=self.lam)
        return {"pooled": pooled, "W": W, "classes": classes, "C": len(p.classes)}

    def score(self, model, p, idx):
        out = np.full((len(idx), model["C"]), -np.inf)
        for k, i in enumerate(idx):
            d = self.design(p, i)
            for c in model["classes"]:
                out[k, c] = loglik(model["W"][c], d) / (d["T"] if self.per_step else 1.0)
        return out
