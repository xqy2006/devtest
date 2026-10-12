"""Drift-aware discriminant: the label difference varies between deployments / environments.

Standard LDA uses w = S_w^-1 * delta, with delta the pooled class-mean difference (dominated by the
largest batch) and S_w the within-class covariance. For a new environment of a new deployment, the
noise also contains the class x environment interaction: the label difference itself varies
(delta_e = delta + eta_e). Every environment holds both labels (paired design), so eta_e is
estimable on the references:
  delta_e    class-mean difference inside each (batch, environment) unit
  delta_bar  equal-weight mean over batches of the per-batch mean of delta_e (no batch dominates)
  S_int      covariance of delta_e - delta_bar minus its expected sampling part S_w (1/nA + 1/nS),
             clipped to positive semi-definite
  w          (S_w + lam * S_int)^-1 delta_bar
lam = 0 with pooled delta is ordinary shrinkage LDA. Same features and preprocessing as before
(standardization on references, group weights, environment-offset projection).
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
from research.gated_fingerprint import LOCAL_FAMILIES  # noqa: E402
from research.pair_robust import protocols, report  # noqa: E402
from research.pair_struct import NPZ, block  # noqa: E402

p, yb = L.p, L.yb


class Hier:
    def __init__(self, score=("val", "dg", "pb"), dense=False, lam=0.0, equal=False):
        self.score, self.dense, self.lam, self.equal = score, dense, lam, equal

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
        Z = (self.X[ref] - self.mu) / self.sd * self.w
        self.B = L.nuisance_basis(Z, ref, 2, 0)
        Z = Z - (Z @ self.B.T) @ self.B
        y = yb[ref]
        m0, m1 = Z[y == 0].mean(0), Z[y == 1].mean(0)
        R = Z - np.where(y[:, None] == 1, m1, m0)
        sc = R.std(0) + 1e-12  # Ledoit-Wolf on standardized residuals, as sklearn's LDA does
        Sw = ledoit_wolf(R / sc, assume_centered=True)[0] * np.outer(sc, sc)
        units = {}
        for k, (b, e) in enumerate(zip(p.role[ref], p.env[ref])):
            units.setdefault((b, e), []).append(k)
        D, noise, ub = [], [], []
        for (b, _), ks in units.items():
            ks = np.array(ks)
            na, ns = (y[ks] == 0).sum(), (y[ks] == 1).sum()
            if na and ns:
                D.append(Z[ks][y[ks] == 1].mean(0) - Z[ks][y[ks] == 0].mean(0))
                noise.append(1 / na + 1 / ns)
                ub.append(b)
        D, noise, ub = np.array(D), np.array(noise), np.array(ub)
        if self.equal:
            delta = np.mean([D[ub == b].mean(0) for b in sorted(set(ub))], axis=0)
        else:
            delta = m1 - m0
        S = Sw
        if self.lam:
            E = D - delta
            # low-rank interaction covariance: eigen-decomposition of the unit deviations, minus
            # the expected sampling variance along each eigen-direction
            _, s, vt = np.linalg.svd(E, full_matrices=False)
            var = s ** 2 / len(E)
            samp = np.array([v @ Sw @ v for v in vt]) * noise.mean()
            keep = np.clip(var - samp, 0, None)
            S = Sw + self.lam * (vt.T * keep) @ vt
        self.coef = np.linalg.solve(S, delta)
        self.icpt = -self.coef @ (m0 + m1) / 2
        return self

    def decision(self, idx):
        Z = (self.X[idx] - self.mu) / self.sd * self.w
        Z = Z - (Z @ self.B.T) @ self.B
        return Z @ self.coef + self.icpt


RUNS = {
    "score pooled": dict(),
    "score equal-batch": dict(equal=True),
    "score int lam1": dict(lam=1.0),
    "score equal int lam1": dict(equal=True, lam=1.0),
    "score equal int lam3": dict(equal=True, lam=3.0),
    "dense+score pooled": dict(dense=True),
    "dense+score equal int lam1": dict(dense=True, equal=True, lam=1.0),
}

if __name__ == "__main__":
    out = {}
    for name, kw in RUNS.items():
        D = protocols(lambda kw=kw: Hier(**kw))
        out[name] = report(D)
        print(f"{name:28}", out[name], flush=True)
        Path(__file__).with_name("out_pair_hier.json").write_text(json.dumps(out, indent=1))
