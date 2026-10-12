"""Fisher-score vectors on the confusable pair (numbers only), same protocols as pair_seq_lab.py.

Per fold, the shared next-number model (fisher_score.fit_base) is fitted on that fold's reference
rows only (both labels pooled, no label used); every response is mapped to its score vector at that
shared model; then standardize -> group weight -> environment-offset projection -> shrinkage LDA,
all on the reference rows. 'raw' = the same features as observed counts (no conditional
expectation subtracted): the histogram analogue, to isolate what conditioning on the history adds.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research import fisher_score as F  # noqa: E402

p, yb = L.p, L.yb
POS = {g: i for i, g in enumerate(L.pair)}
ST = F.Steps([p.rows[i]["numbers"] for i in L.pair])
_THETA, _U = {}, {}
RAW = F.score_vectors(ST, np.zeros(F.DIM), raw=True)
SL = {g: slice(F.OFF[g], F.OFF[g] + n) for g, n in F.GROUPS.items()}


def vectors(ref):
    key = hash(tuple(sorted(ref.tolist())))
    if key not in _U:
        mask = np.zeros(len(L.pair), bool)
        mask[[POS[i] for i in ref]] = True
        t = time.time()
        _THETA[key] = F.fit_base(ST, mask, maxiter=80)
        _U[key] = F.score_vectors(ST, _THETA[key]).astype(np.float32)
        print(f"  [base model fitted on {len(ref)} refs in {time.time()-t:.0f}s]", flush=True)
    return _U[key]


class FisherModel:
    def __init__(self, kind="score", groups=tuple(F.GROUPS), clf="lda", rank=2):
        self.kind, self.groups, self.clf, self.rank = kind, groups, clf, rank

    def _X(self, ref):
        U = RAW if self.kind == "raw" else vectors(ref)
        cols = np.concatenate([np.arange(F.DIM)[SL[g]] for g in self.groups])
        w = np.concatenate([np.full(F.GROUPS[g], 1 / np.sqrt(F.GROUPS[g])) for g in self.groups])
        X = np.zeros((len(p.rows), len(cols)))
        X[L.pair] = U[:, cols]
        return X, w

    def fit(self, ref):
        self.X, w = self._X(ref)
        self.mu, self.sd = self.X[ref].mean(0), self.X[ref].std(0)
        self.sd[self.sd < 1e-12] = np.inf
        self.w = w
        Z = (self.X[ref] - self.mu) / self.sd * w
        self.B = L.nuisance_basis(Z, ref, self.rank, 0)
        Z = Z - (Z @ self.B.T) @ self.B
        y = yb[ref]
        if self.clf == "nc":
            c = np.stack([Z[y == k].mean(0) for k in (0, 1)])
            self.cent = c / np.linalg.norm(c, axis=1, keepdims=True)
        else:
            self.lda = LDA(solver="lsqr", shrinkage="auto").fit(Z, y)
        return self

    def decision(self, idx):
        Z = (self.X[idx] - self.mu) / self.sd * self.w
        Z = Z - (Z @ self.B.T) @ self.B
        if self.clf == "nc":
            s = Z @ self.cent.T
            return s[:, 1] - s[:, 0]
        return self.lda.decision_function(Z)


if __name__ == "__main__":
    out = {}
    runs = {
        "score[all]:lda": lambda: FisherModel("score"),
        "raw[all]:lda": lambda: FisherModel("raw"),
        "score[all]:nc": lambda: FisherModel("score", clf="nc"),
    }
    for g in F.GROUPS:
        runs[f"score[{g}]:lda"] = (lambda g=g: FisherModel("score", (g,)))
        runs[f"raw[{g}]:lda"] = (lambda g=g: FisherModel("raw", (g,)))
    for name, mk in runs.items():
        t = time.time()
        r = L.evaluate(mk)
        out[name] = r
        print(L.line(name, r) + f" ({time.time()-t:.0f}s)", flush=True)
        Path(__file__).with_name("out_pair_fisher.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(Path(__file__).with_name("cache_pair_fisher.npz"),
                        **{str(k): v for k, v in _U.items()})
