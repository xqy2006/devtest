"""General multi-family response scorer.

Features are computed from one response. All standardization and classifier parameters are
fit on the training rows passed to ``fit``. No class-specific branches: every class is a
reference column, and a new model is added by rebuilding from reference responses.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from research.features import extract

_CACHE = {}


def features(p, families):
    key = (id(p), tuple(families))
    if key not in _CACHE:
        F = extract(p.rows, families)
        _CACHE[key] = F
    return _CACHE[key]


class MultiFamily:
    def __init__(self, families, clf="lda", C=0.01, family_weights=None, k3="sum_loglik"):
        self.families = list(families)
        self.clf = clf
        self.C = C
        self.family_weights = family_weights or {}
        self.k3 = k3

    def matrix(self, p, idx=None):
        F = features(p, self.families)
        X = np.hstack([F[f] for f in self.families])
        return X if idx is None else X[idx]

    def _weights(self, p):
        F = features(p, self.families)
        w = []
        for f in self.families:
            # each family contributes equal total variance unless reweighted
            d = F[f].shape[1]
            w.append(np.full(d, self.family_weights.get(f, 1.0) / np.sqrt(d)))
        return np.concatenate(w)

    def fit(self, p, train):
        X = self.matrix(p, train)
        sc = StandardScaler().fit(X)
        w = self._weights(p)
        Z = sc.transform(X) * w
        y = p.y[train]
        if self.clf == "lda":
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z, y)
        elif self.clf == "logreg":
            m = LogisticRegression(C=self.C, max_iter=5000).fit(Z, y)
        else:
            raise ValueError(self.clf)
        assert list(m.classes_) == list(range(len(p.classes)))
        return {"sc": sc, "w": w, "m": m}

    def score(self, model, p, idx):
        Z = model["sc"].transform(self.matrix(p, idx)) * model["w"]
        m = model["m"]
        logits = m.decision_function(Z)
        return logits - logsumexp(logits, axis=1, keepdims=True)

    def score_group(self, model, p, members):
        flat = self.score(model, p, members.ravel()).reshape(len(members), members.shape[1], -1)
        return flat.sum(axis=1)
