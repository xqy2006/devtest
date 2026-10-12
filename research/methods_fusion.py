"""Score-level fusion of several general scorers.

Each component's per-response class scores are standardized across candidates (as production
does), then combined with weights. ``nested=True`` chooses the weight by an inner
leave-one-environment-out CV on the outer training rows only.
"""
from __future__ import annotations

import numpy as np


def cz(s):
    s = np.asarray(s, float)
    return (s - s.mean(1, keepdims=True)) / np.maximum(s.std(1, keepdims=True), 1e-12)


class Fusion:
    def __init__(self, components, weights=None, grid=None, nested=False, objective="k1"):
        self.components = components
        self.weights = weights or [1.0] * len(components)
        self.grid = grid
        self.nested = nested
        self.objective = objective

    def _fit_components(self, p, train):
        return [c.fit(p, train) for c in self.components]

    def _raw(self, models, p, idx):
        return [cz(c.score(m, p, idx)) for c, m in zip(self.components, models)]

    def fit(self, p, train):
        weights = list(self.weights)
        if self.nested and self.grid:
            envs = sorted(set(p.env[train]))
            raws = [np.zeros((len(train), len(p.classes))) for _ in self.components]
            for e in envs:
                inner_te = np.flatnonzero(p.env[train] == e)
                inner_tr = train[p.env[train] != e]
                models = self._fit_components(p, inner_tr)
                for k, r in enumerate(self._raw(models, p, train[inner_te])):
                    raws[k][inner_te] = r
            y = p.y[train]
            best = None
            for w in self.grid:
                s = sum(wi * r for wi, r in zip(w, raws))
                # class-balanced accuracy so one large class does not dominate
                acc = np.mean([(s[y == c].argmax(1) == c).mean() for c in np.unique(y)])
                if best is None or acc > best[0] + 1e-12:
                    best = (acc, list(w))
            weights = best[1]
        return {"models": self._fit_components(p, train), "weights": weights}

    def score(self, model, p, idx):
        return sum(w * r for w, r in zip(model["weights"], self._raw(model["models"], p, idx)))
