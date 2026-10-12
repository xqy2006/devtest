"""Class-conditional smoothed n-gram likelihood over emitted integers.

Each class keeps only count tables from its own reference responses; a new class is
enrolled by adding its counts. Scoring a query uses the query sequence alone.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

V = 355


class NGramClass:
    """Interpolated absolute-discount n-gram model (Kneser-Ney style without continuation counts)."""

    def __init__(self, order=2, discount=0.75, base_alpha=0.5):
        self.order = order
        self.d = discount
        self.base_alpha = base_alpha
        self.tables = [defaultdict(lambda: defaultdict(float)) for _ in range(order)]
        self.unigram = np.zeros(V)

    def add(self, x):
        x = [int(v) - 1 for v in x]
        for v in x:
            self.unigram[v] += 1
        for n in range(1, self.order):
            t = self.tables[n]
            for i in range(n, len(x)):
                t[tuple(x[i - n:i])][x[i]] += 1
        return self

    def finalize(self):
        u = self.unigram + self.base_alpha
        self.p0 = u / u.sum()
        self.ctx = []
        for n in range(1, self.order):
            stats = {}
            for ctx, nxt in self.tables[n].items():
                total = sum(nxt.values())
                stats[ctx] = (total, len(nxt), dict(nxt))
            self.ctx.append(stats)
        return self

    def logprob_seq(self, x, lam_floor=0.0):
        x = [int(v) - 1 for v in x]
        lp = np.zeros(len(x))
        for i, v in enumerate(x):
            p = self.p0[v]
            for n in range(1, self.order):
                if i < n:
                    break
                st = self.ctx[n - 1].get(tuple(x[i - n:i]))
                if st is None:
                    continue
                total, types, nxt = st
                p = max(nxt.get(v, 0.0) - self.d, 0.0) / total + (self.d * types / total) * p
            lp[i] = np.log(p)
        return lp


def class_models(seqs, labels, classes, **kw):
    models = []
    for c in classes:
        m = NGramClass(**kw)
        for s, l in zip(seqs, labels):
            if l == c:
                m.add(s)
        models.append(m.finalize())
    return models


def scores(models, seqs, mode="mean"):
    out = np.zeros((len(seqs), len(models)))
    for i, s in enumerate(seqs):
        for j, m in enumerate(models):
            lp = m.logprob_seq(s)
            out[i, j] = lp.mean() if mode == "mean" else lp.sum()
    return out
