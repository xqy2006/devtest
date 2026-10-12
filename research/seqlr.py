"""Pooled-prior class-conditional sequential likelihood ratio (numbers only).

The cluster members share one generator; a label differs only in how often it takes each branch.
Contexts, shallow to deep: root (value marginal), previous number, previous two numbers, and for
the first T positions the full anchored prefix. A pooled model (all members' references) is a
hierarchical Dirichlet chain: P0(v|ctx) = (n(ctx,v) + lam * P0(v|parent)) / (n(ctx) + lam).
Each label's model at a context is shrunk to the pooled model at the SAME context:
P_c(v|ctx) = (n_c(ctx,v) + lam * P0(v|ctx)) / (n_c(ctx) + lam),
so a context contributes evidence only where that label's own counts there depart from the pool.
Per response the log-ratio is summed over positions separately for each context depth.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

V = 355
LEVELS = ("root", "prev1", "prev2", "prefix")
ANCHORED = ("root", "pos", "pos_prev1", "prefix")


def anchored_contexts(x, t, T):
    """Position-anchored chain for the opening phase: value at position t, then (t, previous
    number), then the full prefix; positions >= T only use the root."""
    if t >= T:
        return [("R",), None, None, None]
    return [("R",), ("T", t), ("TU", t, x[t - 1]) if t >= 1 else None, ("P",) + tuple(x[:t])]


def coarse_contexts(x, t, T, width=12):
    """Opening-phase contexts on coarsened values (near-identical paths pool their support):
    previous two numbers, coarse anchored prefix, exact anchored prefix (deepest)."""
    if t >= T:
        return [("R",), None, None, None]
    c = tuple(v // width for v in x[:t])
    return [("R",), ("CB", t, c[-2:]) if t >= 1 else None, ("CP",) + c, ("P",) + tuple(x[:t])]


def contexts(x, t, T):
    out = [("R",)]
    if t >= 1:
        out.append(("U", x[t - 1]))
    else:
        out.append(None)
    if t >= 2:
        out.append(("B", x[t - 2], x[t - 1]))
    else:
        out.append(None)
    out.append(("P",) + tuple(x[:t]) if t < T else None)
    return out


class SeqLR:
    """min_support: contexts (other than the root) whose pooled count is below this contribute no
    evidence -- the label difference is only read at well-supported decision points."""

    def __init__(self, lam=5.0, T=40, min_support=0, anchored=False, coarse=0):
        self.lam, self.T, self.min_support = lam, T, min_support
        if coarse:
            self.ctxfn = lambda x, t, T: coarse_contexts(x, t, T, coarse)
        else:
            self.ctxfn = anchored_contexts if anchored else contexts

    def fit(self, seqs, labels):
        """seqs: list of int lists; labels: 0/1."""
        self.n = [defaultdict(Counter), defaultdict(Counter)]
        for x, c in zip(seqs, labels):
            for ctx, v in self._events(x):
                self.n[c][ctx][v] += 1
        self.tot = [{k: sum(cnt.values()) for k, cnt in nc.items()} for nc in self.n]
        return self

    def _events(self, x):
        for t, v in enumerate(x):
            for ctx in self.ctxfn(x, t, self.T):
                if ctx is not None:
                    yield ctx, v

    def level_scores(self, x, own_label=None):
        """Summed log P_1 - log P_0 per context depth. If own_label is given, the response's own
        counts are removed first (exact leave-one-out for reference rows)."""
        own = Counter(self._events(x)) if own_label is not None else Counter()
        own_tot = Counter()
        for (ctx, v), k in own.items():
            own_tot[ctx] += k
        lam = self.lam
        out = np.zeros(len(LEVELS))
        for t, v in enumerate(x):
            p0 = 1.0 / V
            for lev, ctx in enumerate(self.ctxfn(x, t, self.T)):
                if ctx is None:
                    continue
                nc = []
                for c in (0, 1):
                    k = self.n[c].get(ctx, {}).get(v, 0)
                    tot = self.tot[c].get(ctx, 0)
                    if own_label == c:
                        k -= own[(ctx, v)]
                        tot -= own_tot[ctx]
                    nc.append((k, tot))
                n_v = nc[0][0] + nc[1][0]
                n_t = nc[0][1] + nc[1][1]
                p0 = (n_v + lam * p0) / (n_t + lam)
                p1 = (nc[1][0] + lam * p0) / (nc[1][1] + lam)
                pa = (nc[0][0] + lam * p0) / (nc[0][1] + lam)
                if lev == 0 or n_t >= self.min_support:
                    out[lev] += np.log(p1) - np.log(pa)
        return out
