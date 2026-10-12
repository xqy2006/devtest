"""Typicality test: do the two labels differ in effective sampling entropy under a shared generator?

A single pooled model of the cluster's generator (both labels' reference rows together) gives each
number a conditional probability. Per response we take dense typicality statistics:
  mean log p, SD of log p, mean log p per position third, and the mode-following rate (fraction of
  positions where the response emitted the most frequent continuation seen in the references).
A temperature/quantization difference moves these statistics in a fixed direction for every
response, with every number contributing.

Leakage control: statistics of reference rows are computed out-of-fold (pooled model refit without
the row's environment); test rows use the model fitted on all reference rows of that protocol.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.ngram import NGramClass  # noqa: E402
import research.pair_seq_lab as L  # noqa: E402

p, yb = L.p, L.yb


def pooled(ref, order):
    m = NGramClass(order=order)
    for i in ref:
        m.add(p.rows[i]["numbers"])
    return m.finalize()


def mode_table(ref, ctx_len=1):
    """Most frequent continuation per context of the previous ctx_len numbers."""
    from collections import Counter, defaultdict
    t = defaultdict(Counter)
    for i in ref:
        x = p.rows[i]["numbers"]
        for k in range(ctx_len, len(x)):
            t[tuple(x[k - ctx_len:k])][x[k]] += 1
    return {c: cnt.most_common(1)[0] for c, cnt in t.items()}


def stats(model, modes, i):
    x = p.rows[i]["numbers"]
    lp = model.logprob_seq(x)
    thirds = [c.mean() for c in np.array_split(lp, 3)]
    hits = seen = 0
    for k in range(2, len(x)):
        m = modes.get(tuple(x[k - 2:k]))
        if m and m[1] >= 2:
            seen += 1
            hits += int(x[k] == m[0])
    return [lp.mean(), lp.std(), *thirds, hits / max(seen, 1), seen / len(x)]


def features(ref, test, order):
    """Out-of-fold statistics for reference rows, full-reference statistics for test rows."""
    F = {}
    for e in sorted(set(p.env[ref])):
        inner = ref[p.env[ref] != e]
        m, modes = pooled(inner, order), mode_table(inner, 2)
        for i in ref[p.env[ref] == e]:
            F[i] = stats(m, modes, i)
    m, modes = pooled(ref, order), mode_table(ref, 2)
    for i in test:
        F[i] = stats(m, modes, i)
    return F


class Typicality:
    def __init__(self, order=3):
        self.order = order

    def fit(self, ref):
        self.ref = ref
        return self

    def decision(self, idx):
        F = features(self.ref, idx, self.order)
        Xr = np.array([F[i] for i in self.ref])
        Xt = np.array([F[i] for i in idx])
        sc = StandardScaler().fit(Xr)
        m = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(Xr), yb[self.ref])
        self.last_ref_stats = (Xr, yb[self.ref])
        return m.decision_function(sc.transform(Xt))


if __name__ == "__main__":
    order = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    # 1. effect sizes of each typicality statistic, per batch (out-of-fold inside the batch)
    names = ["mean_logp", "sd_logp", "logp_1st", "logp_2nd", "logp_3rd", "mode_follow", "ctx_seen"]
    for b in L.TRB + ("old_outside", "fresh_outside"):
        r = L.rows((b,))
        ref = L.rows(L.TRB) if b in ("old_outside", "fresh_outside") else L.rows(L.TRB)
        F = features(ref, r if b in ("old_outside", "fresh_outside") else np.array([], int), order)
        X = np.array([F[i] for i in r])
        y = yb[r]
        d = (X[y == 1].mean(0) - X[y == 0].mean(0)) / np.sqrt((X[y == 1].var(0) + X[y == 0].var(0)) / 2)
        print(f"{b[-12:]:>12} d(Sol-Astra): " + "  ".join(f"{n} {v:+.2f}" for n, v in zip(names, d)), flush=True)
    # 2. classifier on typicality statistics alone
    r = L.evaluate(lambda: Typicality(order))
    print(L.line(f"typicality(order {order})", r), flush=True)
