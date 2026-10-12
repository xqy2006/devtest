"""Triple-level evidence for k=3: how similar a model's three independent responses are.

The three responses of a native triple are independent requests in the same environment with
near-identical prompts. Their mutual similarity estimates how concentrated the model's sequence
distribution is (a collision-probability / determinism measure), which a quantized or distilled
copy can change while keeping the same number preferences. One response alone cannot measure it.

Statistics per pair of responses (averaged over the three pairs of the triple), numbers only and
dense (every position or every shared value contributes):
  prefix     common opening prefix length (capped at 20)
  open10/30  overlap of the value sets of the first 10 / 30 positions
  bigram     shared exact consecutive pairs / length
  band2      shared 36-band consecutive pairs (multiset overlap) / length
  timing     Spearman correlation of first-appearance positions over values used by both
  timing100  same, restricted to values that appear within the first 100 positions of both
  jumpcorr   correlation of the two signed-jump histograms (24 bins)
Batch consistency: Welch t of Sol minus Astra in each training batch, plus the outside panels.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.pair_seq_lab as L  # noqa: E402

p, yb = L.p, L.yb
NAMES = ["prefix", "open10", "open30", "bigram", "band2", "timing", "timing100", "jumpcorr"]


def first_pos(x):
    d = {}
    for t, v in enumerate(x):
        d.setdefault(v, t)
    return d


def jhist(x):
    d = np.diff(np.asarray(x))
    return np.histogram(d, bins=np.linspace(-354, 354, 25))[0].astype(float)


def pair_stats(x, y):
    n = min(len(x), len(y))
    pre = 0
    while pre < min(20, n) and x[pre] == y[pre]:
        pre += 1
    o10 = len(set(x[:10]) & set(y[:10])) / 10
    o30 = len(set(x[:30]) & set(y[:30])) / 30
    bx = set(zip(x[:-1], x[1:]))
    by = set(zip(y[:-1], y[1:]))
    big = len(bx & by) / n
    from collections import Counter
    cx = Counter(zip([(v - 1) * 36 // 355 for v in x[:-1]], [(v - 1) * 36 // 355 for v in x[1:]]))
    cy = Counter(zip([(v - 1) * 36 // 355 for v in y[:-1]], [(v - 1) * 36 // 355 for v in y[1:]]))
    band2 = sum((cx & cy).values()) / n
    fx, fy = first_pos(x), first_pos(y)
    common = sorted(set(fx) & set(fy))
    tim = spearmanr([fx[v] for v in common], [fy[v] for v in common])[0]
    c100 = [v for v in common if fx[v] < 100 and fy[v] < 100]
    tim100 = spearmanr([fx[v] for v in c100], [fy[v] for v in c100])[0] if len(c100) > 5 else 0.0
    jc = np.corrcoef(jhist(x), jhist(y))[0, 1]
    return [pre, o10, o30, big, band2, tim, tim100, jc]


def triple_stats(members):
    seqs = [list(map(int, p.rows[i]["numbers"])) for i in members]
    return np.mean([pair_stats(a, b) for a, b in itertools.combinations(seqs, 2)], axis=0)


def all_triples():
    roles = L.TRB + ("old_outside", "fresh_outside")
    G = L.groups(roles)
    S = np.array([triple_stats(g) for g in G])
    return G, S


if __name__ == "__main__":
    G, S = all_triples()
    np.savez(Path(__file__).with_name("cache_pair_triple.npz"), G=G, S=S)
    y = yb[G[:, 0]]
    role = p.role[G[:, 0]]
    print(f"{'statistic':10} " + " ".join(f"{r[-10:]:>22}" for r in L.TRB + ('old_outside', 'fresh_outside')))
    for j, nm in enumerate(NAMES):
        cells = []
        for r in L.TRB + ("old_outside", "fresh_outside"):
            a, s = S[(role == r) & (y == 0), j], S[(role == r) & (y == 1), j]
            t = (s.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / len(a) + s.var(ddof=1) / len(s) + 1e-12)
            cells.append(f"A {a.mean():6.3f} S {s.mean():6.3f} t{t:+5.1f}")
        print(f"{nm:10} " + " | ".join(cells))
