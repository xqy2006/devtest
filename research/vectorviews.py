"""Systematic vector representations of one response (numbers only).

rff_*     kernel mean embeddings: every consecutive pair (x_{t-1}, x_t), triple (x_{t-2}, x_{t-1}, x_t)
          or jump pair (d_t, d_{t+1}) is a point; the response is the mean of random Fourier
          features of its points, i.e. a smoothed density of its transitions (MMD geometry).
          Several bandwidths, from single-value resolution to band resolution.
tokens    hashed bag of n-grams (exact values 1-3, 36-band 2-4, jump bins 1-3, opening prefixes)
          with sublinear term frequency, as in text mining.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy import sparse

V = 355
_RNG = np.random.default_rng(20261011)
D_RFF = 384
_W = {}


def _weights(dim, bw):
    key = (dim, bw)
    if key not in _W:
        _W[key] = (_RNG.normal(0, 1 / bw, size=(dim, D_RFF)), _RNG.uniform(0, 2 * np.pi, D_RFF))
    return _W[key]


def _rff(points, bw):
    W, b = _weights(points.shape[1], bw)
    return np.sqrt(2 / D_RFF) * np.cos(points @ W + b).mean(0)


def rff_pairs(x, bw):
    u = (x.astype(float) - 1) / (V - 1)
    return _rff(np.column_stack([u[:-1], u[1:]]), bw)


def rff_triples(x, bw):
    u = (x.astype(float) - 1) / (V - 1)
    return _rff(np.column_stack([u[:-2], u[1:-1], u[2:]]), bw)


def rff_jumps(x, bw):
    d = np.diff(x.astype(float)) / (2 * V)
    return _rff(np.column_stack([d[:-1], d[1:]]), bw)


RFF_VIEWS = {}
for _bw in (0.01, 0.03, 0.1):
    RFF_VIEWS[f"rffpair{_bw}"] = (lambda bw: (lambda x: rff_pairs(x, bw)))(_bw)
    RFF_VIEWS[f"rfftri{_bw}"] = (lambda bw: (lambda x: rff_triples(x, bw)))(_bw)
    RFF_VIEWS[f"rffjump{_bw}"] = (lambda bw: (lambda x: rff_jumps(x, bw)))(_bw)

HASH_DIM = 2 ** 16


def _h(s):
    return zlib.crc32(s.encode()) % HASH_DIM


def tokens(x):
    x = list(map(int, x))
    band = [(v - 1) // 10 for v in x]
    jb = [int(np.clip((b - a + V) * 24 // (2 * V + 1), 0, 23)) for a, b in zip(x[:-1], x[1:])]
    toks = []
    for n in (1, 2, 3):
        toks += ["v%d:%s" % (n, ",".join(map(str, x[t:t + n]))) for t in range(len(x) - n + 1)]
        toks += ["j%d:%s" % (n, ",".join(map(str, jb[t:t + n]))) for t in range(len(jb) - n + 1)]
    for n in (2, 3, 4):
        toks += ["b%d:%s" % (n, ",".join(map(str, band[t:t + n]))) for t in range(len(band) - n + 1)]
    toks += ["p:%s" % ",".join(map(str, x[:t + 1])) for t in range(min(12, len(x)))]
    idx, cnt = np.unique([_h(t) for t in toks], return_counts=True)
    return idx, 1 + np.log(cnt)


def token_matrix(seqs):
    rows, cols, vals = [], [], []
    for r, x in enumerate(seqs):
        idx, w = tokens(x)
        rows += [r] * len(idx)
        cols += idx.tolist()
        vals += w.tolist()
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(seqs), HASH_DIM))
