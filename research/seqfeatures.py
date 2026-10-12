"""Dense number-sequence feature families (numbers only; every emitted number contributes).

Excluded on purpose: anything driven by rare per-response events (count vs request, duplicate
counts), and anything read from the text layout. Each family is a smoothed proportion vector in
Hellinger (square-root) form, as in the production fingerprint, or a small set of dense moments.
"""
from __future__ import annotations

import numpy as np

V = 355


def _hist(values, bins, lo, hi, alpha=0.5):
    c, _ = np.histogram(values, bins=bins, range=(lo, hi))
    c = c.astype(float) + alpha
    return np.sqrt(c / c.sum())


def _cat(codes, k, alpha=0.5):
    c = np.bincount(codes, minlength=k).astype(float) + alpha
    return np.sqrt(c / c.sum())


def value(x, bins):
    """Marginal value preference."""
    return _hist(x, bins, 0.5, V + 0.5)


def residues(x, mods=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)):
    """Arithmetic preferences (parity, multiples of 5, ...)."""
    return np.concatenate([_cat(x % m, m) for m in mods])


def digit_structure(x):
    """Distribution of last, tens and hundreds digit, and the last-digit transition table."""
    last, tens, hund = x % 10, (x // 10) % 10, x // 100
    trans = last[:-1] * 10 + last[1:]
    return np.concatenate([_cat(last, 10), _cat(tens, 10), _cat(hund, 4), _cat(trans, 100, 0.2)])


def jumps(x, lag=1, bins=72):
    """Signed difference x_t - x_{t-lag}: how the generator moves through the value range."""
    return _hist(x[lag:] - x[:-lag], bins, -V, V)


def joint_jumps(x, bins=6):
    """Joint distribution of consecutive signed jumps (d_t, d_{t+1}): zig-zag / rebound structure."""
    d = x[1:] - x[:-1]
    b = np.clip(((d + V) * bins) // (2 * V + 1), 0, bins - 1)
    return _cat(b[:-1] * bins + b[1:], bins * bins, 0.2)


def band_ngrams(x, bands, order):
    """n-grams of coarse value bands: the order in which regions of the range are visited."""
    b = np.minimum((x - 1) * bands // V, bands - 1)
    codes = np.zeros(len(b) - order + 1, dtype=np.int64)
    for j in range(order):
        codes = codes * bands + b[j:len(b) - order + 1 + j]
    return _cat(codes, bands ** order, 0.2)


def coverage_dynamics(x, blocks=3, bins=10):
    """How each new number relates to the numbers already written (state tracking):
    distance to the nearest previously used value, and the relative rank of the new value
    among the still-unused values; both histogrammed per position block."""
    used = np.zeros(V + 2, dtype=bool)
    tree = np.zeros(V + 1, dtype=np.int64)  # Fenwick tree over used values

    def add(i):
        while i <= V:
            tree[i] += 1
            i += i & -i

    def prefix(i):
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    sorted_used: list[int] = []
    import bisect
    dnn, urank = [], []
    for t, v in enumerate(x.tolist()):
        if t:
            k = bisect.bisect_left(sorted_used, v)
            near = min(abs(v - sorted_used[j]) for j in (k - 1, k) if 0 <= j < len(sorted_used))
            dnn.append(near)
            unused_below = (v - 1) - prefix(v - 1)
            unused_total = V - len(sorted_used)
            urank.append(unused_below / max(unused_total, 1))
        if not used[v]:
            used[v] = True
            add(v)
            bisect.insort(sorted_used, v)
    dnn, urank = np.array(dnn, float), np.array(urank, float)
    out = []
    for chunk_d, chunk_r in zip(np.array_split(dnn, blocks), np.array_split(urank, blocks)):
        out.append(_hist(np.log1p(chunk_d), bins, 0, np.log1p(60)))
        out.append(_hist(chunk_r, bins, 0, 1))
    return np.concatenate(out)


def ordered(x, blocks=4, bins=16):
    """Production-style ordered blocks: value distribution in successive position blocks."""
    return np.concatenate([_hist(c, bins, 0.5, V + 0.5) for c in np.array_split(x, blocks)])


def moments(x):
    """Dense scalar moments (no counts): autocorrelations, sign alternation, jump size moments.
    (No small-jump rate: GPT-6 outputs almost never jump by < 30, so that rate is a rare-event count.)"""
    z = (x - x.mean()) / max(x.std(), 1e-9)
    d = np.diff(x)
    ad = np.abs(d)
    s = np.sign(d)
    return np.array([*(np.mean(z[l:] * z[:-l]) for l in (1, 2, 3, 4, 5, 6, 8, 10)),
                     np.mean(s[1:] != s[:-1]), ad.mean() / V, ad.std() / V, np.median(ad) / V,
                     np.mean(ad > 250), x.mean() / V, x.std() / V,
                     np.corrcoef(ad[:-1], ad[1:])[0, 1]])


def updown(x, order=5):
    s = (np.diff(x) > 0).astype(np.int64)
    out = []
    for n in range(2, order + 1):
        codes = np.zeros(len(s) - n + 1, dtype=np.int64)
        for j in range(n):
            codes = codes * 2 + s[j:len(s) - n + 1 + j]
        out.append(_cat(codes, 2 ** n))
    return np.concatenate(out)


SEQ_FAMILIES = {
    "value355": lambda x: value(x, 355),
    "value71": lambda x: value(x, 71),
    "value36": lambda x: value(x, 36),
    "residues": residues,
    "digits": digit_structure,
    "jump1": lambda x: jumps(x, 1),
    "jump2": lambda x: jumps(x, 2),
    "jump3": lambda x: jumps(x, 3),
    "jointjump": joint_jumps,
    "band4tri": lambda x: band_ngrams(x, 4, 3),
    "band6bi": lambda x: band_ngrams(x, 6, 2),
    "band10bi": lambda x: band_ngrams(x, 10, 2),
    "coverage": coverage_dynamics,
    "ordered": ordered,
    "moments": moments,
    "updown": updown,
}


def extract_seq(rows, families=None):
    families = families or list(SEQ_FAMILIES)
    out = {f: [] for f in families}
    for row in rows:
        x = np.asarray(row["numbers"], dtype=np.int64)
        for f in families:
            out[f].append(SEQ_FAMILIES[f](x))
    return {f: np.asarray(v, dtype=float) for f, v in out.items()}
