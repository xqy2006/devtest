"""Third round of dense number-sequence statistics (numbers only).

pairsum     complement / reflection moves: distribution of x_t + x_{t-k}, k = 1, 2, 3
recency     short-term avoidance radius: min |x_t - x_{t-j}| over the last 3 / 5 / 10 numbers
window      local uniformity of sliding windows (5 / 10 / 20): distinct deciles, range, local mean
kernel      transition kernel: signed jump distribution given the band of the current number
reversion   local mean reversion: corr(x_t, mean of previous k) and the balance of local means
gap         where the new number falls inside the unused interval that contains it, and how wide
            that interval is (gap-filling strategy), per position third
"""
from __future__ import annotations

import bisect

import numpy as np

V = 355


def _hist(values, bins, lo, hi, alpha=0.5):
    c, _ = np.histogram(values, bins=bins, range=(lo, hi))
    c = c.astype(float) + alpha
    return np.sqrt(c / c.sum())


def pairsum(x):
    x = x.astype(float)
    return np.concatenate([_hist(x[k:] + x[:-k], 48, 2, 2 * V) for k in (1, 2, 3)])


def recency(x):
    x = x.astype(float)
    out = []
    for k in (3, 5, 10):
        d = np.array([np.min(np.abs(x[t] - x[t - k:t])) for t in range(k, len(x))])
        out.append(_hist(np.log1p(d), 16, 0, np.log1p(V)))
    return np.concatenate(out)


def window(x):
    x = x.astype(float)
    out = []
    for w in (5, 10, 20):
        if len(x) < w + 1:
            out.append(np.zeros(5))
            continue
        win = np.lib.stride_tricks.sliding_window_view(x, w)
        dec = np.minimum((win - 1) * 10 // V, 9)
        distinct = np.array([len(set(r)) for r in dec]) / min(w, 10)
        rng = (win.max(1) - win.min(1)) / V
        lm = win.mean(1) / V
        out.append(np.array([distinct.mean(), distinct.std(), rng.mean(), np.abs(lm - 0.5).mean(), lm.std()]))
    return np.concatenate(out)


def kernel(x, bands=6, bins=16):
    x = x.astype(float)
    b = np.minimum((x[:-1] - 1) * bands // V, bands - 1).astype(int)
    d = x[1:] - x[:-1]
    out = []
    for k in range(bands):
        m = b == k
        out.append(_hist(d[m], bins, -V, V))
    return np.concatenate(out)


def reversion(x):
    x = x.astype(float)
    out = []
    for k in (2, 3, 5, 10):
        prev = np.array([x[t - k:t].mean() for t in range(k, len(x))])
        cur = x[k:]
        out.append(np.corrcoef(prev, cur)[0, 1] if prev.std() > 0 else 0.0)
        out.append(np.mean(np.abs((prev * k + cur) / (k + 1) - (V + 1) / 2)) / V)
    return np.array(out)


def gap(x, blocks=3):
    used = []
    pos, width = [], []
    for v in map(int, x):
        k = bisect.bisect_left(used, v)
        if k < len(used) and used[k] == v:
            bisect.insort(used, v)
            continue
        lo = used[k - 1] if k > 0 else 0
        hi = used[k] if k < len(used) else V + 1
        pos.append((v - lo) / (hi - lo))
        width.append(hi - lo - 1)
        bisect.insort(used, v)
    pos, width = np.array(pos), np.array(width, float)
    out = []
    for cp, cw in zip(np.array_split(pos, blocks), np.array_split(width, blocks)):
        out.append(_hist(cp, 10, 0, 1))
        out.append(_hist(np.log1p(cw), 10, 0, np.log1p(V)))
    return np.concatenate(out)


SEQ3_FAMILIES = {"pairsum": pairsum, "recency": recency, "window": window, "kernel": kernel,
                 "reversion": reversion, "gap": gap}


def _recurrence(codes, k, maxgap=30):
    last = {}
    gaps = []
    for t, c in enumerate(codes):
        if c in last:
            gaps.append(min(t - last[c], maxgap))
        last[c] = t
    g = np.array(gaps if gaps else [maxgap])
    c = np.bincount(g, minlength=maxgap + 1)[1:].astype(float) + 0.5
    return np.sqrt(c / c.sum())


def recurrence(x):
    """Steps until the same last digit / tens digit / 10-wide band / 30-wide band recurs."""
    x = x.astype(np.int64)
    return np.concatenate([_recurrence(x % 10, 10), _recurrence((x // 10) % 10, 10),
                           _recurrence((x - 1) // 10, 36), _recurrence((x - 1) // 30, 12)])


def _lis(seq):
    tails = []
    for v in seq:
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
        else:
            tails[k] = v
    return len(tails)


def permutation_shape(x):
    """Global order structure: longest increasing / decreasing subsequence, inversion rate,
    peak rate, mean run length (all normalized by length)."""
    x = list(map(int, x))
    n = len(x)
    arr = np.array(x)
    inv = sum(int((arr[i + 1:] < arr[i]).sum()) for i in range(n)) / (n * (n - 1) / 2)
    d = np.sign(np.diff(arr))
    peaks = np.mean((d[:-1] > 0) & (d[1:] < 0))
    runs = np.diff(np.flatnonzero(np.r_[True, d[1:] != d[:-1], True]))
    return np.array([_lis(x) / np.sqrt(n), _lis([-v for v in x]) / np.sqrt(n), inv, peaks, runs.mean(), runs.std()])


SEQ3_FAMILIES.update({"recurrence": recurrence, "permshape": permutation_shape})
