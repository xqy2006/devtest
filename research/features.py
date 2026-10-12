"""Response-level feature families. Every feature is computed from one response only.

No feature reads labels, environment, batch or any other response.
"""
from __future__ import annotations

import re

import numpy as np

VMAX = 355


def _sqrt_hist(values, bins, lo, hi, alpha=0.5):
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    counts = counts.astype(float) + alpha
    return np.sqrt(counts / counts.sum())


def value_hist(x, bins=355):
    return _sqrt_hist(x, bins, 0.5, VMAX + 0.5)


def signed_diff_hist(x, bins=36, lag=1):
    d = x[lag:] - x[:-lag]
    return _sqrt_hist(d, bins, -VMAX, VMAX)


def mod_diff_hist(x, bins=36, lag=1):
    d = (x[lag:] - x[:-lag]) % VMAX
    return _sqrt_hist(d, bins, 0, VMAX)


def abs_diff_hist(x, bins=24):
    return _sqrt_hist(np.abs(np.diff(x)), bins, 0, VMAX)


def digit_features(x):
    last = x % 10
    tens = (x // 10) % 10
    out = [_sqrt_hist(last, 10, -0.5, 9.5), _sqrt_hist(tens, 10, -0.5, 9.5),
           _sqrt_hist(x // 100, 4, -0.5, 3.5)]
    trans = np.zeros((10, 10)) + 0.1
    np.add.at(trans, (last[:-1], last[1:]), 1)
    out.append(np.sqrt(trans.ravel() / trans.sum()))
    ndig = np.where(x >= 100, 2, np.where(x >= 10, 1, 0))
    out.append(_sqrt_hist(ndig, 3, -0.5, 2.5))
    return np.concatenate(out)


def bin_transition(x, k=6):
    b = np.minimum((x - 1) * k // VMAX, k - 1)
    trans = np.zeros((k, k)) + 0.5
    np.add.at(trans, (b[:-1], b[1:]), 1)
    trans /= trans.sum(axis=1, keepdims=True)
    return np.sqrt(trans.ravel())


def updown_patterns(x, order=4):
    s = (np.diff(x) > 0).astype(int)
    feats = []
    for n in range(2, order + 1):
        if len(s) < n:
            feats.append(np.full(2 ** n, 2 ** (-n / 2)))
            continue
        codes = np.zeros(len(s) - n + 1, dtype=int)
        for j in range(n):
            codes = codes * 2 + s[j:len(s) - n + 1 + j]
        c = np.bincount(codes, minlength=2 ** n).astype(float) + 0.5
        feats.append(np.sqrt(c / c.sum()))
    return np.concatenate(feats)


def autocorr(x, lags=(1, 2, 3, 4, 5, 6, 8, 10)):
    z = (x - x.mean()) / max(x.std(), 1e-9)
    return np.array([np.mean(z[l:] * z[:-l]) for l in lags])


def repeat_features(x):
    n = len(x)
    uniq = len(set(x.tolist()))
    last_seen = {}
    gaps = []
    for i, v in enumerate(x.tolist()):
        if v in last_seen:
            gaps.append(i - last_seen[v])
        last_seen[v] = i
    gaps = np.array(gaps) if gaps else np.array([n])
    coverage = uniq / VMAX
    return np.array([1 - uniq / n, coverage, np.log1p(np.min(gaps)), np.log1p(np.median(gaps)),
                     np.mean(gaps < 10) if len(gaps) else 0.0])


def scalar_features(x, requested):
    d = np.diff(x)
    ad = np.abs(d)
    over = len(x) - requested
    return np.array([
        over, float(over == 0), np.sign(over), np.log1p(abs(over)), len(x) / requested,
        x.mean() / VMAX, x.std() / VMAX, np.median(x) / VMAX,
        ad.mean() / VMAX, ad.std() / VMAX, np.median(ad) / VMAX,
        np.mean(ad < 20), np.mean(ad > 250), np.mean(d > 0),
        np.mean(np.diff(np.sign(d)) != 0),
        x[:len(x) // 2].mean() / VMAX - x[len(x) // 2:].mean() / VMAX,
    ])


def position_features(x, head=12, bins=8):
    """Coarse value bins of the first ``head`` positions and the final positions."""
    out = []
    for v in list(x[:head]) + list(x[-4:]):
        one = np.zeros(bins)
        one[min(int((v - 1) * bins // VMAX), bins - 1)] = 1
        out.append(one)
    return np.concatenate(out)


def ordered_blocks(x, blocks=4, bins=16):
    pieces = []
    for chunk in np.array_split(x.astype(float), blocks):
        pieces.append(_sqrt_hist(chunk, bins, 1.0, 356.0))
    return np.concatenate(pieces)


def format_features(text):
    t = text.strip()
    seps = re.findall(r"\d+(\D*?)(?=\d)", t[:4000])
    sep_kinds = {",": 0, ", ": 1, " ": 2, "\n": 3, ",\n": 4}
    c = np.zeros(6)
    for s in seps:
        c[sep_kinds.get(s, 5)] += 1
    c = (c + 0.1) / (c.sum() + 0.6)
    newlines = t.count("\n")
    return np.concatenate([np.sqrt(c), [float(t.startswith("[")), float(t.endswith("]")),
                                        np.log1p(newlines), float(bool(re.search(r"[A-Za-z一-鿿]", t))),
                                        np.log1p(len(t) - len(re.sub(r"\D", "", t)))]])


FAMILIES = {
    "value355": lambda r: value_hist(r["x"], 355),
    "value71": lambda r: value_hist(r["x"], 71),
    "value36": lambda r: value_hist(r["x"], 36),
    "ordered": lambda r: ordered_blocks(r["x"]),
    "sdiff36": lambda r: signed_diff_hist(r["x"], 36),
    "sdiff72": lambda r: signed_diff_hist(r["x"], 72),
    "sdiff_lag2": lambda r: signed_diff_hist(r["x"], 36, 2),
    "sdiff_lag3": lambda r: signed_diff_hist(r["x"], 36, 3),
    "moddiff36": lambda r: mod_diff_hist(r["x"], 36),
    "absdiff": lambda r: abs_diff_hist(r["x"]),
    "digits": lambda r: digit_features(r["x"]),
    "bintrans6": lambda r: bin_transition(r["x"], 6),
    "bintrans10": lambda r: bin_transition(r["x"], 10),
    "updown": lambda r: updown_patterns(r["x"], 5),
    "autocorr": lambda r: autocorr(r["x"]),
    "repeat": lambda r: repeat_features(r["x"]),
    "scalar": lambda r: scalar_features(r["x"], r["requested_count"]),
    "position": lambda r: position_features(r["x"]),
    "format": lambda r: format_features(r["text"]),
}


def extract(rows, families=None):
    families = families or list(FAMILIES)
    out = {f: [] for f in families}
    for row in rows:
        r = {"x": np.asarray(row["numbers"], dtype=np.int64), "text": row["text"],
             "requested_count": row["requested_count"]}
        for f in families:
            out[f].append(FAMILIES[f](r))
    return {f: np.asarray(v, dtype=float) for f, v in out.items()}


def count_shape(x, requested):
    """Discretized count-precision and repetition behaviour (generic, label-free)."""
    over = len(x) - requested
    dups = len(x) - len(set(x.tolist()))
    ob = [over < 0, over == 0, 0 < over <= 5, 5 < over <= 15, over > 15]
    db = [dups == 0, dups == 1, dups == 2, 3 <= dups <= 5, dups > 5]
    first_dup = 1.0
    seen = set()
    for i, v in enumerate(x.tolist()):
        if v in seen:
            first_dup = i / len(x)
            break
        seen.add(v)
    return np.array(ob + db + [first_dup, np.log1p(dups)], dtype=float)


def chunk_structure(x, size=10):
    n = len(x) // size
    if n < 2:
        return np.zeros(6)
    blocks = x[: n * size].reshape(n, size)
    rng = (blocks.max(1) - blocks.min(1)) / VMAX
    quarters = np.array([len(set(((b - 1) * 4 // VMAX).tolist())) for b in blocks])
    within = np.array([len(set(b.tolist())) for b in blocks]) / size
    bm = blocks.mean(1)
    return np.array([rng.mean(), rng.std(), quarters.mean() / 4, within.mean(),
                     np.std(bm) / VMAX, np.corrcoef(bm[:-1], bm[1:])[0, 1] if n > 2 and np.std(bm) > 0 else 0.0])


FAMILIES["countshape"] = lambda r: count_shape(r["x"], r["requested_count"])
FAMILIES["chunk10"] = lambda r: chunk_structure(r["x"], 10)
FAMILIES["chunk5"] = lambda r: chunk_structure(r["x"], 5)


def scalar_no_request(x):
    """scalar_features without anything derived from the requested count."""
    return scalar_features(x, len(x))[5:]


FAMILIES["scalar_noreq"] = lambda r: scalar_no_request(r["x"])
