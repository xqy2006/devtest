"""Second round of dense number-sequence statistics (numbers only).

periodicity  internal chunking of the generator: log power spectrum of the |jump| series and of the
             band series on a fixed frequency grid, plus position-mod-10 profiles anchored at the start
digitstyle   digit aesthetics of the chosen values: roundness, digit sum, repeated digits,
             palindromes, primes, number of digits, last-digit steps between neighbours
condcurve    conditional behaviour given the current value: mean next value, mean |jump| and
             up-probability for each decile of the current value
"""
from __future__ import annotations

import numpy as np

V = 355
_PRIMES = np.array([all(n % k for k in range(2, int(n ** 0.5) + 1)) and n > 1 for n in range(V + 1)])
_FREQ = np.linspace(0.02, 0.5, 33)


def _cat(codes, k, alpha=0.5):
    c = np.bincount(codes, minlength=k).astype(float) + alpha
    return np.sqrt(c / c.sum())


def _logspec(series):
    z = (series - series.mean()) / max(series.std(), 1e-9)
    n = len(z)
    f = np.fft.rfftfreq(n)
    pw = np.abs(np.fft.rfft(z)) ** 2 / n
    out = []
    for lo, hi in zip(_FREQ[:-1], _FREQ[1:]):
        m = (f >= lo) & (f < hi)
        out.append(np.log(pw[m].mean() + 1e-6) if m.any() else 0.0)
    return np.array(out)


def periodicity(x):
    x = x.astype(float)
    ad = np.abs(np.diff(x))
    band = np.minimum((x - 1) * 4 // V, 3)
    prof_jump = np.array([ad[r::10].mean() for r in range(10)]) / V
    prof_val = np.array([x[r::10].mean() for r in range(10)]) / V
    return np.concatenate([_logspec(ad), _logspec(band), prof_jump - prof_jump.mean(), prof_val - prof_val.mean()])


def digitstyle(x):
    s = x.astype(np.int64)
    digits = [s % 10, (s // 10) % 10, s // 100]
    nd = np.where(s >= 100, 3, np.where(s >= 10, 2, 1))
    dsum = digits[0] + digits[1] + digits[2]
    rep = np.where(nd == 3, (digits[0] == digits[1]) | (digits[1] == digits[2]) | (digits[0] == digits[2]),
                   np.where(nd == 2, digits[0] == digits[1], False))
    pal = np.where(nd == 3, digits[0] == digits[2], np.where(nd == 2, digits[0] == digits[1], True))
    last = digits[0]
    step = (last[1:] - last[:-1]) % 10
    rates = np.array([np.mean(s % 5 == 0), np.mean(s % 10 == 0), np.mean(s % 25 == 0), np.mean(rep),
                      np.mean(pal), np.mean(_PRIMES[s]), np.mean(last[1:] == last[:-1]), np.mean(s % 2 == 1)])
    return np.concatenate([np.sqrt(rates + 0.005), _cat(nd - 1, 3), _cat(dsum % 9, 9), _cat(step, 10)])


def condcurve(x):
    x = x.astype(float)
    cur, nxt = x[:-1], x[1:]
    dec = np.minimum((cur - 1) * 10 // V, 9).astype(int)
    out = []
    for k in range(10):
        m = dec == k
        if m.sum() == 0:
            out += [0.5, 0.5, 0.5]
            continue
        out += [nxt[m].mean() / V, np.abs(nxt[m] - cur[m]).mean() / V, (np.mean(nxt[m] > cur[m]) * m.sum() + 0.5) / (m.sum() + 1)]
    return np.array(out)


SEQ2_FAMILIES = {"periodicity": periodicity, "digitstyle": digitstyle, "condcurve": condcurve}
