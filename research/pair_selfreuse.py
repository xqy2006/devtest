"""In-context self-reuse statistics (dense; numbers only) and their per-batch label effects.

For each position t, does the response reuse something it has already written?
  jump_exact   d_t equals an earlier jump of this response
  jump_near    |d_t - d_s| <= 2 for some earlier s
  lag2_exact   x_t - x_{t-2} equals an earlier lag-2 offset
  digitpair    (last digit of x_{t-1}, last digit of x_t) seen earlier
  tens_pair    (x_{t-1}//10, x_t//10) seen earlier
  near_value   x_t within +-2 of an earlier value (excluding exact repeats)
  mirror       355+1-x_t was written earlier (reflection about the range centre)
  succ_reuse   x_t followed the same predecessor value range (x_{t-1}//36) as before with |x_t - x_s| <= 5
Each statistic is a rate over ~300 positions (first and second half separately)."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L

p, yb = L.p, L.yb
NAMES = ["jump_exact", "jump_near", "lag2_exact", "digitpair", "tens_pair", "near_value", "mirror", "succ_reuse"]


def selfreuse(x):
    x = list(map(int, x))
    n = len(x)
    seen_j, seen_l2, seen_dp, seen_tp, seen_v = set(), set(), set(), set(), set()
    js = []
    succ = {}
    rec = np.zeros((n, len(NAMES)))
    for t in range(n):
        v = x[t]
        if t >= 1:
            d = v - x[t - 1]
            rec[t, 0] = d in seen_j
            rec[t, 1] = any(abs(d - e) <= 2 for e in js[-400:])
            dp = (x[t - 1] % 10, v % 10)
            tp = (x[t - 1] // 10, v // 10)
            rec[t, 3] = dp in seen_dp
            rec[t, 4] = tp in seen_tp
            key = x[t - 1] // 36
            rec[t, 7] = any(abs(v - w) <= 5 for w in succ.get(key, []))
            seen_j.add(d); js.append(d); seen_dp.add(dp); seen_tp.add(tp)
            succ.setdefault(key, []).append(v)
        if t >= 2:
            l2 = v - x[t - 2]
            rec[t, 2] = l2 in seen_l2
            seen_l2.add(l2)
        rec[t, 5] = (v not in seen_v) and any((v + k) in seen_v for k in (-2, -1, 1, 2))
        rec[t, 6] = (356 - v) in seen_v and (356 - v) != v
        seen_v.add(v)
    h = n // 2
    return np.concatenate([rec[:h].mean(0), rec[h:].mean(0)])


idx = np.flatnonzero(np.isin(p.y, [L.A, L.S]))
F = np.zeros((len(p.rows), 2 * len(NAMES)))
for i in idx:
    F[i] = selfreuse(p.rows[i]["numbers"])
np.save(Path(__file__).with_name("cache_pair_selfreuse.npy"), F)
cols = [f"{n}_{h}" for h in ("1st", "2nd") for n in NAMES]
print(f"{'statistic':16}" + "".join(f"{b[-8:]:>10}" for b in L.TRB + ("old_outside", "fresh_outside")) + "   (t of Sol-Astra)")
for j, c in enumerate(cols):
    ts = []
    for b in L.TRB + ("old_outside", "fresh_outside"):
        r = L.rows((b,))
        a, s = F[r][yb[r] == 0, j], F[r][yb[r] == 1, j]
        ts.append((s.mean() - a.mean()) / np.sqrt(a.var() / len(a) + s.var() / len(s) + 1e-18))
    print(f"{c:16}" + "".join(f"{t:10.2f}" for t in ts))
