"""Systematic behaviour at decision points: does one label follow the pooled modal branch more
often (lower effective temperature at low-entropy points)?

Pooled counts from all training rows of both labels (exact leave-one-out for the row scored).
Decision point = position whose deepest available context (anchored prefix for t < 40, else the
previous two numbers) has pooled support >= S_MIN after removing the row itself. Per response:
mode-follow rate, mean pooled log-probability of the chosen value, mean rank, number of points.
Statistics are computed per batch; t of Sol - Astra."""
import sys, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L

p, yb = L.p, L.yb
S_MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rows = np.concatenate([L.rows(L.TRB), L.rows(("old_outside", "fresh_outside"))])
tr = L.rows(L.TRB)
SEQ = {int(i): list(map(int, p.rows[i]["numbers"])) for i in rows}


def ctxs(x, t):
    out = []
    if t < 40:
        out.append(("P",) + tuple(x[:t]))
    if t >= 2:
        out.append(("B", x[t - 2], x[t - 1]))
    return out


cnt = defaultdict(Counter)
for i in tr:
    x = SEQ[int(i)]
    for t, v in enumerate(x):
        for c in ctxs(x, t):
            cnt[c][v] += 1


def stats(i, loo):
    x = SEQ[int(i)]
    own = Counter((c, v) for t, v in enumerate(x) for c in ctxs(x, t)) if loo else Counter()
    own_t = Counter()
    for (c, v), k in own.items():
        own_t[c] += k
    mf, lp, rk = [], [], []
    for t, v in enumerate(x):
        for c in ctxs(x, t):  # deepest first
            d = cnt.get(c)
            if not d:
                continue
            tot = sum(d.values()) - own_t[c]
            if tot < S_MIN:
                continue
            adj = {u: n - own[(c, u)] for u, n in d.items() if n - own[(c, u)] > 0}
            mode = max(adj, key=adj.get)
            mf.append(v == mode)
            lp.append(np.log((adj.get(v, 0) + 0.5) / (tot + 0.5 * 355)))
            order = sorted(adj.values(), reverse=True)
            rk.append(order.index(adj[v]) if v in adj else len(order))
            break
    return [np.mean(mf) if mf else np.nan, np.mean(lp) if lp else np.nan, np.mean(rk) if rk else np.nan, len(mf)]


F = {int(i): stats(i, p.role[i] in L.TRB) for i in rows}
names = ["mode_follow", "mean_logp0", "mean_rank", "n_points"]
print(f"S_MIN={S_MIN}")
for b in L.TRB + ("old_outside", "fresh_outside"):
    r = L.rows((b,))
    X = np.array([F[int(i)] for i in r]); y = yb[r]
    line = []
    for k, n in enumerate(names):
        a, s = X[y == 0, k], X[y == 1, k]
        a, s = a[~np.isnan(a)], s[~np.isnan(s)]
        t = (s.mean() - a.mean()) / np.sqrt(a.var() / len(a) + s.var() / len(s))
        line.append(f"{n} A {a.mean():.3f} S {s.mean():.3f} t {t:+.2f}")
    print(f"{b[-12:]:>12}: " + " | ".join(line))
