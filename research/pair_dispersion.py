"""Do the two labels differ in the SPREAD of dense statistics across responses (not the mean)?

Per feature j and batch b: t_bj = log(var_S / var_A) / SE, SE ~ sqrt(2/(nA-1) + 2/(nS-1)), with
variances computed after removing environment means within label (environment is not a label
effect). Consistency: same sign in all three training batches and min |t| over batches.
Null: label permutation inside batch x environment cells. Outside panels are reported only."""
import sys, warnings, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L

p, yb = L.p, L.yb
fams = list(L.FEAT)
X = np.hstack([L.FEAT[f] for f in fams])
names = np.concatenate([[f"{f}[{k}]" for k in range(L.FEAT[f].shape[1])] for f in fams])
famof = np.concatenate([[f] * L.FEAT[f].shape[1] for f in fams])


def logvar_t(rows, y):
    Z = X[rows].copy()
    for c in (0, 1):
        for e in set(p.env[rows]):
            m = (y == c) & (p.env[rows] == e)
            if m.sum():
                Z[m] -= Z[m].mean(0)
    a, s = Z[y == 0], Z[y == 1]
    va = (a ** 2).sum(0) / max(len(a) - 1, 1) + 1e-12
    vs = (s ** 2).sum(0) / max(len(s) - 1, 1) + 1e-12
    se = np.sqrt(2 / (len(a) - 1) + 2 / (len(s) - 1))
    return np.log(vs / va) / se


def consistency(ylab):
    T = np.array([logvar_t(L.rows((b,)), ylab[L.rows((b,))]) for b in L.TRB])
    same = (np.sign(T) == np.sign(T[0])).all(0)
    return np.where(same, np.abs(T).min(0), 0.0), T


cons, T = consistency(yb)
rng = np.random.default_rng(9)
nulls = []
for _ in range(100):
    yp = yb.copy()
    for b in L.TRB:
        r = L.rows((b,))
        for e in set(p.env[r]):
            m = r[p.env[r] == e]
            yp[m] = rng.permutation(yb[m])
    c, _ = consistency(yp)
    nulls.append([(c > t).sum() for t in (1.0, 1.5, 2.0)] + [c.max()])
nulls = np.array(nulls)
for k, t in enumerate((1.0, 1.5, 2.0)):
    print(f"dims with same-sign log-variance ratio in 3 batches and min|t|>{t}: observed {(cons > t).sum()}, "
          f"null mean {nulls[:, k].mean():.1f} (95%: {np.percentile(nulls[:, k], 95):.0f})")
print("max consistency", round(float(cons.max()), 2), "null 95%", round(float(np.percentile(nulls[:, 3], 95)), 2))
# family-level: mean signed t across dims, per batch (is one label systematically more variable?)
print("\nper family: mean log-variance t (Sol vs Astra) per training batch | outside old, fresh")
To = np.array([logvar_t(L.rows((b,)), yb[L.rows((b,))]) for b in ("old_outside", "fresh_outside")])
for f in fams:
    m = famof == f
    print(f"{f:10} " + " ".join(f"{T[b, m].mean():+6.2f}" for b in range(3)) + " | " + " ".join(f"{To[b, m].mean():+6.2f}" for b in range(2)))
top = np.argsort(-cons)[:15]
print("\ntop consistent dims:")
for j in top:
    print(f"{names[j]:18} " + " ".join(f"{T[b, j]:+6.2f}" for b in range(3)) + " | " + " ".join(f"{To[b, j]:+6.2f}" for b in range(2)))
