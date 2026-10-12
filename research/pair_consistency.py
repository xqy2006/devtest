"""Which dense sequence statistics differ between the two labels in the same direction in every
collection batch? (selection on the three training batches only; outside panels reported as dev)

For every feature dimension j and batch b: standardized difference z_bj = (mean_S - mean_A) /
pooled SD within the batch, using only that batch's rows. Consistency = min over training batches of
sign-agreeing |z| (0 if the signs disagree). Then: how many dimensions are consistent beyond what
label-permutation (within batch and environment) produces, and do the selected dimensions keep their
sign on the outside panels?"""
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
TRB = L.TRB
OUT = ("old_outside", "fresh_outside")


def zdiff(rows, y):
    a, s = X[rows[y == 0]], X[rows[y == 1]]
    sd = np.sqrt(((a - a.mean(0)) ** 2).sum(0) + ((s - s.mean(0)) ** 2).sum(0)) / np.sqrt(len(rows) - 2)
    sd[sd < 1e-9] = np.inf
    return (s.mean(0) - a.mean(0)) / sd / np.sqrt(1 / len(a) + 1 / len(s))  # t statistic


def consistency(ylab):
    T = np.array([zdiff(L.rows((b,)), ylab[L.rows((b,))]) for b in TRB])
    same = (np.sign(T) == np.sign(T[0])).all(0)
    return np.where(same, np.abs(T).min(0), 0.0), T


cons, T = consistency(yb)
rng = np.random.default_rng(5)
null_max, null_counts = [], []
for _ in range(100):
    yp = yb.copy()
    for b in TRB:  # permute labels within batch x environment cells
        r = L.rows((b,))
        for e in set(p.env[r]):
            m = r[p.env[r] == e]
            yp[m] = rng.permutation(yb[m])
    c, _ = consistency(yp)
    null_max.append(c.max())
    null_counts.append([(c > t).sum() for t in (1.5, 2.0, 2.5)])
null_counts = np.array(null_counts)
print("dims:", len(cons))
for k, t in enumerate((1.5, 2.0, 2.5)):
    print(f"dims with min-batch |t| > {t} and same sign in all 3 training batches: observed {(cons > t).sum()}, "
          f"permutation null mean {null_counts[:, k].mean():.1f} (95%: {np.percentile(null_counts[:, k], 95):.0f})")
print("max consistency observed", round(float(cons.max()), 2), "null 95%", round(float(np.percentile(null_max, 95)), 2))
top = np.argsort(-cons)[:25]
To = np.array([zdiff(L.rows((b,)), yb[L.rows((b,))]) for b in OUT])
print(f"{'feature':18} {'t(orig)':>8} {'t(supp)':>8} {'t(10-11)':>8} | {'t(old)':>7} {'t(fresh)':>8}")
for j in top:
    print(f"{names[j]:18} {T[0, j]:8.2f} {T[1, j]:8.2f} {T[2, j]:8.2f} | {To[0, j]:7.2f} {To[1, j]:8.2f}")
sel = cons > 2.0
agree = (np.sign(To[:, sel]) == np.sign(T[0, sel])).mean(1)
print("selected (min |t|>2):", int(sel.sum()), "sign agreement on old/fresh outside:", agree.round(3))
Path(__file__).with_name("out_pair_consistency.json").write_text(json.dumps({
    "top": [[str(names[j]), T[:, j].round(2).tolist(), To[:, j].round(2).tolist()] for j in top],
    "null_counts_mean": null_counts.mean(0).tolist()}, indent=1))
