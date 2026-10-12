"""Does adding reference samples raise OUT-OF-LIBRARY accuracy on the hardest pair?

The pair is the automatically discovered confusable cluster (fixed here). The scorer is the
cluster-local shrinkage LDA on the frozen feature families chosen earlier from training
protocols only. For each reference size n per class we draw random reference subsets,
fit, and score outside rows that are never in the reference set.

Curve A: references drawn only from the training batches (n <= 72); tests: old and fresh outside.
Curve B: references may also include the *other* outside panel (n up to ~96/108); test is the
remaining panel. Outside labels are exposed, so all of this is development evidence.

Uncertainty: paired bootstrap over native test groups (triples), averaged over subsample draws.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import Panel  # noqa: E402
from research.features import extract  # noqa: E402

FAMS = ["value355", "sdiff72", "scalar", "repeat", "digits"]
REPS = 40
rng = np.random.default_rng(20261011)

p = Panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
pair = np.flatnonzero(np.isin(p.y, [A, S]))
F = extract([p.rows[i] for i in pair], FAMS)
Xp = np.hstack([F[f] for f in FAMS])
X = np.zeros((len(p.rows), Xp.shape[1]))
X[pair] = Xp
yb = (p.y == S).astype(int)


def draw(pool_by_class, n):
    """Random n references per class, spread evenly over that class's batches."""
    out = []
    for batches in pool_by_class:
        k = len(batches)
        sizes = [n // k + (1 if j < n % k else 0) for j in range(k)]
        # if a batch is too small, move the remainder to the others
        caps = [len(b) for b in batches]
        extra = 0
        for j in range(k):
            if sizes[j] > caps[j]:
                extra += sizes[j] - caps[j]
                sizes[j] = caps[j]
        for j in range(k):
            take = min(extra, caps[j] - sizes[j])
            sizes[j] += take
            extra -= take
        assert extra == 0, "not enough references"
        for b, s in zip(batches, sizes):
            out.append(rng.choice(b, s, replace=False))
    return np.concatenate(out)


def fit_predict(ref, test):
    sc = StandardScaler().fit(X[ref])
    m = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X[ref]), yb[ref])
    return m.decision_function(sc.transform(X[test]))


def batches(cls, roles):
    return [np.flatnonzero((p.y == cls) & (p.role == r)) for r in roles if ((p.y == cls) & (p.role == r)).any()]


def curve(ref_roles, test_role, sizes):
    test = np.flatnonzero(np.isin(p.y, [A, S]) & (p.role == test_role))
    gmask = (p.group_role == test_role) & np.isin(p.group_y, [A, S])
    groups = p.group_members[gmask]
    pos = {int(i): k for k, i in enumerate(test)}
    gidx = np.array([[pos[int(i)] for i in g] for g in groups])
    pools = [batches(A, ref_roles), batches(S, ref_roles)]
    res = {}
    for n in sizes:
        k1 = np.zeros((REPS, len(test)))
        k3 = np.zeros((REPS, len(groups)))
        for r in range(REPS):
            ref = draw(pools, n)
            d = fit_predict(ref, test)
            k1[r] = (d > 0) == yb[test]
            k3[r] = (d[gidx].sum(1) > 0) == yb[groups[:, 0]]
        res[n] = {"k1_item": k1.mean(0), "k3_group": k3.mean(0), "k1_rep_sd": float(k1.mean(1).std()),
                  "k1": float(k1.mean()), "k3": float(k3.mean())}
    return res, gidx


def bootstrap_diff(res, gidx, n_lo, n_hi, B=4000):
    """Paired bootstrap over native test groups of acc(n_hi) - acc(n_lo)."""
    G = len(gidx)
    d1 = np.array([res[n_hi]["k1_item"][g].mean() - res[n_lo]["k1_item"][g].mean() for g in gidx])
    d3 = res[n_hi]["k3_group"] - res[n_lo]["k3_group"]
    bs1, bs3 = [], []
    brng = np.random.default_rng(7)
    for _ in range(B):
        s = brng.integers(0, G, G)
        bs1.append(d1[s].mean())
        bs3.append(d3[s].mean())
    return {"k1_diff": float(d1.mean()), "k1_ci95": [float(np.percentile(bs1, 2.5)), float(np.percentile(bs1, 97.5))],
            "k1_p_le0": float(np.mean(np.array(bs1) <= 0)),
            "k3_diff": float(d3.mean()), "k3_ci95": [float(np.percentile(bs3, 2.5)), float(np.percentile(bs3, 97.5))]}


def main():
    out = {}
    TR = ("train_original", "train_supplemental")
    plans = {
        "A_train_refs->old": (TR, "old_outside", [12, 24, 36, 48, 60, 72]),
        "A_train_refs->fresh": (TR, "fresh_outside", [12, 24, 36, 48, 60, 72]),
        "B_train+fresh_refs->old": (TR + ("fresh_outside",), "old_outside", [24, 48, 72, 96, 108]),
        "B_train+old_refs->fresh": (TR + ("old_outside",), "fresh_outside", [24, 48, 72, 96]),
    }
    for name, (roles, test_role, sizes) in plans.items():
        res, gidx = curve(roles, test_role, sizes)
        line = "  ".join(f"n={n}: k1 {v['k1']:.3f} (±{v['k1_rep_sd']:.3f}) k3 {v['k3']:.3f}" for n, v in res.items())
        print(f"{name}: {line}", flush=True)
        diff = bootstrap_diff(res, gidx, sizes[0], sizes[-1])
        mid = bootstrap_diff(res, gidx, sizes[len(sizes) // 2 - 1], sizes[-1])
        print(f"   n={sizes[0]}->{sizes[-1]}: {diff}", flush=True)
        print(f"   n={sizes[len(sizes)//2-1]}->{sizes[-1]}: {mid}", flush=True)
        out[name] = {"curve": {n: {k: v[k] for k in ("k1", "k3", "k1_rep_sd")} for n, v in res.items()},
                     "lo_vs_hi": diff, "mid_vs_hi": mid, "test_rows": int(sum(len(g) for g in gidx))}
        np.savez(Path(__file__).with_name(f"lc_{name.replace('>', '')}.npz"), sizes=np.array(sizes), gidx=gidx,
                 k1=np.stack([res[n]["k1_item"] for n in sizes]), k3=np.stack([res[n]["k3_group"] for n in sizes]))
    Path(__file__).with_name("out_outside_learning_curve.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
