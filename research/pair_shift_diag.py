"""Diagnostics on the confusable pair with the extended library (labels used only for scoring).

1. Separability inside each batch (env-CV inside the batch) vs across batches.
2. Random (group-wise) CV vs environment CV on the pooled training rows: size of the env/batch shift.
3. Error correlation inside native triples (why k3 does not improve).
4. Per-environment accuracy of the pooled model on held-out environments.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import PAIR_ROLE, extended_panel  # noqa: E402
from research.features import extract  # noqa: E402

FAMS = ["value355", "sdiff72", "scalar", "repeat", "digits"]
p = extended_panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
pair = np.flatnonzero(np.isin(p.y, [A, S]))
F = extract([p.rows[i] for i in pair], FAMS)
X = np.zeros((len(p.rows), sum(F[f].shape[1] for f in FAMS)))
X[pair] = np.hstack([F[f] for f in FAMS])
yb = (p.y == S).astype(int)
OLD = ("train_original", "train_supplemental")


def fit_predict(ref, test):
    sc = StandardScaler().fit(X[ref])
    m = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X[ref]), yb[ref])
    return m.decision_function(sc.transform(X[test]))


def rows(roles):
    return np.flatnonzero(np.isin(p.y, [A, S]) & np.isin(p.role, roles))


def env_cv(idx):
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[idx])):
        te, tr = idx[p.env[idx] == e], idx[p.env[idx] != e]
        d[te] = fit_predict(tr, te)
    return d


def group_cv(idx, folds=12, seed=0):
    """Random folds over native groups (rows outside complete groups are their own unit)."""
    unit = {}
    for g, mem in enumerate(p.group_members):
        for i in mem:
            unit[int(i)] = g
    units = np.array([unit.get(int(i), -1 - int(i)) for i in idx])
    u = np.unique(units)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(u, rng.permutation(len(u)) % folds))
    f = np.array([fold_of[x] for x in units])
    d = np.zeros(len(p.rows))
    for k in range(folds):
        d[idx[f == k]] = fit_predict(idx[f != k], idx[f == k])
    return d


def acc(d, idx):
    return float(((d[idx] > 0) == yb[idx]).mean())


def k3(d, roles):
    g = p.group_members[np.isin(p.group_role, roles) & np.isin(p.group_y, [A, S])]
    return float(((d[g].sum(1) > 0) == yb[g[:, 0]]).mean()), len(g)


def triple_pattern(d, roles):
    g = p.group_members[np.isin(p.group_role, roles) & np.isin(p.group_y, [A, S])]
    right = ((d[g] > 0) == yb[g]).sum(1)
    obs = np.bincount(right, minlength=4) / len(g)
    q = right.mean() / 3
    ind = np.array([(1 - q) ** 3, 3 * q * (1 - q) ** 2, 3 * q * q * (1 - q), q ** 3])
    return obs.round(3), ind.round(3)


old, new = rows(OLD), rows((PAIR_ROLE,))
allr = rows(OLD + (PAIR_ROLE,))
print("1. inside-batch env-CV: old", round(acc(env_cv(old), old), 3), "codex", round(acc(env_cv(new), new), 3))
d = np.zeros(len(p.rows)); d[new] = fit_predict(old, new); print("   old->codex k1", round(acc(d, new), 3), "k3", k3(d, (PAIR_ROLE,)))
d = np.zeros(len(p.rows)); d[old] = fit_predict(new, old); print("   codex->old k1", round(acc(d, old), 3), "k3", k3(d, OLD))
de, dg = env_cv(allr), group_cv(allr)
print("2. pooled 3 batches: env-CV k1", round(acc(de, allr), 3), "group-CV k1", round(acc(dg, allr), 3))
for name, dd in (("env-CV", de), ("group-CV", dg)):
    for roles in (OLD, (PAIR_ROLE,)):
        k, n = k3(dd, roles)
        obs, ind = triple_pattern(dd, roles)
        print(f"   {name} {'+'.join(roles)}: k1 {acc(dd, rows(roles)):.3f} k3 {k:.3f}/{n}  #right per triple obs {obs} vs indep {ind}")
for role in ("old_outside", "fresh_outside"):
    d = np.zeros(len(p.rows)); te = rows((role,)); d[te] = fit_predict(allr, te)
    obs, ind = triple_pattern(d, (role,))
    print(f"3. pooled -> {role}: k1 {acc(d, te):.3f} k3 {k3(d, (role,))}  #right per triple obs {obs} vs indep {ind}")
    per = {}
    for e in sorted(set(p.env[te])):
        m = te[p.env[te] == e]
        per[e[-2:]] = f"{int(((d[m] > 0) == yb[m]).sum())}/{len(m)}"
    print("   per env:", per)
per = {}
for e in sorted(set(p.env[allr])):
    m = allr[p.env[allr] == e]
    per[e[-2:]] = round(acc(de, m), 2)
print("4. pooled env-CV per env:", per)
