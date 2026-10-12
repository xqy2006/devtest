"""Out-of-library check of the 2026-10-11 paired reference supplement (pair-local scorer).

Same scorer and feature families as ``outside_learning_curve.py`` (frozen before the new data
existed). The supplement is reference-only: it is added to the library, never used as a test
panel for a claim. Tests are the exposed historical outside panels (development evidence) and,
for the frozen pre-supplement library only, a forward check on the supplement itself.

C: references = training batches + supplement, n per class drawn evenly over batches.
D: references = training + supplement + the other outside panel; test the remaining panel.
F: forward check: frozen 72-reference library -> supplement rows/triples.
R: reverse: supplement only as references -> historical training and outside rows.
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
from research.data import PAIR_ROLE, extended_panel  # noqa: E402
from research.features import extract  # noqa: E402

FAMS = ["value355", "sdiff72", "scalar", "repeat", "digits"]
REPS = 30
rng = np.random.default_rng(20261012)

p = extended_panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
pair = np.flatnonzero(np.isin(p.y, [A, S]))
F = extract([p.rows[i] for i in pair], FAMS)
X = np.zeros((len(p.rows), sum(F[f].shape[1] for f in FAMS)))
X[pair] = np.hstack([F[f] for f in FAMS])
yb = (p.y == S).astype(int)
TR = ("train_original", "train_supplemental")


def draw(pools, n):
    out = []
    for batches in pools:
        caps = np.array([len(b) for b in batches])
        sizes = np.zeros(len(batches), int)
        left = n
        while left:  # water-filling: spread evenly, respecting batch sizes
            open_ = np.flatnonzero(sizes < caps)
            assert len(open_), "not enough references"
            share = max(1, left // len(open_))
            for j in open_:
                take = min(share, caps[j] - sizes[j], left)
                sizes[j] += take
                left -= take
                if not left:
                    break
        out += [rng.choice(b, s, replace=False) for b, s in zip(batches, sizes)]
    return np.concatenate(out)


def fit_predict(ref, test):
    sc = StandardScaler().fit(X[ref])
    m = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X[ref]), yb[ref])
    return m.decision_function(sc.transform(X[test]))


def pools(roles):
    return [[np.flatnonzero((p.y == c) & (p.role == r)) for r in roles if ((p.y == c) & (p.role == r)).any()]
            for c in (A, S)]


def test_set(roles):
    test = np.flatnonzero(np.isin(p.y, [A, S]) & np.isin(p.role, roles))
    groups = p.group_members[np.isin(p.group_role, roles) & np.isin(p.group_y, [A, S])]
    pos = {int(i): k for k, i in enumerate(test)}
    return test, groups, np.array([[pos[int(i)] for i in g] for g in groups])


def evaluate(ref_roles, test_roles, sizes, reps=REPS):
    test, groups, gidx = test_set(test_roles)
    pl = pools(ref_roles)
    res = {}
    for n in sizes:
        k1 = np.zeros((reps, len(test)))
        k3 = np.zeros((reps, len(groups)))
        for r in range(reps):
            d = fit_predict(draw(pl, n), test)
            k1[r] = (d > 0) == yb[test]
            k3[r] = (d[gidx].sum(1) > 0) == yb[groups[:, 0]]
        res[n] = {"k1_item": k1.mean(0), "k3_group": k3.mean(0), "k1": float(k1.mean()), "k3": float(k3.mean()),
                  "A": float(k1.mean(0)[yb[test] == 0].mean()), "S": float(k1.mean(0)[yb[test] == 1].mean())}
    return res, gidx


def boot(res, gidx, lo, hi, B=4000):
    d1 = np.array([res[hi]["k1_item"][g].mean() - res[lo]["k1_item"][g].mean() for g in gidx])
    d3 = res[hi]["k3_group"] - res[lo]["k3_group"]
    r = np.random.default_rng(7)
    s = r.integers(0, len(gidx), (B, len(gidx)))
    b1, b3 = d1[s].mean(1), d3[s].mean(1)
    return {"k1_diff": float(d1.mean()), "k1_ci95": np.percentile(b1, [2.5, 97.5]).round(4).tolist(),
            "k1_p_le0": float((b1 <= 0).mean()),
            "k3_diff": float(d3.mean()), "k3_ci95": np.percentile(b3, [2.5, 97.5]).round(4).tolist(),
            "k3_p_le0": float((b3 <= 0).mean())}


def show(name, res, gidx, lo, hi):
    line = "  ".join(f"n={n}: k1 {v['k1']:.3f} (A {v['A']:.2f} S {v['S']:.2f}) k3 {v['k3']:.3f}" for n, v in res.items())
    print(f"{name}: {line}", flush=True)
    b = boot(res, gidx, lo, hi)
    print(f"   n={lo}->{hi}: {b}", flush=True)
    return {"curve": {n: {k: v[k] for k in ("k1", "k3", "A", "S")} for n, v in res.items()}, f"{lo}_vs_{hi}": b,
            "test_rows": int(len(res[lo]["k1_item"])), "test_groups": int(len(gidx))}


def main():
    out = {}
    npair = min(int(((p.y == c) & np.isin(p.role, TR + (PAIR_ROLE,))).sum()) for c in (A, S))
    for test_role in ("old_outside", "fresh_outside"):
        sizes = [72, 108, 144, 180, npair]
        res, gidx = evaluate(TR + (PAIR_ROLE,), (test_role,), sizes)
        out[f"C_train+codex->{test_role}"] = show(f"C train+codex -> {test_role}", res, gidx, 72, npair)
    for ref_extra, test_role in (("fresh_outside", "old_outside"), ("old_outside", "fresh_outside")):
        roles = TR + (PAIR_ROLE, ref_extra)
        nmax = min(int(((p.y == c) & np.isin(p.role, roles)).sum()) for c in (A, S))
        res, gidx = evaluate(roles, (test_role,), [72, 144, nmax])
        out[f"D_train+codex+{ref_extra}->{test_role}"] = show(f"D train+codex+{ref_extra} -> {test_role}", res, gidx, 72, nmax)
    # forward check with the full frozen 72-reference library (deterministic: one fit)
    test, groups, gidx = test_set((PAIR_ROLE,))
    ref = np.flatnonzero(np.isin(p.y, [A, S]) & np.isin(p.role, TR))
    d = fit_predict(ref, test)
    k1 = (d > 0) == yb[test]
    k3 = (d[gidx].sum(1) > 0) == yb[groups[:, 0]]
    out["F_frozen72->codex"] = {"k1": [int(k1.sum()), len(k1)], "A": [int(k1[yb[test] == 0].sum()), int((yb[test] == 0).sum())],
                                "S": [int(k1[yb[test] == 1].sum()), int((yb[test] == 1).sum())], "k3": [int(k3.sum()), len(k3)]}
    print("F frozen 72 refs -> codex supplement:", out["F_frozen72->codex"], flush=True)
    # reverse: supplement as the only references
    for test_roles in (TR, ("old_outside",), ("fresh_outside",)):
        test, groups, gidx = test_set(test_roles)
        ref = np.flatnonzero(np.isin(p.y, [A, S]) & (p.role == PAIR_ROLE))
        d = fit_predict(ref, test)
        k1 = (d > 0) == yb[test]
        k3 = (d[gidx].sum(1) > 0) == yb[groups[:, 0]]
        key = f"R_codex_only->{'+'.join(test_roles)}"
        out[key] = {"k1": [int(k1.sum()), len(k1)], "k3": [int(k3.sum()), len(k3)]}
        print(key, out[key], flush=True)
    Path(__file__).with_name("out_codex_pair_curve.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
