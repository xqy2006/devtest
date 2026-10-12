"""Stacked Astra/Sol evidence on full-length (1000-number) pilot responses, one outer fold per job.

Run 1 showed that the evidence sources are strongly rank-correlated (dense vs score Spearman 0.68).
It also showed single extreme decisions: a normal-looking response got a score decision of
+4553 / -24595 against a median of about 6, from standardized features whose reference variance
is almost zero. The fixes here are generic, with nothing specific to one response:
  - a variance floor in the standardization (sd >= 0.1 * median sd);
  - each source's query decision divided by the robust scale (MAD) of its out-of-fold reference
    decisions and clipped to the range of those decisions (no extrapolation beyond the
    reference support).
Then the sources are combined.

Sources (all fitted on the outer references only):
  dense     v3 families, dense LDA (as run 1), with and without the variance floor
  score     Fisher-score LDA (val, dg, pb); the shared label-free model is fitted on the outer
            references, as in the deliverable
  llr10 / llr100   log-likelihood ratio of label models shrunk towards the shared model
            (lam2 = 10 / 100)
  win0lib   frozen library cluster model (v4 local) on the first 250 numbers; the library never
            contains pilot rows
Out-of-fold reference decisions come from an inner leave-one-environment-out inside the outer
references. Combinations, all pre-specified:
  - equal-weight sums of the scaled, clipped sources;
  - L2 logistic stacking on the reference out-of-fold decisions.

Usage:
  python research/stack_pair.py fold <j>     (j = 0..11 LOEO environments, 12..13 dev<->test)
  python research/stack_pair.py merge <dir>  (reads stack_fold_*.json, writes stack.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.runner_pair import (FS, G, MEMBERS, OUT, SMOKE, V3, Saver, cluster_decisions, family_matrix,  # noqa: E402
                                  folds, library_pair_rows, load_pilot, log, metrics)
from research.pilot_length_matched import env_basis  # noqa: E402

T = 1000
SG = ("val", "dg", "pb")
LAMS = (10.0, 100.0)
SOURCES = ("dense", "dense_floor", "score", "score_floor", "llr10", "llr100", "win0lib")
COMBOS = {
    "dense+score": ("dense_floor", "score_floor"),
    "dense+score+llr10": ("dense_floor", "score_floor", "llr10"),
    "dense+score+llr100": ("dense_floor", "score_floor", "llr100"),
    "dense+score+win0lib": ("dense_floor", "score_floor", "win0lib"),
    "dense+llr100": ("dense_floor", "llr100"),
    "all_floor": ("dense_floor", "score_floor", "llr10", "llr100", "win0lib"),
}


def lda_fit(X, w, y, env, ref, floor):
    mu, sd = X[ref].mean(0), X[ref].std(0)
    if floor:
        sd = np.maximum(sd, floor * np.median(sd[sd > 1e-12]))
    sd[sd < 1e-12] = np.inf
    Z = (X[ref] - mu) / sd * w
    B = env_basis(Z, y[ref], env[ref])
    m = LDA(solver="lsqr", shrinkage="auto").fit(Z - (Z @ B.T) @ B, y[ref])

    def dec(idx):
        Q = (X[idx] - mu) / sd * w
        return m.decision_function(Q - (Q @ B.T) @ B)

    def zmax(idx):
        Q = (X[idx] - mu) / sd
        return np.abs(Q).max(1), np.abs(Q).argmax(1)
    return dec, zmax


class Sources:
    """All sources for one reference set; decisions(idx) for any pilot rows."""

    def __init__(self, ctx, ref, theta0):
        rows, y, env, st, sq, Xd, wd, U, wu, win0 = ctx
        self.dec = {}
        self.dec["dense"], _ = lda_fit(Xd, wd, y, env, ref, None)
        self.dec["dense_floor"], _ = lda_fit(Xd, wd, y, env, ref, 0.1)
        self.dec["score"], self.score_z = lda_fit(U, wu, y, env, ref, None)
        self.dec["score_floor"], _ = lda_fit(U, wu, y, env, ref, 0.1)
        mask = np.zeros(len(y), bool)
        mask[ref] = True
        for lam2 in LAMS:
            th = [FS.fit_base(st, mask & (y == c), lam=lam2, maxiter=G.SCORE_MAXITER, center=theta0) for c in (0, 1)]
            llr = FS.row_loglik(sq, th[1]) - FS.row_loglik(sq, th[0])
            self.dec[f"llr{int(lam2)}"] = lambda idx, v=llr: v[idx]
        self.dec["win0lib"] = lambda idx: win0[idx]

    def __call__(self, idx):
        return {k: f(idx) for k, f in self.dec.items()}


def context():
    rows, y, env, split, groups = load_pilot()
    seqs = [r["numbers"][:T] for r in rows]
    F = family_matrix(seqs, list(V3))
    Xd = np.hstack([F[f] for f in V3])
    wd = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in V3])
    st = FS.Steps(seqs)
    lib = library_pair_rows()
    cl = G.fit_cluster(lib, MEMBERS, [0, 1])
    del lib
    win0 = np.array([cluster_decisions(cl, x[:250])["local"] for x in seqs])
    log("library window-0 decisions done")
    return rows, y, env, split, groups, seqs, Xd, wd, st, win0


def run_fold(j):
    rows, y, env, split, groups, seqs, Xd, wd, st, win0 = context()
    fl = folds(env, split)
    allf = fl["loeo"] + fl.get("devtest", [])
    ref, te = allf[j]
    proto = "loeo" if j < len(fl["loeo"]) else "devtest"
    sv = Saver(f"stack_fold_{j}")
    sv["meta"] = {"proto": proto, "test_rows": te.tolist()}
    theta0 = FS.fit_base(st, np.isin(np.arange(len(y)), ref), lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
    U = FS.score_vectors(st, theta0)
    cols = [U[:, FS.OFF[g]:FS.OFF[g] + FS.GROUPS[g]] for g in SG]
    Us = np.hstack(cols)
    wu = np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols])
    ctx = (rows, y, env, st, st, Xd, wd, Us, wu, win0)
    log(f"fold {j} ({proto}): shared model fitted on {len(ref)} references")
    # inner out-of-fold decisions on the references
    oof = {k: np.zeros(len(y)) for k in SOURCES}
    renv = sorted(set(env[ref]))
    for e in renv[:2] if SMOKE else renv:
        iref, ite = ref[env[ref] != e], ref[env[ref] == e]
        for k, v in Sources(ctx, iref, theta0)(ite).items():
            oof[k][ite] = v
        log(f"  inner {e}")
    src = Sources(ctx, ref, theta0)
    q = src(te)
    zmax, zarg = src.score_z(te)
    out = {"raw": {k: v.tolist() for k, v in q.items()}, "oof_ref": {k: oof[k][ref].tolist() for k in SOURCES}}
    # scaled and clipped
    sc = {}
    for k in SOURCES:
        r = oof[k][ref]
        s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
        sc[k] = (np.clip(q[k], r.min(), r.max()) / s, np.clip(r, r.min(), r.max()) / s)
    out["scaled"] = {k: v[0].tolist() for k, v in sc.items()}
    for name, parts in COMBOS.items():
        out["scaled"][name] = sum(sc[k][0] for k in parts).tolist()
    for name, parts in (("stack_all", SOURCES), ("stack_floor", COMBOS["all_floor"])):
        R = np.column_stack([sc[k][1] for k in parts])
        Q = np.column_stack([sc[k][0] for k in parts])
        lr = LogisticRegression(C=1.0).fit(R, y[ref])
        out["scaled"][name] = lr.decision_function(Q).tolist()
        out[f"{name}_coef"] = dict(zip(parts, np.round(lr.coef_[0], 3).tolist()))
    big = np.flatnonzero(zmax > 20)
    out["score_extreme_features"] = [{"row": int(te[i]), "max_abs_z": round(float(zmax[i]), 1), "feature": int(zarg[i]),
                                      "group": SG[int(np.searchsorted(np.cumsum([FS.GROUPS[g] for g in SG]), zarg[i], side="right"))]}
                                     for i in big]
    sv["result"] = out


def merge(d):
    rows, y, env, split, groups = load_pilot()
    files = sorted(Path(d).glob("stack_fold_*.json"))
    res = {}
    for proto in ("loeo", "devtest"):
        F = [json.loads(f.read_text()) for f in files]
        F = [f for f in F if f.get("meta", {}).get("proto") == proto and "result" in f]
        if not F:
            continue
        mask = np.zeros(len(y), bool)
        names = list(F[0]["result"]["scaled"])
        D = {("raw", k): np.zeros(len(y)) for k in SOURCES}
        D.update({("scaled", k): np.zeros(len(y)) for k in names})
        for f in F:
            te = np.array(f["meta"]["test_rows"])
            mask[te] = True
            for (kind, k) in D:
                D[(kind, k)][te] = f["result"][kind][k]
        res[proto] = {"folds": len(F), **{f"{kind}:{k}": metrics(v, y, groups, mask) for (kind, k), v in D.items()}}
        res[proto + "_stack_coef"] = [f["result"].get("stack_all_coef") for f in F]
        res[proto + "_extreme"] = [x for f in F for x in f["result"]["score_extreme_features"]]
    OUT.mkdir(exist_ok=True)
    (OUT / "stack.json").write_text(json.dumps(res, indent=1))
    for proto in ("loeo", "devtest"):
        if proto in res:
            for k, v in res[proto].items():
                if isinstance(v, dict):
                    print(f"{proto:8} {k:28} k1 {v['k1']:.3f} k3 {v['k3']:.3f} auc {v['auc']:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "fold":
        run_fold(int(sys.argv[2]))
    else:
        merge(sys.argv[2])
