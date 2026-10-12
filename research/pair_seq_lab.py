"""Dense sequence families on the confusable pair, production-style classifiers (numbers only).

Classifiers (all fitted on reference rows only):
  nc    production-style: standardize, project out nuisance directions, cosine to class centroids
  dlda  diagonal LDA after the same projection (per-feature pooled within-class variance)
  lda   shrinkage LDA after the same projection
Nuisance directions = top singular vectors of within-class (class, environment) cell offsets,
optionally plus (class, collection batch) offsets: the deployment-drift directions.

Protocols: leave-one-training-batch-out (primary; deployment drift), env-CV inside the 2026-10-11
batch, pooled env-CV, and the exposed outside panels (development only).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import PAIR_ROLE, extended_panel  # noqa: E402
from research.seqfeatures import SEQ_FAMILIES, extract_seq  # noqa: E402

CACHE = Path(__file__).with_name("cache_pair_seqfeatures.npz")
p = extended_panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
pair = np.flatnonzero(np.isin(p.y, [A, S]))
if CACHE.exists() and set(np.load(CACHE).files) >= set(SEQ_FAMILIES):
    z = np.load(CACHE)
    FEAT = {f: z[f] for f in SEQ_FAMILIES}
else:
    t = time.time()
    F = extract_seq([p.rows[i] for i in pair])
    FEAT = {}
    for f, v in F.items():
        FEAT[f] = np.zeros((len(p.rows), v.shape[1]))
        FEAT[f][pair] = v
    np.savez_compressed(CACHE, **FEAT)
    print(f"features extracted in {time.time()-t:.0f}s", flush=True)
yb = (p.y == S).astype(int)
TRB = ("train_original", "train_supplemental", PAIR_ROLE)


def design(families):
    X = np.hstack([FEAT[f] for f in families])
    w = np.concatenate([np.full(FEAT[f].shape[1], 1 / np.sqrt(FEAT[f].shape[1])) for f in families])
    return X, w


def nuisance_basis(Z, ref, rank_env, rank_batch):
    basis = []
    for key, rank in ((p.env, rank_env), (p.batch, rank_batch)):
        if not rank:
            continue
        offs = []
        for c in (0, 1):
            rc = ref[yb[ref] == c]
            mu = Z[yb[ref] == c].mean(0)
            for k in sorted(set(key[rc])):
                m = (yb[ref] == c) & (key[ref] == k)
                offs.append(Z[m].mean(0) - mu)
        offs = np.array(offs)
        if len(offs) > 2:
            _, s, vt = np.linalg.svd(offs, full_matrices=False)
            basis.append(vt[:min(rank, int((s > s[0] * 1e-8).sum()))])
    if not basis:
        return np.zeros((0, Z.shape[1]))
    B = np.vstack(basis)
    q, _ = np.linalg.qr(B.T)
    return q.T


class Model:
    def __init__(self, families, kind="nc", rank_env=2, rank_batch=0):
        self.families, self.kind, self.re, self.rb = families, kind, rank_env, rank_batch

    def fit(self, ref):
        X, self.w = design(self.families)
        self.X = X
        self.mu = X[ref].mean(0)
        self.sd = X[ref].std(0)
        self.sd[self.sd < 1e-12] = 1.0
        Z = (X[ref] - self.mu) / self.sd * self.w
        self.B = nuisance_basis(Z, ref, self.re, self.rb)
        Z = Z - (Z @ self.B.T) @ self.B
        y = yb[ref]
        if self.kind == "nc":
            c = np.stack([Z[y == k].mean(0) for k in (0, 1)])
            self.cent = c / np.linalg.norm(c, axis=1, keepdims=True)
        elif self.kind == "dlda":
            m0, m1 = Z[y == 0].mean(0), Z[y == 1].mean(0)
            var = (((Z[y == 0] - m0) ** 2).sum(0) + ((Z[y == 1] - m1) ** 2).sum(0)) / (len(y) - 2)
            self.coef = (m1 - m0) / (var + 1e-6 * var.mean())
            self.icpt = -self.coef @ (m0 + m1) / 2
        else:
            self.lda = LDA(solver="lsqr", shrinkage="auto").fit(Z, y)
        return self

    def decision(self, idx):
        Z = (self.X[idx] - self.mu) / self.sd * self.w
        Z = Z - (Z @ self.B.T) @ self.B
        if self.kind == "nc":
            s = Z @ self.cent.T
            return s[:, 1] - s[:, 0]
        if self.kind == "dlda":
            return Z @ self.coef + self.icpt
        return self.lda.decision_function(Z)


class MetaModel:
    """Batch-consistent diagonal discriminant (numbers only, dense statistics).

    For each training batch b and feature j: t_bj = standardized mean difference (second label minus
    first). Features whose sign agrees in every training batch and whose Stouffer-combined
    z = sum_b t_bj / sqrt(B) exceeds ``tau`` are kept; weight = inverse-variance pooled effect
    size (fixed-effect meta-analysis), applied to features standardized by the pooled within-batch,
    within-class SD. Decision = weighted sum minus the midpoint between the two class means."""

    def __init__(self, families, tau=2.0):
        self.families, self.tau = families, tau

    def fit(self, ref):
        self.X = np.hstack([FEAT[f] for f in self.families])
        X, y, bt = self.X[ref], yb[ref], p.batch[ref]
        T, D, W = [], [], []
        resid = np.zeros_like(X)
        for b in sorted(set(bt)):
            for c in (0, 1):
                m = (bt == b) & (y == c)
                resid[m] = X[m] - X[m].mean(0)
        self.sd = np.sqrt((resid ** 2).sum(0) / max(len(ref) - 2 * len(set(bt)), 1))
        self.sd[self.sd < 1e-12] = np.inf
        for b in sorted(set(bt)):
            a, s = X[(bt == b) & (y == 0)], X[(bt == b) & (y == 1)]
            d = (s.mean(0) - a.mean(0)) / self.sd
            v = 1 / len(a) + 1 / len(s)
            D.append(d); W.append(1 / v); T.append(d / np.sqrt(v))
        T, D, W = np.array(T), np.array(D), np.array(W)
        same = (np.sign(T) == np.sign(T[0])).all(0)
        z = T.sum(0) / np.sqrt(len(T))
        keep = same & (np.abs(z) > self.tau)
        self.coef = np.where(keep, (W[:, None] * D).sum(0) / W.sum(), 0.0)
        Zr = X / self.sd
        self.mid = (Zr[y == 0].mean(0) + Zr[y == 1].mean(0)) / 2
        self.kept = int(keep.sum())
        return self

    def decision(self, idx):
        return (self.X[idx] / self.sd - self.mid) @ self.coef


class SumModels:
    """Sum of member decisions, each scaled by its spread on the reference rows (no test statistics)."""

    def __init__(self, makers):
        self.makers = makers

    def fit(self, ref):
        self.ms = [mk().fit(ref) for mk in self.makers]
        self.sc = [np.std(m.decision(ref)) + 1e-12 for m in self.ms]
        return self

    def decision(self, idx):
        return sum(m.decision(idx) / s for m, s in zip(self.ms, self.sc))


DENSE = list(SEQ_FAMILIES)


def rows(roles):
    return np.flatnonzero(np.isin(p.y, [A, S]) & np.isin(p.role, roles))


def groups(roles):
    return p.group_members[np.isin(p.group_role, roles) & np.isin(p.group_y, [A, S])]


def acc(d, idx, g):
    return float(np.mean((d[idx] > 0) == yb[idx])), float(np.mean((d[g].sum(1) > 0) == yb[g[:, 0]]))


def evaluate(make):
    res = {}
    tr, g = rows(TRB), groups(TRB)
    d = np.zeros(len(p.rows))
    for b in TRB:
        te = rows((b,))
        d[te] = make().fit(rows(tuple(x for x in TRB if x != b))).decision(te)
    res["lobo"] = acc(d, tr, g)
    res["lobo_each"] = {b[-8:]: round(float(np.mean((d[rows((b,))] > 0) == yb[rows((b,))])), 3) for b in TRB}
    cx, gx = rows((PAIR_ROLE,)), groups((PAIR_ROLE,))
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[cx])):
        d[cx[p.env[cx] == e]] = make().fit(cx[p.env[cx] != e]).decision(cx[p.env[cx] == e])
    res["codex_env"] = acc(d, cx, gx)
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[tr])):
        d[tr[p.env[tr] == e]] = make().fit(tr[p.env[tr] != e]).decision(tr[p.env[tr] == e])
    res["env"] = acc(d, tr, g)
    m = make().fit(tr)
    for role in ("old_outside", "fresh_outside"):
        te = rows((role,))
        d = np.zeros(len(p.rows))
        d[te] = m.decision(te)
        res[role] = acc(d, te, groups((role,)))
    return res


def line(name, r):
    return (f"{name:34} LOBO {r['lobo'][0]:.3f}/{r['lobo'][1]:.3f} {r['lobo_each']} | 10-11 env {r['codex_env'][0]:.3f}/{r['codex_env'][1]:.3f}"
            f" | env {r['env'][0]:.3f}/{r['env'][1]:.3f} | old {r['old_outside'][0]:.3f}/{r['old_outside'][1]:.3f}"
            f" fresh {r['fresh_outside'][0]:.3f}/{r['fresh_outside'][1]:.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "families"
    out = {}
    if mode == "families":
        for f in SEQ_FAMILIES:
            for kind in ("nc", "dlda", "lda"):
                r = evaluate(lambda: Model([f], kind))
                out[f"{f}:{kind}"] = r
                print(line(f"{f}:{kind}", r), flush=True)
    if mode == "meta":
        for tau in (1.0, 1.5, 2.0, 2.5, 3.0):
            r = evaluate(lambda: MetaModel(DENSE, tau))
            out[f"meta_tau{tau}"] = r
            print(line(f"meta(all dense) tau={tau}", r), flush=True)
        for name, mk in {
            "nc(value355)+meta2": lambda: SumModels([lambda: Model(["value355"], "nc"), lambda: MetaModel(DENSE, 2.0)]),
            "lda(dense7)": lambda: Model(["value355", "digits", "jump1", "moments", "residues", "coverage", "ordered"], "lda"),
            "lda(dense7) no-proj": lambda: Model(["value355", "digits", "jump1", "moments", "residues", "coverage", "ordered"], "lda", 0),
            "lda(dense7)+batchproj": lambda: Model(["value355", "digits", "jump1", "moments", "residues", "coverage", "ordered"], "lda", 2, 1),
            "nc(dense7)": lambda: Model(["value355", "digits", "jump1", "moments", "residues", "coverage", "ordered"], "nc"),
            "lda(dense7)+meta2": lambda: SumModels([lambda: Model(["value355", "digits", "jump1", "moments", "residues", "coverage", "ordered"], "lda"), lambda: MetaModel(DENSE, 2.0)]),
        }.items():
            r = evaluate(mk)
            out[name] = r
            print(line(name, r), flush=True)
    if mode == "select":
        # rule fixed in advance: families whose single-family LOBO k1 (lda) >= 0.56 in
        # out_pair_seq_lab_families.json (training batches only); plus 'all dense' without selection
        fam_res = json.loads(Path(__file__).with_name("out_pair_seq_lab_families.json").read_text())
        chosen = [f for f in SEQ_FAMILIES if fam_res[f"{f}:lda"]["lobo"][0] >= 0.56]
        print("rule-selected families:", chosen, flush=True)
        for name, mk in {
            "lda(rule-selected)": lambda: Model(chosen, "lda"),
            "lda(rule-selected)+batchproj": lambda: Model(chosen, "lda", 2, 1),
            "lda(all dense)": lambda: Model(DENSE, "lda"),
            "lda(all dense)+batchproj": lambda: Model(DENSE, "lda", 2, 1),
        }.items():
            r = evaluate(mk)
            out[name] = r
            print(line(name, r), flush=True)
    Path(__file__).with_name(f"out_pair_seq_lab_{mode}.json").write_text(json.dumps(out, indent=1))
