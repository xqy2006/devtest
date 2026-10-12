"""Round 2 of numbers-only pair research: anchored sequential likelihood ratio and new dense
families (periodicity, digit style, conditional curves). Same protocols as pair_seq_lab.py."""
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
import research.pair_seq_lab as L  # noqa: E402
from research.seqfeatures2 import SEQ2_FAMILIES  # noqa: E402
from research.seqlr import ANCHORED, LEVELS, SeqLR  # noqa: E402

p, yb = L.p, L.yb
for f, fn in SEQ2_FAMILIES.items():
    L.FEAT[f] = np.zeros((len(p.rows), len(fn(np.asarray(p.rows[L.pair[0]]["numbers"])))))
    for i in L.pair:
        L.FEAT[f][i] = fn(np.asarray(p.rows[i]["numbers"]))
SEQ = {int(i): list(map(int, p.rows[i]["numbers"])) for i in L.pair}


class LRModel:
    """Sequential LR. mode: 'sum' (all depths), a single depth name, or 'stack' (LDA on the four
    depth sums, fitted on exact leave-one-out sums of the reference rows)."""

    def __init__(self, mode="sum", lam=5.0, T=40, min_support=0, anchored=False, coarse=0):
        self.mode, self.lam, self.T, self.ms, self.anch, self.coarse = mode, lam, T, min_support, anchored, coarse

    def fit(self, ref):
        self.m = SeqLR(self.lam, self.T, self.ms, self.anch, self.coarse).fit([SEQ[int(i)] for i in ref], [int(yb[i]) for i in ref])
        if self.mode == "stack":
            Z = np.array([self.m.level_scores(SEQ[int(i)], int(yb[i])) for i in ref])
            self.lda = LDA(solver="lsqr", shrinkage="auto").fit(Z, yb[ref])
        return self

    def decision(self, idx):
        Z = np.array([self.m.level_scores(SEQ[int(i)]) for i in idx])
        if self.mode == "sum":
            return Z.sum(1)
        if self.mode == "stack":
            return self.lda.decision_function(Z)
        if self.mode == "nonroot":
            return Z[:, 1:].sum(1)
        if self.mode == "deep":  # the two deepest levels
            return Z[:, 2:].sum(1)
        return Z[:, (ANCHORED if self.anch else LEVELS).index(self.mode)]


class Stacked:
    """Evidence from different sources, combined by logistic regression on out-of-fold scores:
    the dense-distribution LDA (v2 local families) and the sequential LR depth sums. Out-of-fold
    scores come from leave-one-environment-out inside the reference rows only."""

    def __init__(self, components=("lda", "root", "prev2", "prefix"), min_support=10, C=1.0):
        self.components, self.ms, self.C = components, min_support, C

    def _fit_parts(self, ref):
        from research.gated_fingerprint import LOCAL_FAMILIES
        lda = L.Model(list(LOCAL_FAMILIES), "lda").fit(ref) if "lda" in self.components else None
        lr = SeqLR(5.0, 40, self.ms).fit([SEQ[int(i)] for i in ref], [int(yb[i]) for i in ref])
        return lda, lr

    def _scores(self, parts, idx):
        lda, lr = parts
        Z = np.array([lr.level_scores(SEQ[int(i)]) for i in idx])
        cols = []
        for c in self.components:
            cols.append(lda.decision(idx) if c == "lda" else Z[:, LEVELS.index(c)])
        return np.column_stack(cols)

    def fit(self, ref):
        from sklearn.linear_model import LogisticRegression
        oof = np.zeros((len(ref), len(self.components)))
        env = p.env[ref]
        for e in sorted(set(env)):
            parts = self._fit_parts(ref[env != e])
            oof[env == e] = self._scores(parts, ref[env == e])
        self.oof = oof
        self.stack = LogisticRegression(C=self.C).fit(oof, yb[ref])
        self.parts = self._fit_parts(ref)
        return self

    def decision(self, idx):
        return self.stack.decision_function(self._scores(self.parts, idx))


class V3Local:
    """Exactly the deliverable's cluster model (gated_fingerprint.fit_cluster / local_logits)."""

    def fit(self, ref):
        from research.gated_fingerprint import fit_cluster
        self.cl = fit_cluster([p.rows[i] for i in ref], p.classes, [L.A, L.S])
        return self

    def decision(self, idx):
        from research.gated_fingerprint import local_logits
        s = np.array([local_logits(self.cl, p.rows[i]["numbers"]) for i in idx])
        return s[:, 1] - s[:, 0]


class LateFusion:
    """Late fusion: one shrinkage-LDA score per dense family (each low-dimensional) plus the branch
    LR; fusion weights by L2 logistic regression on inner leave-one-environment-out scores of the
    reference rows only."""

    def __init__(self, families, C=0.3, branch=True):
        self.families, self.C, self.branch = families, C, branch

    def _parts(self, ref):
        ms = [L.Model([f], "lda").fit(ref) for f in self.families]
        br = SeqLR(5.0, 40, 10).fit([SEQ[int(i)] for i in ref], [int(yb[i]) for i in ref]) if self.branch else None
        return ms, br

    def _scores(self, parts, idx):
        ms, br = parts
        cols = [m.decision(idx) for m in ms]
        if br is not None:
            Z = np.array([br.level_scores(SEQ[int(i)]) for i in idx])
            cols.append(Z[:, 2] + Z[:, 3])
        return np.column_stack(cols)

    def fit(self, ref):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        env = p.env[ref]
        oof = None
        for e in sorted(set(env)):
            part = self._parts(ref[env != e])
            sc = self._scores(part, ref[env == e])
            if oof is None:
                oof = np.zeros((len(ref), sc.shape[1]))
            oof[env == e] = sc
        self.sc = StandardScaler().fit(oof)
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(self.sc.transform(oof), yb[ref])
        self.parts = self._parts(ref)
        return self

    def decision(self, idx):
        return self.lr.decision_function(self.sc.transform(self._scores(self.parts, idx)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    runs = {}
    if mode in ("all", "families"):
        for f in SEQ2_FAMILIES:
            for kind in ("lda", "dlda"):
                runs[f"{f}:{kind}"] = (lambda f=f, kind=kind: L.Model([f], kind))
    if mode in ("all", "lr"):
        for m in ("sum", "root", "prev1", "prev2", "prefix", "stack"):
            runs[f"seqLR:{m}"] = (lambda m=m: LRModel(m))
        runs["seqLR:sum lam=20"] = lambda: LRModel("sum", 20.0)
        runs["seqLR:prefix T=80"] = lambda: LRModel("prefix", 5.0, 80)
    if mode in ("all", "lr2"):
        for ms in (10, 30):
            for m in ("prev1", "prev2", "prefix", "nonroot"):
                runs[f"seqLR:{m} support>={ms}"] = (lambda m=m, ms=ms: LRModel(m, 5.0, 40, ms))
        for T in (20, 60):
            for m in ("pos", "pos_prev1", "prefix", "nonroot"):
                runs[f"anchored:{m} T={T}"] = (lambda m=m, T=T: LRModel(m, 5.0, T, 0, True))
    if mode in ("all", "coarse"):
        for w in (6, 12, 24, 36):
            for ms in (5, 10):
                runs[f"coarse w={w} support>={ms}: deep"] = (lambda w=w, ms=ms: LRModel("deep", 5.0, 40, ms, False, w))
                runs[f"coarse w={w} support>={ms}: nonroot"] = (lambda w=w, ms=ms: LRModel("nonroot", 5.0, 40, ms, False, w))
    if mode in ("all", "sweep"):
        for lam in (1.0, 2.0, 5.0, 10.0):
            for ms in (4, 6, 10):
                runs[f"branch lam={lam} support>={ms}"] = (lambda lam=lam, ms=ms: LRModel("deep", lam, 40, ms))
    if mode in ("all", "fusion"):
        from research.seqfeatures3 import SEQ3_FAMILIES
        for f, fn in SEQ3_FAMILIES.items():
            L.FEAT[f] = np.zeros((len(p.rows), len(fn(np.asarray(p.rows[L.pair[0]]["numbers"])))))
            for i in L.pair:
                L.FEAT[f][i] = fn(np.asarray(p.rows[i]["numbers"]))
        ALLF = [f for f in L.FEAT if f not in ("permshape",)]
        V2F = ["value355", "ordered", "digits", "jump1", "moments"]
        runs["fusion v2 families + branch"] = lambda: LateFusion(V2F)
        runs["fusion all families + branch C=0.3"] = lambda: LateFusion(ALLF, 0.3)
        runs["fusion all families + branch C=0.05"] = lambda: LateFusion(ALLF, 0.05)
    if mode in ("all", "v3"):
        runs["v3 local (deliverable fit_cluster)"] = lambda: V3Local()
    if mode in ("all", "stack"):
        runs["stack lda+root+prev2+prefix"] = lambda: Stacked()
        runs["stack lda+prev2+prefix"] = lambda: Stacked(("lda", "prev2", "prefix"))
        runs["stack root+prev2+prefix"] = lambda: Stacked(("root", "prev2", "prefix"))
    for name, mk in runs.items():
        t = time.time()
        r = L.evaluate(mk)
        out[name] = r
        print(L.line(name, r) + f" ({time.time()-t:.0f}s)", flush=True)
    Path(__file__).with_name(f"out_pair_seq_lab2_{mode}.json").write_text(json.dumps(out, indent=1))
