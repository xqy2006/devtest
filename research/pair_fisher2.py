"""Round 2 of Fisher-score vectors: group combinations and combination with the v3 evidence.

Uses the per-fold score vectors cached by pair_fisher.py (shared model fitted on each fold's
reference rows only). Combination with v3 = the deliverable's recipe: out-of-fold scores of each
evidence source from leave-one-environment-out inside the reference rows, combined by logistic
regression (Stacked in pair_seq_lab2.py) — the shared model of the outer fold is label-free and
fitted on that fold's reference rows, so the inner folds reuse it.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_fisher as PF  # noqa: E402
import research.pair_seq_lab as L  # noqa: E402
from research.gated_fingerprint import LOCAL_FAMILIES  # noqa: E402
from research.pair_seq_lab2 import SEQ  # noqa: E402
from research.seqlr import LEVELS, SeqLR  # noqa: E402

p, yb = L.p, L.yb
z = np.load(Path(__file__).with_name("cache_pair_fisher.npz"))
for k in z.files:
    PF._U[int(k)] = z[k]


class EarlyFusion(PF.FisherModel):
    """One LDA on [v3 dense families | Fisher score groups] (same preprocessing for both)."""

    def _X(self, ref):
        Xs, ws = super()._X(ref)
        Xd, wd = L.design(list(LOCAL_FAMILIES))
        return np.hstack([Xd, Xs]), np.concatenate([wd, ws])


class StackedFisher:
    """dense LDA (v3 families) + branch LR (prev2 + prefix) + Fisher-score LDA, logistic stacking on
    inner leave-one-environment-out scores of the reference rows."""

    def __init__(self, groups=("val",), use=("lda", "branch", "fisher"), C=1.0):
        self.groups, self.use, self.C = groups, use, C

    def _parts(self, ref, outer_ref):
        parts = {}
        if "lda" in self.use:
            parts["lda"] = L.Model(list(LOCAL_FAMILIES), "lda").fit(ref)
        if "branch" in self.use:
            parts["branch"] = SeqLR(5.0, 40, 10).fit([SEQ[int(i)] for i in ref], [int(yb[i]) for i in ref])
        if "fisher" in self.use:
            m = PF.FisherModel("score", self.groups)
            m.vec_ref = outer_ref
            parts["fisher"] = _FixedBase(m, outer_ref).fit(ref)
        return parts

    def _scores(self, parts, idx):
        cols = []
        for c in self.use:
            if c == "branch":
                Z = np.array([parts[c].level_scores(SEQ[int(i)]) for i in idx])
                cols.append(Z[:, LEVELS.index("prev2")] + Z[:, LEVELS.index("prefix")])
            else:
                cols.append(parts[c].decision(idx))
        return np.column_stack(cols)

    def fit(self, ref):
        env = p.env[ref]
        oof = np.zeros((len(ref), len(self.use)))
        for e in sorted(set(env)):
            oof[env == e] = self._scores(self._parts(ref[env != e], ref), ref[env == e])
        self.stack = LogisticRegression(C=self.C).fit(oof, yb[ref])
        self.parts = self._parts(ref, ref)
        return self

    def decision(self, idx):
        return self.stack.decision_function(self._scores(self.parts, idx))


class _FixedBase:
    """FisherModel whose score vectors come from the shared model of `base_ref` (the outer fold's
    reference rows), while the LDA is fitted on `ref` (an inner subset)."""

    def __init__(self, model, base_ref):
        self.m, self.base_ref = model, base_ref

    def fit(self, ref):
        U = PF.vectors(self.base_ref)
        cols = np.concatenate([np.arange(PF.F.DIM)[PF.SL[g]] for g in self.m.groups])
        w = np.concatenate([np.full(PF.F.GROUPS[g], 1 / np.sqrt(PF.F.GROUPS[g])) for g in self.m.groups])
        X = np.zeros((len(p.rows), len(cols)))
        X[L.pair] = U[:, cols]
        self.m._X = lambda _ref: (X, w)
        self.m.fit(ref)
        return self

    def decision(self, idx):
        return self.m.decision(idx)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "groups"
    out = {}
    runs = {}
    if mode == "groups":
        for gs in (("val", "dg"), ("val", "pb"), ("val", "dg", "pb"), ("val", "dg", "pb", "j1")):
            runs[f"score[{'+'.join(gs)}]:lda"] = (lambda gs=gs: PF.FisherModel("score", gs))
        for gs in (("val",), ("val", "dg", "pb")):
            runs[f"v3dense+score[{'+'.join(gs)}]:lda"] = (lambda gs=gs: EarlyFusion("score", gs))
    if mode == "stack":
        runs["stack lda+branch (v3 recipe)"] = lambda: StackedFisher(use=("lda", "branch"))
        runs["stack lda+branch+fisher[val]"] = lambda: StackedFisher(("val",))
        runs["stack lda+branch+fisher[val+dg+pb]"] = lambda: StackedFisher(("val", "dg", "pb"))
        runs["stack branch+fisher[val+dg+pb]"] = lambda: StackedFisher(("val", "dg", "pb"), ("branch", "fisher"))
    for name, mk in runs.items():
        t = time.time()
        r = L.evaluate(mk)
        out[name] = r
        print(L.line(name, r) + f" ({time.time()-t:.0f}s)", flush=True)
        Path(__file__).with_name(f"out_pair_fisher2_{mode}.json").write_text(json.dumps(out, indent=1))
