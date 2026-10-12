"""Fisher-score vectors from the richer shared model (fisher_score2.Base), same protocols as
pair_seq_lab.py. Per fold the shared model is fitted on that fold's reference rows only (labels
unused); score vectors are cached on disk per fold. Classifier pipeline as in pair_fisher.py."""
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
from research import fisher_score2 as F2  # noqa: E402

p, yb = L.p, L.yb
POS = {g: i for i, g in enumerate(L.pair)}
ST = F2.Steps2([p.rows[i]["numbers"] for i in L.pair])
BASE = F2.Base()
CACHE = Path(__file__).with_name("cache_fisher2")
CACHE.mkdir(exist_ok=True)
_MEM = {}


def vectors(ref):
    key = hash(tuple(sorted(ref.tolist()))) & 0xFFFFFFFFFFFF
    if key in _MEM:
        return _MEM[key]
    path = CACHE / f"{key:012x}.npy"
    if path.exists():
        U = np.load(path)
    else:
        mask = np.zeros(len(L.pair), bool)
        mask[[POS[i] for i in ref]] = True
        t = time.time()
        b = F2.Base().fit(ST, mask)
        U = b.scores(ST).astype(np.float32)
        np.save(path, U)
        np.save(CACHE / f"{key:012x}_theta.npy", b.theta)
        print(f"  [shared model fitted on {len(ref)} refs in {time.time()-t:.0f}s]", flush=True)
    _MEM.clear()
    _MEM[key] = U
    return U


class Fisher2:
    def __init__(self, groups, clf="lda", rank=2, base_ref=None):
        self.groups, self.clf, self.rank, self.base_ref = groups, clf, rank, base_ref

    def _X(self, ref):
        U = vectors(ref if self.base_ref is None else self.base_ref)
        cols = np.concatenate([np.arange(BASE.dim)[BASE.sl[g]] for g in self.groups])
        w = np.concatenate([np.full(BASE.size[g], 1 / np.sqrt(BASE.size[g])) for g in self.groups])
        X = np.zeros((len(p.rows), len(cols)))
        X[L.pair] = U[:, cols]
        return X, w

    def fit(self, ref):
        self.X, self.w = self._X(ref)
        self.mu, self.sd = self.X[ref].mean(0), self.X[ref].std(0)
        self.sd[self.sd < 1e-12] = np.inf
        Z = (self.X[ref] - self.mu) / self.sd * self.w
        self.B = L.nuisance_basis(Z, ref, self.rank, 0)
        Z = Z - (Z @ self.B.T) @ self.B
        if self.clf == "nc":
            c = np.stack([Z[yb[ref] == k].mean(0) for k in (0, 1)])
            self.cent = c / np.linalg.norm(c, axis=1, keepdims=True)
        else:
            self.lda = LDA(solver="lsqr", shrinkage="auto").fit(Z, yb[ref])
        return self

    def decision(self, idx):
        Z = (self.X[idx] - self.mu) / self.sd * self.w
        Z = Z - (Z @ self.B.T) @ self.B
        if self.clf == "nc":
            s = Z @ self.cent.T
            return s[:, 1] - s[:, 0]
        return self.lda.decision_function(Z)


if __name__ == "__main__":
    sets = [("val", "dg", "pb"), ("vst", "dg", "pb"), ("val", "dg", "pb", "bd2", "cvb"),
            ("val", "vst", "dg", "pb"), tuple(F2.GROUP_DEFS), ("val",), ("vst",), ("bd2",), ("cvb",)]
    out = {}
    for gs in sets:
        name = f"score2[{'+'.join(gs)}]:lda"
        t = time.time()
        r = L.evaluate(lambda gs=gs: Fisher2(gs))
        out[name] = r
        print(L.line(name, r) + f" ({time.time()-t:.0f}s)", flush=True)
        Path(__file__).with_name("out_pair_fisher3.json").write_text(json.dumps(out, indent=1))
