"""Model-identity space learned from all reference classes, then the pair inside it.

Vectors: the dense families of seqfeatures for every class (cache_all_seqfeatures.npz).
Space: multi-class shrinkage LDA on the training rows of all classes (environment-nuisance
projected first), giving <= C-1 discriminant directions that separate models in general.
The pair is then classified inside that space by nearest centroid (production-style cosine) or by
a pair-local LDA. Variants: identity space learned from all classes, or from all classes except the
pair (pure transfer). Rows of the held-out environment / batch are excluded from every fit."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402

p, yb = L.p, L.yb
z = np.load(Path(__file__).with_name("cache_all_seqfeatures.npz"))
FAMS = [f for f in z.files]
X = np.hstack([z[f] for f in FAMS])
W = np.concatenate([np.full(z[f].shape[1], 1 / np.sqrt(z[f].shape[1])) for f in FAMS])
TRAIN_ALL = np.flatnonzero(np.isin(p.role, L.TRB))


class IdentitySpace:
    def __init__(self, include_pair=True, local="lda", rank=2):
        self.include_pair, self.local, self.rank = include_pair, local, rank

    def fit(self, ref):
        envs = set(p.env[ref])
        others = TRAIN_ALL[~np.isin(p.y[TRAIN_ALL], [L.A, L.S]) & np.isin(p.env[TRAIN_ALL], list(envs))]
        use = np.concatenate([ref, others]) if self.include_pair else others
        self.mu, self.sd = X[use].mean(0), X[use].std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X[use] - self.mu) / self.sd * W
        offs = np.array([Z[p.env[use] == e].mean(0) for e in sorted(envs)])
        offs -= offs.mean(0)
        _, s, vt = np.linalg.svd(offs, full_matrices=False)
        self.B = vt[:self.rank]
        Z = Z - (Z @ self.B.T) @ self.B
        self.space = LDA(solver="eigen", shrinkage="auto").fit(Z, p.y[use])
        E = self.embed(ref)
        if self.local == "lda":
            self.m = LDA(solver="lsqr", shrinkage="auto").fit(E, yb[ref])
        else:
            c = np.stack([E[yb[ref] == k].mean(0) for k in (0, 1)])
            self.cent = c / np.linalg.norm(c, axis=1, keepdims=True)
        return self

    def embed(self, idx):
        Z = (X[idx] - self.mu) / self.sd * W
        Z = Z - (Z @ self.B.T) @ self.B
        return self.space.transform(Z)

    def decision(self, idx):
        E = self.embed(idx)
        if self.local == "lda":
            return self.m.decision_function(E)
        s = (E / np.linalg.norm(E, axis=1, keepdims=True)) @ self.cent.T
        return s[:, 1] - s[:, 0]


if __name__ == "__main__":
    for inc in (True, False):
        for loc in ("lda", "nc"):
            name = f"identity space ({'all classes' if inc else 'other classes only'}) + {loc}"
            print(L.line(name, L.evaluate(lambda: IdentitySpace(inc, loc))), flush=True)
