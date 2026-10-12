"""Vector-space data mining on the confusable pair (numbers only), production-style computation.

Views: kernel mean embeddings (vectorviews.RFF_VIEWS), hashed n-gram TF-IDF (vectorviews.tokens),
the v3 dense families, and a model-identity subspace learned from all 17 classes.
Computation per view: standardize on reference rows -> environment-nuisance projection -> either
production nearest-centroid cosine ("nc") or shrinkage LDA ("lda"); TF-IDF uses L2-logistic or
NB-SVM. Same protocols as pair_seq_lab.py (LOBO primary; outside panels development only)."""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research.vectorviews import RFF_VIEWS, token_matrix  # noqa: E402

p, yb = L.p, L.yb
CACHE = Path(__file__).with_name("cache_pair_rff.npz")
if CACHE.exists():
    z = np.load(CACHE)
    for f in z.files:
        L.FEAT[f] = z[f]
else:
    t = time.time()
    for f, fn in RFF_VIEWS.items():
        L.FEAT[f] = np.zeros((len(p.rows), len(fn(np.asarray(p.rows[L.pair[0]]["numbers"])))))
        for i in L.pair:
            L.FEAT[f][i] = fn(np.asarray(p.rows[i]["numbers"]))
    np.savez_compressed(CACHE, **{f: L.FEAT[f] for f in RFF_VIEWS})
    print(f"rff views in {time.time()-t:.0f}s", flush=True)
TOK = token_matrix([p.rows[i]["numbers"] if i in set(L.pair.tolist()) else [1, 2, 3] for i in range(len(p.rows))])


class TokenModel:
    """TF-IDF (idf from reference rows) + L2 normalisation; 'logit' or NB-SVM (log-count-ratio
    scaled features, Wang & Manning 2012)."""

    def __init__(self, kind="logit", C=1.0):
        self.kind, self.C = kind, C

    def _tf(self, idx):
        X = TOK[idx].multiply(self.idf).tocsr()
        n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-12
        return X.multiply(1 / n[:, None]).tocsr()

    def fit(self, ref):
        df = np.asarray((TOK[ref] > 0).sum(0)).ravel()
        self.idf = np.log((1 + len(ref)) / (1 + df)) + 1
        self.idf[df < 3] = 0  # tokens seen in fewer than 3 references carry no estimate
        X = self._tf(ref)
        if self.kind == "nbsvm":
            pa = np.asarray(X[yb[ref] == 0].sum(0)).ravel() + 1
            ps = np.asarray(X[yb[ref] == 1].sum(0)).ravel() + 1
            self.r = np.log((ps / ps.sum()) / (pa / pa.sum()))
            X = X.multiply(self.r).tocsr()
        self.m = LogisticRegression(C=self.C, max_iter=3000).fit(X, yb[ref])
        return self

    def decision(self, idx):
        X = self._tf(idx)
        if self.kind == "nbsvm":
            X = X.multiply(self.r).tocsr()
        return self.m.decision_function(X)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rff"
    out = {}
    runs = {}
    if mode == "rff":
        for f in RFF_VIEWS:
            for kind in ("nc", "lda"):
                runs[f"{f}:{kind}"] = (lambda f=f, kind=kind: L.Model([f], kind))
    if mode == "tokens":
        for kind, C in (("logit", 1.0), ("logit", 10.0), ("nbsvm", 1.0), ("nbsvm", 0.1)):
            runs[f"tokens:{kind} C={C}"] = (lambda kind=kind, C=C: TokenModel(kind, C))
    for name, mk in runs.items():
        t = time.time()
        r = L.evaluate(mk)
        out[name] = r
        print(L.line(name, r) + f" ({time.time()-t:.0f}s)", flush=True)
    Path(__file__).with_name(f"out_pair_vector_lab_{mode}.json").write_text(json.dumps(out, indent=1))
