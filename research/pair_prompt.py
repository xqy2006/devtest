"""Prompt-template nuisance projection.

Every query prompt is one of 12 wordings (English/Chinese templates) with a requested count N
(218-339), and within each batch the same prompt is sent to both labels (paired design). The
prompt is the probe itself, known at query time; it is neither the label nor the environment,
batch or any test-group statistic. Its effect on the number features is a nuisance, estimated on
the references only (class-centred template offsets, as for environments) and projected out:
  tmpl r     top-r directions of the per-template, class-centred offsets of the references
  N          the within-class linear effect of the requested count
Feature sources: Fisher score groups (cached per fold, pair_fisher.py) and the dense v3 families.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research.gated_fingerprint import LOCAL_FAMILIES  # noqa: E402
from research.pair_struct import block, lobo_batches  # noqa: E402

p, yb = L.p, L.yb
NPZ = np.load(Path(__file__).with_name("cache_pair_fisher.npz"))
_t = {}
TMPL = np.full(len(p.rows), -1)
NREQ = np.zeros(len(p.rows))
for i in L.pair:
    TMPL[i] = _t.setdefault(re.sub(r"\d+", "N", p.rows[i]["base_prompt"]), len(_t))
    NREQ[i] = float(p.rows[i]["requested_count"])


def offsets_basis(Z, ref, key, rank):
    offs = []
    for c in (0, 1):
        m = yb[ref] == c
        mu = Z[m].mean(0)
        for k in sorted(set(key[ref][m])):
            mk = m & (key[ref] == k)
            if mk.sum() >= 2:
                offs.append(Z[mk].mean(0) - mu)
    _, s, vt = np.linalg.svd(np.array(offs), full_matrices=False)
    return vt[:min(rank, int((s > s[0] * 1e-8).sum()))]


def n_direction(Z, ref):
    rows = []
    for c in (0, 1):
        m = yb[ref] == c
        n = NREQ[ref][m] - NREQ[ref][m].mean()
        rows.append((Z[m] - Z[m].mean(0)).T @ n / (n @ n))
    return np.array(rows)


class PromptModel:
    def __init__(self, score=(), dense=False, tmpl=0, use_n=False, rank=2):
        self.score, self.dense, self.tmpl, self.use_n, self.rank = score, dense, tmpl, use_n, rank

    def fit(self, ref):
        Xs, ws = [], []
        if self.dense:
            X, w = L.design(list(LOCAL_FAMILIES))
            Xs.append(X); ws.append(w)
        if self.score:
            U = NPZ[str(hash(tuple(sorted(ref.tolist()))))]
            cols = [block(U, b) for b in self.score]
            X = np.zeros((len(p.rows), sum(c.shape[1] for c in cols)))
            X[L.pair] = np.hstack(cols)
            Xs.append(X); ws.append(np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols]))
        self.X, self.w = np.hstack(Xs), np.concatenate(ws)
        self.mu, self.sd = self.X[ref].mean(0), self.X[ref].std(0)
        self.sd[self.sd < 1e-12] = np.inf
        Z = (self.X[ref] - self.mu) / self.sd * self.w
        B = [L.nuisance_basis(Z, ref, self.rank, 0)]
        if self.tmpl:
            B.append(offsets_basis(Z, ref, TMPL, self.tmpl))
        if self.use_n:
            B.append(n_direction(Z, ref))
        q, _ = np.linalg.qr(np.vstack(B).T)
        self.B = q.T
        Z = Z - (Z @ self.B.T) @ self.B
        self.lda = LDA(solver="lsqr", shrinkage="auto").fit(Z, yb[ref])
        return self

    def decision(self, idx):
        Z = (self.X[idx] - self.mu) / self.sd * self.w
        Z = Z - (Z @ self.B.T) @ self.B
        return self.lda.decision_function(Z)


SG = ("val", "dg", "pb")
RUNS = {
    "score": dict(score=SG),
    "score tmpl2": dict(score=SG, tmpl=2),
    "score tmpl5": dict(score=SG, tmpl=5),
    "score tmpl11": dict(score=SG, tmpl=11),
    "score N": dict(score=SG, use_n=True),
    "score tmpl5+N": dict(score=SG, tmpl=5, use_n=True),
    "dense": dict(dense=True),
    "dense tmpl5": dict(dense=True, tmpl=5),
    "dense tmpl5+N": dict(dense=True, tmpl=5, use_n=True),
    "dense+score": dict(dense=True, score=SG),
    "dense+score tmpl5+N": dict(dense=True, score=SG, tmpl=5, use_n=True),
}

if __name__ == "__main__":
    out = {}
    for name, kw in RUNS.items():
        mk = (lambda kw=kw: PromptModel(**kw))
        r = L.evaluate(mk)
        r["lobo_batch"] = lobo_batches(mk)
        out[name] = r
        print(L.line(name, r), "| LOBO k1/k3 by batch", r["lobo_batch"], flush=True)
        Path(__file__).with_name("out_pair_prompt.json").write_text(json.dumps(out, indent=1))
