"""Distribution evidence with the noise covariance borrowed from all reference classes.

The within-model sampling noise of the dense sequence statistics is assumed to have a similar
covariance across models; it is estimated from the training rows of ALL classes (residuals after
removing each class x environment mean), with Ledoit-Wolf shrinkage. Only the mean difference is
estimated from the cluster's own references. w = Sigma^-1 (mu_S - mu_A).

Protocol hygiene: rows of the held-out environment (env-CV) or held-out pair batch (LOBO) are
excluded from the covariance as well, for every class."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research.seqfeatures import SEQ_FAMILIES  # noqa: E402

p, yb = L.p, L.yb
TRAIN_ALL = np.flatnonzero(np.isin(p.role, L.TRB))
OTHERS = TRAIN_ALL[~np.isin(p.y[TRAIN_ALL], [L.A, L.S])]
FAMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["value355", "ordered", "digits", "jump1", "moments"]
cache = Path(__file__).with_name("cache_all_seqfeatures.npz")
if cache.exists():
    z = np.load(cache)
    FALL = {f: z[f] for f in z.files}
else:
    rows = np.flatnonzero(np.isin(p.role, L.TRB + ("old_outside", "fresh_outside")))
    FALL = {f: np.zeros((len(p.rows), len(SEQ_FAMILIES[f](np.asarray(p.rows[0]["numbers"]))))) for f in SEQ_FAMILIES}
    for i in rows:
        x = np.asarray(p.rows[i]["numbers"])
        for f in SEQ_FAMILIES:
            FALL[f][i] = SEQ_FAMILIES[f](x)
    np.savez_compressed(cache, **FALL)
X = np.hstack([FALL[f] for f in FAMS])
W = np.concatenate([np.full(FALL[f].shape[1], 1 / np.sqrt(FALL[f].shape[1])) for f in FAMS])


class PooledCov:
    def __init__(self, borrow=True):
        self.borrow = borrow

    def fit(self, ref):
        # environments / batches present in the pair references define what may be used
        envs, batches = set(p.env[ref]), set(p.batch[ref])
        other = OTHERS[np.isin(p.env[OTHERS], list(envs))] if self.borrow else np.array([], int)
        use = np.concatenate([ref, other])
        self.mu = X[use].mean(0)
        self.sd = X[use].std(0)
        self.sd[self.sd < 1e-12] = 1
        Z = (X[use] - self.mu) / self.sd * W
        lab = p.y[use]
        R = np.zeros_like(Z)
        for c in set(lab):
            for e in set(p.env[use]):
                m = (lab == c) & (p.env[use] == e)
                if m.sum():
                    R[m] = Z[m] - Z[m].mean(0)
        cov = LedoitWolf().fit(R).covariance_
        Zr = (X[ref] - self.mu) / self.sd * W
        # environment-balanced class means of the pair
        means = []
        for c in (0, 1):
            em = [Zr[(yb[ref] == c) & (p.env[ref] == e)].mean(0) for e in envs if ((yb[ref] == c) & (p.env[ref] == e)).any()]
            means.append(np.mean(em, axis=0))
        self.coef = np.linalg.solve(cov, means[1] - means[0])
        self.mid = (means[0] + means[1]) / 2
        return self

    def decision(self, idx):
        return ((X[idx] - self.mu) / self.sd * W - self.mid) @ self.coef


if __name__ == "__main__":
    print("families:", FAMS, flush=True)
    for name, mk in (("pair-only covariance", lambda: PooledCov(False)), ("covariance borrowed from all classes", lambda: PooledCov(True))):
        print(L.line(name, L.evaluate(mk)), flush=True)
