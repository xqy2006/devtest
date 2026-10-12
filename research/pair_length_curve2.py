"""How does single-response separability of the pair grow with response length? (extended data)

Each response is truncated to its first L numbers; features are the sequence-content families
(no count/request features, which truncation would fake). Separability is measured by
leave-one-environment-out shrinkage LDA over all pair training batches pooled, as accuracy and as
d' of the out-of-fold decision values. The count-precision cue (over/under-shoot) is reported
separately because it does not grow with length.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import PAIR_ROLE, extended_panel  # noqa: E402
from research.features import FAMILIES  # noqa: E402

p = extended_panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
TRB = ("train_original", "train_supplemental", PAIR_ROLE)
if len(sys.argv) > 1:  # optional subset of batches, e.g. old or codex
    TRB = {"old": ("train_original", "train_supplemental"), "codex": (PAIR_ROLE,)}[sys.argv[1]]
TAG = sys.argv[1] if len(sys.argv) > 1 else "pooled"
idx = np.flatnonzero(np.isin(p.y, [A, S]) & np.isin(p.role, TRB))
yb = (p.y[idx] == S).astype(int)
env = p.env[idx]
SEQFAMS = ["value355", "sdiff72", "repeat", "digits", "scalar_noreq"]


def feats(L):
    out = []
    for i in idx:
        x = np.asarray(p.rows[i]["numbers"][:L], dtype=np.int64)
        r = {"x": x, "text": "", "requested_count": len(x)}
        out.append(np.concatenate([FAMILIES[f](r) * (1 / np.sqrt(len(FAMILIES[f](r)))) for f in SEQFAMS]))
    return np.array(out)


def env_cv(X):
    d = np.zeros(len(idx))
    for e in sorted(set(env)):
        tr, te = env != e, env == e
        sc = StandardScaler().fit(X[tr])
        m = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X[tr]), yb[tr])
        d[te] = m.decision_function(sc.transform(X[te]))
    return d


res = {}
for L in (50, 100, 150, 200, 250, 290):
    d = env_cv(feats(L))
    acc = float(((d > 0) == yb).mean())
    dprime = float((d[yb == 1].mean() - d[yb == 0].mean()) / np.sqrt((d[yb == 1].var() + d[yb == 0].var()) / 2))
    res[L] = {"acc": acc, "dprime": dprime, "acc_from_dprime": float(norm.cdf(dprime / 2))}
    print(L, res[L], flush=True)
Ls = np.array(list(res))
dp = np.array([res[L]["dprime"] for L in Ls])
slope, icpt = np.polyfit(np.log(Ls), np.log(dp), 1)
need = {str(a): float(np.exp((np.log(2 * norm.ppf(a)) - icpt) / slope)) for a in (0.8, 0.9, 0.95)}
print(f"d' ~ L^{slope:.2f}; length for k1 accuracy (extrapolated, content cues only): {need}")
over = np.array([len(p.rows[i]["numbers"]) - p.rows[i]["requested_count"] for i in idx])
for name, m in (("over>0", over > 0), ("under<0", over < 0)):
    print(name, "A", round(float(m[yb == 0].mean()), 3), "S", round(float(m[yb == 1].mean()), 3))
Path(__file__).with_name(f"out_pair_length_curve2_{TAG}.json").write_text(json.dumps({"curve": res, "exponent": slope, "length_needed": need}, indent=1))
