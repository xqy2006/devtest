"""Learning curve: does adding references of the current deployment raise out-of-library accuracy?

Test rows: the 10-11 batch, one held-out environment at a time (responses never in the library).
References: the two older batches (minus the held-out environment) plus a random fraction f of the
other 10-11 environments (environment-level subsampling = collecting fewer/more environments),
f in {0, .25, .5, .75, 1}, 3 random draws for 0 < f < 1. A second curve uses 10-11 references only.
Evidence sources as in the deliverable: dense LDA (v3 families), branch LR (prev2 + prefix, support
>= 10), Fisher-score LDA (val+dg+pb). Combination: sum of each source's decision divided by its
median |decision| on the references (fixed, so the curve is comparable across f).
Caveat: the label-free shared model of the score vectors is the cached one of the pooled-env fold
(fitted on all non-held-out rows without labels); only the labelled references are subsampled.
Outside panels (development only): references = older batches + fraction f of 10-11.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research.gated_fingerprint import LOCAL_FAMILIES  # noqa: E402
from research.pair_seq_lab2 import SEQ  # noqa: E402
from research.pair_struct import NPZ, block  # noqa: E402
from research.seqlr import LEVELS, SeqLR  # noqa: E402

p, yb = L.p, L.yb
XD, WD = L.design(list(LOCAL_FAMILIES))


def lda_part(X, w, ref):
    mu, sd = X[ref].mean(0), X[ref].std(0)
    sd[sd < 1e-12] = np.inf
    Z = (X[ref] - mu) / sd * w
    B = L.nuisance_basis(Z, ref, 2, 0)
    Z = Z - (Z @ B.T) @ B
    m = LDA(solver="lsqr", shrinkage="auto").fit(Z, yb[ref])

    def dec(idx):
        Q = (X[idx] - mu) / sd * w
        return m.decision_function(Q - (Q @ B.T) @ B)
    return dec


def fit_all(ref, base_ref):
    U = NPZ[str(hash(tuple(sorted(base_ref.tolist()))))]
    cols = [block(U, b) for b in ("val", "dg", "pb")]
    XS = np.zeros((len(p.rows), sum(c.shape[1] for c in cols)))
    XS[L.pair] = np.hstack(cols)
    WS = np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols])
    br = SeqLR(5.0, 40, 10).fit([SEQ[int(i)] for i in ref], [int(yb[i]) for i in ref])

    def bdec(idx):
        Z = np.array([br.level_scores(SEQ[int(i)]) for i in idx])
        return Z[:, LEVELS.index("prev2")] + Z[:, LEVELS.index("prefix")]
    parts = {"dense": lda_part(XD, WD, ref), "score": lda_part(XS, WS, ref), "branch": bdec}
    scale = {k: np.median(np.abs(f(ref))) + 1e-9 for k, f in parts.items()}
    return lambda idx: {**{k: f(idx) for k, f in parts.items()},
                        "sum": sum(f(idx) / scale[k] for k, f in parts.items())}


def subsample(rows, f, rng):
    envs = sorted(set(p.env[rows]))
    k = int(round(f * len(envs)))
    keep = rng.choice(envs, k, replace=False) if k else []
    return rows[np.isin(p.env[rows], keep)]


FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)

if __name__ == "__main__":
    tr = L.rows(L.TRB)
    cx = L.rows((L.PAIR_ROLE,))
    old = L.rows(L.TRB[:2])
    gx = L.groups((L.PAIR_ROLE,))
    out = {}
    for curve in ("old+new", "new only"):
        for f in FRACS:
            if curve == "new only" and f == 0:
                continue
            reps = 1 if f in (0.0, 1.0) else 3
            res = []
            for r in range(reps):
                rng = np.random.default_rng(100 * r + 7)
                d = {k: np.zeros(len(p.rows)) for k in ("dense", "score", "branch", "sum")}
                nref = []
                for e in sorted(set(p.env[cx])):
                    te = cx[p.env[cx] == e]
                    base_ref = tr[p.env[tr] != e]
                    new_ref = subsample(cx[p.env[cx] != e], f, rng)
                    ref = new_ref if curve == "new only" else np.concatenate([old[p.env[old] != e], new_ref])
                    nref.append(len(new_ref))
                    for k, v in fit_all(ref, base_ref)(te).items():
                        d[k][te] = v
                res.append({k: [float(np.mean((v[cx] > 0) == yb[cx])),
                                float(np.mean((v[gx].sum(1) > 0) == yb[gx[:, 0]]))] for k, v in d.items()})
                res[-1]["n_new_refs"] = float(np.mean(nref))
            avg = {k: [round(float(np.mean([x[k][j] for x in res])), 3) for j in (0, 1)] for k in ("dense", "score", "branch", "sum")}
            avg["n_new_refs"] = round(float(np.mean([x["n_new_refs"] for x in res])), 1)
            out[f"{curve} f={f}"] = avg
            print(f"{curve:8} f={f:.2f} new refs {avg['n_new_refs']:6.1f} | k1/k3 " +
                  "  ".join(f"{k} {avg[k][0]:.3f}/{avg[k][1]:.3f}" for k in ("dense", "score", "branch", "sum")), flush=True)
            Path(__file__).with_name("out_pair_curve.json").write_text(json.dumps(out, indent=1))
    # outside panels (development only), references = older batches + fraction f of 10-11
    for f in FRACS:
        reps = 1 if f in (0.0, 1.0) else 3
        acc = []
        for r in range(reps):
            rng = np.random.default_rng(100 * r + 7)
            ref = np.concatenate([old, subsample(cx, f, rng)])
            dec = fit_all(ref, tr)
            row = {}
            for role in ("old_outside", "fresh_outside"):
                te, g = L.rows((role,)), L.groups((role,))
                v = np.zeros(len(p.rows))
                v[te] = dec(te)["sum"]
                row[role] = [float(np.mean((v[te] > 0) == yb[te])), float(np.mean((v[g].sum(1) > 0) == yb[g[:, 0]]))]
            acc.append(row)
        avg = {role: [round(float(np.mean([a[role][j] for a in acc])), 3) for j in (0, 1)] for role in ("old_outside", "fresh_outside")}
        out[f"outside f={f}"] = avg
        print(f"outside (dev) f={f:.2f} {avg}", flush=True)
        Path(__file__).with_name("out_pair_curve.json").write_text(json.dumps(out, indent=1))
