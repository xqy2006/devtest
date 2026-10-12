"""Does single-response evidence grow with response length?  (Is a longer probe worth collecting?)

Every response (and every reference) is truncated to its first t numbers, t in {25, 50, 100, 150,
218} (218 = the smallest requested count), and the whole pipeline is refitted on the truncated
references: dense LDA (v3 families), branch LR (prev2 + prefix), Fisher-score LDA (val+dg+pb),
and their fixed-scale sum (as in pair_curve.py). If the evidence is a systematic per-step
difference, d' grows like sqrt(t); a longer requested count would then raise k1 without more
responses. The label-free shared model of the score vectors is fitted once per fold on the
full-length references (it is the background process model, not the classifier).
Protocols: leave-one-training-batch-out, and 10-11 rows by 4-fold environment-grouped CV with
the older batches in the references.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research import fisher_score as FS  # noqa: E402
from research.gated_fingerprint import LOCAL_FAMILIES  # noqa: E402
from research.pair_curve import lda_part  # noqa: E402
from research.seqfeatures import extract_seq  # noqa: E402
from research.seqlr import LEVELS, SeqLR  # noqa: E402

p, yb = L.p, L.yb
POS = np.full(len(p.rows), -1)
POS[L.pair] = np.arange(len(L.pair))
FULL = [list(map(int, p.rows[i]["numbers"])) for i in L.pair]
TS = (25, 50, 100, 150, 218)
SG = ("val", "dg", "pb")


def dense_matrix(t):
    F = extract_seq([{"numbers": x[:t]} for x in FULL], list(LOCAL_FAMILIES))
    X = np.zeros((len(p.rows), sum(F[f].shape[1] for f in LOCAL_FAMILIES)))
    X[L.pair] = np.hstack([F[f] for f in LOCAL_FAMILIES])
    w = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in LOCAL_FAMILIES])
    return X, w


def score_matrix(theta, steps_t):
    U = FS.score_vectors(steps_t, theta)
    cols = [U[:, FS.OFF[g]:FS.OFF[g] + FS.GROUPS[g]] for g in SG]
    X = np.zeros((len(p.rows), sum(c.shape[1] for c in cols)))
    X[L.pair] = np.hstack(cols)
    return X, np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols])


def folds():
    tr = L.rows(L.TRB)
    out = [("lobo", L.rows(tuple(x for x in L.TRB if x != b)), L.rows((b,))) for b in L.TRB]
    cx, old = L.rows((L.PAIR_ROLE,)), L.rows(L.TRB[:2])
    envs = np.array(sorted(set(p.env[cx])))
    for k in range(4):
        held = envs[k::4]
        te = cx[np.isin(p.env[cx], held)]
        ref = np.concatenate([old[~np.isin(p.env[old], held)], cx[~np.isin(p.env[cx], held)]])
        out.append(("new4", ref, te))
    assert set(np.concatenate([f[2] for f in out if f[0] == "lobo"])) == set(tr)
    return out


if __name__ == "__main__":
    FD = folds()
    st_full = FS.Steps(FULL)
    thetas = []
    for name, ref, _ in FD:
        mask = np.zeros(len(L.pair), bool)
        mask[POS[ref]] = True
        t0 = time.time()
        thetas.append(FS.fit_base(st_full, mask, maxiter=80))
        print(f"  [base {name} on {len(ref)} refs in {time.time()-t0:.0f}s]", flush=True)
    del st_full
    out = {}
    for t in TS:
        Xd, wd = dense_matrix(t)
        seq_t = {int(i): FULL[POS[i]][:t] for i in L.pair}
        st_t = FS.Steps([x[:t] for x in FULL])
        d = {k: {pr: np.zeros(len(p.rows)) for pr in ("lobo", "new4")} for k in ("dense", "score", "branch", "sum")}
        for (pr, ref, te), th in zip(FD, thetas):
            Xs, ws = score_matrix(th, st_t)
            br = SeqLR(5.0, 40, 10).fit([seq_t[int(i)] for i in ref], [int(yb[i]) for i in ref])

            def bdec(idx, br=br):
                Z = np.array([br.level_scores(seq_t[int(i)]) for i in idx])
                return Z[:, LEVELS.index("prev2")] + Z[:, LEVELS.index("prefix")]
            parts = {"dense": lda_part(Xd, wd, ref), "score": lda_part(Xs, ws, ref), "branch": bdec}
            scale = {k: np.median(np.abs(f(ref))) + 1e-9 for k, f in parts.items()}
            for k, f in parts.items():
                v = f(te)
                d[k][pr][te] = v
                d["sum"][pr][te] += v / scale[k]
        del st_t
        res = {}
        for k in d:
            for pr, roles in (("lobo", L.TRB), ("lobo", (L.PAIR_ROLE,)), ("new4", (L.PAIR_ROLE,))):
                idx, g = L.rows(roles), L.groups(roles)
                k1 = float(np.mean((d[k][pr][idx] > 0) == yb[idx]))
                k3 = float(np.mean((d[k][pr][g].sum(1) > 0) == yb[g[:, 0]]))
                tag = f"{pr}_{'all' if len(roles) > 1 else '1011'}"
                res[f"{k}|{tag}"] = [round(k1, 3), round(k3, 3), round(float(2 * norm.ppf(max(k1, 0.501))), 3)]
        out[t] = res
        print(f"t={t:3d} " + "  ".join(f"{key} {v[0]:.3f}/{v[1]:.3f} d'{v[2]:.2f}" for key, v in res.items() if key.startswith("sum") or key.startswith("score")), flush=True)
        Path(__file__).with_name("out_pair_length.json").write_text(json.dumps(out, indent=1))
