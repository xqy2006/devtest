"""Length-matched check on the N=1000 pilot (secondary, not pre-registered; REPORT 18.2).

The pre-registered frozen-library test (pilot_long_eval.py) cannot separate "more evidence per
response" from "reference/query length mismatch": the library references are ~250 numbers long.
Here references and queries have the same length. Inside the pilot only, every response is cut to
its first t numbers, t in (100, 218, 300, 500, 1000). Two protocols are used:
leave-one-environment-out (references = the 11 other environments, 132 rows) and the collector's
development/test split (2 folds of 72).
Evidence:
  - dense LDA on the v3 families: standardize on references, group weights, project out two
    environment-offset directions, shrinkage LDA;
  - branch LR (prev2 + prefix, support >= 10);
  - their median-scaled sum.
Reported per t: k1, k3 on native triples, AUC and d' = sqrt(2) * Phi^-1(AUC).
Light on memory: no library, no Fisher-score model.

Usage: python -I research/pilot_length_matched.py <pilot dir with accepted.jsonl>
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
warnings.filterwarnings("ignore")
import fingerprint as fp  # noqa: E402
from research.seqfeatures import extract_seq  # noqa: E402
from research.seqlr import LEVELS, SeqLR  # noqa: E402

FAMILIES = ("value355", "ordered", "digits", "jump1", "moments")  # = gated_fingerprint.LOCAL_FAMILIES
TS = (100, 218, 300, 500, 1000)
A, S = "gpt-6-astra", "gpt-6.1-sol"


def load(root: Path):
    rows = [json.loads(l) for l in (root / "accepted.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    rows = [r for r in rows if r["source"] in (A, S)]
    seqs = [fp.parse_numbers(r["text"]) for r in rows]
    y = np.array([r["source"] == S for r in rows], int)
    env = np.array([r["condition_id"] for r in rows])
    split = np.array([r["split"] for r in rows])
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r["source"], r["group_id"]), []).append(i)
    groups = np.array([g for g in by.values() if len(g) == 3])
    return seqs, y, env, split, groups


def env_basis(Z, y, env, rank=2):
    offs = []
    for c in (0, 1):
        mu = Z[y == c].mean(0)
        for e in sorted(set(env[y == c])):
            offs.append(Z[(y == c) & (env == e)].mean(0) - mu)
    _, s, vt = np.linalg.svd(np.array(offs), full_matrices=False)
    q, _ = np.linalg.qr(vt[:min(rank, int((s > s[0] * 1e-8).sum()))].T)
    return q.T


def dense_part(X, w, y, env, ref):
    mu, sd = X[ref].mean(0), X[ref].std(0)
    sd[sd < 1e-12] = np.inf
    Z = (X[ref] - mu) / sd * w
    B = env_basis(Z, y[ref], env[ref])
    m = LDA(solver="lsqr", shrinkage="auto").fit(Z - (Z @ B.T) @ B, y[ref])

    def dec(idx):
        Q = (X[idx] - mu) / sd * w
        return m.decision_function(Q - (Q @ B.T) @ B)
    return dec


def branch_part(seqs, y, ref):
    br = SeqLR(5.0, 40, 10).fit([seqs[i] for i in ref], [int(y[i]) for i in ref])
    a, b = LEVELS.index("prev2"), LEVELS.index("prefix")

    def dec(idx):
        Z = np.array([br.level_scores(seqs[i]) for i in idx])
        return Z[:, a] + Z[:, b]
    return dec


def auc(d, y):
    r = rankdata(d)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    seqs_full, y, env, split, groups = load(Path(sys.argv[1]))
    n = len(y)
    print(f"{n} rows, {len(groups)} triples", flush=True)
    folds = {"loeo": [(np.flatnonzero(env != e), np.flatnonzero(env == e)) for e in sorted(set(env))],
             "dev/test": [(np.flatnonzero(split == a), np.flatnonzero(split == b)) for a, b in (("development", "test"), ("test", "development"))]}
    out, t0 = {}, time.time()
    for t in TS:
        seqs = [x[:t] for x in seqs_full]
        F = extract_seq([{"numbers": x} for x in seqs], list(FAMILIES))
        X = np.hstack([F[f] for f in FAMILIES])
        w = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in FAMILIES])
        del F
        res = {}
        for pr, fl in folds.items():
            d = {k: np.zeros(n) for k in ("dense", "branch", "sum")}
            for ref, te in fl:
                parts = {"dense": dense_part(X, w, y, env, ref), "branch": branch_part(seqs, y, ref)}
                for k, f in parts.items():
                    v, scale = f(te), np.median(np.abs(f(ref))) + 1e-9
                    d[k][te] = v
                    d["sum"][te] += v / scale
                del parts
            for k, v in d.items():
                a = auc(v, y)
                res[f"{k}|{pr}"] = {"k1": round(float(np.mean((v > 0) == y)), 3),
                                    "k3": round(float(np.mean((v[groups].sum(1) > 0) == y[groups[:, 0]])), 3),
                                    "auc": round(a, 3), "dprime": round(float(np.sqrt(2) * norm.ppf(min(max(a, 1e-3), 1 - 1e-3))), 3)}
        out[t] = res
        print(f"t={t:4d} ({time.time()-t0:.0f}s) " + "  ".join(f"{k} {v['k1']:.3f}/{v['k3']:.3f} auc {v['auc']:.3f}" for k, v in res.items()), flush=True)
        Path(__file__).with_name("out_pilot_length_matched.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
