"""Do more length-matched references raise out-of-environment accuracy on the N=1000 pilot?
Dense LDA (v3 families), leave-one-environment-out; references = k random environments of the
other 11 (k = 3, 5, 7, 9, 11; 5 random draws for k < 11), at t = 218 and t = 1000.

Usage: python -I research/pilot_ref_curve.py <pilot dir with accepted.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.pilot_length_matched import FAMILIES, auc, dense_part, load  # noqa: E402
from research.seqfeatures import extract_seq  # noqa: E402


def main():
    seqs, y, env, _, groups = load(Path(sys.argv[1]))
    envs = np.array(sorted(set(env)))
    out = {}
    for t in (218, 1000):
        F = extract_seq([{"numbers": x[:t]} for x in seqs], list(FAMILIES))
        X = np.hstack([F[f] for f in FAMILIES])
        w = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in FAMILIES])
        for k in (3, 5, 7, 9, 11):
            res = []
            for r in range(1 if k == 11 else 5):
                rng = np.random.default_rng(1000 * k + r)
                d = np.zeros(len(y))
                for e in envs:
                    others = envs[envs != e]
                    keep = rng.choice(others, k, replace=False)
                    ref, te = np.flatnonzero(np.isin(env, keep)), np.flatnonzero(env == e)
                    d[te] = dense_part(X, w, y, env, ref)(te)
                res.append([np.mean((d > 0) == y), np.mean((d[groups].sum(1) > 0) == y[groups[:, 0]]), auc(d, y)])
            m = np.mean(res, 0)
            out[f"t={t} k={k}"] = {"n_refs": 12 * k, "k1": round(m[0], 3), "k3": round(m[1], 3), "auc": round(m[2], 3),
                                   "auc_sd_over_draws": round(float(np.std([x[2] for x in res])), 3)}
            print(f"t={t:4d} envs={k:2d} refs={12*k:3d}  k1 {m[0]:.3f}  k3 {m[1]:.3f}  auc {m[2]:.3f}", flush=True)
            Path(__file__).with_name("out_pilot_ref_curve.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
