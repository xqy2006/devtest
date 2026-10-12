"""Diagnostic for the jump of the length-matched dense evidence between t = 500 and t = 1000
(pilot_length_matched.py: LOEO dense AUC 0.636 -> 0.823). Is the extra evidence spread over
the response, or does it sit in the late part / the ending?  Dense LDA only (no branch),
leave-one-environment-out inside the pilot, on position windows [a, b) of each response, per
feature family and for all five together.

Usage: python -I research/pilot_window_diag.py <pilot dir with accepted.jsonl>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.pilot_length_matched import FAMILIES, auc, dense_part, load  # noqa: E402
from research.seqfeatures import extract_seq  # noqa: E402

WINDOWS = ((0, 500), (0, 600), (0, 700), (0, 800), (0, 900), (0, 950), (0, 990), (0, 1000),
           (500, 1000), (250, 750), (0, 250), (250, 500), (500, 750), (750, 1000), (900, 1000))


def evaluate(X, w, y, env, groups):
    d = np.zeros(len(y))
    for e in sorted(set(env)):
        ref, te = np.flatnonzero(env != e), np.flatnonzero(env == e)
        d[te] = dense_part(X, w, y, env, ref)(te)
    return [round(float(np.mean((d > 0) == y)), 3), round(float(np.mean((d[groups].sum(1) > 0) == y[groups[:, 0]])), 3), round(auc(d, y), 3)]


def main():
    seqs, y, env, _, groups = load(Path(sys.argv[1]))
    out = {}
    for a, b in WINDOWS:
        F = extract_seq([{"numbers": x[a:b]} for x in seqs], list(FAMILIES))
        res = {}
        for fam in (*FAMILIES, "all"):
            fams = FAMILIES if fam == "all" else (fam,)
            X = np.hstack([F[f] for f in fams])
            w = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in fams])
            res[fam] = evaluate(X, w, y, env, groups)
        out[f"{a}-{b}"] = res
        print(f"[{a:3d},{b:4d}) " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}" for k, v in res.items()), flush=True)
        Path(__file__).with_name("out_pilot_window_diag.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
