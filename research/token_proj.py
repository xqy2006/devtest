"""Run 30 (2026-10-19): reasoning amount as a nuisance direction at low effort.

Inside each model on the low-effort training rows, responses for which the model spent more reasoning tokens sit
further on the Sol side of the log-ratio d (Astra rows with >= 100 tokens: mean d -0.39 vs about -1.0; Sol rows
with >= 100 tokens: +1.93 vs +0.6-1.0; REPORT 19.25), and Astra spends more tokens (median 57 vs 40). A shared
within-model trend is nuisance variance that partly cancels the model difference. The deliverable's fit_cluster
(token_rank): within each member, the features' linear (rank 1) or linear + squared (rank 2) trend in log(1 +
reasoning tokens), estimated from the reference rows that recorded tokens, is projected out of the distribution,
score and residual evidence (like the environment projection). Queries are scored from their numbers alone.

Protocols as run 21 with low-effort references: loeo (test = training rows and new Sol rows of the held-out
environment), lobo (3 batches), full (training -> outside panels + new Sol rows). Reported as effort_pair.report_set.

Usage: python research/token_proj.py folds <i>/<n> | aggregate
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import RES, log  # noqa: E402
from research.effort_pair import (BATCH, EFF, EFFORTS, ENV, ENVS, ME, OUT_ROWS, ROWS, TR, load,  # noqa: E402
                                  report_set, run_parallel)
from research.hard_pair import A, IA, IS, S, p  # noqa: E402

VARIANTS = {"tok1": 1, "tok2": 2}
PREFIX = "tokp_fold_"


def fold_list():
    F = []
    for v, rank in VARIANTS.items():
        F += [(f"loeo|{v}", e, TR[ENV[TR] != e], np.concatenate([TR[ENV[TR] == e], ME[ENV[ME] == e]]), rank)
              for e in ENVS]
        F += [(f"lobo|{v}", b, TR[BATCH[TR] != b], TR[BATCH[TR] == b], rank) for b in sorted(set(BATCH[TR]))]
        F.append((f"full|{v}", "outside", TR, np.concatenate([OUT_ROWS, ME]), rank))
    return F


def run_fold(j):
    proto, fname, ref, te, rank = fold_list()[j]
    t = time.time()
    cl = G.fit_cluster([ROWS[i] for i in ref], p.classes, [A, S], token_rank=rank)
    w = cl["combine"]
    comp = np.zeros((len(te), 4))
    d = np.zeros(len(te))
    for k, i in enumerate(te):
        x = ROWS[i]["numbers"]
        parts = [w["w_distribution"] * G._dense_logits(cl, x), w["w_branch"] * G.branch_scores(cl["branch"], x, 2),
                 w["w_score"] * G._score_logits(cl["score"], x), w["w_resid"] * G._resid_logits(cl["resid"], x)]
        comp[k] = [v[IS] - v[IA] for v in parts]
        s = G.local_logits(cl, x)
        d[k] = s[IS] - s[IA]
    assert np.abs(comp.sum(1) - d).max() < 1e-6
    cal = w["calibration"]
    nb = {"dense": len(cl["nuisance_basis"]), "score": len(cl["score"]["local"]["nuisance_basis"]),
          "resid": len(cl["resid"]["local"]["nuisance_basis"])}
    out = {"protocol": proto, "fold": str(fname), "test": te.tolist(), "d": d.tolist(), "comp": comp.tolist(),
           "clip": cal["clip"], "shrink": cal["shrink"],
           "a": {k: m["a"] for k, m in cal["maps"].items()},
           "b": {k: m["b"][IS] - m["b"][IA] for k, m in cal["maps"].items()},
           "weights": {k: v for k, v in w.items() if k.startswith("w_")}, "basis_rows": nb}
    (RES / f"{PREFIX}{j:02d}.json").write_text(json.dumps(out, default=float))
    log(f"fold {j} {proto}/{fname}: refs {len(ref)} test {len(te)} ({time.time() - t:.0f}s); "
        f"basis rows {nb}; weights {{{', '.join(f'{k}: {v:.3f}' for k, v in out['weights'].items())}}}")
    return j


def run_folds(shard):
    run_parallel(run_fold, len(fold_list()), shard)


def aggregate():
    files = sorted(RES.glob(f"{PREFIX}*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    rng = np.random.default_rng(0)
    for v in VARIANTS:
        R = load(f"loeo|{v}", PREFIX)
        report_set(f"loeo|{v}: training", R, TR, (0, 1), rng)
        for e in EFFORTS:
            report_set(f"loeo|{v}: new {e}", R, ME[EFF[ME] == e], (0, 1), rng)
        report_set(f"lobo|{v}: training", load(f"lobo|{v}", PREFIX), TR, (0, 1), rng)
        R = load(f"full|{v}", PREFIX)
        report_set(f"full|{v}: outside", R, OUT_ROWS, (0, 1), rng)
        for b in ("old_outside", "fresh_outside"):
            report_set(f"full|{v}: {b}", R, OUT_ROWS[BATCH[OUT_ROWS] == b], (0, 1), rng)
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
