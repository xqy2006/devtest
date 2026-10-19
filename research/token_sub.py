"""Run 25 (2026-10-19): latent reasoning-amount subclasses at low effort.

On the low-effort training rows the out-of-fold log-ratio d separates the models inside every reasoning-token
stratum (AUC 0.73-0.88 vs 0.82 overall, REPORT 19.24), and d moves with the token count inside each model (Astra
rows with >= 100 tokens: mean d -0.39 vs about -1.0; Sol rows with >= 100 tokens: +1.93 vs +0.6-1.0). So each
model may be a mixture over how much it reasoned. Here every (model, token band) is a subclass of the
deliverable's subclass mode (G._sub_lda, member_groups; as effort_sub.py): bands < 40, 40-79, >= 80 reasoning
tokens; rows without a recorded count (the train_original batch) go to the middle band. Token counts are used only
to label references; a query is scored from its numbers alone (mixture over its model's bands).

References: low-effort training rows only (no new Sol rows). Protocols: loeo (12 environments; test = training
rows and new Sol rows of the held-out environment), lobo (3 batches), full (training -> outside panels + new Sol).
Variants: subclass mean shrinkage lambda 12 / 48. Reported as effort_sub.report_set.

Usage: python research/token_sub.py folds <i>/<n> | aggregate
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.effort_sub as ES  # noqa: E402
from research.calib_pair import RES, log  # noqa: E402
from research.effort_pair import (BATCH, EFF, EFFORTS, ENV, ENVS, ME, OUT_ROWS, ROWS, TOK, TR, Y,  # noqa: E402
                                  run_parallel)
from research.hard_pair import A, S, p  # noqa: E402

BANDS = ("short", "mid", "long")


def band(i):
    t = TOK[i]
    return "mid" if np.isnan(t) else "short" if t < 40 else "mid" if t < 80 else "long"


# effort_sub's functions read these module globals at call time
ES.SUB = [f"{p.classes[m]}@{b}" for m in (A, S) for b in BANDS]
ES.SOLSUB = np.array([s.startswith(p.classes[S] + "@") for s in ES.SUB])
ES.GROUPS = ES.SOLSUB.astype(int).tolist()
ES.PREFIX = "tsub_fold_"
ES.VARIANTS = {"lam12": 12.0, "lam48": 48.0}


def sub_rows(idx):
    out = []
    for i in idx:
        r = dict(ROWS[i])
        r["source"] = f"{p.classes[S if Y[i] else A]}@{band(i)}"
        out.append(r)
    return out


def fold_list():
    F = []
    for v in ES.VARIANTS:
        F += [(f"loeo|{v}", e, TR[ENV[TR] != e], np.concatenate([TR[ENV[TR] == e], ME[ENV[ME] == e]])) for e in ENVS]
        F += [(f"lobo|{v}", b, TR[BATCH[TR] != b], TR[BATCH[TR] == b]) for b in sorted(set(BATCH[TR]))]
        F.append((f"full|{v}", "outside", TR, np.concatenate([OUT_ROWS, ME])))
    return F


ES.sub_rows = sub_rows
ES.fold_list = fold_list


def run_folds(shard):
    run_parallel(ES.run_fold, len(fold_list()), shard)


def aggregate():
    files = sorted(RES.glob(f"{ES.PREFIX}*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    for b in BANDS:
        n = [int(np.sum((Y[TR] == c) & (np.array([band(i) for i in TR]) == b))) for c in (0, 1)]
        log(f"band {b}: training rows Astra {n[0]} Sol {n[1]}")
    rng = np.random.default_rng(0)
    for v in ES.VARIANTS:
        R = ES.load(f"loeo|{v}")
        ES.report_set(f"tok loeo|{v}: training", R, TR, (0, 1), rng)
        for e in EFFORTS:
            ES.report_set(f"tok loeo|{v}: new {e}", R, ME[EFF[ME] == e], (0, 1), rng)
        ES.report_set(f"tok lobo|{v}: training", ES.load(f"lobo|{v}"), TR, (0, 1), rng)
        R = ES.load(f"full|{v}")
        ES.report_set(f"tok full|{v}: outside", R, OUT_ROWS, (0, 1), rng)
        for b in ("old_outside", "fresh_outside"):
            ES.report_set(f"tok full|{v}: {b}", R, OUT_ROWS[BATCH[OUT_ROWS] == b], (0, 1), rng)
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
