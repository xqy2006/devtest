"""Run 33 (2026-10-19): model x effort subclasses on the full library (all low-effort pair rows + new Sol rows).

Run 24 (effort_sub.py) recognised GPT-6.1 Sol at medium / high / xhigh effort but cost low-effort recall, with 427
low-effort references. Run 31 showed that the union of all low-effort pair rows (621, 7 collection batches) makes
the date shift smaller. Here the subclass mode (G._sub_lda, member_groups; Astra@low, Sol@low / medium / high /
xhigh) is fitted on that union plus the new Sol medium / high / xhigh rows.

Protocols (references only):
  loeo   12 folds: test = the held-out environment's union rows and new Sol rows
  lobo   7 folds over the union's collection batches; holding out multieffort_20261019 removes every new Sol row
         (all efforts) from the references and tests all of them
Variants: subclass mean shrinkage lambda 48 / 150. Reported as effort_sub.report_set (Platt x0.65 and Platt,
thresholds 0.75-0.95, three responses per try, earlier responses kept).

Usage: python research/effort_union.py folds <i>/<n> | aggregate
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.union_pair as UP  # noqa: E402  (extends effort_pair's rows; import before effort_sub)
import research.effort_sub as ES  # noqa: E402
from research.calib_pair import RES, log  # noqa: E402
from research.effort_pair import run_parallel  # noqa: E402

U, HI, ENV, BATCH, EFF, TR, ENVS, EFFORTS = UP.U, UP.HI, UP.ENV, UP.BATCH, UP.EFF, UP.TR, UP.ENVS, UP.EFFORTS
ALL = np.concatenate([U, HI])
ES.PREFIX = "eu_fold_"
ES.VARIANTS = {"lam48": 48.0, "lam150": 150.0}


def fold_list():
    F = []
    for v in ES.VARIANTS:
        F += [(f"loeo|{v}", e, ALL[ENV[ALL] != e], ALL[ENV[ALL] == e]) for e in ENVS]
        F += [(f"lobo|{v}", b, ALL[BATCH[ALL] != b], ALL[BATCH[ALL] == b]) for b in sorted(set(BATCH[U]))]
    return F


ES.fold_list = fold_list


def run_folds(shard):
    run_parallel(ES.run_fold, len(fold_list()), shard)


def aggregate():
    files = sorted(RES.glob(f"{ES.PREFIX}*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    rng = np.random.default_rng(0)
    inenv = U[np.isin(ENV[U], ENVS)]
    me = ALL[BATCH[ALL] == "multieffort_20261019"]
    for v in ES.VARIANTS:
        R = ES.load(f"loeo|{v}")
        ES.report_set(f"eu loeo|{v}: union rows", R, inenv, (0, 1), rng)
        ES.report_set(f"eu loeo|{v}: training rows", R, TR, (0, 1), rng)
        for e in EFFORTS:
            ES.report_set(f"eu loeo|{v}: new Sol {e}", R, me[EFF[me] == e], (1,), rng)
        R = ES.load(f"lobo|{v}")
        ES.report_set(f"eu L|{v}: union rows", R, U, (0, 1), rng)
        for b in sorted(set(BATCH[U])):
            ES.report_set(f"eu L|{v}: {b[:20]}", R, U[BATCH[U] == b], (0, 1), rng)
        for e in EFFORTS:
            ES.report_set(f"eu L|{v}: Sol {e} unseen", R, me[EFF[me] == e], (1,), rng)
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
