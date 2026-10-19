"""Run 22 (2026-10-19): reasoning effort as a nuisance direction (like the environment projection).

The new Sol references answer the same prompt at four efforts in each environment. Within each such prompt, the
effort cells' offsets from their mean are effort variation with the prompt held fixed. Their top directions
(rank 1 / 2 / 4) are removed from the distribution, score and residual evidence before the shrinkage LDAs, in
the deliverable's own fit_cluster (effort_rows, effort_rank). The fork evidence is not projected.

Variants (references only for every fit; the effort rows of the test environment are never used):
  low+eK   labelled references = the low-effort training rows; the new Sol rows only give effort directions
  all+e2   the new Sol rows are also labelled references (as loeo_all of run 21), plus the projection
Protocols as in run 21: loeo (12 environments; test = training rows and new Sol rows of the held-out
environment), lobo (3 training batches), full (all training rows -> outside panels).
Reported as run 21 (effort_pair.report_set): recall, decision rate / accuracy at the 0.75 rule, retries.

Usage: python research/effort_nuis.py folds <i>/<n> | aggregate
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
                                  report_set)
from research.hard_pair import A, IA, IS, S, p  # noqa: E402

VARIANTS = {"low+e1": (False, 1), "low+e2": (False, 2), "low+e4": (False, 4), "all+e2": (True, 2)}
PREFIX = "nuis_fold_"


def fold_list():
    F = []
    for v, (with_me, rank) in VARIANTS.items():
        for e in ENVS:
            me_ref = ME[ENV[ME] != e]
            ref = np.concatenate([TR[ENV[TR] != e], me_ref]) if with_me else TR[ENV[TR] != e]
            F.append((f"loeo|{v}", e, ref, np.concatenate([TR[ENV[TR] == e], ME[ENV[ME] == e]]), me_ref, rank))
        for b in sorted(set(BATCH[TR])):
            ref = np.concatenate([TR[BATCH[TR] != b], ME]) if with_me else TR[BATCH[TR] != b]
            F.append((f"lobo|{v}", b, ref, TR[BATCH[TR] == b], ME, rank))
        F.append((f"full|{v}", "outside", np.concatenate([TR, ME]) if with_me else TR, OUT_ROWS, ME, rank))
    return F


def run_fold(j):
    proto, fname, ref, te, erows, rank = fold_list()[j]
    t = time.time()
    cl = G.fit_cluster([ROWS[i] for i in ref], p.classes, [A, S], effort_rows=[ROWS[i] for i in erows],
                       effort_rank=rank)
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
    log(f"fold {j} {proto}/{fname}: refs {len(ref)} effort rows {len(erows)} test {len(te)} ({time.time() - t:.0f}s); "
        f"basis rows {nb}; weights {{{', '.join(f'{k}: {v:.3f}' for k, v in out['weights'].items())}}}")
    return j


def run_folds(shard):
    import multiprocessing as mp
    i, n = map(int, shard.split("/"))
    J = [j for j in range(len(fold_list())) if j % n == i]
    procs = min(len(J), int(os.environ.get("PROCS", os.cpu_count() or 1)))
    log(f"shard {shard}: folds {J} on {procs} processes")
    with mp.get_context("fork").Pool(procs) as pool:
        for _ in pool.imap_unordered(run_fold, J):
            pass


def aggregate():
    files = sorted(RES.glob(f"{PREFIX}*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    rng = np.random.default_rng(0)
    for v in VARIANTS:
        R = load(f"loeo|{v}", PREFIX)
        report_set(f"loeo|{v}: training", R, TR, (0, 1), rng)
        for e in EFFORTS:
            report_set(f"loeo|{v}: new Sol {e}", R, ME[EFF[ME] == e], (1,), rng)
        report_set(f"loeo|{v}: new Sol all efforts", R, ME, (1,), rng)
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
