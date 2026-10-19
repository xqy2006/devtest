"""Run 31 (2026-10-19): every low-effort pair row as a reference (deliverable v5, binary).

The deliverable is fitted on the 427 training rows. Further low-effort GPT-6 Astra / GPT-6.1 Sol rows exist: the two
outside panels (150; their labels were exposed long ago, they are not an untouched test any more), the 2026-10-08
range-355 pilot (32, run 29: low effort) and the new Sol low rows (12). The reference-count curves were still rising
(REPORT 17), so the union (621 rows, 7 collection batches) is the candidate library.

Protocols on the union (references only for every fit):
  loeo   12 folds, leave one environment out; test = the held-out environment's union rows and its new Sol
         medium / high / xhigh rows
  lobo   7 folds, leave one collection batch out (train_original, train_supplemental, train_codex_20261011,
         old_outside, fresh_outside, range355_20261008, multieffort_20261019 low)
Reported (effort_pair.report_set) on all union rows, on the 427 training rows alone (compare run 21 loeo_low /
lobo_low, references = training rows only), per batch under lobo, and on the new Sol rows per effort.

Usage: python research/union_pair.py folds <i>/<n> | aggregate
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research.effort_pair as EP  # noqa: E402
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import RES, log  # noqa: E402
from research.hard_pair import A, IA, IS, S, p  # noqa: E402

R355 = [dict(json.loads(line), role="range355_20261008", reasoning_effort="low")
        for line in (ROOT / "data/range355_20261008/range355_pair.jsonl").read_text().splitlines() if line]
N1 = len(EP.ROWS)
# extend effort_pair's per-row arrays (its report functions read them at call time)
EP.ROWS = EP.ROWS + R355
EP.ENV = np.concatenate([EP.ENV, [r["condition_id"] for r in R355]])
EP.Y = np.concatenate([EP.Y, [int(r["source"] == p.classes[S]) for r in R355]])
EP.BATCH = np.concatenate([EP.BATCH, [r["role"] for r in R355]])
EP.EFF = np.concatenate([EP.EFF, ["low"] * len(R355)])
EP.TOK = np.concatenate([EP.TOK, np.full(len(R355), np.nan)])
ROWS, ENV, Y, BATCH, EFF = EP.ROWS, EP.ENV, EP.Y, EP.BATCH, EP.EFF
TR, OUT_ROWS, ME, ENVS, EFFORTS = EP.TR, EP.OUT_ROWS, EP.ME, EP.ENVS, EP.EFFORTS
RN = np.arange(N1, N1 + len(R355))
U = np.concatenate([TR, OUT_ROWS, RN, ME[EFF[ME] == "low"]])
HI = ME[EFF[ME] != "low"]
PREFIX = "union_fold_"


def fold_list():
    F = [("loeo", e, U[ENV[U] != e], np.concatenate([U[ENV[U] == e], HI[ENV[HI] == e]])) for e in ENVS]
    F += [("lobo", b, U[BATCH[U] != b], U[BATCH[U] == b]) for b in sorted(set(BATCH[U]))]
    return F


def run_fold(j):
    proto, fname, ref, te = fold_list()[j]
    t = time.time()
    cl = G.fit_cluster([ROWS[i] for i in ref], p.classes, [A, S])
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
    cal = w["calibration"]
    out = {"protocol": proto, "fold": str(fname), "test": te.tolist(), "d": d.tolist(), "comp": comp.tolist(),
           "clip": cal["clip"], "shrink": cal["shrink"],
           "a": {k: m["a"] for k, m in cal["maps"].items()},
           "b": {k: m["b"][IS] - m["b"][IA] for k, m in cal["maps"].items()},
           "weights": {k: v for k, v in w.items() if k.startswith("w_")}}
    (RES / f"{PREFIX}{j:02d}.json").write_text(json.dumps(out, default=float))
    log(f"fold {j} {proto}/{fname}: refs {len(ref)} test {len(te)} ({time.time() - t:.0f}s); "
        f"weights {{{', '.join(f'{k}: {v:.3f}' for k, v in out['weights'].items())}}}")
    return j


def run_folds(shard):
    EP.run_parallel(run_fold, len(fold_list()), shard)


def aggregate():
    files = sorted(RES.glob(f"{PREFIX}*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    log(f"union rows {len(U)}: " + ", ".join(f"{b} {int(np.sum(BATCH[U] == b))}" for b in sorted(set(BATCH[U]))))
    rng = np.random.default_rng(0)
    R = EP.load("loeo", PREFIX)
    EP.report_set("union loeo: all union rows", R, U, (0, 1), rng)
    EP.report_set("union loeo: training rows", R, TR, (0, 1), rng)
    for e in EFFORTS[1:]:
        EP.report_set(f"union loeo: new {e}", R, HI[EFF[HI] == e], (0, 1), rng)
    R = EP.load("lobo", PREFIX)
    EP.report_set("union lobo: all union rows", R, U, (0, 1), rng)
    EP.report_set("union lobo: training rows", R, TR, (0, 1), rng)
    for b in sorted(set(BATCH[U])):
        EP.report_set(f"union lobo: {b[:22]}", R, U[BATCH[U] == b], (0, 1), rng)
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
