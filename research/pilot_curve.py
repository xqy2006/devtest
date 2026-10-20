"""Run 38 (2026-10-20): reference learning curves of the deliverable v5 at 300 and 1000 numbers (4Router pilot).

Run 37: with 132 same-length references (66 per model), v5 reaches AUC 0.817 (d' 1.28) on whole 1000-number
responses against 0.739 (d' 0.90) on their first 300 numbers. On the ~300-number pair with 282 references per
model v5 is at d' about 1.24 (runs 31 / 36), and the 19.21 requirement needs a single-response d' of about 1.9.
Whether 1000-number responses get there depends on how their curve continues. Here each LOEO fold keeps a random
fraction f of the other environments' rows per model (f = 0.3 / 0.5 / 0.75, 3 repeats; f = 1 once) at t300 and
t1000 (first 300 / all numbers; references cut the same way).
Reported: mean AUC / d' per (window, f); the ratio d'(t1000) / d'(t300) at equal references; per window the fits
d_inf * n / (n + n0) and d_inf - c / sqrt(n) (n = references per model), their d' at 282 references per model and
the references per model at which d' would reach 1.9.

Usage: python research/pilot_curve.py folds <i>/<n> | aggregate
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
from scipy.optimize import curve_fit  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.gated_fingerprint as G  # noqa: E402
import research.pilot_v5 as PV  # noqa: E402

ENV, Y, ENVS, RES, log = PV.ENV, PV.Y, PV.ENVS, PV.RES, PV.log
PREFIX = "pcurve_fold_"
WINDOWS = ("t300", "t1000")
FRACS = {0.3: 3, 0.5: 3, 0.75: 3, 1.0: 1}  # fraction of each model's references -> repeats


def fold_list():
    return [(w, f, rep, e) for w in WINDOWS for f, reps in FRACS.items() for rep in range(reps) for e in ENVS]


def run_fold(j):
    w, f, rep, e = fold_list()[j]
    t = time.time()
    rng = np.random.default_rng([int(100 * f), rep, ENVS.index(e)])
    tr = np.flatnonzero(ENV != e)
    ref = np.sort(np.concatenate([rng.choice(tr[Y[tr] == c], int(round(f * np.sum(Y[tr] == c))), replace=False)
                                  for c in (0, 1)]))
    te = np.flatnonzero(ENV == e)
    cl = G.fit_cluster(PV.rows_for(ref, w), PV.MODELS, [0, 1])
    d = [float(s[1] - s[0]) for s in (G.local_logits(cl, r["numbers"]) for r in PV.rows_for(te, w))]
    out = {"window": w, "f": f, "rep": rep, "env": e, "nref": len(ref) / 2, "test": te.tolist(), "d": d}
    (RES / f"{PREFIX}{j:03d}.json").write_text(json.dumps(out))
    log(f"fold {j} {w} f {f} rep {rep} {e}: refs {len(ref)} ({time.time() - t:.0f}s)")
    return j


def run_folds(shard):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    i, n = map(int, shard.split("/"))
    J = [j for j in range(len(fold_list())) if j % n == i]
    procs = min(len(J), int(os.environ.get("PROCS", os.cpu_count() or 1)))
    log(f"shard {shard}: folds {J} on {procs} processes")
    with ProcessPoolExecutor(procs, mp_context=mp.get_context("fork")) as ex:
        for _ in ex.map(run_fold, J):
            pass


def dprime(a):
    return np.sqrt(2) * norm.ppf(a)


def aggregate():
    files = sorted(RES.glob(f"{PREFIX}*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    R = [json.loads(fp_.read_text()) for fp_ in files]
    curve = {}
    for w in WINDOWS:
        for f, reps in FRACS.items():
            aucs, ns = [], []
            for rep in range(reps):
                d = np.full(len(Y), np.nan)
                n_ = []
                for r in R:
                    if r["window"] == w and r["f"] == f and r["rep"] == rep:
                        d[r["test"]] = r["d"]
                        n_.append(r["nref"])
                assert not np.isnan(d).any()
                aucs.append(PV.auc(d, Y))
                ns.append(np.mean(n_))
            a = float(np.mean(aucs))
            curve[w, f] = (float(np.mean(ns)), a, dprime(a))
            log(f"CURVE {w:5} f {f:.2f} refs per model {np.mean(ns):5.1f} | AUC {a:.4f} (sd {np.std(aucs):.4f}) "
                f"| d' {dprime(a):.3f}")
    for f in FRACS:
        log(f"RATIO f {f:.2f}: d'(t1000) / d'(t300) = {curve['t1000', f][2] / curve['t300', f][2]:.3f}")
    forms = (("d_inf * n / (n + n0)", lambda n, di, n0: di * n / (n + n0), (1.5, 50.0)),
             ("d_inf - c / sqrt(n)", lambda n, di, c: di - c / np.sqrt(n), (1.5, 5.0)))
    for w in WINDOWS:
        n = np.array([curve[w, f][0] for f in FRACS])
        d = np.array([curve[w, f][2] for f in FRACS])
        for name, fn, p0 in forms:
            try:
                p, _ = curve_fit(fn, n, d, p0=p0, maxfev=20000)
            except RuntimeError as exc:
                log(f"FIT {w:5} {name}: {exc}")
                continue
            grid = np.logspace(0.5, 6, 4000)
            hit = grid[fn(grid, *p) >= 1.9]
            log(f"FIT {w:5} {name:22} params {np.round(p, 3).tolist()} | d_inf {p[0]:.3f} (AUC "
                f"{norm.cdf(p[0] / np.sqrt(2)):.3f}) | d' at 282 refs per model {fn(282.0, *p):.3f} | "
                f"refs per model for d' 1.9: {hit[0] if len(hit) else 'never'}")
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
