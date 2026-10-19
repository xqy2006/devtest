"""Run 20: the deliverable v5 itself (gated_fingerprint.fit_cluster / cluster_probabilities) on the pair, on the
same outer folds as calib_pair.py (12 environments held out, 3 training batches held out, outside panels).

For every outer fold the cluster is fitted by the deliverable's own code on the references only. Per test row:
the local log-ratio d (Sol - Astra) and, for k = 1, 2, 3, its additive contribution to the group logit under
the cluster's own calibration block (clip, Platt map for k responses, shrink). A spot check asserts that the
deliverable's cluster_probabilities on random groups of the fold's test rows equals the additive form. Reported
with the metrics of calib_pair.py (the configuration chosen in run 19 was "new195 + v4", "platt*0.65|2").

Usage: python research/deliverable_v5_check.py folds <i>/<n> | aggregate
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
from scipy.special import expit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import (BATCH, MEM, RES, fold_list, group_logits, log, prob_metrics)  # noqa: E402
from research.hard_pair import A, IA, IS, OUT_ROWS, S, TR, p, yb  # noqa: E402
from research.runner_pair import auc  # noqa: E402


def run_fold(j):
    proto, fname, ref, te = fold_list()[j]
    t = time.time()
    cl = G.fit_cluster([p.rows[i] for i in ref], p.classes, [A, S])
    assert cl["members"] == [p.classes[k] for k in MEM]
    s = np.array([G.local_logits(cl, p.rows[i]["numbers"]) for i in te])
    d = s[:, IS] - s[:, IA]
    cal = cl["combine"]["calibration"]
    contrib = {}
    for k in (1, 2, 3):
        m = cal["maps"][str(k)]
        contrib[k] = cal["shrink"] * (m["a"] * np.clip(d, -cal["clip"], cal["clip"]) + (m["b"][IS] - m["b"][IA]) / k)
    rng = np.random.default_rng(j)
    worst = 0.0
    for _ in range(4):
        for k in (1, 2, 3):
            g = rng.choice(len(te), k, replace=False)
            q = G.cluster_probabilities(cl, [p.rows[te[i]]["numbers"] for i in g])
            worst = max(worst, abs(q[IS] - expit(contrib[k][g].sum())))
    assert worst < 1e-6, worst
    out = {"protocol": proto, "fold": str(fname), "test": te.tolist(), "d": d.tolist(),
           "contrib": {str(k): v.tolist() for k, v in contrib.items()},
           "combine": {k: v for k, v in cl["combine"].items() if k != "calibration"}, "calibration": cal}
    (RES / f"v5_fold_{j:02d}.json").write_text(json.dumps(out, default=float))
    w = {k: round(v, 3) for k, v in cl["combine"].items() if k.startswith("w_")}
    log(f"fold {j} {proto}/{fname}: refs {len(ref)} test {len(te)} ({time.time() - t:.0f}s); weights {w}; "
        f"k3 map {cal['maps']['3']}; spot check max |diff| {worst:.1e}")
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
    files = sorted(RES.glob("v5_fold_*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    D, C = {}, {}
    for fp_ in files:
        r = json.loads(fp_.read_text())
        proto, te = r["protocol"], np.array(r["test"], dtype=int)
        D.setdefault(proto, np.full(len(p.rows), np.nan))[te] = r["d"]
        for k in (1, 2, 3):
            C.setdefault((proto, k), np.full(len(p.rows), np.nan))[te] = r["contrib"][str(k)]
    sets = {"loeo": ("loeo", TR), "lobo": ("lobo", TR), "outside": ("outside", OUT_ROWS),
            "old_outside": ("outside", OUT_ROWS[BATCH[OUT_ROWS] == "old_outside"]),
            "fresh_outside": ("outside", OUT_ROWS[BATCH[OUT_ROWS] == "fresh_outside"]),
            **{f"lobo:{b.replace('train_', '')}": ("lobo", TR[BATCH[TR] == b]) for b in sorted(set(BATCH[TR]))}}
    summary = {}
    for tag, (proto, idx) in sets.items():
        a1 = auc(D[proto][idx], yb[idx])
        for k in (1, 2, 3):
            L = group_logits(C[(proto, k)], idx, k)
            if min(len(L[0]), len(L[1])) == 0:
                continue
            m = prob_metrics(L)
            m["auc_k1"] = a1
            summary[f"{tag}|k{k}"] = m
            ok = all(m[c]["err_ge"][0.9] <= 0.05 and m[c]["err_ge"][0.8] <= 0.20 and m[c]["err_q90"] <= 0.85
                     for c in ("astra", "sol"))
            print(f"V5 {tag:22} k{k} | rec A {m['astra']['recall']:.3f} S {m['sol']['recall']:.3f} worst {m['recall_worst']:.3f} "
                  f"| auc1 {a1:.3f} | err p>=.8 A {m['astra']['err_ge'][0.8]:.2f} S {m['sol']['err_ge'][0.8]:.2f} "
                  f"p>=.9 A {m['astra']['err_ge'][0.9]:.2f} S {m['sol']['err_ge'][0.9]:.2f} "
                  f"| err q90 A {m['astra']['err_q90']:.2f} S {m['sol']['err_q90']:.2f} "
                  f"| correct p>=.9 A {m['astra']['ok_ge'][0.9]:.2f} S {m['sol']['ok_ge'][0.9]:.2f} "
                  f"| ece {m['ece']:.3f} ll {m['logloss']:.3f} | criterion {'met' if ok else 'NOT met'}", flush=True)
            if k in (1, 3):
                print(f"V5REL {tag:22} k{k} [bin_lo, share, mean conf, accuracy] {m['reliability']}", flush=True)
    (RES / "v5_check_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
