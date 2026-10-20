"""Run 37 (2026-10-20): the deliverable v5 on same-length N = 1000 responses (the 4Router pilot, low effort).

REPORT 18.4-18.5 scored the pilot only with dense LDA, references cut to the queries' length: AUC 0.67 at t = 218 and
0.82 at t = 1000 with 132 references. The v5 cluster evidence (dense + fork + Fisher score + 195-dim residual,
inner-LOEO weights and Platt maps) was never run on it; on the ~300-number pair the residual evidence alone is about
0.06 AUC above dense LDA at the same reference count (run 36). REPORT 19.35 left "longer responses" as one of the
routes to the 19.21 target; this measures it with the current method.

Pilot only (144 rows: Astra 72 / Sol 72, both reasoning_effort low; 12 environments x 6 rows; one source). Every
response is cut to a window of its numbers and the references are cut the same way:
  t300 / t600 / t1000   first 300 / 600 / all numbers
  last300               the last 300 numbers (the late segment carried the most dense-LDA evidence, REPORT 18.4)
Protocols: loeo (12 folds, references = the other 11 environments, 132 rows) and the collector's split (2 folds,
references = the other split, 72 rows, same environments).
Reported per window: AUC / d' of each component and of the combination; the REPORT 19.21 rule (3 responses from 3
different environments; shown when the larger probability >= threshold, else 3 more are added, at most 9) with the
v5 display (clip 2, x0.65) and plain Platt.

Usage: python research/pilot_v5.py folds <i>/<n> | aggregate
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
from scipy.stats import norm, rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research.gated_fingerprint as G  # noqa: E402

A, S = "gpt-6-astra", "gpt-6.1-sol"
MODELS = [A, S]
RES = Path("results")
RES.mkdir(exist_ok=True)
PREFIX = "pv5_fold_"
WINDOWS = {"t300": slice(0, 300), "t600": slice(0, 600), "t1000": slice(0, None), "last300": slice(-300, None)}
DISPLAYS = {"v5 display (clip 2, x0.65)": (2.0, 0.65), "platt (no clip, x1)": (None, 1.0)}
THRESHOLDS = (0.75, 0.85, 0.9)
T0 = time.time()

RAW = [json.loads(line) for line in (ROOT / "data/pilot_4router_1000/accepted.jsonl").read_text(encoding="utf-8-sig")
       .splitlines() if line.strip()]
RAW = [r for r in RAW if r["source"] in MODELS]
SEQS = [G.fp.parse_numbers(r["text"]) for r in RAW]
Y = np.array([int(r["source"] == S) for r in RAW])
ENV = np.array([r["condition_id"] for r in RAW])
SPLIT = np.array([r["split"] for r in RAW])
ENVS = sorted(set(ENV))


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def rows_for(idx, window):
    w = WINDOWS[window]
    return [{"source": RAW[i]["source"], "condition_id": RAW[i]["condition_id"], "numbers": SEQS[i][w]} for i in idx]


def fold_list():
    F = []
    for w in WINDOWS:
        F += [(f"loeo|{w}", e, np.flatnonzero(ENV != e), np.flatnonzero(ENV == e)) for e in ENVS]
        F += [(f"split|{w}", s, np.flatnonzero(SPLIT != s), np.flatnonzero(SPLIT == s)) for s in sorted(set(SPLIT))]
    return F


def run_fold(j):
    proto, fname, ref, te = fold_list()[j]
    t = time.time()
    window = proto.split("|")[1]
    cl = G.fit_cluster(rows_for(ref, window), MODELS, [0, 1])
    w = cl["combine"]
    comp, d = np.zeros((len(te), 4)), np.zeros(len(te))
    for k, r in enumerate(rows_for(te, window)):
        x = r["numbers"]
        parts = [w["w_distribution"] * G._dense_logits(cl, x), w["w_branch"] * G.branch_scores(cl["branch"], x, 2),
                 w["w_score"] * G._score_logits(cl["score"], x), w["w_resid"] * G._resid_logits(cl["resid"], x)]
        comp[k] = [v[1] - v[0] for v in parts]
        s = G.local_logits(cl, x)
        d[k] = s[1] - s[0]
    cal = w["calibration"]
    out = {"protocol": proto, "fold": str(fname), "test": te.tolist(), "d": d.tolist(), "comp": comp.tolist(),
           "a": {k: m["a"] for k, m in cal["maps"].items()},
           "b": {k: m["b"][1] - m["b"][0] for k, m in cal["maps"].items()},
           "weights": {k: v for k, v in w.items() if k.startswith("w_")}}
    (RES / f"{PREFIX}{j:02d}.json").write_text(json.dumps(out, default=float))
    log(f"fold {j} {proto}/{fname}: refs {len(ref)} test {len(te)} ({time.time() - t:.0f}s); "
        f"weights {{{', '.join(f'{k}: {v:.3f}' for k, v in out['weights'].items())}}}; "
        f"maps {{{', '.join(f'{k}: ({m[chr(97)]:.3f}, {m[chr(98)][1] - m[chr(98)][0]:+.3f})' for k, m in cal['maps'].items())}}}")
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


# ----------------------------------------------------------------------------- aggregate

def auc(d, y):
    r = rankdata(d)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(proto):
    n = len(RAW)
    R = {"d": np.full(n, np.nan), "comp": np.full((n, 4), np.nan),
         **{f"{c}{k}": np.full(n, np.nan) for c in "ab" for k in (1, 2, 3)}}
    for fp_ in sorted(RES.glob(f"{PREFIX}*.json")):
        r = json.loads(fp_.read_text())
        if r["protocol"] != proto:
            continue
        te = np.array(r["test"], dtype=int)
        R["d"][te], R["comp"][te] = r["d"], r["comp"]
        for k in (1, 2, 3):
            R[f"a{k}"][te], R[f"b{k}"][te] = r["a"][str(k)], r["b"][str(k)]
    return R


def retry_sim(R, c, clip, shrink, thr, rng, n=40000, m=3):
    """True label c; m responses per try from m different environments, earlier ones kept, at most 3 tries."""
    cl = [np.flatnonzero((ENV == e) & (Y == c)) for e in ENVS]
    tries = min(3, len(cl) // m)
    M = np.array(cl)  # 6 rows per environment and label
    perm = np.argsort(rng.random((n, len(cl))), axis=1)[:, :m * tries]
    rows = M[perm, rng.integers(0, M.shape[1], perm.shape)]
    d = R["d"][rows]
    d = np.clip(d, -clip, clip) if clip else d
    th = np.log(thr / (1 - thr))
    state, when = np.zeros(n, int), np.full(n, tries)
    first = None
    for t in range(1, tries + 1):
        s = slice(0, m * t)
        L = shrink * (R["a3"][rows[:, s]].mean(1) * d[:, s].sum(1) + R["b3"][rows[:, s]].mean(1))
        L = L if c == 1 else -L
        new = (state == 0) & (np.abs(L) >= th)
        state[new & (L > 0)], state[new & (L < 0)] = 1, -1
        when[new] = t
        if t == 1:
            first = (np.mean(np.abs(L) >= th), np.sum(L >= th) / max(np.sum(np.abs(L) >= th), 1), np.mean(L > 0))
    return first + (np.mean(state == -1), np.mean(state == 0), when.mean())


def aggregate():
    files = sorted(RES.glob(f"{PREFIX}*.json"))
    assert len(files) == len(fold_list()), f"{len(files)} fold files"
    log(f"pilot rows {len(RAW)} (Sol {Y.sum()}), environments {len(ENVS)}, lengths "
        f"{min(map(len, SEQS))}-{max(map(len, SEQS))}")
    rng = np.random.default_rng(0)
    names = ("dense", "fork", "score", "resid")
    for proto in ("loeo", "split"):
        for w in WINDOWS:
            R = load(f"{proto}|{w}")
            ok = ~np.isnan(R["d"])
            a = auc(R["d"][ok], Y[ok])
            comps = " ".join(f"{nm} {auc(R['comp'][ok, i], Y[ok]):.3f}" for i, nm in enumerate(names))
            log(f"AUC {proto:5} {w:7} | combined {a:.3f} d' {np.sqrt(2) * norm.ppf(a):.2f} | {comps} | "
                f"k1 acc Astra {np.mean(R['d'][ok & (Y == 0)] < 0):.3f} Sol {np.mean(R['d'][ok & (Y == 1)] > 0):.3f}")
            for disp, (clip, shrink) in DISPLAYS.items():
                for thr in (THRESHOLDS[:1] if clip else THRESHOLDS):
                    out = [retry_sim(R, c, clip, shrink, thr, rng) for c in (0, 1)]
                    print(f"SEL {proto:5} {w:7} {disp:27} thr {thr:.2f} | " + " | ".join(
                        f"{lab}: k3 rec {o[2]:.3f} first decide {o[0]:.3f} acc {o[1]:.3f} final wrong {o[3]:.3f} "
                        f"undecided {o[4]:.3f} tries {o[5]:.2f}" for lab, o in zip(("Astra", "Sol"), out)), flush=True)
    log("done")


if __name__ == "__main__":
    if sys.argv[1] == "folds":
        run_folds(sys.argv[2])
    else:
        aggregate()
