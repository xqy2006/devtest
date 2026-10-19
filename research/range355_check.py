"""Run 29 (2026-10-19): the unused 2026-10-08 numeric-range pilot rows with range 1-355 (GPT-6 Astra 17, GPT-6.1 Sol
15; Chinese prompt, 4 environments; reasoning effort not recorded, possibly the CLI default).

1. Where they sit on GPT-6.1 Sol's reasoning-effort axes and on the model axis (as effort_probe.py, run 26). If they
   were not collected at low effort, both models' shifts show how GPT-6 Astra reacts to effort.
2. Scored as a fresh outside batch by (a) the deliverable v5 (binary, references = all low-effort training rows)
   and (b) the model x effort subclass mode (references = training rows + new Sol rows, lambda 48; Platt).
   Reported per label: recall of single responses, mean d, and recall of triples from three different
   environments.

Usage: python research/range355_check.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from sklearn.covariance import shrunk_covariance  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research.effort_sub as ES  # noqa: E402
import research.gated_fingerprint as G  # noqa: E402
from research.calib_pair import log  # noqa: E402
from research.effort_pair import BATCH, EFF, EFFORTS, ENV, ME, ROWS, TR, Y  # noqa: E402
from research.hard_pair import A, IA, IS, S, p  # noqa: E402

NEW = [r for r in (json.loads(line) for line in (ROOT / "data/range355_20261008/range355_pair.jsonl").read_text().splitlines()
                   if line)
       if r["source"] in (p.classes[A], p.classes[S]) and r["range_max"] == 355 and r.get("strict_valid", True)]
NY = np.array([int(r["source"] == p.classes[S]) for r in NEW])
NENV = np.array([r["condition_id"] for r in NEW])


def triples(score, lab):
    """Recall of all same-label triples from three different environments (score > 0 = Sol)."""
    idx = np.flatnonzero(NY == lab)
    hit = tot = 0
    for a, b, c in itertools.combinations(idx, 3):
        if len({NENV[a], NENV[b], NENV[c]}) == 3:
            s = score[a] + score[b] + score[c]
            hit += int((s > 0) == bool(lab))
            tot += 1
    return hit / max(tot, 1), tot


def probe():
    seqs_old = [list(map(int, ROWS[i]["numbers"])) for i in np.concatenate([TR, ME])]
    pi = G.fit_pl(seqs_old)
    Xo = G.resid_vectors(pi, seqs_old)
    Xn = G.resid_vectors(pi, [list(map(int, r["numbers"])) for r in NEW])
    ntr = len(TR)
    mu, sd = Xo[:ntr].mean(0), Xo[:ntr].std(0) + 1e-9
    Zo, Zn = (Xo - mu) / sd, (Xn - mu) / sd
    Ztr, Zme = Zo[:ntr], Zo[ntr:]
    eme, yme = EFF[ME], Y[ME]
    low, hx, xh = (Zme[(yme == 1) & (eme == "low")], Zme[(yme == 1) & np.isin(eme, ["high", "xhigh"])],
                   Zme[(yme == 1) & (eme == "xhigh")])
    axes = {"effort LDA": LDA(solver="lsqr", shrinkage=0.5).fit(np.vstack([low, hx]),
                                                                 np.r_[np.zeros(len(low)), np.ones(len(hx))]).coef_[0]}
    env_me = ENV[ME]
    lo_i = np.flatnonzero((yme == 1) & (eme == "low"))
    xh_i = np.flatnonzero((yme == 1) & (eme == "xhigh"))
    delta = np.mean([Zme[b] - Zme[a] for a in lo_i for b in xh_i if env_me[a] == env_me[b]], 0)
    ytr, btr = Y[TR], BATCH[TR]
    R = np.vstack([Ztr[(ytr == c) & (btr == b)] - Ztr[(ytr == c) & (btr == b)].mean(0)
                   for c in (0, 1) for b in sorted(set(btr))])
    axes["effort paired"] = np.linalg.solve(shrunk_covariance(R.T @ R / len(R), 0.5), delta)
    axes["model LDA"] = LDA(solver="lsqr", shrinkage=0.5).fit(Ztr, ytr).coef_[0]
    for name, w in axes.items():
        lo_, hi_ = (low @ w).mean(), (xh @ w).mean()
        sc = lambda Z: (Z @ w - lo_) / (hi_ - lo_)  # noqa: E731
        for lab, nm in ((0, "Astra"), (1, "Sol")):
            v = sc(Ztr[ytr == lab])
            print(f"R355PROJ {name:13} {nm:5} low-effort training  n {len(v):3} mean {v.mean():+.3f} se {v.std(ddof=1) / np.sqrt(len(v)):.3f}")
            v = sc(Zn[NY == lab])
            print(f"R355PROJ {name:13} {nm:5} range-355 pilot      n {len(v):3} mean {v.mean():+.3f} se {v.std(ddof=1) / np.sqrt(len(v)):.3f}")
        for e in EFFORTS:
            v = sc(Zme[(yme == 1) & (eme == e)])
            print(f"R355PROJ {name:13} Sol   new {e:6}           n {len(v):3} mean {v.mean():+.3f}")
    # does the pilot differ from the low-effort training rows of the same model? (LDA, AUC by leave-one-row-out
    # on the pilot side is too small; report the mean shift in the model-axis units instead)


def score():
    cl = G.fit_cluster([ROWS[i] for i in TR], p.classes, [A, S])
    d = np.array([G.local_logits(cl, r["numbers"])[IS] - G.local_logits(cl, r["numbers"])[IA] for r in NEW])
    cal = cl["combine"]["calibration"]
    m1, m3 = cal["maps"]["1"], cal["maps"]["3"]
    for lab, nm in ((0, "Astra"), (1, "Sol")):
        v = d[NY == lab]
        rec1 = np.mean(v > 0) if lab else np.mean(v < 0)
        r3, n3 = triples(d, lab)
        print(f"R355SCORE v5 binary (low refs)       {nm:5} n {len(v):2} mean d {v.mean():+.2f} | k1 recall {rec1:.3f} "
              f"| k3 recall {r3:.3f} ({n3} triples)", flush=True)
    log(f"v5 maps k1 {m1} k3 {m3}")
    G.SUB_MEAN_LAMBDA = 48.0
    ref = np.concatenate([TR, ME])
    cs = G.fit_cluster(ES.sub_rows(ref), ES.SUB, list(range(len(ES.SUB))), member_groups=ES.GROUPS)
    s = np.array([G.local_logits(cs, r["numbers"]) for r in NEW])
    sol = ES.SOLSUB
    m1 = cs["combine"]["calibration"]["maps"]["1"]
    loc = m1["a"] * s + np.asarray(m1["b"])
    L1 = logsumexp(loc[:, sol], axis=1) - logsumexp(loc[:, ~sol], axis=1)
    m3 = cs["combine"]["calibration"]["maps"]["3"]
    for lab, nm in ((0, "Astra"), (1, "Sol")):
        idx = np.flatnonzero(NY == lab)
        rec1 = np.mean(L1[idx] > 0) if lab else np.mean(L1[idx] < 0)
        hit = tot = 0
        for a, b, c in itertools.combinations(idx, 3):
            if len({NENV[a], NENV[b], NENV[c]}) == 3:
                lc = m3["a"] * (s[a] + s[b] + s[c]) + np.asarray(m3["b"])
                L = logsumexp(lc[sol]) - logsumexp(lc[~sol])
                hit += int((L > 0) == bool(lab))
                tot += 1
        best = [ES.SUB[k] for k in np.argmax(s[idx], axis=1)]
        top = {k: best.count(k) for k in sorted(set(best))}
        print(f"R355SCORE subclass lambda 48         {nm:5} n {len(idx):2} | k1 recall {rec1:.3f} | k3 recall "
              f"{hit / max(tot, 1):.3f} ({tot} triples) | top subclass {top}", flush=True)


if __name__ == "__main__":
    log(f"range-355 pilot rows: Astra {int(np.sum(NY == 0))}, Sol {int(np.sum(NY == 1))}, environments {sorted(set(NENV))}")
    probe()
    score()
    log("done")
