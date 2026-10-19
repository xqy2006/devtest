"""Evaluation harness: tell two models apart from the numbers they write.

Usage
  python eval_harness.py                                   # GPT-6 Astra vs GPT-6.1 Sol, example baseline
  python eval_harness.py gpt-6-sol gpt-6.1-sol             # any other pair (all other pairs separate easily)

Plug in your own method: write fit(pair_refs, other_refs) -> score, where score(rows) returns one number per
row (log-odds, positive = the second model of the pair), and call evaluate(fit, rows, model_a, model_b).

Protocol (the same one used for every number in PROMPT.md)
  - Only the numbers of a response may be used (the end user pastes the reply; no prompt, no metadata).
  - Fitting uses reference rows only. "environment" is used only to form folds, never as a feature.
  - train: leave one environment out over the 12 environments (the held-out environment's rows of every model
    are removed from the references).
  - outside (only GPT-6 Astra / GPT-6.1 Sol): scored by the model fitted on all training rows; these were
    collected on other dates. Do not tune on them.
  - Report each model separately; the headline is the worse model.
  - k1: one response, decision = sign of the score. k3: three responses of the same model from three different
    environments (all such combinations), decision = sign of the sum of the three scores.
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

HERE = Path(__file__).resolve().parent


def load(path=HERE / "data" / "sequences.csv"):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["numbers"] = [int(v) for v in r["numbers"].split()]
    return rows


def auc(d, y):
    order = np.argsort(d, kind="mergesort")
    ranks = np.empty(len(d))
    ranks[order] = np.arange(1, len(d) + 1)
    for v in np.unique(d):  # average ranks for ties
        m = d == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def k3_all(d, y, env, label):
    """Share of correct decisions over all same-label triples whose rows come from three different environments."""
    cells = [d[(y == label) & (env == e)] for e in sorted(set(env[y == label]))]
    sign = 1 if label == 1 else -1
    correct = total = 0
    for a, b, c in itertools.combinations(range(len(cells)), 3):
        s = cells[a][:, None, None] + cells[b][None, :, None] + cells[c][None, None, :]
        correct += int(np.sum(sign * s > 0))
        total += s.size
    return correct / total if total else float("nan")


def metrics(d, rows, model_b):
    y = np.array([r["model"] == model_b for r in rows], dtype=int)
    env = np.array([r["environment"] for r in rows])
    ra, rb = float(np.mean(d[y == 0] <= 0)), float(np.mean(d[y == 1] > 0))
    ka, kb = k3_all(d, y, env, 0), k3_all(d, y, env, 1)
    return {"k1_a": ra, "k1_b": rb, "k1_worst": min(ra, rb), "k3_a": ka, "k3_b": kb, "k3_worst": min(ka, kb),
            "auc": auc(d, y)}


def evaluate(fit, rows, model_a="gpt-6-astra", model_b="gpt-6.1-sol"):
    pair = [r for r in rows if r["model"] in (model_a, model_b)]
    train = [r for r in pair if r["split"] == "train"]
    outside = [r for r in pair if r["split"] == "outside"]
    others = [r for r in rows if r["model"] not in (model_a, model_b) and r["split"] == "train"]
    d = np.zeros(len(train))
    for e in sorted(set(r["environment"] for r in train)):
        held = [i for i, r in enumerate(train) if r["environment"] == e]
        score = fit([r for r in train if r["environment"] != e], [r for r in others if r["environment"] != e])
        d[held] = score([train[i] for i in held])
    out = {"train (leave one environment out)": metrics(d, train, model_b)}
    if len(set(r["model"] for r in outside)) == 2:  # the outside panels only have Astra and 6.1 Sol
        out["outside (other dates)"] = metrics(fit(train, others)(outside), outside, model_b)
    return out


def report(res, model_a, model_b):
    a, b = model_a, model_b
    for tag, m in res.items():
        print(f"{tag:34} k1 {a} {m['k1_a']:.3f} {b} {m['k1_b']:.3f} worst {m['k1_worst']:.3f} | "
              f"k3 {m['k3_a']:.3f} / {m['k3_b']:.3f} worst {m['k3_worst']:.3f} | AUC {m['auc']:.3f}")


# ----------------------------------------------------------------------------- example baseline

def features(x):
    """A small set of dense statistics every response has: value histogram (36 bins), last digit, signed jump
    histogram, a few moments."""
    x = np.asarray(x, dtype=float)
    j = np.diff(x)
    h = np.histogram(x, bins=36, range=(0.5, 360.5))[0] / len(x)
    last = np.bincount(x.astype(int) % 10, minlength=10)[:10] / len(x)
    jh = np.histogram(j, bins=20, range=(-355, 355))[0] / len(j)
    mom = [x.mean() / 355, x.std() / 355, np.corrcoef(x[:-1], x[1:])[0, 1], np.abs(j).mean() / 355]
    return np.concatenate([h, last, jh, mom])


def baseline_fit(ref, others, model_b="gpt-6.1-sol", rank=2):
    """Standardize, remove the main within-model environment offsets (references only), shrinkage LDA."""
    X = np.array([features(r["numbers"]) for r in ref])
    y = np.array([r["model"] == model_b for r in ref], dtype=int)
    env = np.array([r["environment"] for r in ref])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    off = [Z[(y == c) & (env == e)].mean(0) - Z[y == c].mean(0) for c in (0, 1) for e in sorted(set(env[y == c]))]
    U = np.linalg.svd(np.array(off), full_matrices=False)[2][:rank]

    def proj(M):
        return M - (M @ U.T) @ U
    lda = LDA(solver="lsqr", shrinkage="auto").fit(proj(Z), y)

    def score(rows):
        Q = (np.array([features(r["numbers"]) for r in rows]) - mu) / sd
        return lda.decision_function(proj(Q))
    return score


if __name__ == "__main__":
    a, b = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("gpt-6-astra", "gpt-6.1-sol")
    rows = load()
    res = evaluate(lambda ref, oth: baseline_fit(ref, oth, model_b=b), rows, a, b)
    print(f"example baseline: {a} vs {b}")
    report(res, a, b)
