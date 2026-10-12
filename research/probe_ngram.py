"""Pair + 17-class diagnostic for class-conditional n-gram likelihood scoring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.data import Panel  # noqa: E402
from research.ngram import class_models, scores  # noqa: E402


def run(p, train_mask, test_idx, classes_idx, **kw):
    seqs = [p.rows[i]["numbers"] for i in range(len(p.rows))]
    tr = np.flatnonzero(train_mask)
    models = class_models([seqs[i] for i in tr], p.y[tr], classes_idx, **kw)
    return scores(models, [seqs[i] for i in test_idx])


def main():
    p = Panel()
    a, b = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
    is_train = np.isin(p.role, ["train_original", "train_supplemental"])
    for kw in ({"order": 1}, {"order": 2, "discount": 0.75}, {"order": 2, "discount": 0.9}, {"order": 3, "discount": 0.8}):
        res = {}
        pair = np.isin(p.y, [a, b])
        # env-held-out on the pair (training rows only)
        correct = total = 0
        for e in sorted(set(p.env[is_train])):
            te = np.flatnonzero(is_train & pair & (p.env == e))
            s = run(p, is_train & pair & (p.env != e), te, [a, b], **kw)
            correct += (np.array([a, b])[s.argmax(1)] == p.y[te]).sum()
            total += len(te)
        res["envCV"] = correct / total
        for src, dst in (("train_original", "train_supplemental"), ("train_supplemental", "train_original")):
            te = np.flatnonzero(pair & (p.role == dst))
            s = run(p, pair & (p.role == src), te, [a, b], **kw)
            res[src[6:10] + "->" + dst[6:10]] = (np.array([a, b])[s.argmax(1)] == p.y[te]).mean()
        for o in ("old_outside", "fresh_outside"):
            te = np.flatnonzero(pair & (p.role == o))
            s = run(p, is_train & pair, te, [a, b], **kw)
            res[o] = (np.array([a, b])[s.argmax(1)] == p.y[te]).mean()
        print(kw, {k: round(v, 3) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
