"""Diagnostic: which single-response feature families separate the hardest confusable pair.

The pair is chosen from training data alone as the most-confused class pair of the
baseline; nothing here is used at inference. Selection metrics use training rows only;
the outside panels are printed as exposed development diagnostics.
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from research.data import Panel  # noqa: E402
from research.features import FAMILIES, extract  # noqa: E402


def fit_predict(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def main(pair=("gpt-6-astra", "gpt-6.1-sol")):
    p = Panel()
    a, b = (p.classes.index(c) for c in pair)
    sel = np.flatnonzero(np.isin(p.y, [a, b]))
    rows = [p.rows[i] for i in sel]
    F = extract(rows)
    y = (p.y[sel] == b).astype(int)
    env, role = p.env[sel], p.role[sel]
    tr = np.isin(role, ["train_original", "train_supplemental"])
    print(f"{'family':12s} dim  envCV  orig->supp supp->orig  old_out fresh_out")
    for fam in list(FAMILIES) + ["seq_all"]:
        if fam == "seq_all":
            X = np.hstack([F[f] for f in ("sdiff36", "sdiff_lag2", "moddiff36", "updown", "autocorr", "bintrans6", "absdiff", "scalar")])
        else:
            X = F[fam]
        for C in (0.01, 0.1):
            # environment-held-out CV within training rows
            correct = 0
            for e in sorted(set(env[tr])):
                te = tr & (env == e)
                trn = tr & (env != e)
                correct += (fit_predict(X[trn], y[trn], X[te], C) == y[te]).sum()
            env_acc = correct / tr.sum()
            cross = []
            for src, dst in (("train_original", "train_supplemental"), ("train_supplemental", "train_original")):
                trn, te = role == src, role == dst
                cross.append((fit_predict(X[trn], y[trn], X[te], C) == y[te]).mean())
            outs = []
            for o in ("old_outside", "fresh_outside"):
                te = role == o
                outs.append((fit_predict(X[tr], y[tr], X[te], C) == y[te]).mean())
            print(f"{fam:12s} {X.shape[1]:4d} C={C:<5} {env_acc:.3f}  {cross[0]:.3f}  {cross[1]:.3f}   {outs[0]:.3f}  {outs[1]:.3f}")


if __name__ == "__main__":
    main()
