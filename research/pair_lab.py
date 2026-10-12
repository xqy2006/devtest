"""Do higher-capacity single-response models pay off now that the pair has ~3x references?

Pair-only (the automatically discovered cluster). Protocols, all fitted on training rows only:
  env   leave-one-environment-out over the three training batches pooled (427 rows)
  batch leave-one-training-batch-out (each batch scored by a library built from the other two)
  out   full training library -> exposed old/fresh outside panels (development only)
Every method is generic (binary or multi-class within a cluster); labels are used only for fitting
on training rows and for scoring.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import PAIR_ROLE, extended_panel  # noqa: E402
from research.features import FAMILIES, extract  # noqa: E402

p = extended_panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
pair = np.flatnonzero(np.isin(p.y, [A, S]))
yb = (p.y == S).astype(int)
TRB = ("train_original", "train_supplemental", PAIR_ROLE)
fams = [f for f in FAMILIES if f not in ("format", "position")]
F = extract([p.rows[i] for i in pair], fams)
FX = {f: np.zeros((len(p.rows), F[f].shape[1])) for f in fams}
for f in fams:
    FX[f][pair] = F[f]
SEQ = [None] * len(p.rows)
for i in pair:
    SEQ[i] = np.asarray(p.rows[i]["numbers"], dtype=np.int64)


def stack(families):
    X = np.hstack([FX[f] for f in families])
    w = np.concatenate([np.full(FX[f].shape[1], 1 / np.sqrt(FX[f].shape[1])) for f in families])
    return X, w


class LinearFam:
    """Standardize on reference rows, weight each family by 1/sqrt(dim), shrinkage LDA or L2 logit."""

    def __init__(self, families, kind="lda", C=1.0):
        self.families, self.kind, self.C = families, kind, C

    def fit(self, ref):
        self.X, self.w = stack(self.families)
        self.sc = StandardScaler().fit(self.X[ref])
        Z = self.sc.transform(self.X[ref]) * self.w
        if self.kind == "lda":
            self.m = LDA(solver="lsqr", shrinkage="auto").fit(Z, yb[ref])
        else:
            self.m = LogisticRegression(C=self.C, max_iter=3000).fit(Z, yb[ref])
        return self

    def decision(self, idx):
        return self.m.decision_function(self.sc.transform(self.X[idx]) * self.w)


class GBM:
    def __init__(self, families):
        self.families = families

    def fit(self, ref):
        self.X = np.hstack([FX[f] for f in self.families])
        self.m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                l2_regularization=1.0, random_state=0).fit(self.X[ref], yb[ref])
        return self

    def decision(self, idx):
        q = np.clip(self.m.predict_proba(self.X[idx])[:, 1], 1e-6, 1 - 1e-6)
        return np.log(q / (1 - q))


class ValueNB:
    """Per-value presence log-ratio (Bernoulli NB with shrinkage) + count of repeats."""

    def __init__(self, alpha=2.0):
        self.alpha = alpha

    def fit(self, ref):
        pres = {c: np.zeros(356) for c in (0, 1)}
        n = {0: 0, 1: 0}
        for i in ref:
            pres[yb[i]][np.unique(SEQ[i])] += 1
            n[yb[i]] += 1
        self.lr = {}
        rate = {c: (pres[c] + self.alpha * (pres[0] + pres[1]) / (n[0] + n[1])) / (n[c] + self.alpha) for c in (0, 1)}
        rate = {c: np.clip(v, 1e-3, 1 - 1e-3) for c, v in rate.items()}
        self.w1 = np.log(rate[1] / rate[0])
        self.w0 = np.log((1 - rate[1]) / (1 - rate[0]))
        self.b = np.log(n[1] / n[0])
        return self

    def decision(self, idx):
        out = []
        for i in idx:
            m = np.zeros(356, bool)
            m[np.unique(SEQ[i])] = True
            m[0] = False
            out.append(self.w1[m].sum() + self.w0[~m][1:].sum() + self.b)
        return np.array(out) / 10.0


class BigramLM:
    """Class-conditional first-order Markov model over value bins, shrunk to the pooled chain."""

    def __init__(self, bins=36, lam=50.0):
        self.bins, self.lam = bins, lam

    def _b(self, x):
        return np.minimum((x - 1) * self.bins // 355, self.bins - 1)

    def fit(self, ref):
        K = self.bins
        T = {c: np.zeros((K, K)) for c in (0, 1)}
        for i in ref:
            b = self._b(SEQ[i])
            np.add.at(T[yb[i]], (b[:-1], b[1:]), 1)
        pooled = (T[0] + T[1] + 1) / (T[0] + T[1] + 1).sum(1, keepdims=True)
        self.logP = {c: np.log((T[c] + self.lam * pooled) / (T[c].sum(1, keepdims=True) + self.lam)) for c in (0, 1)}
        return self

    def decision(self, idx):
        out = []
        for i in idx:
            b = self._b(SEQ[i])
            out.append(self.logP[1][b[:-1], b[1:]].sum() - self.logP[0][b[:-1], b[1:]].sum())
        return np.array(out) / 10.0


class Ensemble:
    def __init__(self, members):
        self.members = members

    def fit(self, ref):
        # each member's decision is standardized by its own training-row spread (no test statistics)
        self.fitted = [m.fit(ref) for m in self.members]
        self.scale = [np.std(m.decision(ref)) + 1e-9 for m in self.fitted]
        return self

    def decision(self, idx):
        return sum(m.decision(idx) / s for m, s in zip(self.fitted, self.scale))


def rows(roles):
    return np.flatnonzero(np.isin(p.y, [A, S]) & np.isin(p.role, roles))


def groups(roles):
    return p.group_members[np.isin(p.group_role, roles) & np.isin(p.group_y, [A, S])]


def score(make, ref, te, d):
    d[te] = make().fit(ref).decision(te)


def evaluate(make):
    res = {}
    tr = rows(TRB)
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[tr])):
        score(make, tr[p.env[tr] != e], tr[p.env[tr] == e], d)
    g = groups(TRB)
    res["env"] = (float(((d[tr] > 0) == yb[tr]).mean()), float(((d[g].sum(1) > 0) == yb[g[:, 0]]).mean()))
    d = np.zeros(len(p.rows))
    for b in TRB:
        score(make, rows(tuple(x for x in TRB if x != b)), rows((b,)), d)
    res["batch"] = (float(((d[tr] > 0) == yb[tr]).mean()), float(((d[g].sum(1) > 0) == yb[g[:, 0]]).mean()))
    res["batch_each"] = {b[-8:]: round(float(((d[rows((b,))] > 0) == yb[rows((b,))]).mean()), 3) for b in TRB}
    d = np.zeros(len(p.rows))
    te = rows(("old_outside", "fresh_outside"))
    score(make, tr, te, d)
    for role in ("old_outside", "fresh_outside"):
        t, gg = rows((role,)), groups((role,))
        res[role] = (float(((d[t] > 0) == yb[t]).mean()), float(((d[gg].sum(1) > 0) == yb[gg[:, 0]]).mean()))
    return res


FAMS = ["value355", "sdiff72", "scalar", "repeat", "digits"]
COMPACT = ["scalar", "repeat", "countshape", "digits", "sdiff36", "value36", "chunk10", "absdiff", "autocorr", "updown"]
METHODS = {
    "lda_fams": lambda: LinearFam(FAMS),
    "lda_all": lambda: LinearFam(fams),
    "lda_compact": lambda: LinearFam(COMPACT),
    "logit_fams_C0.1": lambda: LinearFam(FAMS, "logit", 0.1),
    "logit_fams_C0.01": lambda: LinearFam(FAMS, "logit", 0.01),
    "logit_all_C0.01": lambda: LinearFam(fams, "logit", 0.01),
    "gbm_compact": lambda: GBM(COMPACT),
    "value_nb": lambda: ValueNB(),
    "bigram36": lambda: BigramLM(36),
    "bigram12": lambda: BigramLM(12, 20.0),
    "ens_lda+nb+bigram": lambda: Ensemble([LinearFam(FAMS), ValueNB(), BigramLM(36)]),
    "ens_lda+gbm": lambda: Ensemble([LinearFam(FAMS), GBM(COMPACT)]),
}

def count_category(n_parsed, requested):
    over = n_parsed - requested
    return 0 if over < 0 else 1 if over == 0 else 2 if over < 8 else 3 if over <= 15 else 4


CAT = np.array([count_category(len(r["numbers"]), r["requested_count"]) for r in p.rows])


class CountNB:
    """Count-precision likelihood ratio (numbers only: parsed count vs requested count)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, ref):
        rates = [np.bincount(CAT[ref][yb[ref] == c], minlength=5) + self.alpha for c in (0, 1)]
        self.lr = np.log(rates[1] / rates[1].sum()) - np.log(rates[0] / rates[0].sum())
        return self

    def decision(self, idx):
        return self.lr[CAT[idx]]


class SumLR:
    """Sum of member log-likelihood ratios (content model + count model)."""

    def __init__(self, members, weights=None):
        self.members, self.weights = members, weights or [1.0] * len(members)

    def fit(self, ref):
        self.fitted = [m.fit(ref) for m in self.members]
        return self

    def decision(self, idx):
        return sum(w * m.decision(idx) for w, m in zip(self.weights, self.fitted))


class Stacked:
    """Member weights learned by logistic regression on inner leave-one-environment-out decisions
    of the reference rows only."""

    def __init__(self, makers):
        self.makers = makers

    def fit(self, ref):
        Z = np.zeros((len(ref), len(self.makers)))
        envs = p.env[ref]
        for e in sorted(set(envs)):
            tr, te = ref[envs != e], envs == e
            for k, mk in enumerate(self.makers):
                Z[te, k] = mk().fit(tr).decision(ref[te])
        self.lr = LogisticRegression(C=1.0).fit(Z, yb[ref])
        self.fitted = [mk().fit(ref) for mk in self.makers]
        return self

    def decision(self, idx):
        Z = np.column_stack([m.decision(idx) for m in self.fitted])
        return self.lr.decision_function(Z)


CONTENT = ["value355", "sdiff72", "scalar_noreq", "repeat", "digits"]
METHODS.update({
    "count_nb": lambda: CountNB(),
    "lda_content": lambda: LinearFam(CONTENT),
    "content+count_nb": lambda: SumLR([LinearFam(CONTENT), CountNB()]),
    "lda_fams+count_nb": lambda: SumLR([LinearFam(FAMS), CountNB()]),
    "stack_content_count": lambda: Stacked([lambda: LinearFam(CONTENT), lambda: CountNB()]),
})


class ExactNGram:
    """Class-conditional exact-value n-gram LM (research/ngram.py); decision = summed log-ratio / 10."""

    def __init__(self, order=3, discount=0.75):
        self.order, self.discount = order, discount

    def fit(self, ref):
        from research.ngram import NGramClass
        self.m = []
        for c in (0, 1):
            m = NGramClass(order=self.order, discount=self.discount)
            for i in ref:
                if yb[i] == c:
                    m.add(SEQ[i])
            self.m.append(m.finalize())
        return self

    def decision(self, idx):
        return np.array([(self.m[1].logprob_seq(SEQ[i]).sum() - self.m[0].logprob_seq(SEQ[i]).sum()) / 10.0 for i in idx])


METHODS.update({
    "ngram2": lambda: ExactNGram(2),
    "ngram3": lambda: ExactNGram(3),
    "ngram4": lambda: ExactNGram(4),
    "lda_fams+ngram3": lambda: Ensemble([LinearFam(FAMS), ExactNGram(3)]),
})


if __name__ == "__main__":
    out = {}
    only = sys.argv[1:] or list(METHODS)
    for name in only:
        t = time.time()
        r = evaluate(METHODS[name])
        out[name] = r
        print(f"{name:20} env k1 {r['env'][0]:.3f} k3 {r['env'][1]:.3f} | batch k1 {r['batch'][0]:.3f} k3 {r['batch'][1]:.3f} "
              f"{r['batch_each']} | old {r['old_outside'][0]:.3f}/{r['old_outside'][1]:.3f} "
              f"fresh {r['fresh_outside'][0]:.3f}/{r['fresh_outside'][1]:.3f} ({time.time()-t:.0f}s)", flush=True)
    Path(__file__).with_name("out_pair_lab.json").write_text(json.dumps(out, indent=1))
