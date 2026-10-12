"""Confusable-cluster refinement on top of a global multi-family discriminant.

1. Global scorer over all classes.
2. Inner leave-one-environment-out on the training rows only gives an OOF confusion matrix.
3. Classes whose symmetric OOF confusion rate >= tau are linked; connected components with
   two or more classes become refinement clusters (discovered automatically; no class names).
4. Each cluster gets its own discriminant fit only on that cluster's training rows.
5. Inference: p(c) = P_global(cluster) * q_cluster(c | cluster).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax

from research.methods_multi import MultiFamily


def clusters_from_confusion(conf, tau):
    n = len(conf)
    rate = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    sym = (rate + rate.T) / 2
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if sym[i, j] >= tau:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


class SubsetLDA(MultiFamily):
    """MultiFamily restricted to a subset of class indices."""

    def fit_subset(self, p, train, subset):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        rows = train[np.isin(p.y[train], subset)]
        X = self.matrix(p, rows)
        sc = StandardScaler().fit(X)
        w = self._weights(p)
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(sc.transform(X) * w, p.y[rows])
        return {"sc": sc, "w": w, "m": m, "subset": list(m.classes_)}

    def score_subset(self, model, p, idx):
        Z = model["sc"].transform(self.matrix(p, idx)) * model["w"]
        logits = model["m"].decision_function(Z)
        if logits.ndim == 1:  # binary: sklearn returns log-odds of class 1
            logits = np.stack([np.zeros_like(logits), logits], axis=1)
        return logits - logsumexp(logits, axis=1, keepdims=True)


class Refined:
    def __init__(self, families, tau=0.1, local_families=None):
        self.globl = MultiFamily(families)
        self.local = SubsetLDA(local_families or families)
        self.tau = tau

    def fit(self, p, train):
        C = len(p.classes)
        conf = np.zeros((C, C))
        for e in sorted(set(p.env[train])):
            te = train[p.env[train] == e]
            tr = train[p.env[train] != e]
            m = self.globl.fit(p, tr)
            pred = self.globl.score(m, p, te).argmax(1)
            np.add.at(conf, (p.y[te], pred), 1)
        clusters = clusters_from_confusion(conf, self.tau)
        model = {"global": self.globl.fit(p, train), "clusters": []}
        for K in clusters:
            model["clusters"].append(self.local.fit_subset(p, train, K))
        model["oof_confusion"] = conf
        return model

    def score(self, model, p, idx):
        logp = self.globl.score(model["global"], p, idx)
        out = logp.copy()
        for cm in model["clusters"]:
            K = cm["subset"]
            mass = logsumexp(logp[:, K], axis=1)
            out[:, K] = mass[:, None] + self.local.score_subset(cm, p, idx)
        return out

    def score_group(self, model, p, members):
        flat = self.score(model, p, members.ravel()).reshape(len(members), members.shape[1], -1)
        return flat.sum(axis=1)


class GatedRefine:
    """Any global scorer + hard gating into automatically discovered confusable clusters.

    Clusters come from the global scorer's own inner leave-one-environment-out confusion on the
    training rows. Inside a cluster, the dedicated discriminant (summed log-likelihood over the
    query's own responses for k3) picks the class; elsewhere the global decision stands.
    """

    def __init__(self, global_method, local_families, tau=0.1, local_inner=None):
        self.globl = global_method
        self.local = SubsetLDA(local_families)
        self.tau = tau

    def fit(self, p, train):
        C = len(p.classes)
        conf = np.zeros((C, C))
        for e in sorted(set(p.env[train])):
            te = train[p.env[train] == e]
            tr = train[p.env[train] != e]
            m = self.globl.fit(p, tr)
            np.add.at(conf, (p.y[te], self.globl.score(m, p, te).argmax(1)), 1)
        clusters = clusters_from_confusion(conf, self.tau)
        return {"global": self.globl.fit(p, train),
                "clusters": [self.local.fit_subset(p, train, K) for K in clusters],
                "cluster_sets": clusters, "oof_confusion": conf}

    def _combine(self, g, locals_):
        # g: (n, C) global scores; locals_: list of (subset, (n, |K|) local log-probs)
        out = np.zeros_like(g)
        top = g.argmax(1)
        out[np.arange(len(g)), top] = 1.0
        for K, lp in locals_:
            inside = np.isin(top, K)
            out[inside] = 0.0
            out[np.flatnonzero(inside), np.asarray(K)[lp[inside].argmax(1)]] = 1.0
        # tie-free scores: one-hot decision plus tiny global term for ranking
        return out + 1e-6 * (g - g.max(1, keepdims=True))

    def score(self, model, p, idx):
        g = self.globl.score(model["global"], p, idx)
        locs = [(cm["subset"], self.local.score_subset(cm, p, idx)) for cm in model["clusters"]]
        return self._combine(g, locs)

    def score_group(self, model, p, members):
        flat = members.ravel()
        g = self.globl.score(model["global"], p, flat).reshape(len(members), members.shape[1], -1).mean(1)
        locs = [(cm["subset"], self.local.score_subset(cm, p, flat).reshape(len(members), members.shape[1], -1).sum(1))
                for cm in model["clusters"]]
        return self._combine(g, locs)
