"""Deliverable candidate: production scorer + automatic confusable-cluster refinement.

Build (reference rows only):
  1. Production bank (unchanged code: ``bank_builder.build_bank``).
  2. Inner leave-one-environment-out over the reference rows with the production robust scorer
     gives an out-of-fold confusion matrix.
  3. Classes whose symmetric OOF confusion rate >= TAU are linked; connected components of size
     >= 2 become clusters. No model names appear anywhere in this code.
  4. Each cluster gets two kinds of local evidence, fit only on that cluster's reference rows
     (numbers only; no count-vs-request, duplicate-count or text-layout features):
     a. distribution evidence: shrinkage LDA on dense number-sequence families after the
        production-style environment-nuisance projection;
     b. branch evidence: the members share one generator and differ in how often they take each
        branch at low-entropy decision points (mostly the opening path). A pooled hierarchical
        Dirichlet model of the next number given (previous two numbers) and (anchored prefix) is
        fit on all members; each member's branch frequencies are shrunk to the pooled ones, and
        only contexts seen at least BRANCH_SUPPORT times contribute log-ratios.
     c. score evidence (v4): a shared conditional log-linear next-number model is fit on all
        members' reference rows without labels (research/fisher_score.py). Each response is
        mapped to its score vector at that shared model, (1/T) sum_t [features of the chosen
        number - expected features given that exact history], for the value, last-digit
        transition and position x band groups; then the same standardize / nuisance projection
        / shrinkage LDA as (a). For near-identical members this is the locally most powerful
        statistic (Rao score test); for the value group it measures how early each number is
        chosen relative to its availability, which raw histograms (saturated: ~300 of 355
        values used) cannot see.
     d. residual evidence (v5): the first-occurrence order under a label-free Plackett-Luce model shared
        by the members, as availability-corrected residuals of value / last digit / tens digit at three
        time scales (96), plus time-weighted value, digit, hundreds and jump statistics (99); shrinkage
        LDA (0.5). Every response yields it; it was the evidence that stayed least overconfident across
        collection dates.
     The four are combined with weights fit by conditional likelihood on inner
     leave-one-environment-out scores of the cluster's reference rows (the shared models of (c) and (d)
     are label-free and fit once on the cluster's reference rows). All parameters are JSON.

Score (one response for k=1, or the query's own fixed responses for k=3):
  production analysis -> global probabilities; if the global top class lies in a cluster, the
  cluster's probability mass is redistributed by the query's own responses: each response's local
  logits clipped (LOCAL_CLIP), summed, a Platt map for k responses fitted on inner
  leave-one-environment-out groups, times LOCAL_SHRINK (v5; so a wrong attribution inside a cluster
  does not come with a high probability). Nothing else is read: no labels, environment, batch or
  other queries.

Adding a new model = append its reference rows and rebuild. Clusters are rediscovered.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))
sys.path.insert(0, str(ROOT))
import fingerprint as fp  # noqa: E402
from bank_builder import build_bank, fit_robust_artifacts  # noqa: E402
from research import fisher_score as FS  # noqa: E402
from research.seqfeatures import SEQ_FAMILIES  # noqa: E402

TAU = 0.1
# production's own families (value histogram, ordered blocks) plus the families carrying the
# label differences that are consistent across collection batches (digits, jumps, jump moments)
LOCAL_FAMILIES = ("value355", "ordered", "digits", "jump1", "moments")
NUISANCE_RANK = 2
BRANCH_LAMBDA = 5.0     # Dirichlet prior strength towards the parent / pooled distribution
BRANCH_PREFIX = 40      # anchored-prefix contexts are used for the first positions only
BRANCH_SUPPORT = 10     # a context carries evidence only if seen this often in the references
V = 355
SCORE_GROUPS = ("val", "dg", "pb")  # score-vector groups used by the score evidence
SCORE_LAMBDA = 1.0                  # L2 penalty of the shared next-number model
SCORE_MAXITER = 80                  # L-BFGS iterations of the shared next-number model
# Covariance of the two linear discriminants (distribution, score). "auto" (v4): sklearn shrinkage,
# which standardizes every column inside the Ledoit-Wolf estimate, so the family / group weights
# have no effect on the covariance. "weighted": Ledoit-Wolf towards a scaled identity in the
# family-weighted space, so large families are shrunk more. On the N=1000 pilot this reduced the
# GPT-6.1 Sol -> GPT-6 Astra bias across collection batches (research/REPORT.md 19.6-19.7).
LDA_COV = "auto"
LDA_COV_PARTS = ("dense", "score")
# Cluster probabilities (v5, research/calib_pair.py runs 18-19; user target 2026-10-19: when the attribution is
# wrong, the reported probability must not be high). Each response's centered local logits are clipped to
# +-LOCAL_CLIP / 2 (two members: the log-ratio to +-LOCAL_CLIP), the summed logits of k responses go through a
# Platt map fitted on inner leave-one-environment-out groups of k responses, then are multiplied by
# LOCAL_SHRINK. Without the clip and shrink, the reported probability was calibrated for references and
# queries of the same dates but too confident across collection dates (wrong model >= 0.9 in 30% of the
# misjudged triples when a whole batch was left out); with them, in every protocol (environment held out,
# batch held out, outside panels) the wrong model got >= 0.9 in at most 5% of misjudged triples and the 90%
# quantile stayed <= 0.85, single responses never above 0.8. Chosen on training rows only.
LOCAL_CLIP = 2.0
LOCAL_SHRINK = 0.65


def _lda(part):
    if LDA_COV == "weighted" and part in LDA_COV_PARTS:
        return LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=LedoitWolf())
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def response_features(numbers, families=LOCAL_FAMILIES):
    x = np.asarray(numbers, dtype=np.int64)
    return np.concatenate([SEQ_FAMILIES[f](x) for f in families])


def _family_weights(numbers, families):
    x = np.asarray(numbers, dtype=np.int64)
    return np.concatenate([np.full(len(SEQ_FAMILIES[f](x)), 1.0 / math.sqrt(len(SEQ_FAMILIES[f](x)))) for f in families])


def _prepare(rows):
    out = []
    for row in rows:
        numbers = row.get("numbers")
        if numbers is None or isinstance(numbers, str):
            numbers = fp.parse_numbers(row["text"])
        out.append({**row, "numbers": numbers, "counts": fp.count_numbers(numbers)})
    return out


def oof_confusion(rows, model_ids):
    index = {m: i for i, m in enumerate(model_ids)}
    conf = np.zeros((len(model_ids), len(model_ids)))
    envs = sorted({r["condition_id"] for r in rows})
    for env in envs:
        train = [r for r in rows if r["condition_id"] != env]
        test = [r for r in rows if r["condition_id"] == env]
        bank = {"robust": fit_robust_artifacts(train, model_ids)}
        for r in test:
            s = fp.robust_score_numbers(r["numbers"], bank)["fused"]
            conf[index[r["source"]], int(np.argmax(s))] += 1
    return conf


def discover_clusters(conf, tau=TAU):
    rate = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    sym = (rate + rate.T) / 2
    n = len(conf)
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


def fit_local(rows, model_ids, members, families=LOCAL_FAMILIES, nuisance_rank=NUISANCE_RANK):
    sel = [r for r in rows if r["source"] in {model_ids[k] for k in members}]
    X = np.stack([response_features(r["numbers"], families) for r in sel])
    y = np.array([model_ids.index(r["source"]) for r in sel])
    env = np.array([r["condition_id"] for r in sel])
    w = _family_weights(sel[0]["numbers"], families)
    sc = StandardScaler().fit(X)
    Z = sc.transform(X) * w
    # environment nuisance: top directions of within-class environment offsets (production-style)
    offsets = [Z[(y == c) & (env == e)].mean(0) - Z[y == c].mean(0)
               for c in sorted(set(y)) for e in sorted(set(env[y == c]))]
    basis = np.zeros((0, Z.shape[1]))
    if nuisance_rank and len(offsets) > 1:
        _, singular, right = np.linalg.svd(np.array(offsets), full_matrices=False)
        basis = right[:min(nuisance_rank, int(np.sum(singular > singular[0] * 1e-8)))]
        Z = Z - (Z @ basis.T) @ basis
    lda = _lda("dense").fit(Z, y)
    coef = np.atleast_2d(lda.coef_)
    intercept = np.atleast_1d(lda.intercept_)
    if len(lda.classes_) == 2:  # sklearn binary form: log-odds of the second class
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.array([0.0, intercept[0]])
    return {"members": [model_ids[k] for k in lda.classes_], "families": list(families),
            "feature_mean": sc.mean_.tolist(), "feature_scale": sc.scale_.tolist(), "family_weight": w.tolist(),
            "nuisance_basis": basis.tolist(), "coef": coef.tolist(), "intercept": intercept.tolist(),
            "training_rows": len(sel)}


def _branch_chain(x, t):
    """Contexts for position t, shallow to deep: root, previous number, previous two numbers,
    anchored prefix (opening phase only). Evidence is read at the last two depths."""
    return [("R",), ("U", x[t - 1]) if t >= 1 else None, ("B", x[t - 2], x[t - 1]) if t >= 2 else None,
            ("P",) + tuple(x[:t]) if t < BRANCH_PREFIX else None]


def _key(ctx):
    return " ".join(map(str, ctx))


def fit_branch(seqs, labels, n_classes):
    counts = {}
    for x, c in zip(seqs, labels):
        x = list(map(int, x))
        for t, v in enumerate(x):
            for ctx in _branch_chain(x, t):
                if ctx is not None:
                    counts.setdefault(_key(ctx), {}).setdefault(v, [0] * n_classes)[c] += 1
    kept = {}
    for k, d in counts.items():
        total = sum(sum(c) for c in d.values())
        if k[0] in "RU" or total >= BRANCH_SUPPORT:  # root/previous-number levels feed the pooled chain
            kept[k] = [[v, *c] for v, c in sorted(d.items())]
    return {"lambda": BRANCH_LAMBDA, "prefix_len": BRANCH_PREFIX, "min_support": BRANCH_SUPPORT,
            "counts": kept}


_INDEX_CACHE: dict = {}  # id(branch) -> (branch, index); bounded, so old banks can be freed


def _branch_index(branch):
    """Lookup key -> (value -> row, class-count matrix, class totals), cached outside the JSON bank."""
    hit = _INDEX_CACHE.get(id(branch))
    if hit is None or hit[0] is not branch:
        idx = {}
        for k, lst in branch["counts"].items():
            arr = np.array(lst, dtype=np.int64)
            idx[k] = ({int(v): j for j, v in enumerate(arr[:, 0])}, arr[:, 1:], arr[:, 1:].sum(0))
        if len(_INDEX_CACHE) >= 8:
            _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
        hit = _INDEX_CACHE[id(branch)] = (branch, idx)
    return hit[1]


def branch_scores(branch, numbers, n_classes):
    """Per member: summed log P_member - log P_pooled over well-supported decision points."""
    idx = _branch_index(branch)
    lam, support = branch["lambda"], branch["min_support"]
    x = list(map(int, numbers))
    out = np.zeros(n_classes)
    zero = np.zeros(n_classes)
    for t, v in enumerate(x):
        p0 = 1.0 / V
        for depth, ctx in enumerate(_branch_chain(x, t)):
            if ctx is None:
                continue
            entry = idx.get(_key(ctx))
            if entry is None:
                continue
            row = entry[0].get(v)
            nv = entry[1][row] if row is not None else zero
            tot = entry[2]
            p0 = (nv.sum() + lam * p0) / (tot.sum() + lam)
            if depth >= 2 and tot.sum() >= support:
                out += np.log((nv + lam * p0) / (tot + lam)) - np.log(p0)
    return out


def _dense_logits(local, numbers):
    z = (response_features(numbers, local["families"]) - np.asarray(local["feature_mean"])) \
        / np.asarray(local["feature_scale"]) * np.asarray(local["family_weight"])
    basis = np.asarray(local["nuisance_basis"]).reshape(-1, len(z))
    z = z - (z @ basis.T) @ basis
    s = np.asarray(local["coef"]) @ z + np.asarray(local["intercept"])
    return s - logsumexp(s)


def _score_columns(groups):
    return np.concatenate([np.arange(FS.OFF[g], FS.OFF[g] + FS.GROUPS[g]) for g in groups])


def score_vectors(theta, seqs, groups=SCORE_GROUPS):
    """Score vectors (rows x selected dims) of responses at the shared model theta."""
    U = FS.score_vectors(FS.Steps(seqs), np.asarray(theta))
    return U[:, _score_columns(groups)]


def fit_score_local(U, y, env, groups=SCORE_GROUPS, nuisance_rank=NUISANCE_RANK):
    """Standardize -> group weights -> environment-offset projection -> shrinkage LDA on score vectors."""
    w = np.concatenate([np.full(FS.GROUPS[g], 1 / np.sqrt(FS.GROUPS[g])) for g in groups])
    sc = StandardScaler().fit(U)
    Z = sc.transform(U) * w
    offsets = [Z[(y == c) & (env == e)].mean(0) - Z[y == c].mean(0)
               for c in sorted(set(y)) for e in sorted(set(env[y == c]))]
    basis = np.zeros((0, Z.shape[1]))
    if nuisance_rank and len(offsets) > 1:
        _, singular, right = np.linalg.svd(np.array(offsets), full_matrices=False)
        basis = right[:min(nuisance_rank, int(np.sum(singular > singular[0] * 1e-8)))]
        Z = Z - (Z @ basis.T) @ basis
    lda = _lda("score").fit(Z, y)
    coef = np.atleast_2d(lda.coef_)
    intercept = np.atleast_1d(lda.intercept_)
    if len(lda.classes_) == 2:
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.array([0.0, intercept[0]])
    return {"groups": list(groups), "feature_mean": sc.mean_.tolist(), "feature_scale": sc.scale_.tolist(),
            "group_weight": w.tolist(), "nuisance_basis": basis.tolist(), "coef": coef.tolist(),
            "intercept": intercept.tolist()}


def _score_logits_from_vector(loc, u):
    z = (u - np.asarray(loc["feature_mean"])) / np.asarray(loc["feature_scale"]) * np.asarray(loc["group_weight"])
    basis = np.asarray(loc["nuisance_basis"]).reshape(-1, len(z))
    z = z - (z @ basis.T) @ basis
    s = np.asarray(loc["coef"]) @ z + np.asarray(loc["intercept"])
    return s - logsumexp(s)


def _score_logits(score, numbers):
    u = score_vectors(score["theta"], [numbers], score["local"]["groups"])[0]
    return _score_logits_from_vector(score["local"], u)


# (d) residual evidence (v5, research/calib_pair.py run 18; proposed by another AI): a 195-dim vector every
# response yields.
#   96: the first-occurrence order under a label-free Plackett-Luce model shared by the members (weight pi_v per
#       value, fit on the cluster's references); at each step the not-yet-used values give a working
#       distribution; the chosen number's features (value Legendre 1-12, last digit, tens digit) minus their
#       expectation, accumulated with time scales 16 / 64 / whole response and divided by the root of the
#       accumulated variance. It measures how a response departs from the shared preference once the numbers
#       already used are taken into account.
#   99: value Legendre 1-8, Fourier coordinates of the last and tens digit, hundreds band, |jump| powers 1-4,
#       normalized exponential averages with time scales 16 / 64 / whole response.
# Standardize, shrinkage LDA (0.5).
RESID_LAMBDA = 20.0   # Gamma(lambda + 1, lambda) smoothing of the Plackett-Luce weights
RESID_ITERS = 40      # MM iterations
RESID_SHRINK = 0.5    # LDA shrinkage
_a = 2 * (np.arange(1, V + 1) - 0.5) / V - 1
_PHI = np.hstack([np.polynomial.legendre.legvander(_a, 12)[:, 1:], np.eye(10)[np.arange(1, V + 1) % 10],
                  np.eye(10)[(np.arange(1, V + 1) // 10) % 10]])  # 355 x 32


def _first_order(numbers):
    return np.array([v - 1 for v in dict.fromkeys(int(t) for t in numbers) if 1 <= v <= V], dtype=int)


def fit_pl(seqs, lam=RESID_LAMBDA, iters=RESID_ITERS):
    """Shared Plackett-Luce weights of the first-occurrence orders (MM updates, mean 1)."""
    Z = [_first_order(x) for x in seqs]
    c = np.bincount(np.concatenate(Z), minlength=V).astype(float)
    pi = np.ones(V)
    for _ in range(iters):
        D = np.zeros(V)
        tot = pi.sum()
        for z in Z:
            w = pi[z]
            cum = np.cumsum(1.0 / (tot - (np.cumsum(w) - w)))
            contrib = np.full(V, cum[-1])
            contrib[z] = cum
            D += contrib
        pi = (c + lam) / (D + lam)
        pi /= pi.mean()
    return pi


def pl_residual(numbers, pi):
    z = _first_order(numbers)
    w = pi[z]
    P1, P2 = w[:, None] * _PHI[z], w[:, None] * _PHI[z] ** 2
    S0 = pi.sum() - (np.cumsum(w) - w)
    E = (_PHI.T @ pi - (np.cumsum(P1, 0) - P1)) / S0[:, None]
    Var = np.maximum(((_PHI ** 2).T @ pi - (np.cumsum(P2, 0) - P2)) / S0[:, None] - E ** 2, 0.0)
    r = _PHI[z] - E
    t = np.arange(len(z))
    out = []
    for tau in (16.0, 64.0, 1e10):
        g = np.exp(-t / tau)
        out.append((g @ r) / np.sqrt(0.1 + (g ** 2) @ Var))
    return np.concatenate(out)


def _fourier(d):
    return np.hstack([np.cos(2 * np.pi * np.outer(d, np.arange(1, 6)) / 10),
                      np.sin(2 * np.pi * np.outer(d, np.arange(1, 5)) / 10)])


def time_stats(numbers):
    x = np.clip(np.asarray(numbers, dtype=float), 1, V)
    xi = x.astype(int)
    M = np.hstack([np.polynomial.legendre.legvander(2 * (x - 1) / 354 - 1, 8)[:, 1:], _fourier(xi % 10),
                   _fourier((xi // 10) % 10), np.polynomial.legendre.legvander(2 * np.floor(x / 100) / 3 - 1, 3)[:, 1:]])
    Jp = np.column_stack([(np.abs(np.diff(x)) / V) ** k for k in range(1, 5)])
    out = []
    for tau in (16.0, 64.0, 1e9):
        g, gj = np.exp(-np.arange(len(x)) / tau), np.exp(-np.arange(1, len(x)) / tau)
        out += [g @ M / g.sum(), gj @ Jp / gj.sum()]
    return np.concatenate(out)


def resid_vectors(pi, seqs):
    return np.array([np.concatenate([pl_residual(x, pi), time_stats(x)]) for x in seqs])


def fit_resid_local(X, y, shrink=RESID_SHRINK):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrink).fit((X - mu) / sd, y)
    coef = np.atleast_2d(lda.coef_)
    intercept = np.atleast_1d(lda.intercept_)
    if len(lda.classes_) == 2:
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.array([0.0, intercept[0]])
    return {"feature_mean": mu.tolist(), "feature_scale": sd.tolist(), "coef": coef.tolist(),
            "intercept": intercept.tolist()}


def _resid_logits_from_vector(loc, u):
    z = (u - np.asarray(loc["feature_mean"])) / np.asarray(loc["feature_scale"])
    s = np.asarray(loc["coef"]) @ z + np.asarray(loc["intercept"])
    return s - logsumexp(s)


def _resid_logits(resid, numbers):
    return _resid_logits_from_vector(resid["local"], resid_vectors(np.asarray(resid["pi"]), [numbers])[0])


def fit_cluster(rows, model_ids, members):
    """Distribution, branch, score and residual evidence + combination weights for one cluster."""
    from scipy.optimize import minimize
    sel = [r for r in rows if r["source"] in {model_ids[k] for k in members}]
    member_ids = [model_ids[k] for k in sorted(members)]
    y = np.array([member_ids.index(r["source"]) for r in sel])
    env = np.array([r["condition_id"] for r in sel])
    K = len(member_ids)
    # shared next-number model: label-free, fit once on all of the cluster's reference rows
    seqs = [list(map(int, r["numbers"])) for r in sel]
    steps = FS.Steps(seqs)
    theta = FS.fit_base(steps, lam=SCORE_LAMBDA, maxiter=SCORE_MAXITER)
    U = FS.score_vectors(steps, theta)[:, _score_columns(SCORE_GROUPS)]
    del steps
    # shared Plackett-Luce weights: label-free, fit once on all of the cluster's reference rows
    pi = fit_pl(seqs)
    X = resid_vectors(pi, seqs)
    parts = np.zeros((4, len(sel), K))  # distribution, branch, score, residual
    for e in sorted(set(env)):  # out-of-fold evidence for the combination weights
        keep = env != e
        tr = [r for r, k in zip(sel, keep) if k]
        loc = fit_local(tr, model_ids, members)
        br = fit_branch([r["numbers"] for r in tr], [member_ids.index(r["source"]) for r in tr], K)
        sloc = fit_score_local(U[keep], y[keep], env[keep])
        rloc = fit_resid_local(X[keep], y[keep])
        for i in np.flatnonzero(env == e):
            parts[:, i] = [_dense_logits(loc, sel[i]["numbers"]), branch_scores(br, sel[i]["numbers"], K),
                           _score_logits_from_vector(sloc, U[i]), _resid_logits_from_vector(rloc, X[i])]

    def nll(w):
        s = np.tensordot(w, parts, 1)
        return -(s[np.arange(len(y)), y] - logsumexp(s, axis=1)).sum() + 1e-3 * (w ** 2).sum()
    w = minimize(nll, np.ones(len(parts)), method="L-BFGS-B").x
    S = np.tensordot(w, parts, 1)
    S = S - logsumexp(S, axis=1, keepdims=True)  # inner out-of-fold local logits, as local_logits returns them
    maps = {str(k): _fit_group_map(S, y, env, k) for k in (1, 2, 3)}
    local = fit_local(sel, model_ids, members)
    assert local["members"] == member_ids
    local["branch"] = fit_branch([r["numbers"] for r in sel], y.tolist(), K)
    local["score"] = {"theta": theta.tolist(), "local": fit_score_local(U, y, env)}
    local["resid"] = {"pi": pi.tolist(), "local": fit_resid_local(X, y)}
    local["combine"] = {"w_distribution": float(w[0]), "w_branch": float(w[1]), "w_score": float(w[2]),
                        "w_resid": float(w[3]),
                        "fitted_on": "inner leave-one-environment-out scores of the cluster's reference rows",
                        "calibration": {"clip": LOCAL_CLIP, "shrink": LOCAL_SHRINK, "maps": maps,
                                        "fitted_on": "inner leave-one-environment-out groups of k responses "
                                                     "from k different environments"}}
    return local


def _fit_group_map(S, y, env, k, n=30000, seed=0):
    """Platt map for k responses: softmax(a * summed logits + b), b[0] = 0, fitted with equal weight per member on
    random same-member groups of k rows from k different environments."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    K = S.shape[1]
    G, lab = [], []
    for c in range(K):
        rows = np.flatnonzero(y == c)
        if k > 1 and len(set(env[rows])) < k:
            continue
        g = rows[:, None] if k == 1 else rows[rng.integers(0, len(rows), size=(4 * n, k))]
        ok = np.ones(len(g), bool)
        for i in range(k):
            for j in range(i + 1, k):
                ok &= env[g[:, i]] != env[g[:, j]]
        g = g[ok][:n]
        G.append(S[g].sum(1))
        lab.append(np.full(len(g), c))
    G, lab = np.vstack(G), np.concatenate(lab)
    wt = (1.0 / np.bincount(lab, minlength=K))[lab]

    def nll(p):
        z = p[0] * G + np.concatenate([[0.0], p[1:]])
        return -(wt * (z[np.arange(len(lab)), lab] - logsumexp(z, axis=1))).sum() / wt.sum()
    p = minimize(nll, np.concatenate([[1.0], np.zeros(K - 1)]), method="L-BFGS-B").x
    return {"a": float(p[0]), "b": [0.0] + [float(v) for v in p[1:]]}


def cluster_probabilities(cluster, responses):
    """Member probabilities from a query's own responses (numbers lists). Banks without a calibration block
    (v4 and earlier) keep the plain softmax of the summed local logits."""
    K = len(cluster["members"])
    if not responses:
        return np.full(K, 1.0 / K)
    s = [local_logits(cluster, n) for n in responses]
    cal = cluster.get("combine", {}).get("calibration")
    if cal is None:
        loc = sum(s)
    else:
        h = cal["clip"] / 2
        m = cal["maps"][str(min(len(responses), 3))]
        loc = cal["shrink"] * (m["a"] * sum(np.clip(v - v.mean(), -h, h) for v in s) + np.asarray(m["b"]))
    return np.exp(loc - logsumexp(loc))


def build_gated_bank(rows, tau=TAU, families=LOCAL_FAMILIES, calibrate=True):
    rows = _prepare(rows)
    model_ids = list(dict.fromkeys(r["source"] for r in rows))
    if calibrate:
        bank = build_bank(rows)
    else:  # research harness: calibration only rescales probabilities, not the decision
        bank = {"models": [{"id": m, "display_name": m, "counts": [0] * fp.DIMENSION} for m in model_ids],
                "robust": fit_robust_artifacts(rows, model_ids),
                "calibration": {str(k): {"beta": 1.0, "cv_accuracy": None} for k in (1, 2, 3)}}
    conf = oof_confusion(rows, model_ids)
    clusters = discover_clusters(conf, tau)
    bank["refinement"] = {
        "tau": tau,
        "oof_confusion": conf.tolist(),
        "clusters": [fit_cluster(rows, model_ids, K) for K in clusters],
        "rule": "if the global top class is in a cluster, redistribute that cluster's probability mass by "
                "summed local log-likelihoods of the query's own responses",
    }
    return bank


def local_logits(cluster, numbers):
    w = cluster.get("combine", {"w_distribution": 1.0, "w_branch": 0.0})
    s = w["w_distribution"] * _dense_logits(cluster, numbers)
    if w["w_branch"]:
        s = s + w["w_branch"] * branch_scores(cluster["branch"], numbers, len(cluster["members"]))
    if w.get("w_score"):
        s = s + w["w_score"] * _score_logits(cluster["score"], numbers)
    if w.get("w_resid"):
        s = s + w["w_resid"] * _resid_logits(cluster["resid"], numbers)
    return s - logsumexp(s)


def analyze(outputs, bank):
    """outputs: list of {"text", "expected_count"}; same validity rule as production."""
    base = fp.analyze_outputs(outputs, bank)
    model_ids = [m["id"] for m in bank["models"]]
    prob = np.zeros(len(model_ids))
    for item in base["results"]:
        prob[model_ids.index(item["model"])] = item["probability"]
    valid = []
    for out in outputs:
        numbers = fp.parse_numbers(str(out.get("text", "")))
        expected = int(out.get("expected_count") or 0)
        minimum = max(80, math.ceil(expected * 0.55)) if expected else 80
        if len(numbers) >= minimum:
            valid.append(numbers)
    final = prob.copy()
    used_cluster = None
    top = int(np.argmax(prob))
    for cluster in bank.get("refinement", {}).get("clusters", []):
        idx = [model_ids.index(m) for m in cluster["members"]]
        if top in idx:
            final[idx] = prob[idx].sum() * cluster_probabilities(cluster, valid)
            used_cluster = cluster["members"]
    order = np.argsort(-final)
    return {"prediction": model_ids[order[0]], "probability": float(final[order[0]]),
            "probabilities": {model_ids[i]: float(final[i]) for i in order},
            "global_prediction": base["prediction"], "refinement_cluster": used_cluster,
            "used_outputs": base["used_outputs"]}


class GatedDeliverable:
    """Harness wrapper that runs exactly the deliverable builder/scorer."""

    def __init__(self, calibrate=False):
        self.calibrate = calibrate

    def fit(self, p, train):
        return build_gated_bank([p.rows[i] for i in train], calibrate=self.calibrate)

    def _score_outputs(self, bank, p, members):
        out = []
        order = [m["id"] for m in bank["models"]]
        for group in members:
            res = analyze([{"text": p.rows[i]["text"], "expected_count": p.rows[i]["requested_count"]} for i in group], bank)
            v = np.full(len(p.classes), -1e9)
            for m, pr in res["probabilities"].items():
                v[p.classes.index(m)] = math.log(max(pr, 1e-300))
            out.append(v)
        return np.array(out)

    def score(self, model, p, idx):
        return self._score_outputs(model, p, [[i] for i in idx])

    def score_group(self, model, p, members):
        return self._score_outputs(model, p, members)
