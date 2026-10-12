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
     The two are combined with two weights fit by conditional likelihood on inner
     leave-one-environment-out scores of the cluster's reference rows. All parameters are JSON.

Score (one response for k=1, or the query's own fixed responses for k=3):
  production analysis -> global probabilities; if the global top class lies in a cluster, the
  cluster's probability mass is redistributed by the summed local log-likelihoods of the query's
  own responses. Nothing else is read: no labels, environment, batch or other queries.

Adding a new model = append its reference rows and rebuild. Clusters are rediscovered.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))
sys.path.insert(0, str(ROOT))
import fingerprint as fp  # noqa: E402
from bank_builder import build_bank, fit_robust_artifacts  # noqa: E402
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
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Z, y)
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


def fit_cluster(rows, model_ids, members):
    """Distribution evidence + branch evidence + combination weights for one cluster."""
    from scipy.optimize import minimize
    sel = [r for r in rows if r["source"] in {model_ids[k] for k in members}]
    member_ids = [model_ids[k] for k in sorted(members)]
    y = np.array([member_ids.index(r["source"]) for r in sel])
    env = np.array([r["condition_id"] for r in sel])
    K = len(member_ids)
    D, B = np.zeros((len(sel), K)), np.zeros((len(sel), K))
    for e in sorted(set(env)):  # out-of-fold evidence for the combination weights
        tr = [r for r, keep in zip(sel, env != e) if keep]
        loc = fit_local(tr, model_ids, members)
        br = fit_branch([r["numbers"] for r in tr], [member_ids.index(r["source"]) for r in tr], K)
        for i in np.flatnonzero(env == e):
            D[i] = _dense_logits(loc, sel[i]["numbers"])
            B[i] = branch_scores(br, sel[i]["numbers"], K)

    def nll(w):
        s = w[0] * D + w[1] * B
        return -(s[np.arange(len(y)), y] - logsumexp(s, axis=1)).sum() + 1e-3 * (w ** 2).sum()
    w = minimize(nll, np.array([1.0, 1.0]), method="L-BFGS-B").x
    local = fit_local(sel, model_ids, members)
    assert local["members"] == member_ids
    local["branch"] = fit_branch([r["numbers"] for r in sel], y.tolist(), K)
    local["combine"] = {"w_distribution": float(w[0]), "w_branch": float(w[1]),
                        "fitted_on": "inner leave-one-environment-out scores of the cluster's reference rows"}
    return local


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
            loc = sum(local_logits(cluster, n) for n in valid)
            q = np.exp(loc - logsumexp(loc))
            final[idx] = prob[idx].sum() * q
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
