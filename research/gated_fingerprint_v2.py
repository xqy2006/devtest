"""Version 2 of the deliverable (dense numbers-only local LDA; kept to reproduce REPORT section 12).

Deliverable candidate: production scorer + automatic confusable-cluster refinement.

Build (reference rows only):
  1. Production bank (unchanged code: ``bank_builder.build_bank``).
  2. Inner leave-one-environment-out over the reference rows with the production robust scorer
     gives an out-of-fold confusion matrix.
  3. Classes whose symmetric OOF confusion rate >= TAU are linked; connected components of size
     >= 2 become clusters. No model names appear anywhere in this code.
  4. Each cluster gets a shrinkage LDA on dense number-sequence families (numbers only: every emitted
     number contributes; no count-vs-request, duplicate-count or text-layout features), after the
     production-style environment-nuisance projection, fit only on that cluster's reference rows.
     Parameters are stored as plain JSON.

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
        "clusters": [fit_local(rows, model_ids, K, families) for K in clusters],
        "rule": "if the global top class is in a cluster, redistribute that cluster's probability mass by "
                "summed local log-likelihoods of the query's own responses",
    }
    return bank


def local_logits(cluster, numbers):
    z = (response_features(numbers, cluster["families"]) - np.asarray(cluster["feature_mean"])) \
        / np.asarray(cluster["feature_scale"]) * np.asarray(cluster["family_weight"])
    basis = np.asarray(cluster["nuisance_basis"]).reshape(-1, len(z))
    z = z - (z @ basis.T) @ basis
    s = np.asarray(cluster["coef"]) @ z + np.asarray(cluster["intercept"])
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
