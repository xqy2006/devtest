"""Run 11: the worst case at the original length: GPT-6.1 Sol responses that the model calls GPT-6 Astra.

Question (2026-10-14): with the original responses (218-300 numbers) and at most three of them, is a detail
left that separates the Sol responses the model calls Astra from Astra, or does one Astra-typical part of
the evidence pull them over?

Data: the original data (extended_panel): Astra / Sol rows of the three training batches (427 rows, leave
one environment out) and the two outside panels (scored by the model fitted on all training batches).
Model: the deliverable v4 cluster model (gated_fingerprint.fit_cluster), unchanged. d = Sol logit - Astra
logit; "misjudged Sol" = Sol with d <= 0.

Experiments (one job each):
  oof     (a) where each row's evidence comes from: the three components (distribution, branch, score) with
              the model's weights, the distribution split by feature family, the score by group, the branch
              evidence by context (anchored opening prefix vs the previous two numbers). Means for correct /
              misjudged Astra and Sol; per-label recall and all-combination k3 with one part left out (same
              fit, no refit).
          (b) residual information in the worst case: among the rows the model calls Astra (d <= 0), is
              anything left that separates the misjudged Sol rows from Astra? For every held-out environment,
              a second-stage shrinkage LDA is fitted on the training environments' own d <= 0 rows (their d
              from the inner leave-one-environment-out inside fit_cluster, so the held-out environment is
              never used), and a logistic combination of d and that score on inner out-of-fold scores.
              Reported among the held-out d <= 0 rows (misjudged Sol vs Astra): AUC of d alone, of the
              second-stage score, of the combination, and an environment-bootstrap interval of the gain.
              Mirror: d > 0 (misjudged Astra vs Sol). Feature sets: readable statistics, new sequence
              statistics (coverage, band periodicity, surprisal under the shared model), the distribution
              features, all Fisher-score groups, the score groups the model does not use.
          (c) readable and new statistics: misjudged Sol vs Astra among d <= 0 rows after adjusting for d,
              per-environment consistency, and the same on the outside panels.
  halves  is "looking like Astra" a property of the whole response? The model fitted on first halves only
          and on second halves only (leave one environment out; outside panels from all training batches).
          Per label: correlation of the two half decisions (raw and within environment), P(second half
          misjudged | first half misjudged) vs P(second half misjudged | first half correct).

Run 12 (after run 11 found that the evidence sits in the first half and that only the low-dimensional
readable statistics carry information the model misses):
  prefix              the model refitted on numbers [0,10), [0,20), [0,40), [0,80) and [40,125) only (LOEO and
                      outside): how much of the evidence the opening carries.
  readable:<mode>     a readable-statistics discriminant (interp_stats without the repeat-based statistics,
                      standardized, environment projection, shrinkage LDA) either replacing the distribution
                      component (mode replace) or as a fourth component (mode add); weights by the same inner
                      leave-one-environment-out; v4 refitted in the same run for a paired comparison.

Run 15 (after the neural model of run 14 and a second opinion): refine = the cheap refinements left on the
original data on top of the readable change: separate weights for the two fork contexts, soft forks (banded
opening prefix, outcome counts smoothed over neighbouring values), an opening position x value density.

Run 16: universal = only the feature vectors every response yields (no fork evidence, which only applies to
responses that pass a well-supported opening node).

Usage: python research/hard_pair.py oof | halves | prefix | readable:replace | readable:add | refine | universal
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.gated_fingerprint as G  # noqa: E402
from research import fisher_score as FS  # noqa: E402
from research.curve_pair import k3_all, lda_fit_clean  # noqa: E402
from research.data import PAIR_ROLE, extended_panel  # noqa: E402
from research.runner_pair import auc  # noqa: E402
from research.seqfeatures import SEQ_FAMILIES  # noqa: E402
from research.sol_pair import interp_stats  # noqa: E402

p = extended_panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
MEM = sorted([A, S])
IA, IS = MEM.index(A), MEM.index(S)  # positions in the member logit vectors
TRB = ("train_original", "train_supplemental", PAIR_ROLE)
OUTS = ("old_outside", "fresh_outside")
pair = np.isin(p.y, [A, S])
TR = np.flatnonzero(pair & np.isin(p.role, TRB))
OUT_ROWS = np.flatnonzero(pair & np.isin(p.role, OUTS))
yb = (p.y == S).astype(int)
ENVS = sorted(set(p.env[TR]))
T0 = time.time()
RES = Path("results")
RES.mkdir(exist_ok=True)
SAVED: dict = {}


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def save(name, key, value):
    SAVED[key] = value
    (RES / f"hard_{name}.json").write_text(json.dumps(SAVED, indent=1))


def seq(i):
    return list(map(int, p.rows[i]["numbers"]))


# ----------------------------------------------------------------------------- model with its inner evidence

def fit_cluster_oof(rows):
    """gated_fingerprint.fit_cluster (same steps), also returning the inner leave-one-environment-out
    evidence (Sol - Astra) of its own reference rows, the combination weights and the shared model."""
    model_ids, members = p.classes, [A, S]
    member_ids = [model_ids[k] for k in MEM]
    y = np.array([member_ids.index(r["source"]) for r in rows])
    env = np.array([r["condition_id"] for r in rows])
    K = len(member_ids)
    seqs = [list(map(int, r["numbers"])) for r in rows]
    steps = FS.Steps(seqs)
    theta = FS.fit_base(steps, lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
    U = FS.score_vectors(steps, theta)[:, G._score_columns(G.SCORE_GROUPS)]
    del steps
    D, B, F = np.zeros((len(rows), K)), np.zeros((len(rows), K)), np.zeros((len(rows), K))
    for e in sorted(set(env)):
        keep = env != e
        tr = [r for r, k in zip(rows, keep) if k]
        loc = G.fit_local(tr, model_ids, members)
        br = G.fit_branch([r["numbers"] for r in tr], [member_ids.index(r["source"]) for r in tr], K)
        sloc = G.fit_score_local(U[keep], y[keep], env[keep])
        for i in np.flatnonzero(env == e):
            D[i] = G._dense_logits(loc, rows[i]["numbers"])
            B[i] = G.branch_scores(br, rows[i]["numbers"], K)
            F[i] = G._score_logits_from_vector(sloc, U[i])

    def nll(w):
        s = w[0] * D + w[1] * B + w[2] * F
        return -(s[np.arange(len(y)), y] - logsumexp(s, axis=1)).sum() + 1e-3 * (w ** 2).sum()
    w = minimize(nll, np.array([1.0, 1.0, 1.0]), method="L-BFGS-B").x
    cl = G.fit_local(rows, model_ids, members)
    cl["branch"] = G.fit_branch([r["numbers"] for r in rows], y.tolist(), K)
    cl["score"] = {"theta": theta.tolist(), "local": G.fit_score_local(U, y, env)}
    cl["combine"] = {"w_distribution": float(w[0]), "w_branch": float(w[1]), "w_score": float(w[2])}
    inner = w[0] * (D[:, IS] - D[:, IA]) + w[1] * (B[:, IS] - B[:, IA]) + w[2] * (F[:, IS] - F[:, IA])
    return cl, inner, w, theta


FAM_DIM = {f: len(SEQ_FAMILIES[f](np.arange(1, 251))) for f in G.LOCAL_FAMILIES}


def dense_parts(cl, x):
    z = (G.response_features(x, cl["families"]) - np.asarray(cl["feature_mean"])) \
        / np.asarray(cl["feature_scale"]) * np.asarray(cl["family_weight"])
    basis = np.asarray(cl["nuisance_basis"]).reshape(-1, len(z))
    z = z - (z @ basis.T) @ basis
    coef, icp = np.asarray(cl["coef"]), np.asarray(cl["intercept"])
    c = (coef[IS] - coef[IA]) * z
    out, o = {}, 0
    for f in cl["families"]:
        out[f] = float(c[o:o + FAM_DIM[f]].sum())
        o += FAM_DIM[f]
    assert o == len(c)
    out["intercept"] = float(icp[IS] - icp[IA])
    return out


def score_parts(loc, u):
    z = (u - np.asarray(loc["feature_mean"])) / np.asarray(loc["feature_scale"]) * np.asarray(loc["group_weight"])
    basis = np.asarray(loc["nuisance_basis"]).reshape(-1, len(z))
    z = z - (z @ basis.T) @ basis
    coef, icp = np.asarray(loc["coef"]), np.asarray(loc["intercept"])
    c = (coef[IS] - coef[IA]) * z
    out, o = {}, 0
    for g in loc["groups"]:
        out[g] = float(c[o:o + FS.GROUPS[g]].sum())
        o += FS.GROUPS[g]
    out["intercept"] = float(icp[IS] - icp[IA])
    return out


def branch_parts(branch, numbers):
    """gated_fingerprint.branch_scores (Sol - Astra), split by context: previous two numbers (depth 2) and
    the anchored opening prefix (depth 3), with the number of decision points that carried evidence."""
    idx = G._branch_index(branch)
    lam, support = branch["lambda"], branch["min_support"]
    x = list(map(int, numbers))
    out = {"bigram": 0.0, "prefix": 0.0, "n_bigram": 0, "n_prefix": 0}
    zero = np.zeros(2)
    for t, v in enumerate(x):
        p0 = 1.0 / G.V
        for depth, ctx in enumerate(G._branch_chain(x, t)):
            if ctx is None:
                continue
            entry = idx.get(G._key(ctx))
            if entry is None:
                continue
            row = entry[0].get(v)
            nv = entry[1][row] if row is not None else zero
            tot = entry[2]
            p0 = (nv.sum() + lam * p0) / (tot.sum() + lam)
            if depth >= 2 and tot.sum() >= support:
                e = np.log((nv + lam * p0) / (tot + lam)) - np.log(p0)
                k = "bigram" if depth == 2 else "prefix"
                out[k] += float(e[IS] - e[IA])
                out["n_" + k] += 1
    return out


# ----------------------------------------------------------------------------- statistics

def new_stats(x):
    """Sequence statistics not in the model: coverage, quarter-band periodicity, longer-lag dependence."""
    x = np.asarray(x, int)
    n = len(x)
    seen, new = np.zeros(400, bool), np.zeros(n, bool)
    for t, v in enumerate(x):
        new[t] = not seen[v]
        seen[v] = True
    s = {f"new-value share in first {m}": float(new[:m].mean()) for m in (50, 100, 150, 200)}
    s["repeats per 100"] = float(100 * (~new).mean())
    band = np.minimum((x - 1) * 4 // 355, 3)
    for k in range(1, 7):
        s[f"same quarter-band as {k} back"] = float(np.mean(band[k:] == band[:-k]))
    f = x.astype(float) - x.mean()
    for k in (2, 3, 4, 5):
        s[f"lag-{k} autocorrelation"] = float((f[k:] @ f[:-k]) / (f @ f))
    d = np.diff(x)
    s["|jump| sd"] = float(np.abs(d).std())
    sg = np.sign(d)
    runs = np.diff(np.flatnonzero(np.concatenate([[True], sg[1:] != sg[:-1], [True]])))
    s["mean run of same jump direction"] = float(runs.mean())
    gaps = []
    for t in range(1, n):  # distance of each new value to the nearest value used before it
        if new[t]:
            used = x[:t]
            gaps.append(np.abs(used - x[t]).min())
    gaps = np.array(gaps, float)
    s["distance of new value to nearest used, first half"] = float(gaps[:len(gaps) // 2].mean())
    s["distance of new value to nearest used, second half"] = float(gaps[len(gaps) // 2:].mean())
    return s


def stat_matrix(fn, idx):
    st = [fn(seq(i)) for i in idx]
    names = list(st[0])
    X = np.full((len(p.rows), len(names)), np.nan)
    X[idx] = np.array([[s[k] for k in names] for s in st])
    return X, names


def boot_ci(fn, idx, reps=300, seed=0):
    """Environment-cluster bootstrap of fn(rows) -> float."""
    rng = np.random.default_rng(seed)
    envs = sorted(set(p.env[idx]))
    vals = []
    for _ in range(reps):
        dr = rng.choice(envs, len(envs), replace=True)
        ni = np.concatenate([idx[p.env[idx] == e] for e in dr])
        try:
            vals.append(fn(ni))
        except (ValueError, ZeroDivisionError, IndexError):
            continue
    v = np.array([x for x in vals if np.isfinite(x)])
    return [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)], round(float(np.mean(v <= 0)), 3)


def recall(d, idx):
    a, s = idx[yb[idx] == 0], idx[yb[idx] == 1]
    return float(np.mean(d[a] <= 0)), float(np.mean(d[s] > 0))


def label_metrics(d, idx):
    ra, rs = recall(d, idx)
    ok3 = all(len(set(p.env[idx][yb[idx] == c])) >= 3 for c in (0, 1))
    k = k3_all(d, yb, p.env, idx) if ok3 else {"k3_all_astra": float("nan"), "k3_all_sol": float("nan")}
    return {"recall_astra": round(ra, 3), "recall_sol": round(rs, 3), "k1_worst": round(min(ra, rs), 3),
            "k3_all_astra": k["k3_all_astra"], "k3_all_sol": k["k3_all_sol"],
            "k3_all_worst": min(k["k3_all_astra"], k["k3_all_sol"]), "auc": round(auc(d[idx], yb[idx]), 3)}


# ----------------------------------------------------------------------------- second stage

def stage2(X, w, ref, d_in, te, d_te, region):
    """Second-stage LDA on the reference rows of one region (d <= 0: 'low', d > 0: 'high') and a logistic
    combination with d, fitted on inner out-of-fold second-stage scores. Returns (s_te, comb_te, coef)."""
    sel = d_in <= 0 if region == "low" else d_in > 0
    rr, din = ref[sel], d_in[sel]
    if min(np.bincount(yb[rr], minlength=2)) < 8:
        return None
    er = p.env[rr]
    s_in = np.full(len(rr), np.nan)
    for e in sorted(set(er)):
        a, b = rr[er != e], rr[er == e]
        if min(np.bincount(yb[a], minlength=2)) < 5:
            continue
        s_in[er == e] = lda_fit_clean(X, w, yb, p.env, a)(b)
    ok = ~np.isnan(s_in)
    Zin = np.c_[din[ok], s_in[ok]]
    mu, sd = Zin.mean(0), Zin.std(0) + 1e-9
    lr = LogisticRegression(C=1.0).fit((Zin - mu) / sd, yb[rr[ok]])
    s_te = lda_fit_clean(X, w, yb, p.env, rr)(te)
    comb = lr.decision_function((np.c_[d_te, s_te] - mu) / sd)
    return s_te, comb, (lr.coef_[0] / sd).round(3).tolist()


SETS = ("readable", "newseq", "dense", "score_all", "score_unused")
UNUSED = tuple(g for g in FS.GROUPS if g not in G.SCORE_GROUPS)


def exp_oof():
    name = "oof"
    both = np.concatenate([TR, OUT_ROWS])
    XR, nr = stat_matrix(interp_stats, both)
    XN0, nn0 = stat_matrix(new_stats, both)
    XD = np.full((len(p.rows), sum(FAM_DIM.values())), np.nan)
    XD[both] = np.array([G.response_features(seq(i)) for i in both])
    wD = G._family_weights(seq(TR[0]), G.LOCAL_FAMILIES)
    wS = np.concatenate([np.full(n, 1 / np.sqrt(n)) for n in FS.GROUPS.values()])
    ucols = G._score_columns(UNUSED)
    log(f"features ready: readable {XR.shape[1]}, new {XN0.shape[1]}, dense {XD.shape[1]}, score {FS.DIM}")

    d = np.full(len(p.rows), np.nan)
    comp = np.full((len(p.rows), 3), np.nan)  # weighted distribution, branch, score
    parts = {}
    s2 = {(s, r): np.full(len(p.rows), np.nan) for s in SETS for r in ("low", "high")}
    c2 = {(s, r): np.full(len(p.rows), np.nan) for s in SETS for r in ("low", "high")}
    lrcoef = {}
    XN = np.full((len(p.rows), XN0.shape[1] + 3), np.nan)
    XS = np.full((len(p.rows), FS.DIM), np.nan)
    folds = [(e, TR[p.env[TR] != e], TR[p.env[TR] == e]) for e in ENVS] + [("outside", TR, OUT_ROWS)]
    for fname, ref, te in folds:
        cl, d_in, w, theta = fit_cluster_oof([p.rows[i] for i in ref])
        allr = np.concatenate([ref, te])
        seqs = [seq(i) for i in allr]
        st = FS.Steps(seqs)
        U = FS.score_vectors(st, theta)
        ll, T = FS.row_loglik(st, theta), st.T
        del st
        st = FS.Steps([s[:40] for s in seqs])
        ll40, T40 = FS.row_loglik(st, theta), st.T
        del st
        XS[allr] = U
        XN[allr] = np.c_[XN0[allr], ll / T, ll40 / T40, (ll - ll40) / np.maximum(T - T40, 1)]
        pos = {i: k for k, i in enumerate(allr)}
        scols = G._score_columns(G.SCORE_GROUPS)
        for i in te:
            x = seq(i)
            dp = dense_parts(cl, x)
            bp = branch_parts(cl["branch"], x)
            sp = score_parts(cl["score"]["local"], U[pos[i], scols])
            Dd, Bd, Fd = sum(dp.values()), bp["bigram"] + bp["prefix"], sum(sp.values())
            comp[i] = w[0] * Dd, w[1] * Bd, w[2] * Fd
            d[i] = comp[i].sum()
            parts[int(i)] = {**{f"dist:{k}": w[0] * v for k, v in dp.items()},
                             "branch:bigram": w[1] * bp["bigram"], "branch:prefix": w[1] * bp["prefix"],
                             "n_bigram": bp["n_bigram"], "n_prefix": bp["n_prefix"],
                             **{f"score:{k}": w[2] * v for k, v in sp.items()}}
        sets = {"readable": (XR, np.full(XR.shape[1], 1 / np.sqrt(XR.shape[1]))),
                "newseq": (XN, np.full(XN.shape[1], 1 / np.sqrt(XN.shape[1]))),
                "dense": (XD, wD), "score_all": (XS, wS),
                "score_unused": (XS[:, ucols], wS[ucols])}
        for sname, (X, wx) in sets.items():
            for region in ("low", "high"):
                r = stage2(X, wx, ref, d_in, te, d[te], region)
                if r is None:
                    continue
                s2[(sname, region)][te], c2[(sname, region)][te] = r[0], r[1]
                lrcoef.setdefault(f"{sname}:{region}", []).append(r[2])
        mis_in = {"astra": int(np.sum((yb[ref] == 0) & (d_in > 0))), "sol": int(np.sum((yb[ref] == 1) & (d_in <= 0)))}
        log(f"fold {fname}: {len(ref)} refs, weights {np.round(w, 3).tolist()}, inner misjudged {mis_in}, "
            f"query recall A/S {np.round(recall(d, te), 3).tolist()}")
    save(name, "rows", {str(i): {"role": p.role[i], "env": p.env[i], "sol": int(yb[i]), "d": round(float(d[i]), 4),
                                 "components": np.round(comp[i], 4).tolist(),
                                 **{k: round(float(v), 4) for k, v in parts[int(i)].items()}} for i in both})

    for tag, idx in (("train (LOEO)", TR), ("outside", OUT_ROWS)):
        # (a) where the evidence comes from
        grp = {"Astra ok": idx[(yb[idx] == 0) & (d[idx] <= 0)], "Astra misjudged": idx[(yb[idx] == 0) & (d[idx] > 0)],
               "Sol ok": idx[(yb[idx] == 1) & (d[idx] > 0)], "Sol misjudged": idx[(yb[idx] == 1) & (d[idx] <= 0)]}
        keys = list(parts[int(idx[0])])
        tab = {k: {g: round(float(np.mean([parts[int(i)][k] for i in v])), 3) if len(v) else None for g, v in grp.items()}
               for k in keys}
        tab["d"] = {g: round(float(d[v].mean()), 3) if len(v) else None for g, v in grp.items()}
        save(name, f"a_parts {tag}", {"n": {g: len(v) for g, v in grp.items()}, "means": tab})
        log(f"PARTS {tag} n {json.dumps({g: len(v) for g, v in grp.items()})}")
        for k, r in tab.items():
            gap = None if None in (r["Sol ok"], r["Sol misjudged"]) else r["Sol misjudged"] - r["Sol ok"]
            print(f"PARTS {tag:12} {k:22} " + " | ".join(f"{g} {v:+.3f}" if v is not None else f"{g} -" for g, v in r.items())
                  + (f" | Sol misjudged - ok {gap:+.3f}" if gap is not None else ""), flush=True)
        # leave one part out (same fit, no refit)
        base = label_metrics(d, idx)
        loo = {"all": base}
        for c, cn in enumerate(("distribution", "branch", "score")):
            loo[f"without {cn}"] = label_metrics(d - comp[:, c], idx)
            loo[f"only {cn}"] = label_metrics(np.nan_to_num(comp[:, c]), idx)
        for k in keys:
            if k.startswith("n_") or k.endswith("intercept"):
                continue
            v = np.zeros(len(p.rows))
            v[idx] = [parts[int(i)][k] for i in idx]
            loo[f"without {k}"] = label_metrics(d - v, idx)
        save(name, f"a_leave_one_out {tag}", loo)
        for k, r in loo.items():
            print(f"LOO {tag:12} {k:26} k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} worst {r['k1_worst']:.3f} | "
                  f"k3_all A {r['k3_all_astra']:.3f} S {r['k3_all_sol']:.3f} worst {r['k3_all_worst']:.3f} | auc {r['auc']:.3f}", flush=True)

        # (b) residual information inside each region
        res_b = {}
        for region, lab_hard in (("low", 1), ("high", 0)):
            reg0 = idx[(d[idx] <= 0) if region == "low" else (d[idx] > 0)]
            sgn = 1 if region == "low" else -1  # the misjudged label scores high
            for sname in SETS:
                reg = reg0[~np.isnan(s2[(sname, region)][reg0])]  # folds without a second stage are left out
                if len(reg) < 10:
                    continue
                s, c = s2[(sname, region)][reg], c2[(sname, region)][reg]
                tgt = (yb[reg] == lab_hard).astype(int)
                base_auc, a_s, a_c = auc(sgn * d[reg], tgt), auc(sgn * s, tgt), auc(sgn * c, tgt)

                def gain(ni, region_=region, sname_=sname):
                    t = (yb[ni] == lab_hard).astype(int)
                    sg_ = 1 if region_ == "low" else -1
                    return auc(sg_ * c2[(sname_, region_)][ni], t) - auc(sg_ * d[ni], t)
                ci, pneg = boot_ci(gain, reg)
                # what a two-stage call would do: flip a row when the region's logistic says the other label
                flip = (c > 0) if region == "low" else (c < 0)
                r = {"n_region": len(reg), "n_misjudged": int(tgt.sum()), "auc_d": round(base_auc, 3),
                     "auc_stage2": round(a_s, 3), "auc_combined": round(a_c, 3), "gain": round(a_c - base_auc, 3),
                     "gain_ci": ci, "boot_share_gain<=0": pneg,
                     "flipped_misjudged": int(np.sum(flip & (tgt == 1))), "flipped_correct": int(np.sum(flip & (tgt == 0)))}
                res_b[f"{region}:{sname}"] = r
                print(f"RESIDUAL {tag:12} {'called Astra' if region == 'low' else 'called Sol':12} {sname:13} "
                      f"n {r['n_region']} (misjudged {r['n_misjudged']}) | AUC misjudged vs correct: d {r['auc_d']:.3f}, "
                      f"stage2 {r['auc_stage2']:.3f}, d+stage2 {r['auc_combined']:.3f}, gain {r['gain']:+.3f} {ci} "
                      f"(boot P<=0 {pneg}) | flips: misjudged {r['flipped_misjudged']}, correct {r['flipped_correct']}", flush=True)
        save(name, f"b_residual {tag}", res_b)

        # (c) readable / new statistics, misjudged Sol vs Astra among d <= 0, adjusted for d
        reg = idx[d[idx] <= 0]
        hs, aa = reg[yb[reg] == 1], reg[yb[reg] == 0]
        sol_ok = idx[(yb[idx] == 1) & (d[idx] > 0)]
        Xc = np.c_[XR, XN0]
        names = nr + nn0
        rows_c = []
        for j, k in enumerate(names):
            v = Xc[reg, j]
            M = np.c_[np.ones(len(reg)), d[reg]]
            beta = np.linalg.lstsq(M, v, rcond=None)[0]
            r_ = v - M @ beta
            sd = r_.std() + 1e-12
            eff = (r_[yb[reg] == 1].mean() - r_[yb[reg] == 0].mean()) / sd
            envs = [e for e in sorted(set(p.env[reg])) if ((p.env[hs] == e).any() and (p.env[aa] == e).any())]
            agree = sum(np.sign(r_[(p.env[reg] == e) & (yb[reg] == 1)].mean() - r_[(p.env[reg] == e) & (yb[reg] == 0)].mean())
                        == np.sign(eff) for e in envs)
            rows_c.append({"stat": k, "effect_adjusted": round(float(eff), 3), "envs_agree": f"{agree}/{len(envs)}",
                           "mean Astra(d<=0)": round(float(Xc[aa, j].mean()), 4), "mean Sol misjudged": round(float(Xc[hs, j].mean()), 4),
                           "mean Sol ok": round(float(Xc[sol_ok, j].mean()), 4)})
        rows_c.sort(key=lambda r: -abs(r["effect_adjusted"]))
        save(name, f"c_stats {tag}", rows_c)
        for r in rows_c[:25]:
            print(f"STAT {tag:12} {r['stat']:48} adj. effect {r['effect_adjusted']:+.2f} ({r['envs_agree']} envs) | "
                  f"Astra(d<=0) {r['mean Astra(d<=0)']} Sol misjudged {r['mean Sol misjudged']} Sol ok {r['mean Sol ok']}", flush=True)
    save(name, "logistic_coef (d, stage2)", lrcoef)


def exp_halves():
    name = "halves"
    both = np.concatenate([TR, OUT_ROWS])
    dh = {}
    for half in ("first", "second"):
        def cut(i, half_=half):
            x = seq(i)
            return x[:len(x) // 2] if half_ == "first" else x[len(x) // 2:]
        rows_h = {int(i): dict(p.rows[i], numbers=cut(i)) for i in both}
        v = np.full(len(p.rows), np.nan)
        for fname, ref, te in [(e, TR[p.env[TR] != e], TR[p.env[TR] == e]) for e in ENVS] + [("outside", TR, OUT_ROWS)]:
            cl = G.fit_cluster([rows_h[int(i)] for i in ref], p.classes, [A, S])
            for i in te:
                s = G.local_logits(cl, rows_h[int(i)]["numbers"])
                v[i] = s[IS] - s[IA]
            log(f"{half} half, fold {fname}: query recall A/S {np.round(recall(v, te), 3).tolist()}")
        dh[half] = v
    d1, d2 = dh["first"], dh["second"]
    save(name, "rows", {str(i): {"role": p.role[i], "env": p.env[i], "sol": int(yb[i]),
                                 "d_first": round(float(d1[i]), 4), "d_second": round(float(d2[i]), 4)} for i in both})
    out = {}
    for tag, idx in (("train (LOEO)", TR), ("outside", OUT_ROWS)):
        for half, v in dh.items():
            out[f"{tag} {half} half"] = label_metrics(v, idx)
            r = out[f"{tag} {half} half"]
            print(f"HALF {tag:12} {half:6} k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} | k3_all A {r['k3_all_astra']:.3f} "
                  f"S {r['k3_all_sol']:.3f} | auc {r['auc']:.3f}", flush=True)
        for c, lab in ((0, "Astra"), (1, "Sol")):
            q = idx[yb[idx] == c]
            w1 = (d1[q] > 0) if c == 0 else (d1[q] <= 0)  # first half misjudged
            w2 = (d2[q] > 0) if c == 0 else (d2[q] <= 0)
            # within environment: remove each environment's mean of this label
            r1, r2 = d1[q].copy(), d2[q].copy()
            for e in set(p.env[q]):
                m = p.env[q] == e
                r1[m] -= r1[m].mean()
                r2[m] -= r2[m].mean()

            def coupling(ni, c_=c):
                qq = ni[yb[ni] == c_]
                a1 = (d1[qq] > 0) if c_ == 0 else (d1[qq] <= 0)
                a2 = (d2[qq] > 0) if c_ == 0 else (d2[qq] <= 0)
                return float(a2[a1].mean() - a2[~a1].mean())
            ci, pneg = boot_ci(coupling, idx)
            res = {"n": len(q), "r_pearson": round(float(np.corrcoef(d1[q], d2[q])[0, 1]), 3),
                   "r_spearman": round(float(spearmanr(d1[q], d2[q])[0]), 3),
                   "r_within_env": round(float(np.corrcoef(r1, r2)[0, 1]), 3),
                   "first misjudged": int(w1.sum()), "second misjudged": int(w2.sum()), "both": int((w1 & w2).sum()),
                   "P(second misjudged | first misjudged)": round(float(w2[w1].mean()), 3) if w1.any() else None,
                   "P(second misjudged | first correct)": round(float(w2[~w1].mean()), 3) if (~w1).any() else None,
                   "difference_ci": ci, "boot_share<=0": pneg}
            out[f"{tag} {lab}"] = res
            print(f"STATE {tag:12} {lab:5} {json.dumps(res)}", flush=True)
    save(name, "summary", out)


# ----------------------------------------------------------------------------- run 12

WINDOWS = ((0, 10), (0, 20), (0, 40), (0, 80), (40, 125))


def exp_prefix():
    """How much of the evidence sits in the opening: the model refitted on numbers [a, b) only."""
    name = "prefix"
    both = np.concatenate([TR, OUT_ROWS])
    out = {}
    # some family statistics are undefined on a few numbers (run 12: NaN -> SVD failure at [0, 10)); this
    # experiment only, the deliverable is unchanged
    rf = G.response_features
    G.response_features = lambda numbers, families=G.LOCAL_FAMILIES: np.nan_to_num(rf(numbers, families), nan=0.0, posinf=0.0, neginf=0.0)
    for a, b in WINDOWS:
        rows_w = {int(i): dict(p.rows[i], numbers=seq(i)[a:b]) for i in both}
        v = np.full(len(p.rows), np.nan)
        for fname, ref, te in [(e, TR[p.env[TR] != e], TR[p.env[TR] == e]) for e in ENVS] + [("outside", TR, OUT_ROWS)]:
            cl = G.fit_cluster([rows_w[int(i)] for i in ref], p.classes, [A, S])
            for i in te:
                s = G.local_logits(cl, rows_w[int(i)]["numbers"])
                v[i] = s[IS] - s[IA]
        for tag, idx in (("train (LOEO)", TR), ("outside", OUT_ROWS)):
            r = label_metrics(v, idx)
            out[f"{tag} [{a},{b})"] = r
            print(f"PREFIX {tag:12} numbers [{a:3},{b:3}) k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} worst {r['k1_worst']:.3f} | "
                  f"k3_all A {r['k3_all_astra']:.3f} S {r['k3_all_sol']:.3f} worst {r['k3_all_worst']:.3f} | auc {r['auc']:.3f}", flush=True)
        save(name, "summary", out)
        save(name, f"decisions [{a},{b})", {str(i): round(float(v[i]), 4) for i in both})


# readable statistics without the repeat-based ones (a repeat is a rare event in 218-300 numbers)
REPEAT_BASED = ("distinct values / 355", "top-10 values share", "max count of one value", "repeat within 5",
                "repeat within 20", "repeat within 100", "median gap to same value")


def exp_readable(mode):
    """The one candidate change suggested by run 11: a low-dimensional readable-statistics discriminant.
    mode replace: it replaces the distribution component (high-dimensional histograms); mode add: it is a
    fourth component. Everything else as fit_cluster; weights by the same inner leave-one-environment-out."""
    name = f"readable_{mode}"
    both = np.concatenate([TR, OUT_ROWS])
    XR, nr = stat_matrix(interp_stats, both)
    cols = [j for j, k in enumerate(nr) if k not in REPEAT_BASED]
    XR = XR[:, cols]
    wR = np.full(len(cols), 1 / np.sqrt(len(cols)))
    log(f"readable statistics used: {[nr[j] for j in cols]}")
    d = np.full(len(p.rows), np.nan)
    base = np.full(len(p.rows), np.nan)
    weights = []
    for fname, ref, te in [(e, TR[p.env[TR] != e], TR[p.env[TR] == e]) for e in ENVS] + [("outside", TR, OUT_ROWS)]:
        rows = [p.rows[i] for i in ref]
        member_ids = [p.classes[k] for k in MEM]
        y = np.array([member_ids.index(r["source"]) for r in rows])
        env = np.array([r["condition_id"] for r in rows])
        seqs = [seq(i) for i in ref]
        steps = FS.Steps(seqs)
        theta = FS.fit_base(steps, lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
        U = FS.score_vectors(steps, theta)[:, G._score_columns(G.SCORE_GROUPS)]
        del steps
        comp = np.zeros((len(ref), 4))  # distribution, branch, score, readable (Sol - Astra)
        for e in sorted(set(env)):
            keep = env != e
            tr = [r for r, k in zip(rows, keep) if k]
            loc = G.fit_local(tr, p.classes, [A, S])
            br = G.fit_branch([r["numbers"] for r in tr], y[keep].tolist(), 2)
            sloc = G.fit_score_local(U[keep], y[keep], env[keep])
            rdec = lda_fit_clean(XR, wR, yb, p.env, ref[keep])
            for i in np.flatnonzero(~keep):
                x = rows[i]["numbers"]
                Dv, Bv, Fv = G._dense_logits(loc, x), G.branch_scores(br, x, 2), G._score_logits_from_vector(sloc, U[i])
                comp[i, :3] = Dv[IS] - Dv[IA], Bv[IS] - Bv[IA], Fv[IS] - Fv[IA]
            comp[~keep, 3] = rdec(ref[~keep])
        use = (3, 1, 2) if mode == "replace" else (0, 1, 2, 3)

        def nll(w, use_=use):
            s = comp[:, use_] @ w  # log-odds of Sol (two classes)
            return -(np.where(y == IS, s, 0) - np.logaddexp(0, s)).sum() + 1e-3 * (w ** 2).sum()

        def nll_base(w):
            s = comp[:, :3] @ w
            return -(np.where(y == IS, s, 0) - np.logaddexp(0, s)).sum() + 1e-3 * (w ** 2).sum()
        w = minimize(nll, np.ones(len(use)), method="L-BFGS-B").x
        wb = minimize(nll_base, np.ones(3), method="L-BFGS-B").x
        weights.append({"fold": fname, "w": dict(zip([("distribution", "branch", "score", "readable")[k] for k in use], np.round(w, 3).tolist())),
                        "w_base": np.round(wb, 3).tolist()})
        cl = G.fit_local(rows, p.classes, [A, S])
        brf = G.fit_branch([r["numbers"] for r in rows], y.tolist(), 2)
        slocf = G.fit_score_local(U, y, env)
        rdec = lda_fit_clean(XR, wR, yb, p.env, ref)
        Ute = FS.score_vectors(FS.Steps([seq(i) for i in te]), theta)[:, G._score_columns(G.SCORE_GROUPS)]
        qc = np.zeros((len(te), 4))
        for k, i in enumerate(te):
            x = seq(i)
            Dv, Bv, Fv = G._dense_logits(cl, x), G.branch_scores(brf, x, 2), G._score_logits_from_vector(slocf, Ute[k])
            qc[k, :3] = Dv[IS] - Dv[IA], Bv[IS] - Bv[IA], Fv[IS] - Fv[IA]
        qc[:, 3] = rdec(te)
        d[te] = qc[:, use] @ w
        base[te] = qc[:, :3] @ wb
        log(f"fold {fname}: weights {weights[-1]}, recall A/S new {np.round(recall(d, te), 3).tolist()} base {np.round(recall(base, te), 3).tolist()}")
    save(name, "weights", weights)
    out = {}
    for tag, idx in (("train (LOEO)", TR), ("outside", OUT_ROWS)):
        for lab, v in (("v4 (same run)", base), (f"readable {mode}", d)):
            r = label_metrics(v, idx)
            out[f"{tag} {lab}"] = r
            print(f"READABLE {tag:12} {lab:18} k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} worst {r['k1_worst']:.3f} | "
                  f"k3_all A {r['k3_all_astra']:.3f} S {r['k3_all_sol']:.3f} worst {r['k3_all_worst']:.3f} | auc {r['auc']:.3f}", flush=True)
        # paired difference in worst-label k3 and AUC, environment bootstrap
        ci, pneg = boot_ci(lambda ni: auc(d[ni], yb[ni]) - auc(base[ni], yb[ni]), idx)
        out[f"{tag} auc gain"] = {"ci": ci, "boot_share<=0": pneg}
        print(f"READABLE {tag:12} AUC gain over v4: {round(auc(d[idx], yb[idx]) - auc(base[idx], yb[idx]), 3):+.3f} {ci} (boot P<=0 {pneg})", flush=True)
    save(name, "summary", out)
    save(name, "decisions", {str(i): [round(float(base[i]), 4), round(float(d[i]), 4)] for i in both})


# ----------------------------------------------------------------------------- run 15

# Soft forks: the branch contexts of gated_fingerprint plus a banded opening prefix; the outcome counts are
# smoothed over neighbouring values (27/28/29 support each other; 22 vs 28 stay apart).
SOFT_K = (1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9)  # triangular kernel over v-2..v+2
QW = 30                                        # band width of the banded prefix (12 bands)
LEVEL = {2: "B", 3: "Q", 4: "P"}


def soft_chain(x, t):
    """Shallow to deep: root, previous number, previous two numbers, banded opening prefix, exact opening
    prefix (the last two only for t < BRANCH_PREFIX). Evidence is read at the last three."""
    op = t < G.BRANCH_PREFIX
    return [("R",), ("U", x[t - 1]) if t >= 1 else None, ("B", x[t - 2], x[t - 1]) if t >= 2 else None,
            ("Q",) + tuple(v // QW for v in x[:t]) if op else None, ("P",) + tuple(x[:t]) if op else None]


def fit_soft(seqs, labels):
    """labels: 1 = Sol, 0 = Astra. Same support rule as fit_branch."""
    counts = {}
    for x, c in zip(seqs, labels):
        for t, v in enumerate(x):
            for ctx in soft_chain(x, t):
                if ctx is not None:
                    counts.setdefault(ctx, {}).setdefault(v, [0, 0])[c] += 1
    model = {}
    for ctx, dv in counts.items():
        ta, ts = sum(c[0] for c in dv.values()), sum(c[1] for c in dv.values())
        if ctx[0] in ("R", "U") or ta + ts >= G.BRANCH_SUPPORT:
            model[ctx] = (dv, ta, ts)
    return model


def soft_parts(model, x):
    """log P_Sol - log P_Astra summed per evidence level, with kernel-smoothed outcome counts."""
    lam, sup = G.BRANCH_LAMBDA, G.BRANCH_SUPPORT
    out = {"B": 0.0, "Q": 0.0, "P": 0.0}
    for t, v in enumerate(x):
        p0 = 1.0 / G.V
        for lev, ctx in enumerate(soft_chain(x, t)):
            if ctx is None:
                continue
            e = model.get(ctx)
            if e is None:
                continue
            dv, ta, ts = e
            na = ns = 0.0
            for j, kw in zip(range(-2, 3), SOFT_K):
                c = dv.get(v + j)
                if c:
                    na += kw * c[0]
                    ns += kw * c[1]
            p0 = (na + ns + lam * p0) / (ta + ts + lam)
            if lev >= 2 and ta + ts >= sup:
                out[LEVEL[lev]] += float(np.log((ns + lam * p0) / (ts + lam)) - np.log((na + lam * p0) / (ta + lam)))
    return out


# Opening density: per position t < 40, the label's value density from the references' numbers at positions
# t-2..t+2, Gaussian-smoothed over values (bandwidth 4), mixed with 10% uniform.
WT, WS, WH = 40, 2, 4.0


def fit_density(seqs, labels):
    C = np.zeros((2, WT + WS, 358))
    for x, c in zip(seqs, labels):
        for s, v in enumerate(x[:WT + WS]):
            C[c, s, min(max(v, 0), 357)] += 1
    u = np.arange(-12, 13)
    g = np.exp(-u ** 2 / (2 * WH ** 2))
    g /= g.sum()
    Cv = np.apply_along_axis(lambda r: np.convolve(r, g, mode="same"), 2, C)
    M = np.stack([Cv[:, max(0, t - WS):t + WS + 1].sum(1) for t in range(WT)], 1)
    M /= M.sum(2, keepdims=True)
    return 0.9 * M + 0.1 / 358


def density_llr(M, x):
    return float(sum(np.log(M[1, t, min(max(v, 0), 357)]) - np.log(M[0, t, min(max(v, 0), 357)]) for t, v in enumerate(x[:WT])))


COLS = ("D", "B", "Bb", "Bp", "F", "R", "SB", "SQ", "SP", "W")
REFINE_VARIANTS = {  # D distribution, B fork (Bb previous two numbers, Bp opening prefix), F score, R readable,
    "v4": ("D", "B", "F"),                                   # S* soft forks, W opening density
    "readable (run 12)": ("R", "B", "F"),
    "readable + fork split": ("R", "Bb", "Bp", "F"),
    "readable + soft forks": ("R", "SB", "SQ", "SP", "F"),
    "readable + fork split + density": ("R", "Bb", "Bp", "F", "W"),
    "all (primary)": ("R", "SB", "SQ", "SP", "F", "W"),
}

# run 16: only components every response yields (feature vectors), no fork evidence; primary = v4 without forks
UNIVERSAL = {
    "v4": ("D", "B", "F"),
    "universal: distribution + score (primary)": ("D", "F"),
    "universal: readable + score": ("R", "F"),
    "universal: distribution + readable + score": ("D", "R", "F"),
    "universal: score only": ("F",),
}


def fit_weights(comp, yv, use):
    Z = comp[:, [COLS.index(c) for c in use]]

    def nll(w):
        s = Z @ w
        return -(np.where(yv == 1, s, 0) - np.logaddexp(0, s)).sum() + 1e-3 * (w ** 2).sum()
    return minimize(nll, np.ones(len(use)), method="L-BFGS-B").x


def exp_refine(mode="refine"):
    """Run 15: the cheap refinements left on the original data (after a second opinion), each on top of the
    run 12 readable change, weights by the same inner leave-one-environment-out; v4 refitted in the same run.
    Pre-specified primary variant: all (primary).
    Run 16 (mode universal): only components every response yields (distribution, readable, score vectors),
    no fork evidence; pre-specified primary: v4 without its fork component (distribution + score)."""
    name = mode
    VARIANTS = UNIVERSAL if mode == "universal" else REFINE_VARIANTS
    prim = next(v for v in VARIANTS if "primary" in v)
    both = np.concatenate([TR, OUT_ROWS])
    XR, nr = stat_matrix(interp_stats, both)
    cols = [j for j, k in enumerate(nr) if k not in REPEAT_BASED]
    XR = XR[:, cols]
    wR = np.full(len(cols), 1 / np.sqrt(len(cols)))
    member_ids = [p.classes[k] for k in MEM]

    def components(tr, U_tr, ev, U_ev):
        rows_tr = [p.rows[i] for i in tr]
        ym = np.array([member_ids.index(r["source"]) for r in rows_tr])
        loc = G.fit_local(rows_tr, p.classes, [A, S])
        br = G.fit_branch([seq(i) for i in tr], ym.tolist(), 2)
        sloc = G.fit_score_local(U_tr, ym, p.env[tr])
        r = lda_fit_clean(XR, wR, yb, p.env, tr)(ev)
        soft = fit_soft([seq(i) for i in tr], yb[tr].tolist())
        M = fit_density([seq(i) for i in tr], yb[tr].tolist())
        out = np.zeros((len(ev), len(COLS)))
        for k, i in enumerate(ev):
            x = seq(i)
            Dv, Fv = G._dense_logits(loc, x), G._score_logits_from_vector(sloc, U_ev[k])
            bp, sp = branch_parts(br, x), soft_parts(soft, x)
            out[k] = [Dv[IS] - Dv[IA], bp["bigram"] + bp["prefix"], bp["bigram"], bp["prefix"], Fv[IS] - Fv[IA],
                      r[k], sp["B"], sp["Q"], sp["P"], density_llr(M, x)]
        return out

    d = {v: np.full(len(p.rows), np.nan) for v in VARIANTS}
    comp_q = np.full((len(p.rows), len(COLS)), np.nan)
    weights = []
    for fname, ref, te in [(e, TR[p.env[TR] != e], TR[p.env[TR] == e]) for e in ENVS] + [("outside", TR, OUT_ROWS)]:
        steps = FS.Steps([seq(i) for i in ref])
        theta = FS.fit_base(steps, lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
        U = FS.score_vectors(steps, theta)[:, G._score_columns(G.SCORE_GROUPS)]
        del steps
        env = p.env[ref]
        comp = np.zeros((len(ref), len(COLS)))
        for e in sorted(set(env)):
            keep = env != e
            comp[~keep] = components(ref[keep], U[keep], ref[~keep], U[~keep])
        W = {v: fit_weights(comp, yb[ref], use) for v, use in VARIANTS.items()}
        Ute = FS.score_vectors(FS.Steps([seq(i) for i in te]), theta)[:, G._score_columns(G.SCORE_GROUPS)]
        qc = components(ref, U, te, Ute)
        comp_q[te] = qc
        for v, use in VARIANTS.items():
            d[v][te] = qc[:, [COLS.index(c) for c in use]] @ W[v]
        weights.append({"fold": str(fname), **{v: dict(zip(use, np.round(W[v], 3).tolist())) for v, use in VARIANTS.items()}})
        log(f"fold {fname}: primary weights {weights[-1][prim]}; recall A/S v4 {np.round(recall(d['v4'], te), 3).tolist()} "
            f"primary {np.round(recall(d[prim], te), 3).tolist()}")
        save(name, "weights", weights)
    out = {}
    for tag, idx in (("train (LOEO)", TR), ("outside", OUT_ROWS)):
        for v in VARIANTS:
            r = label_metrics(d[v], idx)
            out[f"{tag} {v}"] = r
            print(f"REFINE {tag:12} {v:32} k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} worst {r['k1_worst']:.3f} | "
                  f"k3_all A {r['k3_all_astra']:.3f} S {r['k3_all_sol']:.3f} worst {r['k3_all_worst']:.3f} | auc {r['auc']:.3f}", flush=True)
        for v in list(VARIANTS)[1:]:
            ci, pneg = boot_ci(lambda ni, v_=v: auc(d[v_][ni], yb[ni]) - auc(d["v4"][ni], yb[ni]), idx)
            g = auc(d[v][idx], yb[idx]) - auc(d["v4"][idx], yb[idx])
            out[f"{tag} {v} auc gain"] = {"gain": round(g, 3), "ci": ci, "boot_share<=0": pneg}
            print(f"REFINE {tag:12} {v:32} AUC gain over v4 {g:+.3f} {ci} (boot P<=0 {pneg})", flush=True)
        for c in COLS:  # each component alone (held-out values)
            print(f"REFINE {tag:12} component {c:3} auc {auc(comp_q[idx, COLS.index(c)], yb[idx]):.3f}", flush=True)
    save(name, "summary", out)
    save(name, "decisions", {str(i): {v: round(float(d[v][i]), 4) for v in VARIANTS} for i in both})
    save(name, "components (held-out)", {str(i): dict(zip(COLS, np.round(comp_q[i], 4).tolist())) for i in both})


if __name__ == "__main__":
    log(f"rows: train {len(TR)} ({len(ENVS)} environments), outside {len(OUT_ROWS)}")
    arg = sys.argv[1]
    if arg.startswith("readable:"):
        exp_readable(arg.split(":")[1])
    else:
        {"oof": exp_oof, "halves": exp_halves, "prefix": exp_prefix, "refine": exp_refine,
         "universal": lambda: exp_refine("universal")}[arg]()
    log("done")
