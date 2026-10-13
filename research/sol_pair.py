"""Run 5: why many GPT-6.1 Sol responses are scored as GPT-6 Astra, and reference-only fixes.

All features are the clean ones of run 4 (every response cut to CLEAN_T = 990 numbers, columns that
are constant in the references left out, query features clipped to the reference range).

Experiments:
  why2       (run 8) readable statistics of every response (value ranges, last digits, repeats, jumps,
             first vs last 100): where the misjudged Sol responses sit between correct Sol and Astra,
             how much the prompt decides it (same prompt in both splits), and a discriminant on these
             statistics only.
  why        diagnosis, nothing here is used to score a query.
             (a) the frozen library cluster model (no pilot rows at all) on four 250-number windows:
                 does it see test-split Sol responses as more Astra-like than development-split Sol
                 responses, environment by environment? A frozen model cannot have learned the split.
             (b) feature-space shift between the splits inside each label, as a fraction of the
                 Astra -> Sol gap, per feature family.
             (c) can the split be predicted inside one label (LOEO dense LDA, AUC)?
             (d) development -> test (and back): how far the query means of each label moved from the
                 reference means along the fitted discriminant, per family and per feature.
             (e) one line per response: decisions, collection time, output tokens (format), parsed length.
  fix:<v>    development -> test and test -> development with a different dense / score fit; win0lib
             is unchanged. A trailing ":loeo" also runs the 12 LOEO folds.
             base         lda_fit_clean (run 4)
             rank:<r>     environment nuisance projection of rank r (base: 2)
             re:<g>:<k>   random-effects LDA: w = (S_w + k * S_n)^-1 (m_Sol - m_Astra), S_w Ledoit-Wolf
                          within-label covariance, S_n the covariance of the within-label offsets of
                          the reference groups g: env (prompt) or envwave (prompt and collection wave).
                          Waves are runs of reference collection times with gaps under 8 minutes,
                          found from the reference rows only. Directions in which prompts or waves
                          move the responses are down-weighted instead of removed. g = none: no
                          nuisance term.
             re:<g>:<k>:<a>  (run 6) the same with the within-label covariance shrunk towards its mean
                          variance by a fixed a (a = 1: nearest-centroid direction in the family-weighted
                          space), a = lw Ledoit-Wolf, a = cv chosen per source from (lw, 0.5, 0.8, 0.95, 1)
                          by the AUC of an inner leave-one-environment-out inside the references.
             Run 5 found that re:env beats base across the splits through the covariance estimate (rank:0,
             no projection, does not), i.e. through stronger shrinkage in the family-weighted space.

Every experiment prints per-label results as soon as it finishes (worse label first), so the live log
shows them while the job runs.

Usage: python research/sol_pair.py <experiment> [<experiment> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.curve_pair import CLEAN_T, NAMES, Context, lda_fit_clean, run_loeo, summarize  # noqa: E402
from research.pilot_length_matched import LDA, env_basis  # noqa: E402
from research.runner_pair import (FS, G, MEMBERS, PILOT, SMOKE, Saver, auc, cluster_decisions,  # noqa: E402
                                  folds, library_pair_rows, log)
from research.stack_pair import SG  # noqa: E402


def row_meta(rows):
    """Collection metadata, in the order of load_pilot rows (diagnosis and reference waves only)."""
    raw = [json.loads(line) for line in PILOT.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if SMOKE:
        keep = {r["condition_id"][len("pilot:"):] for r in rows}
        raw = [r for r in raw if r["condition_id"] in keep]
    assert len(raw) == len(rows)
    return {"minute": np.array([(int(r["collected_at"][11:13]) - 10) * 60 + int(r["collected_at"][14:16])
                                + int(r["collected_at"][17:19]) / 60 for r in raw]),
            "out_tokens": np.array([r["usage"]["output_tokens"] for r in raw]),
            "reasoning": np.array([r["usage"]["output_tokens_details"]["reasoning_tokens"] for r in raw]),
            "seconds": np.array([r["request_seconds"] for r in raw]),
            "sha": np.array([r["messages_sha256"] for r in raw]),
            "out_of_range": np.array([r.get("range_violation_count") or 0 for r in raw])}


def waves(minute, gap=8.0):
    """Collection waves: runs of times without a gap of more than `gap` minutes."""
    order = np.argsort(minute)
    w = np.zeros(len(minute), int)
    w[order] = np.concatenate([[0], np.cumsum(np.diff(minute[order]) > gap)])
    return w


# ----------------------------------------------------------------------------- fits

def standardize(X, w, ref, floor=0.1):
    R = X[ref]
    lo, hi = R.min(0), R.max(0)
    mu, sd = R.mean(0), R.std(0)
    keep = sd > 1e-12
    sd = np.where(keep, np.maximum(sd, floor * np.median(sd[keep])), np.inf)
    return lambda idx: ((np.clip(X[idx], lo, hi) - mu) / sd * w)[:, keep], keep


def fit_rank(X, w, y, env, ref, rank=2):
    z, _ = standardize(X, w, ref)
    Z = z(ref)
    B = env_basis(Z, y[ref], env[ref], rank) if rank else np.zeros((0, Z.shape[1]))
    m = LDA(solver="lsqr", shrinkage="auto").fit(Z - (Z @ B.T) @ B, y[ref])

    def dec(idx):
        Q = z(idx)
        return m.decision_function(Q - (Q @ B.T) @ B)
    return dec


ALPHAS = (None, 0.5, 0.8, 0.95, 1.0)  # None = Ledoit-Wolf


def _re_direction(Z, yr, nuis_r, kappa, alpha):
    m0, m1 = Z[yr == 0].mean(0), Z[yr == 1].mean(0)
    C = np.vstack([Z[yr == 0] - m0, Z[yr == 1] - m1])
    if alpha is None:
        Sw = ledoit_wolf(C, assume_centered=True)[0]
    else:  # fixed shrinkage towards the mean variance; alpha = 1 is the nearest-centroid direction
        S = C.T @ C / len(C)
        Sw = (1 - alpha) * S + alpha * np.trace(S) / len(S) * np.eye(len(S))
    Sn = np.zeros_like(Sw)
    for gr in nuis_r:
        for c, mc in ((0, m0), (1, m1)):
            for v in sorted(set(gr[yr == c])):
                s = (yr == c) & (gr == v)
                d = Z[s].mean(0) - mc
                Sn += s.sum() * np.outer(d, d) / len(yr)
    wv = np.linalg.solve(Sw + kappa * Sn, m1 - m0)
    return wv, -wv @ (m0 + m1) / 2


def fit_re(X, w, y, ref, nuis, kappa, alpha=None, env=None):
    """Random-effects LDA. nuis: list of label arrays over all rows (only reference entries are read).
    alpha: covariance shrinkage (None = Ledoit-Wolf, a number = fixed, "cv" = chosen from ALPHAS by
    the AUC of an inner leave-one-environment-out inside the references)."""
    z, _ = standardize(X, w, ref)
    Z, yr, er = z(ref), y[ref], env[ref] if env is not None else None
    nr = [g[ref] for g in nuis]
    if alpha == "cv":
        best = None
        for a in ALPHAS:
            o = np.zeros(len(yr))
            for e in sorted(set(er)):
                tr = er != e
                wv, b = _re_direction(Z[tr], yr[tr], [g[tr] for g in nr], kappa, a)
                o[~tr] = Z[~tr] @ wv + b
            s = auc(o, yr)
            if best is None or s >= best[0]:  # ties go to the stronger shrinkage
                best = (s, a)
        alpha = best[1]
    wv, b = _re_direction(Z, yr, nr, kappa, alpha)
    return lambda idx: z(idx) @ wv + b


def make_fit(variant, ctx, minute):
    """variant: base | rank:<r> | re:<env|envwave|none>:<kappa>[:<alpha|lw|cv>]"""
    p = variant.split(":")
    if p[0] == "base":
        return lda_fit_clean
    if p[0] == "rank":
        return lambda X, w, y, env, ref: fit_rank(X, w, y, env, ref, int(p[1]))
    if p[0] == "re":
        kappa = float(p[2])
        alpha = None if len(p) < 4 or p[3] == "lw" else "cv" if p[3] == "cv" else float(p[3])

        def fit(X, w, y, env, ref):
            nuis = [] if p[1] == "none" else [env]
            if p[1] == "envwave":
                wv = np.full(len(y), -1)
                wv[ref] = waves(minute[ref])  # from the reference collection times only
                nuis.append(wv)
            return fit_re(X, w, y, ref, nuis, kappa, alpha, env)
        return fit
    raise ValueError(variant)


# ----------------------------------------------------------------------------- reporting

def report(tag, D, ctx, protos=("dev->test", "test->dev")):
    y, env, split, groups = ctx.y, ctx.env, ctx.split, ctx.groups
    idx = {"dev->test": np.flatnonzero(split == "test"), "test->dev": np.flatnonzero(split == "development"),
           "loeo": np.arange(len(y))}
    out = {}
    for proto in protos:
        s = summarize(D, y, env, groups, idx[proto])
        out[proto] = s
        for k in NAMES:
            r = s[k]
            print(f"RESULT {tag:22} {proto:9} {k:20} worst k1 {r['k1_worst']:.3f} k3 {r['k3_all_worst']:.3f} | "
                  f"k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} | k3_all A {r['k3_all_astra']:.3f} "
                  f"S {r['k3_all_sol']:.3f} | auc {r['auc']:.3f}", flush=True)
    return out


def devtest(ctx):
    D = {k: np.zeros(len(ctx.y)) for k in NAMES}
    for ref, te in folds(ctx.env, ctx.split).get("devtest", []):
        for k, v in ctx.fold(ref, te).items():
            D[k][te] = v
    return D


def loeo(ctx):
    D = {k: np.zeros(len(ctx.y)) for k in NAMES}
    for ref, te in folds(ctx.env, ctx.split)["loeo"]:
        for k, v in ctx.fold(ref, te).items():
            D[k][te] = v
    return D


# ----------------------------------------------------------------------------- experiments

def label_shift(v, y, env, test):
    """Inside each label: development vs test for one decision (Sol positive)."""
    out = {}
    for c, lab in ((0, "Astra"), (1, "Sol")):
        s = y == c
        diffs = [v[s & test & (env == e)].mean() - v[s & ~test & (env == e)].mean()
                 for e in sorted(set(env[s])) if (s & test & (env == e)).any() and (s & ~test & (env == e)).any()]
        diffs = np.array(diffs)
        tstat = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)) + 1e-12) if len(diffs) > 1 else 0.0
        out[lab] = {"median_dev": round(float(np.median(v[s & ~test])), 3), "median_test": round(float(np.median(v[s & test])), 3),
                    "env_paired_mean_test_minus_dev": round(float(diffs.mean()), 3), "t_over_envs": round(float(tstat), 2),
                    "envs_lower_in_test": f"{int((diffs < 0).sum())}/{len(diffs)}",
                    "auc_test_vs_dev": round(auc(v[s], test[s].astype(int)), 3),
                    "recall_dev": round(float(np.mean((v[s & ~test] > 0) == c)), 3),
                    "recall_test": round(float(np.mean((v[s & test] > 0) == c)), 3)}
    return out


def exp_why():
    sv = Saver("sol_why")
    ctx = Context(CLEAN_T, True)
    y, env, split, rows = ctx.y, ctx.env, ctx.split, ctx.rows
    test = split == "test"
    meta = row_meta(rows)

    # (a) frozen library model on four windows
    lib = library_pair_rows()
    cl = G.fit_cluster(lib, MEMBERS, [0, 1])
    del lib
    frozen = {}
    for a in (0, 250, 500, 740):
        dd = [cluster_decisions(cl, r["numbers"][a:a + 250]) for r in rows]
        for k in dd[0]:
            frozen[f"{k}@{a}"] = np.array([d[k] for d in dd])
    res = {k: label_shift(v, y, env, test) for k, v in frozen.items()}
    sv["a_frozen_library_split_shift"] = res
    for k, r in res.items():
        print(f"FROZEN {k:18} " + " | ".join(f"{lab}: median dev {x['median_dev']:+.2f} test {x['median_test']:+.2f}, "
                                             f"env-paired {x['env_paired_mean_test_minus_dev']:+.2f} (t {x['t_over_envs']:+.1f}, "
                                             f"{x['envs_lower_in_test']} lower), recall dev {x['recall_dev']:.2f} test {x['recall_test']:.2f}"
                                             for lab, x in r.items()), flush=True)

    # (b) feature-space shift between the splits inside each label, as a fraction of the label gap
    allr = np.arange(len(y))
    z, keep = standardize(ctx.Xd, ctx.wd, allr)
    Z = z(allr)
    fam = np.concatenate([[f] * n for f, n in ctx.fam])[keep]
    B = env_basis(Z, y, env)
    P = Z - (Z @ B.T) @ B
    wl = LDA(solver="lsqr", shrinkage="auto").fit(P, y).coef_[0]
    gap = P[y == 1].mean(0) - P[y == 0].mean(0)
    shift_b = {}
    for f in [None] + [f for f, _ in ctx.fam]:
        cols = np.ones(len(fam), bool) if f is None else fam == f
        r = {"lda_gap": round(float(wl[cols] @ gap[cols]), 3)}
        for c, lab in ((0, "Astra"), (1, "Sol")):
            d = P[(y == c) & test].mean(0) - P[(y == c) & ~test].mean(0)
            r[f"{lab}_test_minus_dev_along_lda_over_gap"] = round(float(wl[cols] @ d[cols] / (wl @ gap)), 3)
            r[f"{lab}_cos_with_mean_gap"] = round(float(d[cols] @ gap[cols] / (np.linalg.norm(d[cols]) * np.linalg.norm(gap[cols]) + 1e-12)), 3)
            r[f"{lab}_shift_norm_over_gap_norm"] = round(float(np.linalg.norm(d[cols]) / (np.linalg.norm(gap[cols]) + 1e-12)), 3)
        shift_b[f or "all"] = r
        print(f"SHIFT {f or 'all':10} {json.dumps(r)}", flush=True)
    sv["b_feature_shift"] = shift_b

    # (c) can the split be predicted inside one label? (LOEO over environments, dense clean LDA)
    yb = test.astype(int)
    det = {}
    for c, lab in ((0, "Astra"), (1, "Sol")):
        rc = np.flatnonzero(y == c)
        o = np.zeros(len(y))
        for e in sorted(set(env[rc])):
            ref, te = rc[env[rc] != e], rc[env[rc] == e]
            o[te] = lda_fit_clean(ctx.Xd, ctx.wd, yb, env, ref)(te)
        det[lab] = round(auc(o[rc], yb[rc]), 3)
    sv["c_split_detectability_auc"] = det
    print(f"SPLIT-DETECT dense LOEO AUC (0.5 = no batch difference): {det}", flush=True)

    # (d) development -> test and back: query mean minus reference mean along the fitted discriminant
    dec = {}
    for ref, te in folds(env, split).get("devtest", []):
        direction = f"{split[ref[0]][:3]}->{split[te[0]][:4]}"
        theta0 = FS.fit_base(ctx.st, np.isin(np.arange(len(y)), ref), lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
        U = FS.score_vectors(ctx.st, theta0)
        cols = [U[:, FS.OFF[g]:FS.OFF[g] + FS.GROUPS[g]] for g in SG]
        sources = {"dense": (ctx.Xd, ctx.wd, np.concatenate([[f + f"[{i}]" for i in range(n)] for f, n in ctx.fam])),
                   "score": (np.hstack(cols), np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols]),
                             np.concatenate([[g + f"[{i}]" for i in range(FS.GROUPS[g])] for g in SG]))}
        for name, (X, w, names) in sources.items():
            z, keep = standardize(X, w, ref)
            Zr = z(ref)
            B = env_basis(Zr, y[ref], env[ref])
            pr = lambda idx: z(idx) - (z(idx) @ B.T) @ B  # noqa: E731
            m = LDA(solver="lsqr", shrinkage="auto").fit(pr(ref), y[ref])
            coef = m.coef_[0]
            nm = names[keep]
            fam_of = np.array([s.split("[")[0] for s in nm])
            Pr, Pt = pr(ref), pr(te)
            r = {"ref_mean_decision": {lab: round(float(np.mean(Pr[y[ref] == c] @ coef + m.intercept_[0])), 3) for c, lab in ((0, "Astra"), (1, "Sol"))},
                 "query_mean_decision": {lab: round(float(np.mean(Pt[y[te] == c] @ coef + m.intercept_[0])), 3) for c, lab in ((0, "Astra"), (1, "Sol"))}}
            for c, lab in ((0, "Astra"), (1, "Sol")):
                contrib = coef * (Pt[y[te] == c].mean(0) - Pr[y[ref] == c].mean(0))
                r[f"{lab}_shift_by_family"] = {f: round(float(contrib[fam_of == f].sum()), 3) for f in sorted(set(fam_of))}
                top = np.argsort(contrib if c == 1 else -contrib)[:12]  # moves this label towards the other one
                raw_idx = np.flatnonzero(keep)[top]
                r[f"{lab}_top_features_towards_other"] = [
                    {"feature": nm[j], "contribution": round(float(contrib[j]), 3), "coef": round(float(coef[j]), 3),
                     "raw_mean A_dev A_test S_dev S_test": [round(float(X[(y == cc) & (split == ss), ri].mean()), 5)
                                                            for cc in (0, 1) for ss in ("development", "test")]}
                    for j, ri in zip(top, raw_idx)]
            dec[f"{direction} {name}"] = r
            print(f"DECOMP {direction} {name}: ref mean {r['ref_mean_decision']} query mean {r['query_mean_decision']} | "
                  f"Sol shift by family {r['Sol_shift_by_family']} | Astra shift by family {r['Astra_shift_by_family']}", flush=True)
            for lab in ("Sol", "Astra"):
                print(f"DECOMP {direction} {name} {lab} top: " + "; ".join(
                    f"{t['feature']} {t['contribution']:+.3f} raw {t['raw_mean A_dev A_test S_dev S_test']}"
                    for t in r[f"{lab}_top_features_towards_other"][:8]), flush=True)
    sv["d_devtest_decomposition"] = dec

    # (e) one line per response
    D = devtest(ctx)
    report("why/base", D, ctx)
    fmt = np.where(meta["out_tokens"] < 2500, "compact", "spaced")
    N = np.array([len(ctx.rows[i]["numbers"]) for i in range(len(y))])
    tab = []
    for i in np.lexsort((D["dense+score+win0lib"], split, y)):
        tab.append({"row": int(i), "label": ["Astra", "Sol"][y[i]], "split": split[i][:4], "env": env[i][-2:],
                    "minute": round(float(meta["minute"][i]), 1), "format": fmt[i], "out_tokens": int(meta["out_tokens"][i]),
                    "reasoning": int(meta["reasoning"][i]), "seconds": float(meta["seconds"][i]), "N": int(N[i]),
                    **{k: round(float(D[k][i]), 2) for k in NAMES},
                    **{k: round(float(frozen[k][i]), 2) for k in ("local@0", "distribution@0", "score@0")}})
    sv["e_rows"] = tab
    for t in tab:
        print("ROW " + " ".join(f"{k}={v}" for k, v in t.items()), flush=True)
    # recall by output format and by wave (diagnosis)
    wv = waves(meta["minute"])
    by = {}
    for c, lab in ((0, "Astra"), (1, "Sol")):
        for s in ("development", "test"):
            for f in ("compact", "spaced"):
                q = (y == c) & (split == s) & (fmt == f)
                if q.any():
                    by[f"{lab} {s[:4]} {f}"] = [int(q.sum()), round(float(np.mean((D['dense+score+win0lib'][q] > 0) == c)), 3)]
            for v in sorted(set(wv[(y == c) & (split == s)])):
                q = (y == c) & (split == s) & (wv == v)
                by[f"{lab} {s[:4]} wave{v}"] = [int(q.sum()), round(float(np.mean((D['dense+score+win0lib'][q] > 0) == c)), 3)]
    sv["e_recall_by_format_and_wave"] = by
    print(f"RECALL by format / wave (dense+score+win0lib, devtest): {json.dumps(by)}", flush=True)

    # (f) the same prompt was sent once in each split: paired test - development differences per label.
    # A whole-batch shift moves every pair a little; a mixture (some test responses from another
    # deployment) moves a few pairs a lot and leaves the rest, and those rows should then also look
    # like the other label in the independent evidence (frozen model, out-of-range 356s, which the
    # parser drops so no feature sees them).
    from scipy.stats import fisher_exact, wilcoxon
    srcs = {**{k: frozen[k] for k in ("local@0", "distribution@0", "score@0", "local@250", "distribution@500")},
            **{f"devtest {k}": D[k] for k in ("dense", "score", "dense+score+win0lib")}}
    paired = {}
    for c, lab in ((0, "Astra"), (1, "Sol")):
        keys = sorted({(env[i], meta["sha"][i]) for i in np.flatnonzero(y == c)})
        pairs = []
        for e, h in keys:
            q = (y == c) & (env == e) & (meta["sha"] == h)
            dv, ts = np.flatnonzero(q & ~test), np.flatnonzero(q & test)
            if len(dv) == 1 and len(ts) == 1:
                pairs.append((dv[0], ts[0]))
        pairs = np.array(pairs)
        for k, v in srcs.items():
            d = v[pairs[:, 1]] - v[pairs[:, 0]]
            p = float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else None
            paired[f"{lab} {k}"] = {"pairs": len(d), "mean_test_minus_dev": round(float(d.mean()), 3),
                                    "median": round(float(np.median(d)), 3), "n_lower_in_test": int((d < 0).sum()),
                                    "wilcoxon_p": round(p, 4) if p is not None else None,
                                    "sd_dev": round(float(v[pairs[:, 0]].std()), 3)}
    sv["f_prompt_paired"] = paired
    for k, r in paired.items():
        print(f"PAIRED {k:28} {json.dumps(r)}", flush=True)
    oor = meta["out_of_range"] > 0
    mix = {}
    for c, lab in ((0, "Astra"), (1, "Sol")):
        a = [[int((oor & (y == c) & (split == s)).sum()), int((~oor & (y == c) & (split == s)).sum())] for s in ("development", "test")]
        mix[f"{lab} rows with a 356 (dev, test)"] = [a[0][0], a[1][0]]
        mix[f"{lab} fisher_p dev vs test"] = round(float(fisher_exact(a).pvalue), 4)
        for s in ("development", "test"):
            for k in ("local@0", "devtest dense+score+win0lib"):
                v = srcs[k]
                q = (y == c) & (split == s)
                mix[f"{lab} {s[:4]} {k} mean with/without 356"] = [round(float(v[q & oor].mean()), 3) if (q & oor).any() else None,
                                                                   round(float(v[q & ~oor].mean()), 3) if (q & ~oor).any() else None]
    sv["f_out_of_range"] = mix
    print(f"OUT-OF-RANGE {json.dumps(mix)}", flush=True)


def exp_fix(variant, with_loeo):
    sv = Saver(f"sol_fix:{variant}{':loeo' if with_loeo else ''}")
    ctx = Context(CLEAN_T, True)
    meta = row_meta(ctx.rows)
    fit, _, sg = variant.partition("@")  # optional @val.dg.pb.j1: Fisher-score groups of the score source
    if sg:
        ctx.sg = tuple(sg.split("."))
    ctx.fit_override = make_fit(fit, ctx, meta["minute"])
    D = devtest(ctx)
    sv["devtest"] = {"decisions": {k: v.round(4).tolist() for k, v in D.items()},
                     "summary": report(f"fix:{variant}", D, ctx)}
    if with_loeo:
        D = loeo(ctx)
        sv["loeo"] = {"decisions": {k: v.round(4).tolist() for k, v in D.items()},
                      "summary": report(f"fix:{variant}", D, ctx, ("loeo",))}


def interp_stats(x):
    """Readable statistics of one response (numbers only), for explaining the decisions."""
    x = np.asarray(x, int)
    f, n = x.astype(float), len(x)
    d = np.diff(f)
    cnt = np.bincount(x, minlength=400)
    last, gap = {}, np.full(n, n, float)
    for t, v in enumerate(x):
        if v in last:
            gap[t] = t - last[v]
        last[v] = t
    s = {"mean": f.mean(), "sd": f.std(), "skew": float(((f - f.mean()) ** 3).mean() / f.std() ** 3)}
    for k in range(10):
        s[f"share {int(k * 35.5) + 1}-{int((k + 1) * 35.5)}"] = np.mean((f - 1) // 35.5 == k)
    for lo, hi in ((1, 9), (10, 99), (100, 199), (200, 299), (300, 355)):
        s[f"share {lo}-{hi}"] = np.mean((x >= lo) & (x <= hi))
    for u in range(10):
        s[f"last digit {u}"] = np.mean(x % 10 == u)
    s["multiple of 5"] = np.mean(x % 5 == 0)
    s["repdigit (11, 22, 111..)"] = np.mean(np.isin(x, [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333]))
    s["distinct values / 355"] = np.count_nonzero(cnt[1:356]) / 355
    s["top-10 values share"] = np.sort(cnt)[-10:].sum() / n
    s["max count of one value"] = cnt.max()
    for w in (5, 20, 100):
        s[f"repeat within {w}"] = np.mean(gap[1:] <= w)
    s["median gap to same value"] = float(np.median(gap[gap < n])) if (gap < n).any() else n
    s["mean |jump|"] = np.abs(d).mean()
    s["|jump| <= 10"] = np.mean(np.abs(d) <= 10)
    s["|jump| >= 200"] = np.mean(np.abs(d) >= 200)
    s["up-down alternation"] = np.mean(np.sign(d[1:]) * np.sign(d[:-1]) < 0)
    s["lag-1 autocorrelation"] = float(np.corrcoef(f[:-1], f[1:])[0, 1])
    s["same last digit as previous"] = np.mean(x[1:] % 10 == x[:-1] % 10)
    s["same hundred as previous"] = np.mean(x[1:] // 100 == x[:-1] // 100)
    s["mean of first 100"], s["mean of last 100"] = f[:100].mean(), f[-100:].mean()
    s["sd of first 100"], s["sd of last 100"] = f[:100].std(), f[-100:].std()
    return {k: float(v) for k, v in s.items()}


def exp_why2():
    """Which readable properties make the misjudged Sol responses look like Astra?"""
    sv = Saver("sol_why2")
    ctx = Context(CLEAN_T, True)
    y, env, split = ctx.y, ctx.env, ctx.split
    test = split == "test"
    meta = row_meta(ctx.rows)
    D = devtest(ctx)["dense+score+win0lib"]  # each row scored by a model fitted on the other split
    seqs = [r["numbers"][:CLEAN_T] for r in ctx.rows]
    st = [interp_stats(x) for x in seqs]
    names = list(st[0])
    T = np.array([[s[k] for k in names] for s in st])
    bad = (y == 1) & (D <= 0)
    grp = {"A dev": (y == 0) & ~test, "A test": (y == 0) & test, "S dev ok": (y == 1) & ~test & ~bad,
           "S dev bad": (y == 1) & ~test & bad, "S test ok": (y == 1) & test & ~bad, "S test bad": (y == 1) & test & bad}
    print(f"GROUPS {json.dumps({k: int(v.sum()) for k, v in grp.items()})}", flush=True)
    # pooled within-(label, split) sd, label effect, environment consistency, prompt dependence
    cell = [(y == c) & (split == s) for c in (0, 1) for s in ("development", "test")]
    resid = np.vstack([T[m] - T[m].mean(0) for m in cell])
    sd = resid.std(0) + 1e-12
    envs = sorted(set(env))
    pairs = []
    for e in envs:
        for h in sorted(set(meta["sha"][env == e])):
            q = (y == 1) & (env == e) & (meta["sha"] == h)
            if (q & ~test).sum() == 1 and (q & test).sum() == 1:
                pairs.append((np.flatnonzero(q & ~test)[0], np.flatnonzero(q & test)[0]))
    pairs = np.array(pairs)
    rows_out = []
    for j, k in enumerate(names):
        m = {g: float(T[v, j].mean()) for g, v in grp.items() if v.any()}
        label_d = (T[y == 1, j].mean() - T[y == 0, j].mean()) / sd[j]
        env_agree = sum(np.sign(T[(y == 1) & (env == e), j].mean() - T[(y == 0) & (env == e), j].mean()) == np.sign(label_d) for e in envs)
        # where do the misjudged Sol rows sit, inside their own split: 0 = like correct Sol, 1 = like Astra
        tow = {}
        for s_, a_ in (("test", "A test"), ("dev", "A dev")):
            ok, bd = m.get(f"S {s_} ok"), m.get(f"S {s_} bad")
            tow[s_] = round((bd - ok) / (m[a_] - ok), 2) if None not in (ok, bd) and abs(m[a_] - ok) > 1e-12 else None
        bad_vs_ok_d = (T[bad, j].mean() - T[(y == 1) & ~bad, j].mean()) / sd[j]
        prompt_r = float(np.corrcoef(T[pairs[:, 0], j], T[pairs[:, 1], j])[0, 1]) if len(pairs) > 2 else None
        rows_out.append({"stat": k, "label_d (Sol-Astra)/sd": round(float(label_d), 3), "envs_agreeing": f"{env_agree}/{len(envs)}",
                         "bad_vs_ok_Sol_d": round(float(bad_vs_ok_d), 3), "bad_Sol_towards_Astra (test, dev)": [tow["test"], tow["dev"]],
                         "sol_prompt_pair_r": None if prompt_r is None else round(prompt_r, 2),
                         "means": {g: round(v, 4) for g, v in m.items()}})
    rows_out.sort(key=lambda r: -abs(r["label_d (Sol-Astra)/sd"]))
    sv["stats"] = rows_out
    for r in rows_out:
        print(f"STAT {r['stat']:30} label d {r['label_d (Sol-Astra)/sd']:+.2f} ({r['envs_agreeing']} envs) | misjudged Sol vs correct Sol d "
              f"{r['bad_vs_ok_Sol_d']:+.2f}, towards Astra (test, dev) {r['bad_Sol_towards_Astra (test, dev)']} | prompt-pair r "
              f"{r['sol_prompt_pair_r']} | means {json.dumps(r['means'])}", flush=True)
    # the decision itself: prompt dependence of Sol decisions (same prompt, other split)
    r_dec = float(np.corrcoef(D[pairs[:, 0]], D[pairs[:, 1]])[0, 1])
    both_bad = int(np.sum(bad[pairs[:, 0]] & bad[pairs[:, 1]]))
    sv["decision_prompt_pairs"] = {"pairs": len(pairs), "r": round(r_dec, 3), "both_misjudged": both_bad,
                                   "dev_bad": int(bad[pairs[:, 0]].sum()), "test_bad": int(bad[pairs[:, 1]].sum())}
    print(f"PROMPT-PAIRS Sol decision dev vs test, same prompt: {json.dumps(sv.data['decision_prompt_pairs'])}", flush=True)
    # a readable discriminant on these statistics only: does it make the same Sol errors, and why
    yb = y
    for ref, te in folds(env, split).get("devtest", []):
        mu, s0 = T[ref].mean(0), T[ref].std(0) + 1e-12
        lda = LDA(solver="lsqr", shrinkage="auto").fit((T[ref] - mu) / s0, yb[ref])
        z = (T[te] - mu) / s0
        dd = lda.decision_function(z)
        mid = ((T[ref][yb[ref] == 0].mean(0) + T[ref][yb[ref] == 1].mean(0)) / 2 - mu) / s0
        contrib = lda.coef_[0] * (z - mid)
        tag = f"{split[ref[0]][:3]}->{split[te[0]][:4]}"
        yt = yb[te]
        badq = (yt == 1) & (dd <= 0)
        print(f"READABLE-LDA {tag}: auc {auc(dd, yt):.3f} recall A {np.mean(dd[yt == 0] <= 0):.3f} S {np.mean(dd[yt == 1] > 0):.3f}; "
              f"agreement with the main model on Sol: {np.mean((dd[yt == 1] > 0) == (D[te][yt == 1] > 0)):.3f}", flush=True)
        if badq.any():
            avg = contrib[badq].mean(0)
            top = np.argsort(avg)[:10]
            print(f"READABLE-LDA {tag} misjudged Sol, stats pushing towards Astra: " + "; ".join(
                f"{names[i]} {avg[i]:+.2f}" for i in top), flush=True)
    # raw beginnings of the misjudged test Sol responses and of their same-prompt development responses
    for i in np.flatnonzero(grp["S test bad"]):
        pr = pairs[pairs[:, 1] == i]
        j = pr[0, 0] if len(pr) else None
        print(f"SEQ test Sol row {i} env {env[i][-2:]} decision {D[i]:+.2f}: {seqs[i][:40]}", flush=True)
        if j is not None:
            print(f"SEQ   same prompt, dev Sol row {j} decision {D[j]:+.2f}: {seqs[j][:40]}", flush=True)


def exp_curve(kind, n, draw, variant):
    """Reference-count curve of one fix variant (LOEO). kind env: the references are all rows of n
    random other environments; kind rep: n random rows per label from each of the 11 other
    environments. Same random draws as curve_pair.py ref / rep, so the subsets match run 4."""
    sv = Saver(f"sol_curve:{kind}:{n}:{draw}:{variant}")
    ctx = Context(CLEAN_T, True)
    meta = row_meta(ctx.rows)
    fit, _, sg = variant.partition("@")
    if sg:
        ctx.sg = tuple(sg.split("."))
    ctx.fit_override = make_fit(fit, ctx, meta["minute"])
    if kind == "env":
        rng = np.random.default_rng(1000 + 100 * n + draw)

        def pick(e, ref):
            others = np.array(sorted(set(ctx.env[ref])))
            return ref[np.isin(ctx.env[ref], rng.choice(others, min(n, len(others)), replace=False))]
    else:
        rng = np.random.default_rng(2000 + 100 * n + draw)

        def pick(e, ref):
            keep = []
            for ee in sorted(set(ctx.env[ref])):
                for c in (0, 1):
                    cell = ref[(ctx.env[ref] == ee) & (ctx.y[ref] == c)]
                    keep.extend(rng.choice(cell, min(n, len(cell)), replace=False))
            return np.sort(np.array(keep))
    run_loeo(ctx, pick, sv, "loeo")
    sv["summary"] = report(f"curve:{kind}:{n}:{draw}", sv.data["loeo"]["decisions"], ctx, ("loeo",))


def run_one(spec):
    log(f"experiment {spec} (smoke={SMOKE})")
    if spec == "why":
        exp_why()
    elif spec == "why2":
        exp_why2()
    elif spec.startswith("fix:"):
        v = spec[4:]
        with_loeo = v.endswith(":loeo")
        exp_fix(v[:-5] if with_loeo else v, with_loeo)
    elif spec.startswith("curve:"):  # curve:<env|rep>:<n>:<draw>:<variant>
        _, kind, n, draw, variant = spec.split(":", 4)
        exp_curve(kind, int(n), int(draw), variant)
    else:
        raise ValueError(spec)
    log(f"done {spec}")


def main():
    for spec in sys.argv[1:]:
        run_one(spec)


if __name__ == "__main__":
    main()
