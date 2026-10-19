"""Run 18 (2026-10-19): another AI's feature vector, and calibrated pair probabilities.

1. The other AI's proposal (description only, reimplemented here): every response yields a 195-dim vector,
   - 96 dims: first-occurrence order z_1..z_m; a label-free Plackett-Luce weight pi_v per value (shared by both
     labels, MM updates with a Gamma(21, 20) smoothing prior, 40 iterations, fit on the references); at each
     step the available values A_t give q_t(v) ~ pi_v; phi(v) = Legendre 1-12 of the value, last-digit and
     tens-digit one-hots (32); r_t = phi(z_t) - E_q[phi], V_t = Var_q[phi]; for tau in 16, 64, 1e10:
     sum_t e^(-(t-1)/tau) r_t / sqrt(0.1 + sum_t e^(-2(t-1)/tau) V_t)  (per coordinate).
   - 99 dims: per position Legendre 1-8 of the value, Fourier coordinates (9) of the last and the tens digit,
     Legendre 1-3 of the hundreds band, powers 1-4 of |jump| / 355; normalized exponential averages with
     tau in 16, 64, 1e9.
   - standardize on the references, shrinkage LDA (0.5). Score = decision function (positive = Sol).
   It is used alone and as one more component next to v4's (distribution D, fork B, score F); combination
   weights by the same inner leave-one-environment-out logistic as v4.
2. The user's new target: when the attribution is wrong, the reported probability must not be high. Every
   scorer gets a probability for k = 1, 2, 3 responses, from maps fitted on the fold's references only:
     raw     p = sigmoid(sum of d)                                   (what the deliverable does now)
     temp    p = sigmoid(a_k * sum d)          inner leave-one-environment-out d, groups of k rows from k envs
     platt   p = sigmoid(a_k * sum d + b_k)    same data
     lobo    p = sigmoid(a_k * sum d + b_k)    inner leave-one-batch-out d (the references' own date shift)

Protocols (references only for every fit; environment / batch only form folds):
  loeo      12 folds, leave one environment out (all rows of the held-out environment leave the references)
  lobo      3 folds, leave one training batch out (references from the other two dates): a date-shift check
            on training data only
  outside   fitted on all training rows, scored on old_outside + fresh_outside (other dates); reported only,
            never used for a choice
Choices are made on loeo + lobo only (pre-specified): a scorer replaces v4 if its worse-label k3 and AUC are
not lower than v4's on both; the calibration map is the one with the fewest confident errors (wrong model
>= 0.8) at k = 3 on lobo among maps whose loeo / lobo log loss is not worse than raw.

Reported per protocol x scorer x map x k (all same-label groups of k rows from k different environments),
per label and the worse label: recall; among errors the share with p(wrong) >= 0.7 / 0.8 / 0.9 and the
median / 90% quantile of p(wrong); confident errors over all groups (p(wrong) >= 0.8, >= 0.9); label-balanced
coverage and accuracy of decisions with confidence >= 0.8 / 0.9; label-balanced ECE (10 bins), log loss, Brier;
and a reliability table.

Usage: python research/calib_pair.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.special import expit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.gated_fingerprint as G  # noqa: E402
from research import fisher_score as FS  # noqa: E402
from research.hard_pair import A, ENVS, IA, IS, MEM, OUT_ROWS, S, TR, branch_parts, p, seq, yb  # noqa: E402
from research.runner_pair import auc  # noqa: E402

T0 = time.time()
RES = Path("results")
RES.mkdir(exist_ok=True)
V = 355
BATCH = p.role
MEMBER_IDS = [p.classes[k] for k in MEM]


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- the other AI's 195-dim vector

_a = 2 * (np.arange(1, V + 1) - 0.5) / V - 1
PHI = np.hstack([legvander(_a, 12)[:, 1:], np.eye(10)[np.arange(1, V + 1) % 10],
                 np.eye(10)[(np.arange(1, V + 1) // 10) % 10]])  # 355 x 32
TAU_R = (16.0, 64.0, 1e10)
TAU_S = (16.0, 64.0, 1e9)


def first_order(x):
    return np.array([v - 1 for v in dict.fromkeys(int(t) for t in x) if 1 <= v <= V], dtype=int)


def fit_pl(seqs, lam=20.0, iters=40):
    """Shared Plackett-Luce weights over the first-occurrence orders (MM updates, Gamma(lam + 1, lam) prior)."""
    Z = [first_order(x) for x in seqs]
    c = np.bincount(np.concatenate(Z), minlength=V).astype(float)
    pi = np.ones(V)
    for _ in range(iters):
        D = np.zeros(V)
        tot = pi.sum()
        for z in Z:
            w = pi[z]
            avail = tot - (np.cumsum(w) - w)
            cum = np.cumsum(1.0 / avail)
            contrib = np.full(V, cum[-1])
            contrib[z] = cum
            D += contrib
        pi = (c + lam) / (D + lam)
        pi /= pi.mean()
    return pi


def pl_residual(x, pi):
    z = first_order(x)
    w = pi[z]
    P1, P2 = w[:, None] * PHI[z], w[:, None] * PHI[z] ** 2
    S0 = pi.sum() - (np.cumsum(w) - w)
    S1 = PHI.T @ pi - (np.cumsum(P1, 0) - P1)
    S2 = (PHI ** 2).T @ pi - (np.cumsum(P2, 0) - P2)
    E = S1 / S0[:, None]
    Var = np.maximum(S2 / S0[:, None] - E ** 2, 0.0)
    r = PHI[z] - E
    t = np.arange(len(z))
    out = []
    for tau in TAU_R:
        g = np.exp(-t / tau)
        out.append((g @ r) / np.sqrt(0.1 + (g ** 2) @ Var))
    return np.concatenate(out)


def _fourier(d):
    return np.hstack([np.cos(2 * np.pi * np.outer(d, np.arange(1, 6)) / 10),
                      np.sin(2 * np.pi * np.outer(d, np.arange(1, 5)) / 10)])


def time_stats(x):
    x = np.clip(np.asarray(x, dtype=float), 1, V)
    xi = x.astype(int)
    M = np.hstack([legvander(2 * (x - 1) / 354 - 1, 8)[:, 1:], _fourier(xi % 10), _fourier((xi // 10) % 10),
                   legvander(2 * np.floor(x / 100) / 3 - 1, 3)[:, 1:]])
    J = np.abs(np.diff(x)) / V
    Jp = np.column_stack([J ** k for k in range(1, 5)])
    out = []
    for tau in TAU_S:
        g = np.exp(-np.arange(len(x)) / tau)
        gj = np.exp(-np.arange(1, len(x)) / tau)
        out += [g @ M / g.sum(), gj @ Jp / gj.sum()]
    return np.concatenate(out)


TSTAT = {}


def oai_vectors(idx, pi):
    for i in idx:
        if i not in TSTAT:
            TSTAT[i] = time_stats(seq(i))
    return (np.array([pl_residual(seq(i), pi) for i in idx]), np.array([TSTAT[i] for i in idx]))


def lda05(Xtr, ytr, Xev):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = LDA(solver="lsqr", shrinkage=0.5).fit((Xtr - mu) / sd, ytr)
    return m.decision_function((Xev - mu) / sd)


# ----------------------------------------------------------------------------- components

COLS = ("D", "B", "F", "N", "N96", "N99")
VARIANTS = {
    "v4": ("D", "B", "F"),
    "new195": ("N",),
    "new195 + v4": ("D", "B", "F", "N"),
    "new195 + score": ("F", "N"),
    "new195 + distribution + score": ("D", "F", "N"),
}


def components(tr, ev, U, R, T):
    """Held-out evidence (Sol - Astra) of rows ev from references tr. U Fisher score vectors, R residual (96),
    T time statistics (99), all as dicts row -> vector (label-free, fitted on the fold's references)."""
    rows_tr = [p.rows[i] for i in tr]
    ym = np.array([MEMBER_IDS.index(r["source"]) for r in rows_tr])
    loc = G.fit_local(rows_tr, p.classes, [A, S])
    br = G.fit_branch([seq(i) for i in tr], ym.tolist(), 2)
    Utr = np.array([U[i] for i in tr])
    sloc = G.fit_score_local(Utr, ym, p.env[tr])
    Rtr, Rev = np.array([R[i] for i in tr]), np.array([R[i] for i in ev])
    Ttr, Tev = np.array([T[i] for i in tr]), np.array([T[i] for i in ev])
    n = lda05(np.hstack([Rtr, Ttr]), yb[tr], np.hstack([Rev, Tev]))
    n96, n99 = lda05(Rtr, yb[tr], Rev), lda05(Ttr, yb[tr], Tev)
    out = np.zeros((len(ev), len(COLS)))
    for k, i in enumerate(ev):
        x = seq(i)
        Dv, Fv = G._dense_logits(loc, x), G._score_logits_from_vector(sloc, U[i])
        bp = branch_parts(br, x)
        out[k] = [Dv[IS] - Dv[IA], bp["bigram"] + bp["prefix"], Fv[IS] - Fv[IA], n[k], n96[k], n99[k]]
    return out


# ----------------------------------------------------------------------------- calibration

def logistic1(s, y, w, intercept=True):
    """Weighted logistic regression of y on one score (Newton)."""
    X = np.column_stack([s, np.ones_like(s)]) if intercept else s[:, None]
    beta = np.zeros(X.shape[1])
    beta[0] = 1.0
    for _ in range(100):
        pz = expit(X @ beta)
        g = X.T @ ((y - pz) * w) - 1e-6 * beta
        H = (X * (pz * (1 - pz) * w)[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return (float(beta[0]), float(beta[1]) if intercept else 0.0)


def sample_groups(idx, k, rng, n=30000):
    """Random same-label groups of k rows from k different environments (both labels), as index arrays."""
    out, lab = [], []
    for c in (0, 1):
        rows = idx[yb[idx] == c]
        if k == 1:
            out.append(rows[:, None])
            lab.append(np.full(len(rows), c))
            continue
        if len(set(p.env[rows])) < k:
            continue
        g = rows[rng.integers(0, len(rows), size=(4 * n, k))]
        ok = np.ones(len(g), bool)
        for a, b in itertools.combinations(range(k), 2):
            ok &= p.env[g[:, a]] != p.env[g[:, b]]
        g = g[ok][:n]
        out.append(g)
        lab.append(np.full(len(g), c))
    return np.vstack(out), np.concatenate(lab)


def fit_maps(d_in, idx_in, rng):
    """Calibration maps (a_k, b_k), k = 1..3, from held-out d of the references idx_in (label-balanced)."""
    maps = {}
    for k in (1, 2, 3):
        g, lab = sample_groups(idx_in, k, rng)
        s = d_in[g].sum(1)
        w = np.where(lab == 1, 0.5 / max(1, (lab == 1).sum()), 0.5 / max(1, (lab == 0).sum())) * len(lab)
        maps[k] = {"temp": logistic1(s, lab, w, intercept=False), "platt": logistic1(s, lab, w)}
    return maps


# ----------------------------------------------------------------------------- folds

def fold_run(ref, te, inner_lobo=True):
    """v4-style fit on ref: inner leave-one-environment-out components (weights, temp/platt maps), inner
    leave-one-batch-out components (lobo map), and held-out components of te."""
    steps = FS.Steps([seq(i) for i in ref])
    theta = FS.fit_base(steps, lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
    Uref = FS.score_vectors(steps, theta)[:, G._score_columns(G.SCORE_GROUPS)]
    del steps
    Ute = FS.score_vectors(FS.Steps([seq(i) for i in te]), theta)[:, G._score_columns(G.SCORE_GROUPS)]
    U = {**dict(zip(ref, Uref)), **dict(zip(te, Ute))}
    pi = fit_pl([seq(i) for i in ref])
    both = np.concatenate([ref, te])
    Rm, Tm = oai_vectors(both, pi)
    R, T = dict(zip(both, Rm)), dict(zip(both, Tm))
    full = np.full((len(p.rows), len(COLS)), np.nan)
    for e in sorted(set(p.env[ref])):
        h = ref[p.env[ref] == e]
        full[h] = components(ref[p.env[ref] != e], h, U, R, T)
    W = {v: fit_weights(full[ref], yb[ref], [COLS.index(c) for c in use]) for v, use in VARIANTS.items()}
    rng = np.random.default_rng(0)
    maps = {}
    for v, use in VARIANTS.items():
        d_in = np.full(len(p.rows), np.nan)
        d_in[ref] = full[ref][:, [COLS.index(c) for c in use]] @ W[v]
        maps[v] = {"loeo": fit_maps(d_in, ref, rng)}
    if inner_lobo:
        fb = np.full((len(p.rows), len(COLS)), np.nan)
        for b in sorted(set(BATCH[ref])):
            h = ref[BATCH[ref] == b]
            fb[h] = components(ref[BATCH[ref] != b], h, U, R, T)
        for v, use in VARIANTS.items():
            d_b = np.full(len(p.rows), np.nan)
            d_b[ref] = fb[ref][:, [COLS.index(c) for c in use]] @ W[v]
            maps[v]["lobo"] = fit_maps(d_b, ref, rng)
    comp_te = components(ref, te, U, R, T)
    return comp_te, W, maps


def fit_weights(comp, yv, cols):
    """Combination weights (no intercept) by logistic likelihood, as hard_pair.fit_weights; cols are indices."""
    from scipy.optimize import minimize
    Z = comp[:, cols]

    def nll(w):
        s = Z @ w
        return -(np.where(yv == 1, s, 0) - np.logaddexp(0, s)).sum() + 1e-3 * (w ** 2).sum()
    return minimize(nll, np.ones(len(cols)), method="L-BFGS-B").x


# ----------------------------------------------------------------------------- metrics

CALS = ("raw", "temp", "platt", "lobo")
BINS = np.linspace(0.5, 1.0, 6)


def group_logits(contrib, idx, k):
    """All same-label groups of k rows from k different environments among idx: summed contributions per label."""
    out = {}
    for c in (0, 1):
        cells = [contrib[idx[(p.env[idx] == e) & (yb[idx] == c)]] for e in sorted(set(p.env[idx]))]
        cells = [x for x in cells if len(x)]
        parts = []
        for comb in itertools.combinations(range(len(cells)), k):
            s = cells[comb[0]]
            for j in comb[1:]:
                s = s[..., None] + cells[j].reshape((1,) * s.ndim + (-1,))
            parts.append(s.ravel())
        out[c] = np.concatenate(parts) if parts else np.zeros(0)
    return out


def prob_metrics(L):
    """L: {label: logits (positive = Sol)} -> calibration / error-confidence metrics."""
    res, pt = {}, {}
    for c in (0, 1):
        x = L[c]
        p_true = expit(x) if c == 1 else expit(-x)
        pt[c] = p_true
        corr = (x > 0) if c == 1 else (x < 0)
        pw = 1 - p_true[~corr]
        res[c] = {"n": int(len(x)), "recall": float(corr.mean()),
                  "err_ge": {t: float(np.mean(pw >= t)) if len(pw) else 0.0 for t in (0.7, 0.8, 0.9)},
                  "err_med": float(np.median(pw)) if len(pw) else float("nan"),
                  "err_q90": float(np.quantile(pw, 0.9)) if len(pw) else float("nan"),
                  "conf_err": {t: float(np.mean((~corr) & (1 - p_true >= t))) for t in (0.8, 0.9)},
                  "logloss": float(np.mean(-np.log(np.clip(p_true, 1e-12, 1)))),
                  "brier": float(np.mean((1 - p_true) ** 2))}
    conf = {c: np.maximum(pt[c], 1 - pt[c]) for c in (0, 1)}
    ok = {c: pt[c] > 0.5 for c in (0, 1)}
    wts = {c: np.full(len(pt[c]), 0.5 / len(pt[c])) for c in (0, 1)}
    cf, okc, wc = (np.concatenate([conf[0], conf[1]]), np.concatenate([ok[0], ok[1]]),
                   np.concatenate([wts[0], wts[1]]))
    ece, rel = 0.0, []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (cf >= lo) & ((cf < hi) if hi < 1 else (cf <= hi))
        if m.any():
            acc, mc, share = np.average(okc[m], weights=wc[m]), np.average(cf[m], weights=wc[m]), wc[m].sum()
            ece += share * abs(acc - mc)
            rel.append([round(float(lo), 1), round(float(share), 4), round(float(mc), 3), round(float(acc), 3)])
    sel = {}
    for t in (0.8, 0.9):
        m = cf >= t
        sel[t] = {"coverage": float(wc[m].sum()), "accuracy": float(np.average(okc[m], weights=wc[m])) if m.any() else float("nan")}
    worst = min((0, 1), key=lambda c: res[c]["recall"])
    return {"astra": res[0], "sol": res[1], "worst_label": "astra" if worst == 0 else "sol",
            "recall_worst": min(res[0]["recall"], res[1]["recall"]),
            "err_ge_worst": {t: max(res[0]["err_ge"][t], res[1]["err_ge"][t]) for t in (0.7, 0.8, 0.9)},
            "conf_err_worst": {t: max(res[0]["conf_err"][t], res[1]["conf_err"][t]) for t in (0.8, 0.9)},
            "err_q90_worst": max(res[0]["err_q90"], res[1]["err_q90"]),
            "err_med_worst": max(res[0]["err_med"], res[1]["err_med"]),
            "ece": float(ece), "logloss": (res[0]["logloss"] + res[1]["logloss"]) / 2,
            "brier": (res[0]["brier"] + res[1]["brier"]) / 2, "confident": sel, "reliability": rel}


def contributions(d, fold_of, maps_of, v, cal, k):
    """Per-row additive contribution to the group logit: a_k d + b_k / k with the row's own fold map."""
    c = np.full(len(d), np.nan)
    for f, idx in fold_of.items():
        if cal == "raw":
            a, b = 1.0, 0.0
        else:
            a, b = maps_of[f][v]["lobo" if cal == "lobo" else "loeo"][k]["temp" if cal == "temp" else "platt"]
        c[idx] = a * d[idx] + b / k
    return c


# ----------------------------------------------------------------------------- main

def main():
    log(f"rows: train {len(TR)} ({len(ENVS)} environments, batches {sorted(set(BATCH[TR]))}), outside {len(OUT_ROWS)}")
    protocols = {
        "loeo": [(e, TR[p.env[TR] != e], TR[p.env[TR] == e]) for e in ENVS],
        "lobo": [(b, TR[BATCH[TR] != b], TR[BATCH[TR] == b]) for b in sorted(set(BATCH[TR]))],
        "outside": [("all training", TR, OUT_ROWS)],
    }
    saved = {"weights": {}, "maps": {}, "d": {}}
    D, FOLD, MAPS, COMP = {}, {}, {}, {}
    for proto, folds in protocols.items():
        D[proto] = {v: np.full(len(p.rows), np.nan) for v in VARIANTS}
        COMP[proto] = np.full((len(p.rows), len(COLS)), np.nan)
        FOLD[proto], MAPS[proto] = {}, {}
        for fname, ref, te in folds:
            comp_te, W, maps = fold_run(ref, te)
            COMP[proto][te] = comp_te
            for v, use in VARIANTS.items():
                D[proto][v][te] = comp_te[:, [COLS.index(c) for c in use]] @ W[v]
            FOLD[proto][str(fname)], MAPS[proto][str(fname)] = te, maps
            saved["weights"][f"{proto}/{fname}"] = {v: np.round(W[v], 4).tolist() for v in VARIANTS}
            saved["maps"][f"{proto}/{fname}"] = maps
            m3 = {v: [round(x, 3) for x in maps[v]["loeo"][3]["platt"] + maps[v]["lobo"][3]["platt"]] for v in VARIANTS}
            log(f"{proto} fold {fname}: refs {len(ref)} test {len(te)}; k3 platt (a, b) loeo / lobo {m3}")
        (RES / "calib_pair.json").write_text(json.dumps(saved, indent=1, default=float))

    sets = {"loeo": ("loeo", TR), "lobo": ("lobo", TR), "outside": ("outside", OUT_ROWS),
            "old_outside": ("outside", OUT_ROWS[BATCH[OUT_ROWS] == "old_outside"]),
            "fresh_outside": ("outside", OUT_ROWS[BATCH[OUT_ROWS] == "fresh_outside"])}
    summary = {}
    for tag, (proto, idx) in sets.items():
        for c in COLS:
            print(f"COMP {tag:13} {c:4} auc {auc(COMP[proto][idx, COLS.index(c)], yb[idx]):.3f}", flush=True)
        for v in VARIANTS:
            a1 = auc(D[proto][v][idx], yb[idx])
            for cal in CALS:
                for k in (1, 2, 3):
                    c = contributions(D[proto][v], FOLD[proto], MAPS[proto], v, cal, k)
                    Lg = group_logits(c, idx, k)
                    if min(len(Lg[0]), len(Lg[1])) == 0:
                        continue
                    m = prob_metrics(Lg)
                    m["auc_k1"] = a1
                    summary[f"{tag}|{v}|{cal}|k{k}"] = m
                    print(f"CAL {tag:13} {v:30} {cal:5} k{k} | rec A {m['astra']['recall']:.3f} S {m['sol']['recall']:.3f} "
                          f"worst {m['recall_worst']:.3f} | auc1 {a1:.3f} | err p>=.7/.8/.9 "
                          f"A {m['astra']['err_ge'][0.7]:.2f}/{m['astra']['err_ge'][0.8]:.2f}/{m['astra']['err_ge'][0.9]:.2f} "
                          f"S {m['sol']['err_ge'][0.7]:.2f}/{m['sol']['err_ge'][0.8]:.2f}/{m['sol']['err_ge'][0.9]:.2f} "
                          f"| err med/q90 A {m['astra']['err_med']:.2f}/{m['astra']['err_q90']:.2f} "
                          f"S {m['sol']['err_med']:.2f}/{m['sol']['err_q90']:.2f} "
                          f"| conf-err>=.8/.9 A {m['astra']['conf_err'][0.8]:.3f}/{m['astra']['conf_err'][0.9]:.3f} "
                          f"S {m['sol']['conf_err'][0.8]:.3f}/{m['sol']['conf_err'][0.9]:.3f} "
                          f"| conf>=.8 cov {m['confident'][0.8]['coverage']:.2f} acc {m['confident'][0.8]['accuracy']:.3f} "
                          f"conf>=.9 cov {m['confident'][0.9]['coverage']:.2f} acc {m['confident'][0.9]['accuracy']:.3f} "
                          f"| ece {m['ece']:.3f} ll {m['logloss']:.3f} brier {m['brier']:.3f}", flush=True)
                    if k in (1, 3):
                        print(f"REL {tag:13} {v:30} {cal:5} k{k} [bin_lo, share, mean conf, accuracy] {m['reliability']}", flush=True)
        (RES / "calib_pair_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    saved["d"] = {proto: {v: {str(i): round(float(D[proto][v][i]), 4) for i in np.flatnonzero(np.isfinite(D[proto][v]))}
                          for v in VARIANTS} for proto in D}
    saved["components"] = {proto: {str(i): np.round(COMP[proto][i], 4).tolist() for i in np.flatnonzero(np.isfinite(COMP[proto][:, 0]))}
                           for proto in COMP}
    (RES / "calib_pair.json").write_text(json.dumps(saved, indent=1, default=float))
    log("done")


if __name__ == "__main__":
    main()
