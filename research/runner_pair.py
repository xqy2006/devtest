"""GPT-6 Astra vs GPT-6.1 Sol on the N=1000 pilot: experiments for a GitHub-hosted runner.

One workflow run executes every experiment below as a separate job; each job writes
results/<experiment>.json (rewritten after every step, so a timeout keeps partial results).

Data: data/pilot_4router_1000/accepted.jsonl (144 rows, 12 environments, 48 native triples,
numbers = what the production parser reads from the text) and the library (historical rows +
10-11 Codex supplement; only its Astra / Sol training rows are used here).
Evaluation: queries are always rows of a held-out pilot environment (LOEO, 12 folds) or the
other split (development <-> test, 2 folds). Nothing about a query (label, environment, batch)
is used to score it. Decision = Sol logit - Astra logit; k1 = sign, k3 = sign of the sum over a
native triple, AUC threshold-free, d' = sqrt(2) * Phi^-1(AUC).

Experiments (argument after the colon):
  v4_matched:t      deliverable v4 cluster model (distribution + branch + Fisher score, inner
                    env-CV weights) refitted on pilot references cut to t numbers, queries cut to t
  v4_matched_lib:t  same, references = pilot references (t) + library Astra/Sol rows (~250)
  window_sum        responses cut into four 250-number windows, each scored by a cluster model
                    whose references are ~250 long: (a) frozen library model; (b) library +
                    windows of the pilot reference rows (LOEO); (c) pilot windows only (LOEO);
                    evidence summed over windows
  families          dense LDA at t=1000 per feature family, the v3 set, all families, and an
                    unbiased nested forward selection (inner LOEO on the references only)
  perm              permutation null of the LOEO dense-LDA AUC at t=218 and t=1000
                    (labels shuffled within environment)
  ref_curve:k       v4 cluster model at t=1000 with references from k random other environments
  score_variants    Fisher-score LDA at t=1000 with different score groups (base model fitted on
                    the references only, label-free)

Usage: python research/runner_pair.py <experiment>[:<arg>]
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "production"))
warnings.filterwarnings("ignore")
import fingerprint as fp  # noqa: E402
import research.gated_fingerprint as G  # noqa: E402
from research import fisher_score as FS  # noqa: E402
from research.pilot_length_matched import dense_part  # noqa: E402
from research.seqfeatures import SEQ_FAMILIES, extract_seq, ordered, value  # noqa: E402

PILOT = ROOT / "data/pilot_4router_1000/accepted.jsonl"
OUT = ROOT / "results"
A, S = "gpt-6-astra", "gpt-6.1-sol"
MEMBERS = [A, S]
SMOKE = bool(os.environ.get("SMOKE"))  # tiny sizes for a local syntax/flow check
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- data

def load_pilot():
    rows = []
    for line in PILOT.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({"source": r["source"], "condition_id": "pilot:" + r["condition_id"], "split": r["split"],
                         "group_id": r["group_id"], "numbers": fp.parse_numbers(r["text"])})
    if SMOKE:  # four environments only
        keep = sorted({r["condition_id"] for r in rows})[:4]
        rows = [r for r in rows if r["condition_id"] in keep]
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r["source"], r["group_id"]), []).append(i)
    groups = np.array([g for g in by.values() if len(g) == 3])
    y = np.array([r["source"] == S for r in rows], int)
    env = np.array([r["condition_id"] for r in rows])
    split = np.array([r["split"] for r in rows])
    return rows, y, env, split, groups


def library_pair_rows():
    from research.data import Panel, extended_panel
    lib = extended_panel(Panel())
    out = [{"source": lib.rows[i]["source"], "condition_id": "lib:" + str(lib.env[i]),
            "numbers": list(map(int, lib.rows[i]["numbers"]))}
           for i in lib.train if lib.classes[lib.y[i]] in MEMBERS]
    return out[::8] if SMOKE else out


def cut(rows, t):
    return [dict(r, numbers=r["numbers"][:t]) for r in rows]


def folds(env, split):
    envs = sorted(set(env))
    f = {"loeo": [(np.flatnonzero(env != e), np.flatnonzero(env == e)) for e in envs],
         "devtest": [(np.flatnonzero(split == a), np.flatnonzero(split == b)) for a, b in (("development", "test"), ("test", "development"))]}
    if SMOKE:
        f = {"loeo": f["loeo"][:2]}
    return f


# ----------------------------------------------------------------------------- metrics

def auc(d, y):
    r = rankdata(d)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(d, y, groups, mask=None):
    """d: decisions for all pilot rows (rows outside mask are ignored)."""
    m = np.ones(len(y), bool) if mask is None else mask
    g = groups[m[groups].all(1)]
    a = auc(d[m], y[m])
    return {"n": int(m.sum()), "k1": round(float(np.mean((d[m] > 0) == y[m])), 4),
            "k3": round(float(np.mean((d[g].sum(1) > 0) == y[g[:, 0]])), 4) if len(g) else None,
            "auc": round(a, 4), "dprime": round(float(np.sqrt(2) * norm.ppf(min(max(a, 1e-3), 1 - 1e-3))), 3),
            "recall_astra": round(float(np.mean(d[m][y[m] == 0] <= 0)), 4),
            "recall_sol": round(float(np.mean(d[m][y[m] == 1] > 0)), 4)}


COMPONENTS = ("distribution", "branch", "score", "local")


def cluster_decisions(cl, x):
    a, s = cl["members"].index(A), cl["members"].index(S)
    parts = {"distribution": G._dense_logits(cl, x), "branch": G.branch_scores(cl["branch"], x, len(cl["members"])),
             "score": G._score_logits(cl["score"], x), "local": G.local_logits(cl, x)}
    return {k: float(v[s] - v[a]) for k, v in parts.items()}


class Saver:
    def __init__(self, name):
        self.path, self.data = OUT / f"{name.replace(':', '_')}.json", {"experiment": name, "smoke": SMOKE}
        OUT.mkdir(exist_ok=True)

    def __setitem__(self, k, v):
        self.data[k] = v
        self.data["elapsed_s"] = round(time.time() - T0)
        self.path.write_text(json.dumps(self.data, indent=1))
        log(f"{k}: {json.dumps(v)[:600]}")


# ----------------------------------------------------------------------------- experiments

def exp_v4_matched(t, with_lib=False):
    sv = Saver(f"v4_matched{'_lib' if with_lib else ''}:{t}")
    rows, y, env, split, groups = load_pilot()
    P = cut(rows, t)
    lib = library_pair_rows() if with_lib else []
    for proto, fl in folds(env, split).items():
        D = {k: np.zeros(len(y)) for k in COMPONENTS}
        weights = []
        for j, (ref, te) in enumerate(fl):
            cl = G.fit_cluster([P[i] for i in ref] + lib, MEMBERS, [0, 1])
            weights.append(cl["combine"])
            for i in te:
                for k, v in cluster_decisions(cl, P[i]["numbers"]).items():
                    D[k][i] = v
            log(f"{proto} fold {j + 1}/{len(fl)} done")
        mask = np.zeros(len(y), bool)
        mask[np.concatenate([te for _, te in fl])] = True
        sv[proto] = {k: metrics(D[k], y, groups, mask) for k in COMPONENTS}
        sv[proto + "_weights"] = weights
        sv[proto + "_decisions"] = {k: [round(float(v), 4) for v in D[k]] for k in COMPONENTS}


def windows(x, w=250, n=4):
    return [x[k * w:(k + 1) * w] for k in range(n) if len(x[k * w:(k + 1) * w]) >= 0.8 * w]


def exp_window_sum():
    sv = Saver("window_sum")
    rows, y, env, split, groups = load_pilot()
    lib = library_pair_rows()

    def evaluate(per_window, tag, mask=None):
        """per_window: rows x 4 x components decisions (nan where a window is missing)."""
        res = {}
        for c, k in enumerate(COMPONENTS):
            for w in range(4):
                res[f"{k}|window{w}"] = metrics(np.nan_to_num(per_window[:, w, c]), y, groups, mask)
            for m in range(1, 5):
                res[f"{k}|sum_first{m}"] = metrics(np.nansum(per_window[:, :m, c], 1), y, groups, mask)
        sv[tag] = res

    # (a) frozen library model
    cl = G.fit_cluster(lib, MEMBERS, [0, 1])
    log("frozen library cluster fitted")
    W = np.full((len(rows), 4, len(COMPONENTS)), np.nan)
    for i, r in enumerate(rows):
        for w, xw in enumerate(windows(r["numbers"])):
            W[i, w] = list(cluster_decisions(cl, xw).values())
    evaluate(W, "a_frozen_library")
    sv["a_frozen_library_full_length"] = {k: metrics(np.array([cluster_decisions(cl, r["numbers"])[k] for r in rows]), y, groups) for k in COMPONENTS}

    # (b) library + pilot reference windows, (c) pilot reference windows only; LOEO
    for tag, use_lib in (("b_library_plus_pilot_windows", True), ("c_pilot_windows_only", False)):
        W = np.full((len(rows), 4, len(COMPONENTS)), np.nan)
        fl = folds(env, split)["loeo"]
        for j, (ref, te) in enumerate(fl):
            wrows = [dict(rows[i], numbers=xw) for i in ref for xw in windows(rows[i]["numbers"])]
            cl = G.fit_cluster(wrows + (lib if use_lib else []), MEMBERS, [0, 1])
            for i in te:
                for w, xw in enumerate(windows(rows[i]["numbers"])):
                    W[i, w] = list(cluster_decisions(cl, xw).values())
            log(f"{tag} fold {j + 1}/{len(fl)}")
        mask = np.zeros(len(y), bool)
        mask[np.concatenate([te for _, te in fl])] = True
        evaluate(W, tag, mask)


EXTRA_FAMILIES = {
    "ordered8": lambda x: ordered(x, 8, 16),
    "ordered4x36": lambda x: ordered(x, 4, 36),
    "value71_halves": lambda x: np.concatenate([value(h, 71) for h in np.array_split(x, 2)]),
    "value71_quarters": lambda x: np.concatenate([value(h, 71) for h in np.array_split(x, 4)]),
}
V3 = ("value355", "ordered", "digits", "jump1", "moments")


def family_matrix(seqs, fams):
    F = extract_seq([{"numbers": x} for x in seqs], [f for f in fams if f in SEQ_FAMILIES])
    for f in fams:
        if f in EXTRA_FAMILIES:
            F[f] = np.array([EXTRA_FAMILIES[f](np.asarray(x, np.int64)) for x in seqs])
    return F


def lda_loeo(F, fams, y, env, idx=None):
    """LOEO decisions of dense LDA on families fams, restricted to rows idx (default all)."""
    idx = np.arange(len(y)) if idx is None else idx
    X = np.hstack([F[f][idx] for f in fams])
    w = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in fams])
    d = np.zeros(len(idx))
    for e in sorted(set(env[idx])):
        ref, te = np.flatnonzero(env[idx] != e), np.flatnonzero(env[idx] == e)
        d[te] = dense_part(X, w, y[idx], env[idx], ref)(te)
    return d


def exp_families():
    sv = Saver("families")
    rows, y, env, split, groups = load_pilot()
    fams = list(SEQ_FAMILIES) + list(EXTRA_FAMILIES)
    if SMOKE:
        fams = fams[:4] + ["ordered8"]
    for t in (218, 1000):
        F = family_matrix([r["numbers"][:t] for r in rows], fams)
        res = {f: metrics(lda_loeo(F, [f], y, env), y, groups) for f in fams}
        res["v3"] = metrics(lda_loeo(F, list(V3), y, env), y, groups) if not SMOKE else None
        res["all"] = metrics(lda_loeo(F, fams, y, env), y, groups)
        sv[f"t={t}"] = res
    # nested forward selection at t=1000: the selection sees only the outer references
    F = family_matrix([r["numbers"][:1000] for r in rows], fams)
    d, chosen = np.zeros(len(y)), []
    for e in sorted(set(env)):
        ref, te = np.flatnonzero(env != e), np.flatnonzero(env == e)
        sel, best = [], -1.0
        while len(sel) < 5:
            cand = {f: auc(lda_loeo(F, sel + [f], y, env, ref), y[ref]) for f in fams if f not in sel}
            f, a = max(cand.items(), key=lambda kv: kv[1])
            if a <= best + 1e-3:
                break
            sel.append(f)
            best = a
        chosen.append(sel)
        X = np.hstack([F[f] for f in sel])
        w = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in sel])
        d[te] = dense_part(X, w, y, env, ref)(te)
        log(f"nested env {e}: {sel} inner auc {best:.3f}")
        if SMOKE:
            break
    sv["nested_selection_t=1000"] = {"outer": metrics(d, y, groups), "chosen_per_fold": chosen}


def exp_perm():
    sv = Saver("perm")
    rows, y, env, split, groups = load_pilot()
    rng = np.random.default_rng(20261012)
    n_perm = 5 if SMOKE else 500
    for t in (218, 1000):
        F = family_matrix([r["numbers"][:t] for r in rows], list(V3))
        obs = auc(lda_loeo(F, list(V3), y, env), y)
        null = []
        for b in range(n_perm):
            yp = y.copy()
            for e in sorted(set(env)):
                m = np.flatnonzero(env == e)
                yp[m] = rng.permutation(y[m])
            null.append(auc(lda_loeo(F, list(V3), yp, env), yp))
            if b % 50 == 0:
                log(f"t={t} perm {b}")
        null = np.array(null)
        sv[f"t={t}"] = {"observed_auc": round(obs, 4), "null_mean": round(float(null.mean()), 4), "null_sd": round(float(null.std()), 4),
                        "null_q95": round(float(np.quantile(null, .95)), 4), "null_q99": round(float(np.quantile(null, .99)), 4),
                        "p_value": round(float((1 + np.sum(null >= obs)) / (1 + len(null))), 4), "n_perm": n_perm}


def exp_ref_curve(k):
    sv = Saver(f"ref_curve:{k}")
    rows, y, env, split, groups = load_pilot()
    P = cut(rows, 1000)
    envs = np.array(sorted(set(env)))
    draws = []
    for r in range(1 if SMOKE else 2):
        rng = np.random.default_rng(100 * k + r)
        D = {c: np.zeros(len(y)) for c in COMPONENTS}
        for e in envs[:2] if SMOKE else envs:
            keep = rng.choice(envs[envs != e], k, replace=False)
            ref, te = np.flatnonzero(np.isin(env, keep)), np.flatnonzero(env == e)
            cl = G.fit_cluster([P[i] for i in ref], MEMBERS, [0, 1])
            for i in te:
                for c, v in cluster_decisions(cl, P[i]["numbers"]).items():
                    D[c][i] = v
            log(f"draw {r} env {e}")
        draws.append({c: metrics(D[c], y, groups) for c in COMPONENTS})
        sv["draws"] = draws
    sv["mean"] = {c: {m: round(float(np.mean([d[c][m] for d in draws])), 4) for m in ("k1", "k3", "auc")} for c in COMPONENTS}


SCORE_SETS = {"default(val,dg,pb)": ("val", "dg", "pb"), "all": tuple(FS.GROUPS),
              **{g: (g,) for g in FS.GROUPS}, "val+cv": ("val", "cv"), "transitions(j1,j2,bd,jj)": ("j1", "j2", "bd", "jj")}


def exp_score_variants():
    sv = Saver("score_variants")
    rows, y, env, split, groups = load_pilot()
    seqs = [r["numbers"][:1000] for r in rows]
    st = FS.Steps(seqs)
    D = {k: np.zeros(len(y)) for k in SCORE_SETS}
    fl = folds(env, split)["loeo"]
    for j, (ref, te) in enumerate(fl):
        mask = np.zeros(len(y), bool)
        mask[ref] = True
        theta = FS.fit_base(st, mask, lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
        U = FS.score_vectors(st, theta)
        for name, gs in SCORE_SETS.items():
            cols = [U[:, FS.OFF[g]:FS.OFF[g] + FS.GROUPS[g]] for g in gs]
            X = np.hstack(cols)
            w = np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols])
            D[name][te] = dense_part(X, w, y, env, ref)(te)
        log(f"score fold {j + 1}/{len(fl)}")
    mask = np.zeros(len(y), bool)
    mask[np.concatenate([te for _, te in fl])] = True
    sv["loeo_t=1000"] = {k: metrics(v, y, groups, mask) for k, v in D.items()}


def main():
    name, _, arg = sys.argv[1].partition(":")
    log(f"experiment {sys.argv[1]} (smoke={SMOKE})")
    {"v4_matched": lambda: exp_v4_matched(int(arg)),
     "v4_matched_lib": lambda: exp_v4_matched(int(arg), with_lib=True),
     "window_sum": exp_window_sum, "families": exp_families, "perm": exp_perm,
     "ref_curve": lambda: exp_ref_curve(int(arg)), "score_variants": exp_score_variants}[name]()
    log("done")


if __name__ == "__main__":
    main()
