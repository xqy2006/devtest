"""Run 3: how the best pre-specified Astra/Sol combination grows with more references and with
longer responses. This is the evidence needed before asking for more data.

The combination (run 2, `stack_pair.py`) has three sources:
  - dense LDA on the v3 families, with the variance floor;
  - Fisher-score LDA (val, dg, pb), with the variance floor;
  - win0lib: the frozen library cluster model on the first 250 numbers.
Each source is divided by the MAD of its out-of-fold reference decisions and clipped to their
range, and the three are summed with equal weights. Out-of-fold reference decisions come from an
inner leave-one-environment-out inside the references, so nothing about a query is used.

Experiments (one job each):
  ref:k:r   LOEO at t=1000. The references are all rows of k random other environments
            (draw r), so 12k references.
  rep:m:r   LOEO at t=1000. The references come from all 11 other environments, but only m
            random rows per label and environment (draw r), so 22m references. Comparing this
            with ref:k at an equal reference count tells whether new prompts or more responses
            per prompt help more.
  len:t     LOEO and development <-> test, with references and queries cut to t numbers
            (win0lib always uses the first 250).
  libwin    the frozen library cluster model on the first t pilot numbers, t in (218, 250, 300),
            all components; plus a library cut to 218 numbers scoring the first 218.
  diag      why the production parser finds fewer numbers than the collector counted.
A trailing ":clean" (run 4) cuts every response to exactly CLEAN_T = 990 numbers and uses
lda_fit_clean. After run 2 we found that the full-length features leaked the response length
(999 vs 1000 parsed numbers) through structurally empty bins, and that one rare jump could
dominate a decision.
Every job saves its decisions for all pilot rows. `merge` reports each label separately with the
worse label as the headline (k1, native k3, all-combination k3 over same-label triples from three
different environments, with a cluster-bootstrap interval for the len experiments). It also
reports single-response recall split by parsed length.

Usage:
  python research/curve_pair.py <experiment> [<experiment> ...]   (for example ref:5:1:clean len:990:clean diag)
  python research/curve_pair.py merge <dir>                         (reads curve_*.json, writes curve.json)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.runner_pair import (FS, G, MEMBERS, OUT, PILOT, SMOKE, V3, Saver, cluster_decisions,  # noqa: E402
                                  family_matrix, folds, fp, library_pair_rows, load_pilot, log, metrics)
from research.pilot_length_matched import LDA, env_basis  # noqa: E402
from research.stack_pair import SG, lda_fit  # noqa: E402

NAMES = ("dense", "score", "win0lib", "dense+score", "dense+score+win0lib")


def library_window(t=250, lib_cut=None):
    """Frozen library cluster model (no pilot rows) on the first t numbers of every pilot row."""
    rows, *_ = load_pilot()
    lib = library_pair_rows()
    if lib_cut:
        lib = [dict(r, numbers=r["numbers"][:lib_cut]) for r in lib]
    cl = G.fit_cluster(lib, MEMBERS, [0, 1])
    del lib
    return [cluster_decisions(cl, r["numbers"][:t]) for r in rows]


def lda_fit_clean(X, w, y, env, ref, floor=0.1, batch=None):
    """Dense LDA without the two artifacts found after run 2.
    - Columns that are constant in the references are left out. stack_pair.lda_fit applied the
      variance floor before that test, so a structurally empty bin (jump1[35], small negative
      jumps) stayed in with a tiny sd. Its only variation was the smoothing denominator, which is
      the response length, and a single rare jump in a query gave z of about 36.
    - Every query feature is clipped to the range seen in the references (no extrapolation
      beyond the reference support), so one rare event cannot dominate a decision.
    batch (optional): the collection batch of every pilot row (development / test split). Only the
    references' own batches are used: the within-label batch offsets are projected out together
    with the environment offsets, so a shift of a whole batch does not move decisions."""
    R = X[ref]
    lo, hi = R.min(0), R.max(0)
    mu, sd = R.mean(0), R.std(0)
    keep = sd > 1e-12
    sd = np.where(keep, np.maximum(sd, floor * np.median(sd[keep])), np.inf)
    Z = (R - mu) / sd * w
    B = env_basis(Z, y[ref], env[ref])
    if batch is not None and len(set(batch[ref])) > 1:
        Bb = env_basis(Z, y[ref], batch[ref], rank=1)
        B = np.linalg.qr(np.vstack([B, Bb]).T)[0].T
    m = LDA(solver="lsqr", shrinkage="auto").fit(Z - (Z @ B.T) @ B, y[ref])

    def dec(idx):
        Q = (np.clip(X[idx], lo, hi) - mu) / sd * w
        return m.decision_function(Q - (Q @ B.T) @ B)
    return dec


class Context:
    def __init__(self, t=1000, clean=False, batch=False):
        """clean: every response cut to exactly t numbers (t <= 996, the shortest pilot response,
        so the length carries no information) and lda_fit_clean instead of stack_pair.lda_fit."""
        self.rows, self.y, self.env, self.split, self.groups = load_pilot()
        self.clean, self.batch = clean, batch
        seqs = [r["numbers"][:t] for r in self.rows]
        if clean:
            assert len({len(x) for x in seqs}) == 1, "clean mode needs t <= the shortest response"
        F = family_matrix(seqs, list(V3))
        self.Xd = np.hstack([F[f] for f in V3])
        self.fam = [(f, F[f].shape[1]) for f in V3]  # column blocks of Xd, for diagnosis
        self.fit_override = None  # optional fit(X, w, y, env, ref) -> dec, used by sol_pair.py
        self.sg = SG  # Fisher-score groups of the score source (sol_pair.py may change them)
        self.wd = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in V3])
        self.st = FS.Steps(seqs)
        self.win0 = np.array([d["local"] for d in library_window()])
        log(f"context ready (t={t}, {len(seqs)} rows)")

    def fold(self, ref, te):
        """Scaled source decisions and combinations for the query rows te, fitted on ref only."""
        y, env = self.y, self.env
        theta0 = FS.fit_base(self.st, np.isin(np.arange(len(y)), ref), lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
        U = FS.score_vectors(self.st, theta0)
        cols = [U[:, FS.OFF[g]:FS.OFF[g] + FS.GROUPS[g]] for g in self.sg]
        Us = np.hstack(cols)
        wu = np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols])

        if self.fit_override is not None:
            fit = self.fit_override
        elif self.clean:
            fit = (lambda *a: lda_fit_clean(*a, batch=self.split)) if self.batch else lda_fit_clean
        else:
            fit = lambda *a: lda_fit(*a, 0.1)[0]  # noqa: E731

        def sources(r):
            return {"dense": fit(self.Xd, self.wd, y, env, r),
                    "score": fit(Us, wu, y, env, r),
                    "win0lib": lambda idx: self.win0[idx]}
        oof = {k: np.zeros(len(y)) for k in ("dense", "score", "win0lib")}
        for e in sorted(set(env[ref])):
            iref, ite = ref[env[ref] != e], ref[env[ref] == e]
            for k, f in sources(iref).items():
                oof[k][ite] = f(ite)
        sc = {}
        for k, f in sources(ref).items():
            r = oof[k][ref]
            s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
            sc[k] = np.clip(f(te), r.min(), r.max()) / s
        sc["dense+score"] = sc["dense"] + sc["score"]
        sc["dense+score+win0lib"] = sc["dense+score"] + sc["win0lib"]
        return sc


def run_loeo(ctx, pick_ref, sv, key):
    """pick_ref(e, ref_rows) -> the references actually used for held-out environment e."""
    D = {k: np.zeros(len(ctx.y)) for k in NAMES}
    n_ref = []
    for e, (ref, te) in zip(sorted(set(ctx.env)), folds(ctx.env, ctx.split)["loeo"]):
        r = pick_ref(e, ref)
        n_ref.append(len(r))
        for k, v in ctx.fold(r, te).items():
            D[k][te] = v
        log(f"{key}: held-out {e}, {len(r)} references")
    sv[key] = {"n_ref": float(np.mean(n_ref)), "decisions": {k: v.round(4).tolist() for k, v in D.items()}}


CLEAN_T = 990  # every pilot response has at least 996 parsed numbers


def exp_ref(k, draw, clean=False):
    sv = Saver(f"curve_ref{'_clean' if clean else ''}:{k}:{draw}")
    ctx = Context(CLEAN_T, True) if clean else Context()
    if SMOKE:
        k = min(k, 2)
    rng = np.random.default_rng(1000 + 100 * k + draw)

    def pick(e, ref):
        others = np.array(sorted(set(ctx.env[ref])))
        keep = rng.choice(others, min(k, len(others)), replace=False)
        return ref[np.isin(ctx.env[ref], keep)]
    run_loeo(ctx, pick, sv, "loeo")


def exp_rep(m, draw, clean=False):
    sv = Saver(f"curve_rep{'_clean' if clean else ''}:{m}:{draw}")
    ctx = Context(CLEAN_T, True) if clean else Context()
    rng = np.random.default_rng(2000 + 100 * m + draw)

    def pick(e, ref):
        keep = []
        for ee in sorted(set(ctx.env[ref])):
            for c in (0, 1):
                cell = ref[(ctx.env[ref] == ee) & (ctx.y[ref] == c)]
                keep.extend(rng.choice(cell, min(m, len(cell)), replace=False))
        return np.sort(np.array(keep))
    run_loeo(ctx, pick, sv, "loeo")


def exp_len(t, clean=False, batch=False):
    sv = Saver(f"curve_len{'_cleanb' if batch else '_clean' if clean else ''}:{t}")
    ctx = Context(t, clean or batch, batch)
    run_loeo(ctx, lambda e, ref: ref, sv, "loeo")
    D = {k: np.zeros(len(ctx.y)) for k in NAMES}
    for ref, te in folds(ctx.env, ctx.split).get("devtest", []):
        for k, v in ctx.fold(ref, te).items():
            D[k][te] = v
        log(f"devtest fold with {len(ref)} references")
    sv["devtest"] = {"decisions": {k: v.round(4).tolist() for k, v in D.items()}}


def exp_diag():
    """Why does the production parser find fewer numbers than the collector counted?"""
    sv = Saver("curve_diag")
    raw = [json.loads(line) for line in PILOT.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    counts, examples = {}, []
    for i, r in enumerate(raw):
        p, c = fp.parse_numbers(r["text"]), list(r["numbers"])
        key = f"{r['source']}|{r['split']}|collector {len(c)}|parser {len(p)}"
        counts[key] = counts.get(key, 0) + 1
        if len(p) != len(c) and len(examples) < 6:
            k = next((j for j in range(min(len(p), len(c))) if p[j] != c[j]), min(len(p), len(c)))
            examples.append({"row": i, "source": r["source"], "collector_len": len(c), "parser_len": len(p),
                             "first_difference": k, "collector_around": c[max(0, k - 3):k + 4],
                             "parser_around": p[max(0, k - 3):k + 4],
                             "parser_equals_collector_without_first": p == c[1:],
                             "parser_equals_collector_without_last": p == c[:-1],
                             "collector_out_of_range": [v for v in c if not 1 <= v <= 355][:5],
                             "range_violation_count": r.get("range_violation_count"),
                             "text_head": r["text"][:160], "text_tail": r["text"][-160:]})
    sv["counts"] = dict(sorted(counts.items()))
    sv["examples"] = examples


def exp_libwin():
    sv = Saver("curve_libwin")
    out = {}
    for t in (218, 250, 300):
        dd = library_window(t)
        out[f"t={t}"] = {c: [round(d[c], 4) for d in dd] for c in dd[0]}
        sv["decisions"] = out
    dd = library_window(218, lib_cut=218)
    out["t=218,lib_cut=218"] = {c: [round(d[c], 4) for d in dd] for c in dd[0]}
    sv["decisions"] = out


# ----------------------------------------------------------------------------- merge

def k3_all(d, y, env, idx):
    """Share of correct same-label triples whose rows come from three different environments."""
    acc = []
    for c in (0, 1):
        cells = [d[idx[(env[idx] == e) & (y[idx] == c)]] for e in sorted(set(env[idx]))]
        sgn, tot, cor = (1 if c else -1), 0, 0
        for a, b, e3 in itertools.combinations(range(len(cells)), 3):
            s = cells[a][:, None, None] + cells[b][None, :, None] + cells[e3][None, None, :]
            cor += np.sum(sgn * s > 0)
            tot += s.size
        acc.append(cor / tot)
    return {"k3_all": round(float(np.mean(acc)), 4), "k3_all_astra": round(float(acc[0]), 4), "k3_all_sol": round(float(acc[1]), 4)}


def k3_boot(d, y, env, idx, reps=200, seed=0):
    """95% interval of the worst-label all-combination k3, resampling environments (clusters)."""
    rng = np.random.default_rng(seed)
    envs = sorted(set(env[idx]))
    out = []
    for _ in range(reps):
        draw = rng.choice(envs, len(envs), replace=True)
        ni = np.concatenate([idx[env[idx] == e] for e in draw])
        ne = np.concatenate([[str(k)] * int(np.sum(env[idx] == e)) for k, e in enumerate(draw)])
        r = k3_all(d[ni], y[ni], ne, np.arange(len(ni)))
        out.append(min(r["k3_all_astra"], r["k3_all_sol"]))
    return [round(float(q), 4) for q in np.quantile(out, [0.025, 0.975])]


def summarize(D, y, env, groups, idx, boot=False, N=None, minute=None, dev=None):
    """Per-label results; the headline is the worse of the two labels."""
    m = np.zeros(len(y), bool)
    m[idx] = True
    out = {}
    for k, v in D.items():
        v = np.array(v)
        r = {**metrics(v, y, groups, m), **k3_all(v, y, env, idx)}
        r["k1_worst"] = min(r["recall_astra"], r["recall_sol"])
        r["k3_all_worst"] = min(r["k3_all_astra"], r["k3_all_sol"])
        if boot:
            r["k3_all_worst_ci"] = k3_boot(v, y, env, idx)
        if N is not None:  # single-response recall by parsed length (diagnosis of the length artifact)
            r["recall_by_length"] = {f"{lab} N{'<' if short else '='}1000": [int(s.sum()), round(float(np.mean((v[s] > 0) == c)), 3)]
                                     for c, lab in ((0, "Astra"), (1, "Sol")) for short in (True, False)
                                     for s in [m & (y == c) & ((N < 1000) == short)] if s.sum()}
        if minute is not None:  # does the decision drift with collection time inside each label?
            r["time"] = {}
            for c, lab in ((0, "Astra"), (1, "Sol")):
                s = m & (y == c)
                rt, rd = np.argsort(np.argsort(minute[s])), np.argsort(np.argsort(v[s]))
                early = s & (minute < 75)  # before 11:15, when the test split started
                r["time"][lab] = {"spearman_decision_vs_time": round(float(np.corrcoef(rt, rd)[0, 1]), 3),
                                  "median_before_11:15": round(float(np.median(v[early])), 3) if early.sum() else None,
                                  "median_after_11:15": round(float(np.median(v[s & ~early])), 3) if (s & ~early).sum() else None,
                                  "n_before/after": [int(early.sum()), int((s & ~early).sum())],
                                  # inside the development split only, so the split itself is not the comparison
                                  "dev_median_before/after_11:15": [round(float(np.median(v[q])), 3) if q.sum() else None
                                                                    for q in (s & dev & early, s & dev & ~early)],
                                  "dev_n_before/after": [int((s & dev & early).sum()), int((s & dev & ~early).sum())]}
        out[k] = r
    return out


def merge(d):
    rows, y, env, split, groups = load_pilot()
    N = np.array([len(r["numbers"]) for r in rows])
    raw = [json.loads(line) for line in PILOT.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if SMOKE:
        keep = {r["condition_id"][len("pilot:"):] for r in rows}
        raw = [r for r in raw if r["condition_id"] in keep]
    minute = np.array([(int(r["collected_at"][11:13]) - 10) * 60 + int(r["collected_at"][14:16]) for r in raw])  # minutes after 10:00 UTC
    del raw
    allrows = np.arange(len(y))
    res = {}
    for f in sorted(Path(d).glob("curve_*.json")):
        j = json.loads(f.read_text())
        name = j["experiment"]
        if name == "curve_diag":
            res[name] = {k: j[k] for k in ("counts", "examples") if k in j}
            continue
        full = name.startswith("curve_len")  # bootstrap intervals and length diagnosis for the main evaluations
        if "loeo" in j:
            res[name] = {"n_ref": j["loeo"]["n_ref"],
                         "loeo": summarize(j["loeo"]["decisions"], y, env, groups, allrows, full, N if full else None,
                                          minute if full else None, split == "development")}
        if "devtest" in j:
            D = j["devtest"]["decisions"]
            res[name]["dev->test"] = summarize(D, y, env, groups, np.flatnonzero(split == "test"), full, N)
            res[name]["test->dev"] = summarize(D, y, env, groups, np.flatnonzero(split == "development"), full, N)
        if "decisions" in j and name == "curve_libwin":
            res[name] = {t: summarize(D, y, env, groups, allrows) for t, D in j["decisions"].items()}
    # mean over draws for the reference curves
    keys = ("k1", "recall_astra", "recall_sol", "k1_worst", "k3", "k3_all", "k3_all_astra", "k3_all_sol", "k3_all_worst", "auc", "dprime")
    for key in sorted({n.rsplit(":", 1)[0] for n in res if n.startswith(("curve_ref", "curve_rep"))}):
        ds = [v for n, v in res.items() if n.rsplit(":", 1)[0] == key]
        res[key + ":mean"] = {"draws": len(ds), "n_ref": float(np.mean([v["n_ref"] for v in ds])),
                              "loeo": {k: {m: round(float(np.mean([v["loeo"][k][m] for v in ds])), 4) for m in keys} for k in NAMES}}
    OUT.mkdir(exist_ok=True)
    (OUT / "curve.json").write_text(json.dumps(res, indent=1))
    for n, v in sorted(res.items()):
        if n == "curve_diag":
            print(json.dumps(v, indent=1, ensure_ascii=False))
            continue
        for proto in ("loeo", "dev->test", "test->dev"):
            if proto in v:
                for k, r in v[proto].items():
                    print(f"{n:26} {proto:9} {k:20} n_ref {v.get('n_ref', 0):6.1f} | k1 A {r['recall_astra']:.3f} S {r['recall_sol']:.3f} "
                          f"worst {r['k1_worst']:.3f} | k3_all A {r['k3_all_astra']:.3f} S {r['k3_all_sol']:.3f} worst {r['k3_all_worst']:.3f}"
                          + (f" CI {r['k3_all_worst_ci']}" if "k3_all_worst_ci" in r else "")
                          + f" | native k3 {r['k3']} | auc {r['auc']:.3f}"
                          + (f" | by length {r['recall_by_length']}" if "recall_by_length" in r else "")
                          + (f" | time {json.dumps(r['time'])}" if "time" in r else ""))
        if n == "curve_libwin":
            for t, r in v.items():
                for k, x in r.items():
                    print(f"{n:26} {t:18} {k:12} k1 A {x['recall_astra']:.3f} S {x['recall_sol']:.3f} | "
                          f"k3_all A {x['k3_all_astra']:.3f} S {x['k3_all_sol']:.3f} | auc {x['auc']:.3f}")


def run_one(spec):
    """spec: ref:k:r, rep:m:r, len:t, libwin or diag; a trailing :clean selects the clean features,
    :cleanb (len only) the clean features plus the reference batch nuisance projection."""
    name, *args = spec.split(":")
    clean = bool(args) and args[-1] in ("clean", "cleanb")
    batch = bool(args) and args[-1] == "cleanb"
    args = [int(a) for a in args if a not in ("clean", "cleanb")]
    log(f"experiment {spec} (smoke={SMOKE})")
    {"ref": lambda: exp_ref(*args, clean=clean), "rep": lambda: exp_rep(*args, clean=clean),
     "len": lambda: exp_len(*args, clean=clean, batch=batch), "libwin": exp_libwin, "diag": exp_diag}[name]()
    log(f"done {spec}")


def main():
    if sys.argv[1] == "merge":
        return merge(sys.argv[2])
    for spec in sys.argv[1:]:  # several experiments in one job, run one after another
        run_one(spec)


if __name__ == "__main__":
    main()
