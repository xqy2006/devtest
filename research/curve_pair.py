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
Every job saves its decisions for all pilot rows. `merge` computes k1, native k3 and the k3 over
all same-label triples from three different environments.

Usage:
  python research/curve_pair.py <experiment>     (for example ref:5:1, rep:3:0, len:700, libwin)
  python research/curve_pair.py merge <dir>      (reads curve_*.json, writes curve.json)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.runner_pair import (FS, G, MEMBERS, OUT, SMOKE, V3, Saver, cluster_decisions, family_matrix,  # noqa: E402
                                  folds, library_pair_rows, load_pilot, log, metrics)
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


class Context:
    def __init__(self, t=1000):
        self.rows, self.y, self.env, self.split, self.groups = load_pilot()
        seqs = [r["numbers"][:t] for r in self.rows]
        F = family_matrix(seqs, list(V3))
        self.Xd = np.hstack([F[f] for f in V3])
        self.wd = np.concatenate([np.full(F[f].shape[1], 1 / np.sqrt(F[f].shape[1])) for f in V3])
        self.st = FS.Steps(seqs)
        self.win0 = np.array([d["local"] for d in library_window()])
        log(f"context ready (t={t}, {len(seqs)} rows)")

    def fold(self, ref, te):
        """Scaled source decisions and combinations for the query rows te, fitted on ref only."""
        y, env = self.y, self.env
        theta0 = FS.fit_base(self.st, np.isin(np.arange(len(y)), ref), lam=G.SCORE_LAMBDA, maxiter=G.SCORE_MAXITER)
        U = FS.score_vectors(self.st, theta0)
        cols = [U[:, FS.OFF[g]:FS.OFF[g] + FS.GROUPS[g]] for g in SG]
        Us = np.hstack(cols)
        wu = np.concatenate([np.full(c.shape[1], 1 / np.sqrt(c.shape[1])) for c in cols])

        def sources(r):
            return {"dense": lda_fit(self.Xd, self.wd, y, env, r, 0.1)[0],
                    "score": lda_fit(Us, wu, y, env, r, 0.1)[0],
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


def exp_ref(k, draw):
    sv = Saver(f"curve_ref:{k}:{draw}")
    ctx = Context()
    if SMOKE:
        k = min(k, 2)
    rng = np.random.default_rng(1000 + 100 * k + draw)

    def pick(e, ref):
        others = np.array(sorted(set(ctx.env[ref])))
        keep = rng.choice(others, min(k, len(others)), replace=False)
        return ref[np.isin(ctx.env[ref], keep)]
    run_loeo(ctx, pick, sv, "loeo")


def exp_rep(m, draw):
    sv = Saver(f"curve_rep:{m}:{draw}")
    ctx = Context()
    rng = np.random.default_rng(2000 + 100 * m + draw)

    def pick(e, ref):
        keep = []
        for ee in sorted(set(ctx.env[ref])):
            for c in (0, 1):
                cell = ref[(ctx.env[ref] == ee) & (ctx.y[ref] == c)]
                keep.extend(rng.choice(cell, min(m, len(cell)), replace=False))
        return np.sort(np.array(keep))
    run_loeo(ctx, pick, sv, "loeo")


def exp_len(t):
    sv = Saver(f"curve_len:{t}")
    ctx = Context(t)
    run_loeo(ctx, lambda e, ref: ref, sv, "loeo")
    D = {k: np.zeros(len(ctx.y)) for k in NAMES}
    for ref, te in folds(ctx.env, ctx.split).get("devtest", []):
        for k, v in ctx.fold(ref, te).items():
            D[k][te] = v
        log(f"devtest fold with {len(ref)} references")
    sv["devtest"] = {"decisions": {k: v.round(4).tolist() for k, v in D.items()}}


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


def summarize(D, y, env, groups, idx):
    m = np.zeros(len(y), bool)
    m[idx] = True
    return {k: {**metrics(np.array(v), y, groups, m), **k3_all(np.array(v), y, env, idx)} for k, v in D.items()}


def merge(d):
    rows, y, env, split, groups = load_pilot()
    allrows = np.arange(len(y))
    res = {}
    for f in sorted(Path(d).glob("curve_*.json")):
        j = json.loads(f.read_text())
        name = j["experiment"]
        if "loeo" in j:
            res[name] = {"n_ref": j["loeo"]["n_ref"], "loeo": summarize(j["loeo"]["decisions"], y, env, groups, allrows)}
        if "devtest" in j:
            D = j["devtest"]["decisions"]
            res[name]["dev->test"] = summarize(D, y, env, groups, np.flatnonzero(split == "test"))
            res[name]["test->dev"] = summarize(D, y, env, groups, np.flatnonzero(split == "development"))
        if "decisions" in j and name == "curve_libwin":
            res[name] = {t: summarize(D, y, env, groups, allrows) for t, D in j["decisions"].items()}
    # mean over draws for the reference curves
    for kind in ("ref", "rep"):
        for key in sorted({n.rsplit(":", 1)[0] for n in res if n.startswith(f"curve_{kind}:")}):
            ds = [v for n, v in res.items() if n.rsplit(":", 1)[0] == key]
            res[key + ":mean"] = {"draws": len(ds), "n_ref": float(np.mean([v["n_ref"] for v in ds])),
                                  "loeo": {k: {m: round(float(np.mean([v["loeo"][k][m] for v in ds])), 4)
                                               for m in ("k1", "k3", "k3_all", "k3_all_astra", "k3_all_sol", "auc", "dprime")}
                                           for k in NAMES}}
    OUT.mkdir(exist_ok=True)
    (OUT / "curve.json").write_text(json.dumps(res, indent=1))
    for n, v in sorted(res.items()):
        for proto in ("loeo", "dev->test", "test->dev"):
            if proto in v:
                for k, r in v[proto].items():
                    print(f"{n:22} {proto:9} {k:20} n_ref {v.get('n_ref', 0):6.1f} k1 {r['k1']:.3f} k3 {r['k3']:.3f} "
                          f"k3_all {r['k3_all']:.3f} (A {r['k3_all_astra']:.3f} S {r['k3_all_sol']:.3f}) auc {r['auc']:.3f}")
        if n == "curve_libwin":
            for t, r in v.items():
                for k, x in r.items():
                    print(f"{n:22} {t:18} {k:12} k1 {x['k1']:.3f} k3 {x['k3']:.3f} k3_all {x['k3_all']:.3f} auc {x['auc']:.3f}")


def main():
    if sys.argv[1] == "merge":
        return merge(sys.argv[2])
    name, *args = sys.argv[1].split(":")
    log(f"experiment {sys.argv[1]} (smoke={SMOKE})")
    {"ref": lambda: exp_ref(int(args[0]), int(args[1])), "rep": lambda: exp_rep(int(args[0]), int(args[1])),
     "len": lambda: exp_len(int(args[0])), "libwin": exp_libwin}[name]()
    log("done")


if __name__ == "__main__":
    main()
