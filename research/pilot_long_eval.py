"""Pre-registered evaluation of the long-response pilot (requested count N = 1000).

Question: does single-response evidence keep growing with response length, as the truncation study
(REPORT 17.4) predicts?  The library is frozen before the pilot is seen: all current training rows
(historical 684 + 10-11 supplement 283 = 967), deliverable v4, production for reference.

Primary analysis (fixed before the data arrived):
  every pilot response is scored with its first t numbers, t in {218, 300, 500, full}, through the
  real deliverable pipeline (text -> parse -> production + cluster refinement), with
  expected_count = t for the truncated versions;
  reported: 17-class k1, pair-restricted k1 (Astra vs Sol only), k3 on complete native triples,
  per label, and the cluster-local components (distribution / branch / score) separately;
  test: pair-restricted correctness at full length vs t = 218 on the same rows (exact McNemar).
Secondary (--adapt): leave-one-environment-out over the pilot environments, cluster model refitted
  on the library's Astra/Sol rows plus the other pilot environments (full length).
Descriptives (not features): parsed length vs requested, distinct values, longest +-1 run,
  longest exactly repeated block.

Usage: python -I research/pilot_long_eval.py <pilot dir containing *.jsonl [+ native_triples.json]> [--adapt]
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.gated_fingerprint as G  # noqa: E402
from research.data import Panel, extended_panel  # noqa: E402
from research.harness import _k3  # noqa: E402
from research.methods_baseline import ProductionBaseline, fp  # noqa: E402

ROLE = "pilot_long"
TS = (218, 300, 500, None)  # None = full response


def load_pilot(root: Path):
    # accepted.jsonl (when present) is the complete accepted set = development + test; the other
    # jsonl files are subsets or rejected extras and must not be counted twice
    files = sorted(root.rglob("accepted.jsonl")) or sorted(f for f in root.rglob("*.jsonl") if "extra" not in f.name)
    rows = []
    for f in files:
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                r = json.loads(line)
                if "source" in r and ("numbers" in r or "text" in r):
                    rows.append(r)
    for r in rows:
        # the numeric sequence is what the production parser reads from the text (in-range values)
        r["numbers"] = fp.parse_numbers(r["text"]) if "text" in r else json.loads(r["numbers"]) if isinstance(r["numbers"], str) else r["numbers"]
    trip = list(root.rglob("native_triples.json"))
    groups = []
    pos = {r["row_id"]: i for i, r in enumerate(rows)}
    if trip:
        for t in json.loads(trip[0].read_text()):
            if t.get("complete") and all(rid in pos for rid in t["accepted_row_ids"]):
                groups.append([pos[rid] for rid in t["accepted_row_ids"]])
    else:
        by = {}
        for i, r in enumerate(rows):
            gid = r.get("native_group_id") or r.get("group_id")
            if gid:
                by.setdefault((r["source"], gid), []).append(i)
        groups = [g for g in by.values() if len(g) == 3]
    print(f"pilot: {len(rows)} rows from {[f.name for f in files]}, {len(groups)} complete triples", flush=True)
    return rows, groups


def panel_with(lib, rows, groups, t):
    """Library panel plus the pilot rows truncated to t numbers (text rewritten as a JSON array)."""
    q = copy.copy(lib)
    n0 = len(lib.rows)
    new = []
    for r in rows:
        x = list(map(int, r["numbers"]))[:t] if t else list(map(int, r["numbers"]))
        r2 = dict(r, numbers=x)
        if t:
            r2["text"], r2["requested_count"] = json.dumps(x), min(t, len(x))
        new.append(r2)
    q.rows = lib.rows + new
    idx = {c: i for i, c in enumerate(lib.classes)}
    q.y = np.concatenate([lib.y, [idx[r["source"]] for r in rows]])
    q.env = np.concatenate([lib.env, [r.get("condition_id", "") for r in rows]])
    q.role = np.concatenate([lib.role, [ROLE] * len(rows)])
    return q, np.arange(n0, n0 + len(rows)), np.array(groups, dtype=int).reshape(-1, 3) + n0


def descriptives(rows):
    out = {}
    for lab in sorted(set(r["source"] for r in rows)):
        R = [r for r in rows if r["source"] == lab]
        L, ratio, uniq, run, rep = [], [], [], [], []
        for r in R:
            x = list(map(int, r["numbers"]))
            L.append(len(x))
            ratio.append(len(x) / max(1, int(r.get("requested_count") or len(x))))
            uniq.append(len(set(x)))
            d = np.diff(x)
            best = cur = 0
            for s in np.abs(d) == 1:
                cur = cur + 1 if s else 0
                best = max(best, cur)
            run.append(best)
            seen, longest = set(), 0
            for k in (8, 16, 32, 64):
                blocks = [tuple(x[i:i + k]) for i in range(len(x) - k + 1)]
                if len(blocks) != len(set(blocks)):
                    longest = k
            rep.append(longest)
        out[lab] = {"n": len(R), "len_median": float(np.median(L)), "len/requested>=0.95": float(np.mean(np.array(ratio) >= 0.95)),
                    "distinct_median": float(np.median(uniq)), "max_pm1_run_median": float(np.median(run)),
                    "rows_with_repeated_block>=16": float(np.mean(np.array(rep) >= 16))}
    return out


def safe(fn, items, k):
    """Scores per item; items rejected by the pipeline's validity rule get ok=False."""
    out, ok = np.zeros((len(items), k)), np.ones(len(items), bool)
    for j, it in enumerate(items):
        try:
            out[j] = fn(it)
        except ValueError:
            ok[j] = False
    return out, ok


def pair_cluster(bank, A, S):
    for c in bank.get("refinement", {}).get("clusters", []):
        if A in c["members"] and S in c["members"]:
            return c
    raise RuntimeError("no Astra/Sol cluster in the bank")


def components(cluster, x, A, S):
    a, s = cluster["members"].index(A), cluster["members"].index(S)
    w = cluster["combine"]
    parts = {"distribution": G._dense_logits(cluster, x),
             "branch": G.branch_scores(cluster["branch"], x, len(cluster["members"])),
             "score": G._score_logits(cluster["score"], x) if "score" in cluster else np.zeros(len(cluster["members"])),
             "local": G.local_logits(cluster, x)}
    return {k: float(v[s] - v[a]) for k, v in parts.items()}, w


def main():
    root = Path(sys.argv[1])
    rows, groups = load_pilot(root)
    lib = extended_panel(Panel())
    train = lib.train
    A, S = "gpt-6-astra", "gpt-6.1-sol"
    ia, isx = lib.classes.index(A), lib.classes.index(S)
    out = {"descriptives": descriptives(rows)}
    print(json.dumps(out["descriptives"], indent=1), flush=True)
    t0 = time.time()
    models = {name: (m, m.fit(lib, train)) for name, m in (("production", ProductionBaseline()), ("deliverable_v4", G.GatedDeliverable()))}
    print(f"library fitted on {len(train)} rows ({time.time()-t0:.0f}s)", flush=True)
    cluster = pair_cluster(models["deliverable_v4"][1], A, S)
    correct_pair = {}
    for t in TS:
        q, idx, g = panel_with(lib, rows, groups, t)
        y = q.y[idx]
        split = np.array([r.get("split", "") for r in rows])
        res = {}
        for name, (m, bank) in models.items():
            s1, ok1 = safe(lambda i: m.score(bank, q, [i])[0], idx, len(lib.classes))
            pair = np.where(s1[:, isx] > s1[:, ia], isx, ia)
            # end-to-end: an invalid (rejected) response counts as an error; valid-only reported apart
            r = {"invalid": int((~ok1).sum()),
                 "k1_17": float(np.mean((s1.argmax(1) == y) & ok1)), "k1_pair": float(np.mean((pair == y) & ok1)),
                 "k1_pair_valid_only": float(np.mean((pair == y)[ok1])) if ok1.any() else None,
                 "k1_pair_by_label": {c: float(np.mean(((pair == y) & ok1)[y == lib.classes.index(c)])) for c in (A, S)},
                 "k1_pair_by_split": {sp: float(np.mean(((pair == y) & ok1)[split == sp])) for sp in sorted(set(split))}}
            if len(g):
                s3, ok3 = safe(lambda tr: _k3(m, bank, q, np.array([tr]))[0], list(g), len(lib.classes))
                p3 = np.where(s3[:, isx] > s3[:, ia], isx, ia)
                r["k3_17"] = float(np.mean((s3.argmax(1) == q.y[g[:, 0]]) & ok3))
                r["k3_pair"] = float(np.mean((p3 == q.y[g[:, 0]]) & ok3))
            pair = np.where(ok1, pair, -1)
            res[name] = r
            if name == "deliverable_v4":
                correct_pair[t] = pair == y
        comp = np.array([[components(cluster, q.rows[i]["numbers"], A, S)[0][k] for k in ("distribution", "branch", "score", "local")] for i in idx])
        ys = (y == isx).astype(int)
        res["local_components_k1"] = {k: float(np.mean((comp[:, j] > 0) == ys)) for j, k in enumerate(("distribution", "branch", "score", "local"))}
        if len(g):
            pos = {int(i): k for k, i in enumerate(idx)}
            res["local_components_k3"] = {k: float(np.mean((np.array([[comp[pos[int(i)], j] for i in tr] for tr in g]).sum(1) > 0) == (q.y[g[:, 0]] == isx)))
                                          for j, k in enumerate(("distribution", "branch", "score", "local"))}
        out[f"t={t or 'full'}"] = res
        print(f"t={t or 'full'}: {json.dumps(res)} ({time.time()-t0:.0f}s)", flush=True)
    a, b = correct_pair[None], correct_pair[218]
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    out["mcnemar_full_vs_218"] = {"full_only_right": n10, "t218_only_right": n01,
                                  "p_one_sided": float(binomtest(n10, n10 + n01, 0.5, alternative="greater").pvalue) if n10 + n01 else 1.0}
    print("McNemar full vs 218:", out["mcnemar_full_vs_218"], flush=True)
    if "--adapt" in sys.argv:
        out["adapt"] = adapt(lib, rows, groups, A, S)
    Path(__file__).with_name("out_pilot_long.json").write_text(json.dumps(out, indent=1))


def adapt(lib, rows, groups, A, S):
    """Leave-one-pilot-environment-out; cluster refs = library Astra/Sol rows + other pilot envs."""
    q, idx, g = panel_with(lib, rows, groups, None)
    lib_pair = [i for i in lib.train if lib.classes[lib.y[i]] in (A, S)]
    env = q.env[idx]
    d = np.zeros(len(q.rows))
    for e in sorted(set(env)):
        te, tr = idx[env == e], idx[env != e]
        cl = G.fit_cluster([q.rows[i] for i in list(lib_pair) + list(tr)], lib.classes, [lib.classes.index(A), lib.classes.index(S)])
        for i in te:
            s = G.local_logits(cl, q.rows[i]["numbers"])
            d[i] = s[cl["members"].index(S)] - s[cl["members"].index(A)]
        print(f"  adapt env {e}: {len(te)} rows", flush=True)
    ys = q.y[idx] == lib.classes.index(S)
    r = {"k1": float(np.mean((d[idx] > 0) == ys))}
    if len(g):
        r["k3"] = float(np.mean((d[g].sum(1) > 0) == (q.y[g[:, 0]] == lib.classes.index(S))))
    print("adapt:", r, flush=True)
    return r


if __name__ == "__main__":
    main()
