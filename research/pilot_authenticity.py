"""Is the 4Router pilot plausibly the claimed models?  17-class placement versus genuine data.

The pilot comes from a third-party relay (labels are the relay's own reports). If the relay routed
to another model, the responses would sit nearer that model's class (for example a GPT-5.x or a
Luna class) than genuine GPT-6 Astra / GPT-6.1 Sol responses do.
Protocol: the production 17-class fingerprint (unchanged original algorithm) fitted on the
historical training rows only (the 10-11 Codex batch left out). Two out-of-library query sets
are scored the same way:
  - genuine: the 10-11 Codex Astra / Sol rows;
  - pilot: the 4Router rows, cut to their first t numbers (t = 250 is in the library length
    range 218-339; full length is shown too).
Reported per set and label: top-1 class counts, top-1 family-group shares, share with Astra or
Sol among the top 3, and the same for native triples (mean of the three score vectors, as the
harness does for production).
Lean on memory (numpy only): ~250MB is all this machine has free.

Usage: python -I research/pilot_authenticity.py <pilot dir with accepted.jsonl>
"""
from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.data import Panel, extended_panel  # noqa: E402
from research.methods_baseline import ProductionBaseline, fp  # noqa: E402

CODEX = "train_codex_20261011"
A, S = "gpt-6-astra", "gpt-6.1-sol"
GROUPS = ("6-astra/6.1-sol", "gpt-6-sol", "luna (5.6/6)", "gpt-5.x (non-luna)", "claude")


def group_of(c):
    if c in (A, S):
        return GROUPS[0]
    if c.startswith("claude"):
        return GROUPS[4]
    if "luna" in c:
        return GROUPS[2]
    if c.startswith("gpt-5"):
        return GROUPS[3]
    return GROUPS[1]


def mem(tag):
    print(f"[{tag}: peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB]", flush=True)


def summarize(sc, labels, classes):
    """sc: production fused scores (n x 17); labels: claimed class names."""
    top = sc.argmax(1)
    top3 = np.argsort(-sc, 1)[:, :3]
    pair = [classes.index(A), classes.index(S)]
    gi = np.array([GROUPS.index(group_of(c)) for c in classes])
    out = {}
    for lab in (A, S):
        m = np.array(labels) == lab
        tc = {}
        for k in top[m]:
            tc[classes[k]] = tc.get(classes[k], 0) + 1
        out[lab] = {"n": int(m.sum()),
                    "top1_class": dict(sorted(tc.items(), key=lambda kv: -kv[1])),
                    "top1_group": {g: round(float(np.mean(gi[top[m]] == j)), 3) for j, g in enumerate(GROUPS)},
                    "astra_or_sol_in_top3": round(float(np.mean(np.isin(top3[m], pair).any(1))), 3)}
    return out


def load_pilot(root: Path):
    rows = []
    for line in (root / "accepted.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({"source": r["source"], "group_id": r["group_id"], "numbers": fp.parse_numbers(r["text"])})
    by = {}
    for i, r in enumerate(rows):
        by.setdefault((r["source"], r["group_id"]), []).append(i)
    return rows, [g for g in by.values() if len(g) == 3]


def main():
    rows, triples = load_pilot(Path(sys.argv[1]))
    print(f"pilot: {len(rows)} rows, {len(triples)} triples", flush=True)
    lib = extended_panel(Panel())
    for r in lib.rows:  # production reads only the numbers
        for k in [k for k in r if k not in ("numbers", "source", "condition_id", "nuisance_condition_id")]:
            del r[k]
    mem("library loaded")
    hist = np.array([i for i in lib.train if lib.role[i] != CODEX])
    gen = np.array([i for i in lib.train if lib.role[i] == CODEX and lib.classes[lib.y[i]] in (A, S)])
    m = ProductionBaseline()
    bank = m.fit(lib, hist)
    mem(f"production fitted on {len(hist)} historical rows")
    score = lambda x: np.array(fp.robust_score_numbers(x, bank)["fused"])  # noqa: E731
    out = {}
    S1 = np.array([score(lib.rows[i]["numbers"]) for i in gen])
    out["genuine_codex_1011"] = summarize(S1, [lib.classes[lib.y[i]] for i in gen], lib.classes)
    gg = lib.group_members[lib.group_role == CODEX]
    gg = gg[np.isin(lib.group_y[lib.group_role == CODEX], [lib.classes.index(A), lib.classes.index(S)])]
    pos = {int(i): k for k, i in enumerate(gen)}
    out["genuine_codex_1011_triples"] = summarize(np.array([S1[[pos[int(i)] for i in g]].mean(0) for g in gg]),
                                                  [lib.classes[lib.y[g[0]]] for g in gg], lib.classes)
    for t in (250, None):
        P = np.array([score(r["numbers"][:t] if t else r["numbers"]) for r in rows])
        key = f"pilot_t={t or 'full'}"
        out[key] = summarize(P, [r["source"] for r in rows], lib.classes)
        out[key + "_triples"] = summarize(np.array([P[g].mean(0) for g in triples]), [rows[g[0]]["source"] for g in triples], lib.classes)
    mem("scored")
    print(json.dumps(out, indent=1), flush=True)
    Path(__file__).with_name("out_pilot_authenticity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
