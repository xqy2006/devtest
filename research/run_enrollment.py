"""Enrollment check: the newest model is added only through its reference rows, then rebuild.

Before: candidate set without the newest label. After: same code, its reference rows appended.
Reports discovered clusters and per-class env-holdout accuracy before/after.
"""
import json, sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
from research.data import Panel, subset_panel
from research.harness import env_holdout
from research.gated_fingerprint import GatedDeliverable, build_gated_bank
p = Panel()
newest = max({r["source"]: r["collected_at"] for r in p.rows if r["role"] == "train_original"}.items(), key=lambda kv: kv[1])
print("most recently collected original label (enrolled last):", newest)
before = subset_panel(p, [c for c in p.classes if c != newest[0]])
out = {}
for name, q in (("before", before), ("after", p)):
    train = np.flatnonzero(np.isin(q.role, ("train_original", "train_supplemental")))
    bank = build_gated_bank([q.rows[i] for i in train], calibrate=False)
    clusters = [c["members"] for c in bank["refinement"]["clusters"]]
    r = env_holdout(GatedDeliverable(False), q)
    out[name] = {"clusters_full_build": clusters, "k1": r["k1"], "k3": {k: r["k3"][k] for k in ("correct", "total")}}
    print(name, "clusters:", clusters, "envCV k1", r["k1"]["correct"], "/", r["k1"]["total"], "k3", r["k3"]["correct"], "/", r["k3"]["total"], flush=True)
common = [c for c in before.classes]
print("per-class k1 before -> after (classes present in both):")
for c in common:
    b, a = out["before"]["k1"]["per_class"][c], out["after"]["k1"]["per_class"][c]
    print(f"   {c:28s} {b[0]}/{b[1]} -> {a[0]}/{a[1]}")
c = newest[0]; print(f"   {c:28s} (new)   -> {out['after']['k1']['per_class'][c]}")
json.dump(out, open("research/out_enrollment.json", "w"), indent=1)
