"""Deliverable vs production after adding the 2026-10-11 paired reference supplement.

1. Forward check: library frozen before the supplement existed (historical 684 training rows)
   scores the supplement rows (k1) and its complete native triples (k3) with all 17 candidates.
2. All protocols on the extended library (684 + 283 training rows): leave-one-environment-out,
   batch cross-holdout (each Astra/Sol training batch held out in turn) and the exposed outside
   panels (development results only).
"""
import json, sys, time, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import PAIR_ROLE, Panel, extended_panel
from research.harness import _k3, _tally, dump, summarize
if "--v1" in sys.argv:  # reproduce earlier deliverable versions
    from research.gated_fingerprint_v1 import GatedDeliverable
elif "--v2" in sys.argv:
    from research.gated_fingerprint_v2 import GatedDeliverable
elif "--v3" in sys.argv:
    from research.gated_fingerprint_v3 import GatedDeliverable
else:
    from research.gated_fingerprint import GatedDeliverable
TAG = "" if "--v1" in sys.argv else "_v2" if "--v2" in sys.argv else "_v3" if "--v3" in sys.argv else "_v4"
from research.methods_baseline import ProductionBaseline

p = extended_panel(Panel())
hist_train = np.flatnonzero(np.isin(p.role, ("train_original", "train_supplemental")))
test = np.flatnonzero(p.role == PAIR_ROLE)
gtest = np.flatnonzero(p.group_role == PAIR_ROLE)
out = {}
t = time.time()
for name, m in (("production", ProductionBaseline()), ("gated_deliverable", GatedDeliverable())):
    model = m.fit(p, hist_train)
    s1 = m.score(model, p, test)
    s3 = _k3(m, model, p, p.group_members[gtest])
    fwd = {"k1": _tally(s1.argmax(1), p.y[test], p.classes), "k3": _tally(s3.argmax(1), p.group_y[gtest], p.classes)}
    if hasattr(model, "get"):
        fwd["clusters"] = [c["members"] for c in model.get("refinement", {}).get("clusters", [])]
    out.setdefault(name, {})["forward_frozen_to_supplement"] = fwd
    wrong = s1.argmax(1) != p.y[test]
    print(f"[{name}] forward frozen-684 -> supplement: k1 {fwd['k1']['correct']}/{fwd['k1']['total']} "
          f"{fwd['k1']['per_class']} k3 {fwd['k3']['correct']}/{fwd['k3']['total']} {fwd['k3']['per_class']} "
          f"errors predicted as {sorted(set(p.classes[k] for k in s1.argmax(1)[wrong]))} ({time.time()-t:.0f}s)", flush=True)
for name, m in (("production", ProductionBaseline()), ("gated_deliverable", GatedDeliverable())):
    out[name].update(summarize(name + "+supplement", m, p))
    print(f"  ({time.time()-t:.0f}s)", flush=True)
dump(out, Path(__file__).with_name(f"final_results_codex{TAG}.json"))
