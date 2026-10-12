"""Extended library (historical + 2026-10-11 supplement), deliverable vs production:
- env-CV accuracy with the confusable pair counted as one family (cluster-level identification),
- per-class env-CV, and a paired McNemar test of the forward check (frozen 684 library -> supplement)."""
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import PAIR_ROLE, Panel, extended_panel
from research.harness import env_holdout
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
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
merge = lambda v: np.where(v == S, A, v)
out = {}
hist = np.flatnonzero(np.isin(p.role, ("train_original", "train_supplemental")))
new = np.flatnonzero(p.role == PAIR_ROLE)
fwd = {}
for name, m in (("production", ProductionBaseline()), ("gated_deliverable", GatedDeliverable())):
    fwd[name] = m.score(m.fit(p, hist), p, new).argmax(1) == p.y[new]
    r = env_holdout(m, p)
    pool, s1 = r["_pool"], r["_s1"]
    pred = s1[pool].argmax(1)
    y = p.y[pool]
    nonpair = ~np.isin(y, [A, S])
    out[name] = {"k1_17class": [int((pred == y).sum()), len(y)],
                 "k1_pair_merged": [int((merge(pred) == merge(y)).sum()), len(y)],
                 "pair_rows_kept_in_pair": [int(np.isin(pred[~nonpair], [A, S]).sum()), int((~nonpair).sum())],
                 "other15_k1": [int((pred[nonpair] == y[nonpair]).sum()), int(nonpair.sum())],
                 "balanced_17": float(np.mean([np.mean(pred[y == c] == c) for c in range(len(p.classes))]))}
    print(name, out[name], flush=True)
b = int((fwd["gated_deliverable"] & ~fwd["production"]).sum()); c = int((~fwd["gated_deliverable"] & fwd["production"]).sum())
out["forward_mcnemar"] = {"gated_only_right": b, "production_only_right": c, "p_two_sided": binomtest(b, b + c).pvalue}
print(out["forward_mcnemar"])
Path(__file__).with_name(f"out_codex_merged_eval{TAG}.json").write_text(json.dumps(out, indent=1))
