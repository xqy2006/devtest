"""The deliverable's own cluster model (gated_fingerprint.fit_cluster / local_logits) on the pair:
leave-one-training-batch-out, 10-11 within-batch env-CV, pooled env-CV and the outside panels
(development only), with per-batch k1/k3 (pair_k3.summary). Usage: [--v3] to run the v3 copy."""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L  # noqa: E402
from research.pair_k3 import summary  # noqa: E402

if "--v3" in sys.argv:
    import research.gated_fingerprint_v3 as G
    TAG = "v3"
else:
    import research.gated_fingerprint as G
    TAG = "v4"
p, yb = L.p, L.yb


class Local:
    def fit(self, ref):
        t = time.time()
        self.cl = G.fit_cluster([p.rows[i] for i in ref], p.classes, [L.A, L.S])
        print(f"  [cluster fitted on {len(ref)} refs in {time.time()-t:.0f}s; weights {self.cl['combine']}]", flush=True)
        return self

    def decision(self, idx):
        s = np.array([G.local_logits(self.cl, p.rows[i]["numbers"]) for i in idx])
        return s[:, 1] - s[:, 0]


if __name__ == "__main__":
    tr = L.rows(L.TRB)
    res = {}
    d = np.zeros(len(p.rows))
    for b in L.TRB:
        te = L.rows((b,))
        d[te] = Local().fit(L.rows(tuple(x for x in L.TRB if x != b))).decision(te)
    res["lobo"] = summary(d)
    print("LOBO", json.dumps(res["lobo"]), flush=True)
    cx = L.rows((L.PAIR_ROLE,))
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[cx])):
        te = cx[p.env[cx] == e]
        d[te] = Local().fit(cx[p.env[cx] != e]).decision(te)
    gx = L.groups((L.PAIR_ROLE,))
    res["codex_env"] = {"k1": round(float(np.mean((d[cx] > 0) == yb[cx])), 3),
                        "k3": round(float(np.mean((d[gx].sum(1) > 0) == yb[gx[:, 0]])), 3)}
    print("10-11 env", res["codex_env"], flush=True)
    d = np.zeros(len(p.rows))
    for e in sorted(set(p.env[tr])):
        te = tr[p.env[tr] == e]
        d[te] = Local().fit(tr[p.env[tr] != e]).decision(te)
    res["env"] = summary(d)
    print("env", json.dumps(res["env"]), flush=True)
    m = Local().fit(tr)
    for role in ("old_outside", "fresh_outside"):
        te, g = L.rows((role,)), L.groups((role,))
        d = np.zeros(len(p.rows))
        d[te] = m.decision(te)
        res[role] = {"k1": round(float(np.mean((d[te] > 0) == yb[te])), 3),
                     "k3": round(float(np.mean((d[g].sum(1) > 0) == yb[g[:, 0]])), 3)}
        print(role, res[role], "(development only)", flush=True)
    Path(__file__).with_name(f"out_pair_deliverable_{TAG}.json").write_text(json.dumps(res, indent=1))
