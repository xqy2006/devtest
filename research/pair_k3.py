"""What limits k=3 on the pair: per-batch k1/k3 and within-triple error correlation.

Out-of-fold decisions (leave-one-training-batch-out and pooled leave-one-environment-out) for the
v3 recipe and for v3 + Fisher-score evidence (pair_fisher2.StackedFisher). For each protocol:
  k1, k3 per held-out batch;
  d' implied by k1 and the k3 expected if the three responses were independent;
  intra-triple correlation of the class-centred decisions (0 = independent errors).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research.pair_seq_lab as L  # noqa: E402
from research.pair_fisher2 import StackedFisher  # noqa: E402

p, yb = L.p, L.yb


def oof(make, protocol):
    tr = L.rows(L.TRB)
    d = np.zeros(len(p.rows))
    if protocol == "lobo":
        for b in L.TRB:
            te = L.rows((b,))
            d[te] = make().fit(L.rows(tuple(x for x in L.TRB if x != b))).decision(te)
    else:
        for e in sorted(set(p.env[tr])):
            te = tr[p.env[tr] == e]
            d[te] = make().fit(tr[p.env[tr] != e]).decision(te)
    return d


def summary(d):
    out = {}
    G = L.groups(L.TRB)
    for b in L.TRB + ("all",):
        g = G if b == "all" else G[p.role[G[:, 0]] == b]
        idx = g.ravel()
        y = yb[idx]
        k1 = float(np.mean((d[idx] > 0) == y))
        k3 = float(np.mean((d[g].sum(1) > 0) == yb[g[:, 0]]))
        dp = 2 * norm.ppf(min(max(k1, 0.501), 0.999))
        k3_ind = float(norm.cdf(dp * np.sqrt(3) / 2))
        # intra-triple correlation of decisions centred by class (one-way ICC)
        r = d[g] - np.where(yb[g] == 1, d[idx][y == 1].mean(), d[idx][y == 0].mean())
        msb = 3 * r.mean(1).var(ddof=1)
        msw = ((r - r.mean(1, keepdims=True)) ** 2).sum() / (len(g) * 2)
        icc = float((msb - msw) / (msb + 2 * msw))
        out[b[-8:]] = {"n_triples": len(g), "k1": round(k1, 3), "k3": round(k3, 3),
                       "k3_if_independent": round(k3_ind, 3), "icc": round(icc, 3)}
    return out


if __name__ == "__main__":
    res = {}
    for name, mk in (("v3 recipe", lambda: StackedFisher(use=("lda", "branch"))),
                     ("v3 + fisher[val+dg+pb]", lambda: StackedFisher(("val", "dg", "pb"))),):
        for prot in ("lobo", "env"):
            d = oof(mk, prot)
            res[f"{name} | {prot}"] = summary(d)
            print(name, prot, json.dumps(res[f"{name} | {prot}"]), flush=True)
    Path(__file__).with_name("out_pair_k3.json").write_text(json.dumps(res, indent=1))
