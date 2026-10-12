"""Is there any real numeric difference between the two labels inside the 2026-10-11 snapshot?

Paired permutation test: the same message went to both labels; under the null (no label effect)
swapping the labels inside a message pair leaves the data distribution unchanged. Statistic:
leave-one-environment-out accuracy of the pair-local LDA (numbers only)."""
import sys, warnings, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_lab as L

p = L.p
idx = L.rows((L.PAIR_ROLE,))
by = {}
for i in idx:
    by.setdefault(p.rows[i]["messages_sha256"], []).append(i)
pairs = np.array([v for v in by.values() if len(v) == 2])
use = pairs.ravel()
X, w = L.stack(L.FAMS)


def stat(y):
    d = np.zeros(len(use))
    env = p.env[use]
    for e in sorted(set(env)):
        tr, te = env != e, env == e
        sc = L.StandardScaler().fit(X[use[tr]])
        m = L.LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X[use[tr]]) * w, y[tr])
        d[te] = m.decision_function(sc.transform(X[use[te]]) * w)
    return float(np.mean((d > 0) == y))


y0 = L.yb[use]
obs = stat(y0)
rng = np.random.default_rng(11)
null = []
for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 200):
    flip = np.repeat(rng.random(len(pairs)) < 0.5, 2)
    null.append(stat(np.where(flip, 1 - y0, y0)))
null = np.array(null)
res = {"pairs": len(pairs), "observed_acc": obs, "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
       "p_value": float((1 + (null >= obs).sum()) / (1 + len(null)))}
over = np.array([len(p.rows[i]["numbers"]) - p.rows[i]["requested_count"] for i in use]) > 0
a, s = over[y0 == 0].sum(), over[y0 == 1].sum()
from scipy.stats import fisher_exact
res["overshoot_A_S"] = [int(a), int(s), int((y0 == 0).sum())]
res["overshoot_fisher_p"] = float(fisher_exact([[a, (y0 == 0).sum() - a], [s, (y0 == 1).sum() - s]])[1])
print(res)
Path(__file__).with_name("out_pair_permutation.json").write_text(json.dumps(res, indent=1))
