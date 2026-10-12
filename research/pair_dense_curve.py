"""Learning curve of the v2 local discriminant (dense numbers-only families) inside the 2026-10-11
snapshot: leave-one-environment-out, n references per class drawn from the other environments."""
import sys, warnings, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L
from research.gated_fingerprint import LOCAL_FAMILIES

p, yb = L.p, L.yb
cx = L.rows((L.PAIR_ROLE,))
g = L.groups((L.PAIR_ROLE,))
rng = np.random.default_rng(1)
res = {}
for n in (24, 48, 72, 96, 120):
    k1, k3 = [], []
    for rep in range(10):
        d = np.zeros(len(p.rows))
        for e in sorted(set(p.env[cx])):
            pool = cx[p.env[cx] != e]
            ref = np.concatenate([rng.choice(pool[yb[pool] == c], n, replace=False) for c in (0, 1)])
            te = cx[p.env[cx] == e]
            d[te] = L.Model(list(LOCAL_FAMILIES), "lda").fit(ref).decision(te)
        k1.append(np.mean((d[cx] > 0) == yb[cx])); k3.append(np.mean((d[g].sum(1) > 0) == yb[g[:, 0]]))
    res[n] = (float(np.mean(k1)), float(np.std(k1)), float(np.mean(k3)))
    print(f"n={n}/class: k1 {res[n][0]:.3f} (sd over draws {res[n][1]:.3f}) k3 {res[n][2]:.3f}", flush=True)
Path(__file__).with_name("out_pair_dense_curve.json").write_text(json.dumps(res, indent=1))
