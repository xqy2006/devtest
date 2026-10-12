"""Numbers-only method sweep inside one collection batch (leave-one-environment-out inside the batch).

Measures how separable the pair is within a single deployment snapshot, e.g. the 2026-10-11 batch."""
import sys, time, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_lab as L

roles = {"codex": (L.PAIR_ROLE,), "old": ("train_original", "train_supplemental"),
         "supp": ("train_supplemental",), "fresh": ("fresh_outside",)}[sys.argv[1]]
names = sys.argv[2:] or ["lda_fams", "lda_content", "lda_all", "logit_fams_C0.1", "gbm_compact", "value_nb",
                         "bigram36", "count_nb", "lda_fams+count_nb", "ens_lda+gbm"]
idx = L.rows(roles)
g = L.groups(roles)
for name in names:
    t = time.time()
    d = np.zeros(len(L.p.rows))
    for e in sorted(set(L.p.env[idx])):
        L.score(L.METHODS[name], idx[L.p.env[idx] != e], idx[L.p.env[idx] == e], d)
    k1 = np.mean((d[idx] > 0) == L.yb[idx]); k3 = np.mean((d[g].sum(1) > 0) == L.yb[g[:, 0]])
    print(f"{sys.argv[1]:6} {name:20} k1 {k1:.3f} (n={len(idx)}) k3 {k3:.3f} (n={len(g)}) ({time.time()-t:.0f}s)", flush=True)
