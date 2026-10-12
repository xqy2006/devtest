"""Is the label difference conditional on the opening path family (read from the response itself)?

Groups: first number 247 / 217 / anything else. Same decomposition as pair_interaction.py, pooled
over all three training batches (cells = label x opener group; batch means removed within label)."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L

p, yb = L.p, L.yb
tr = L.rows(L.TRB)
first = np.array([p.rows[i]["numbers"][0] for i in tr])
grp = np.where(first == 247, 0, np.where(first == 217, 1, 2))
print("group sizes (A,S):", [(int(((grp == g) & (yb[tr] == 0)).sum()), int(((grp == g) & (yb[tr] == 1)).sum())) for g in range(3)])
print(f"{'family':10} {'main/noise':>10} {'inter/noise':>11}")
for f in L.FEAT:
    X = L.FEAT[f][tr].copy()
    for b in set(p.batch[tr]):  # remove batch means within label
        for c in (0, 1):
            m = (p.batch[tr] == b) & (yb[tr] == c)
            X[m] -= X[m].mean(0)
    cells = [(c, g) for c in (0, 1) for g in range(3)]
    resid = np.vstack([X[(yb[tr] == c) & (grp == g)] - X[(yb[tr] == c) & (grp == g)].mean(0) for c, g in cells])
    sd = np.sqrt((resid ** 2).sum(0) / (len(tr) - len(cells)))
    sd[sd < 1e-9] = 1
    Z = X / sd
    D, nz = [], []
    for g in range(3):
        a, s = Z[(yb[tr] == 0) & (grp == g)], Z[(yb[tr] == 1) & (grp == g)]
        D.append(s.mean(0) - a.mean(0)); nz.append(Z.shape[1] * (1 / len(a) + 1 / len(s)))
    D = np.array(D); w = 1 / np.array(nz)
    inter = float(((D - D.mean(0)) ** 2).sum(1).mean()); noise = float(np.mean(nz) * (1 - 1 / 3))
    main = float((D.mean(0) ** 2).sum()); noise1 = float(np.mean(nz) / 3)
    print(f"{f:10} {main/noise1:10.2f} {inter/noise:11.2f}")
