"""Is the Astra-vs-Sol difference environment-specific? (2026-10-11 batch: balanced, same messages)

Per family, in features standardized by the pooled within-cell SD:
  main   = || mean_e Delta_e ||^2                (class main effect)
  inter  = mean_e || Delta_e - mean Delta ||^2   (class x environment interaction)
  noise  = expected value of inter under no interaction = mean_e sum_j (1/nA_e + 1/nS_e) * (1 - 1/E)
where Delta_e = mean(S, e) - mean(A, e). inter >> noise means the direction separating the two
labels changes with the environment. Also reported: the same for the batch factor (old batches vs 10-11)."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L

p, yb = L.p, L.yb
cx = L.rows((L.PAIR_ROLE,))
envs = sorted(set(p.env[cx]))
print(f"{'family':10} {'dim':>4} {'main':>8} {'inter':>8} {'noise':>8} {'inter/noise':>11} {'main/noise1':>11}")
for f in L.FEAT:
    X = L.FEAT[f][cx]
    cell = [(c, e) for c in (0, 1) for e in envs]
    resid = np.vstack([X[(yb[cx] == c) & (p.env[cx] == e)] - X[(yb[cx] == c) & (p.env[cx] == e)].mean(0) for c, e in cell])
    sd = np.sqrt((resid ** 2).sum(0) / (len(cx) - len(cell)))
    sd[sd < 1e-9] = 1
    Z = X / sd
    D, noise = [], []
    for e in envs:
        a, s = Z[(yb[cx] == 0) & (p.env[cx] == e)], Z[(yb[cx] == 1) & (p.env[cx] == e)]
        D.append(s.mean(0) - a.mean(0))
        noise.append(Z.shape[1] * (1 / len(a) + 1 / len(s)))
    D = np.array(D)
    main = float((D.mean(0) ** 2).sum())
    inter = float(((D - D.mean(0)) ** 2).sum(1).mean())
    nz = float(np.mean(noise) * (1 - 1 / len(envs)))
    main_noise = float(np.mean(noise) / len(envs))
    print(f"{f:10} {Z.shape[1]:4d} {main:8.2f} {inter:8.2f} {nz:8.2f} {inter/nz:11.2f} {main/main_noise:11.2f}")
