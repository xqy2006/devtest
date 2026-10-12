"""Out-of-library learning curve of the v3 cluster model (exact deliverable fit_cluster):
n references per class drawn evenly over the three training batches; scored on the exposed
outside panels (development evidence). Paired bootstrap over native groups for n_lo -> n_hi."""
import sys, warnings, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L
from research.gated_fingerprint import fit_cluster, local_logits

p, yb = L.p, L.yb
rng = np.random.default_rng(12)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
SIZES = (72, 108, 144, 180, 212)
tests = {r: (L.rows((r,)), L.groups((r,))) for r in ("old_outside", "fresh_outside")}
allte = np.concatenate([t for t, _ in tests.values()])
pools = [[np.flatnonzero((p.y == c) & (p.role == b)) for b in L.TRB] for c in (L.A, L.S)]


def draw(n):
    out = []
    for batches in pools:
        caps = np.array([len(b) for b in batches]); sizes = np.zeros(len(batches), int); left = n
        while left:
            open_ = np.flatnonzero(sizes < caps); share = max(1, left // len(open_))
            for j in open_:
                take = min(share, caps[j] - sizes[j], left); sizes[j] += take; left -= take
                if not left:
                    break
        out += [rng.choice(b, s, replace=False) for b, s in zip(batches, sizes)]
    return np.concatenate(out)


K1 = {r: np.zeros((len(SIZES), REPS, len(t))) for r, (t, g) in tests.items()}
K3 = {r: np.zeros((len(SIZES), REPS, len(g))) for r, (t, g) in tests.items()}
for a, n in enumerate(SIZES):
    for rep in range(REPS):
        cl = fit_cluster([p.rows[i] for i in draw(n)], p.classes, [L.A, L.S])
        d = np.zeros(len(p.rows))
        for i in allte:
            s = local_logits(cl, p.rows[i]["numbers"]); d[i] = s[1] - s[0]
        for r, (t, g) in tests.items():
            K1[r][a, rep] = (d[t] > 0) == yb[t]
            K3[r][a, rep] = (d[g].sum(1) > 0) == yb[g[:, 0]]
    print(f"n={n}: " + "  ".join(f"{r[:5]} k1 {K1[r][a].mean():.3f} k3 {K3[r][a].mean():.3f}" for r in tests), flush=True)
out = {}
for r, (t, g) in tests.items():
    pos = {int(i): k for k, i in enumerate(t)}
    gi = np.array([[pos[int(i)] for i in gg] for gg in g])
    k1i = K1[r].mean(1); k3g = K3[r].mean(1)
    d1 = np.array([k1i[-1][x].mean() - k1i[0][x].mean() for x in gi]); d3 = k3g[-1] - k3g[0]
    b = np.random.default_rng(7).integers(0, len(gi), (4000, len(gi)))
    out[r] = {"curve": {n: [float(k1i[a].mean()), float(k3g[a].mean())] for a, n in enumerate(SIZES)},
              "k1_gain": float(d1.mean()), "k1_ci95": np.percentile(d1[b].mean(1), [2.5, 97.5]).round(3).tolist(),
              "k3_gain": float(d3.mean()), "k3_ci95": np.percentile(d3[b].mean(1), [2.5, 97.5]).round(3).tolist()}
    print(r, {k: v for k, v in out[r].items() if k != "curve"}, flush=True)
Path(__file__).with_name("out_pair_v3_curve.json").write_text(json.dumps(out, indent=1))
