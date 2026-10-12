"""Fit saturating learning curves d'^2(n) = D2 * n / (n + K) to out-of-library curves.

d' = 2 * Phi^-1(acc) (equal-variance two-class model). Bootstrap over native test groups.
Reports the asymptotic accuracy (n -> inf) and n needed for 90% (inf if asymptote < 90%).
"""
import glob, json
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

def fit(n, acc):
    acc = np.clip(acc, 0.501, 0.999)
    d2 = (2 * norm.ppf(acc)) ** 2
    f = lambda n, D2, K: D2 * n / (n + K)
    try:
        (D2, K), _ = curve_fit(f, n, d2, p0=[d2[-1] * 2, n[-1]], bounds=([1e-6, 1e-6], [1e3, 1e6]), maxfev=20000)
    except Exception:
        return np.nan, np.inf
    asym = norm.cdf(np.sqrt(D2) / 2)
    need = np.inf
    target = (2 * norm.ppf(0.9)) ** 2
    if D2 > target:
        need = K * target / (D2 - target)
    return asym, need

rng = np.random.default_rng(3)
out = {}
for path in sorted(glob.glob("research/lc_*.npz")):
    z = np.load(path); n = z["sizes"].astype(float); gidx = z["gidx"]
    k1g = np.stack([z["k1"][:, g].mean(2) for g in [gidx]])[0]  # (sizes, groups)
    k3g = z["k3"]
    res = {}
    for name, mat in (("k1", k1g), ("k3", k3g)):
        point = fit(n, mat.mean(1))
        boots = [fit(n, mat[:, rng.integers(0, mat.shape[1], mat.shape[1])].mean(1)) for _ in range(500)]
        asy = np.array([b[0] for b in boots]); need = np.array([b[1] for b in boots])
        res[name] = {"observed": [round(float(a), 3) for a in mat.mean(1)],
                     "asymptote": round(float(point[0]), 3), "asymptote_ci90": [round(float(np.nanpercentile(asy, 5)), 3), round(float(np.nanpercentile(asy, 95)), 3)],
                     "n_for_90_point": float(point[1]), "n_for_90_median_boot": float(np.median(need)),
                     "p_reach_90": float(np.mean(np.isfinite(need)))}
        print(path.split('/')[-1], name, res[name])
    out[path] = res
json.dump(out, open("research/out_learning_curve_fit.json", "w"), indent=1)
