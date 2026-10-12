import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.data import Panel
from research import seqlogit as sl
p = Panel()
a, b = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
pair = np.flatnonzero(np.isin(p.y, [a, b]))
D = {i: sl.seq_design(p.rows[i]["numbers"]) for i in pair}
tr = np.isin(p.role, ["train_original", "train_supplemental"])
def run(train, test, lam):
    pooled = sl.fit([D[i] for i in train])
    W = [sl.fit([D[i] for i in train if p.y[i] == c], prior=pooled, lam=lam) for c in (a, b)]
    pred = np.array([[sl.loglik(w, D[i]) for w in W] for i in test]).argmax(1)
    return (np.array([a, b])[pred] == p.y[test])
t0 = time.time()
for lam in (float(sys.argv[1]),) if len(sys.argv) > 1 else (0.01, 0.1, 1.0):
    trp = pair[tr[pair]]
    env = p.env[trp]
    ok = np.concatenate([run(trp[env != e], trp[env == e], lam) for e in sorted(set(env))])
    r = [ok.mean()]
    for s, d in (("train_original", "train_supplemental"), ("train_supplemental", "train_original")):
        r.append(run(pair[p.role[pair] == s], pair[p.role[pair] == d], lam).mean())
    for o in ("old_outside", "fresh_outside"):
        r.append(run(trp, pair[p.role[pair] == o], lam).mean())
    print(f"lam={lam}: env {r[0]:.3f} o>s {r[1]:.3f} s>o {r[2]:.3f} | old {r[3]:.3f} fresh {r[4]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
