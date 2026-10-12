"""Pair-local discriminant selection on training protocols (env-CV + batch cross); outside = dev."""
import sys, warnings, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from research.data import Panel
from research.features import extract
p = Panel()
a, b = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
sel = np.flatnonzero(np.isin(p.y, [a, b]))
F = extract([p.rows[i] for i in sel])
y = (p.y[sel] == b).astype(int); env = p.env[sel]; role = p.role[sel]
tr = np.isin(role, ["train_original", "train_supplemental"])
def mk(kind):
    if kind == "lda": return lambda: LDA(solver="lsqr", shrinkage="auto")
    C = float(kind[2:]); return lambda: LogisticRegression(C=C, max_iter=5000)
def run(fams, kind):
    X = np.hstack([F[f] for f in fams]); make = mk(kind)
    def fp(trn, te):
        sc = StandardScaler().fit(X[trn]); return make().fit(sc.transform(X[trn]), y[trn]).predict(sc.transform(X[te]))
    c = sum((fp(tr & (env != e), tr & (env == e)) == y[tr & (env == e)]).sum() for e in sorted(set(env[tr])))
    r = [c / tr.sum()]
    for s, d in (("train_original", "train_supplemental"), ("train_supplemental", "train_original")):
        r.append((fp(role == s, role == d) == y[role == d]).mean())
    for o in ("old_outside", "fresh_outside"):
        r.append((fp(tr, role == o) == y[role == o]).mean())
    return r
base = ["value355", "sdiff72", "scalar", "repeat", "digits"]
cands = {
 "base": base,
 "base+countshape": base + ["countshape"],
 "base+chunk10": base + ["chunk10"],
 "base+countshape+chunk10+chunk5": base + ["countshape", "chunk10", "chunk5"],
 "value71+scalar+repeat+countshape": ["value71", "scalar", "repeat", "countshape"],
 "value355+scalar+repeat+countshape": ["value355", "scalar", "repeat", "countshape"],
 "scalar+repeat+countshape+chunk10": ["scalar", "repeat", "countshape", "chunk10"],
 "base+value71+countshape": base + ["value71", "countshape"],
}
print(f"{'features':42s} clf     env   o>s   s>o  |old  fresh  (train mean)")
for name, fams in cands.items():
    for kind in ("lda", "lr0.01", "lr0.1"):
        r = run(fams, kind)
        print(f"{name:42s} {kind:6s} " + " ".join(f"{v:.3f}" for v in r[:3]) + " | " + " ".join(f"{v:.3f}" for v in r[3:]) + f"  ({np.mean(r[:3]):.3f})", flush=True)
