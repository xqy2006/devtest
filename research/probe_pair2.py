"""More representations/classifiers for the hardest pair (training-only selection; outside = dev)."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from research.data import Panel
from research.features import extract

p = Panel()
a, b = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
sel = np.flatnonzero(np.isin(p.y, [a, b]))
rows = [p.rows[i] for i in sel]
y = (p.y[sel] == b).astype(int); env = p.env[sel]; role = p.role[sel]
tr = np.isin(role, ["train_original", "train_supplemental"])

def presence(x, bins=355):
    v = np.zeros(bins); v[np.unique((np.asarray(x) - 1) * bins // 355)] = 1; return v
def firstpos(x, bins=71):
    x = np.asarray(x); n = len(x); out = np.full(bins, 1.0)
    for i, v in enumerate(x):
        k = (v - 1) * bins // 355
        out[k] = min(out[k], i / n)
    return out
def meanpos(x, bins=36):
    x = np.asarray(x); n = len(x); s = np.zeros(bins); c = np.zeros(bins)
    for i, v in enumerate(x):
        k = (v - 1) * bins // 355; s[k] += i / n; c[k] += 1
    return np.where(c > 0, s / np.maximum(c, 1), 0.5)
F = extract(rows)
F["presence"] = np.array([presence(r["numbers"]) for r in rows])
F["firstpos"] = np.array([firstpos(r["numbers"]) for r in rows])
F["meanpos"] = np.array([meanpos(r["numbers"]) for r in rows])
F["combo"] = np.hstack([F[k] for k in ("value355", "sdiff72", "scalar", "repeat", "digits")])

def evaluate(X, make):
    correct = 0
    for e in sorted(set(env[tr])):
        te = tr & (env == e); trn = tr & (env != e)
        correct += (make().fit(X[trn], y[trn]).predict(X[te]) == y[te]).sum()
    res = [correct / tr.sum()]
    for src, dst in (("train_original", "train_supplemental"), ("train_supplemental", "train_original")):
        res.append((make().fit(X[role == src], y[role == src]).predict(X[role == dst]) == y[role == dst]).mean())
    m = make().fit(X[tr], y[tr])
    for o in ("old_outside", "fresh_outside"):
        res.append((m.predict(X[role == o]) == y[role == o]).mean())
    return res

makers = {
    "logreg.01": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)),
    "lda_shrink": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(C=1.0)),
    "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=2, random_state=0),
}
print("feature      clf         envCV ori>sup sup>ori old  fresh")
for fam in ("value355", "presence", "firstpos", "meanpos", "sdiff72", "combo"):
    for name, mk in makers.items():
        r = evaluate(F[fam], mk)
        print(f"{fam:12s} {name:10s} " + " ".join(f"{v:.3f}" for v in r), flush=True)
