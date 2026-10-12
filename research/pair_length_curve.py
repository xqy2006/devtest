"""Pair separability vs. number of integers used per response (truncation to first m)."""
import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from research.data import Panel
from research.features import extract
p=Panel()
a,b=p.classes.index('gpt-6-astra'),p.classes.index('gpt-6.1-sol')
sel=np.flatnonzero(np.isin(p.y,[a,b]))
y=(p.y[sel]==b).astype(int); env=p.env[sel]; role=p.role[sel]; tr=np.isin(role,['train_original','train_supplemental'])
fams=["value355","sdiff72","digits"]
for m in (50,100,150,200,250,10**6):
    rows=[{**p.rows[i],"numbers":p.rows[i]["numbers"][:m]} for i in sel]
    F=extract(rows,fams); X=np.hstack([F[f] for f in fams])
    dv=np.zeros(len(sel))
    for e in sorted(set(env[tr])):
        te=tr&(env==e); trn=tr&(env!=e)
        sc=StandardScaler().fit(X[trn]); mdl=LDA(solver='lsqr',shrinkage='auto').fit(sc.transform(X[trn]),y[trn])
        dv[te]=mdl.decision_function(sc.transform(X[te]))
    acc=((dv[tr]>0)==y[tr]).mean()
    n_eff=np.mean([min(len(p.rows[i]['numbers']),m) for i in sel])
    print(f"first {min(m,9999):5d} numbers (mean used {n_eff:5.1f}): envCV acc {acc:.3f}  d'={2*norm.ppf(acc):.2f}")
