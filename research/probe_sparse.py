import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from research.data import Panel
from research.features import extract
p=Panel()
a,b=p.classes.index('gpt-6-astra'),p.classes.index('gpt-6.1-sol')
sel=np.flatnonzero(np.isin(p.y,[a,b])); rows=[p.rows[i] for i in sel]
y=(p.y[sel]==b).astype(int); env=p.env[sel]; role=p.role[sel]; tr=np.isin(role,['train_original','train_supplemental'])
for fams in (["value355","sdiff72","scalar","repeat","digits"],["value355","scalar","repeat"]):
    F=extract(rows,fams); X=np.hstack([F[f] for f in fams])
    for pen,C in (('l1',0.05),('l1',0.1),('l1',0.3),('elasticnet',0.1)):
        mk=lambda: LogisticRegression(penalty=pen,C=C,solver='saga',l1_ratio=0.5 if pen=='elasticnet' else None,max_iter=20000)
        def fp(trn,te):
            sc=StandardScaler().fit(X[trn]); return mk().fit(sc.transform(X[trn]),y[trn]).predict(sc.transform(X[te]))
        c=sum((fp(tr&(env!=e),tr&(env==e))==y[tr&(env==e)]).sum() for e in sorted(set(env[tr])))
        r=[c/tr.sum()]
        for s_,d_ in (('train_original','train_supplemental'),('train_supplemental','train_original')):
            r.append((fp(role==s_,role==d_)==y[role==d_]).mean())
        for o in ('old_outside','fresh_outside'): r.append((fp(tr,role==o)==y[role==o]).mean())
        print(f'{"+".join(fams):38s} {pen} C={C}: env {r[0]:.3f} o>s {r[1]:.3f} s>o {r[2]:.3f} | old {r[3]:.3f} fresh {r[4]:.3f}',flush=True)
