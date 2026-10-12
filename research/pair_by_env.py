import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from research.data import Panel
from research.features import extract
sys.path.insert(0,'production')
from challenge_suite import ENVIRONMENTS
p=Panel()
a,b=p.classes.index('gpt-6-astra'),p.classes.index('gpt-6.1-sol')
sel=np.flatnonzero(np.isin(p.y,[a,b])); rows=[p.rows[i] for i in sel]
fams=["value355","sdiff72","scalar","repeat","digits"]
F=extract(rows,fams); X=np.hstack([F[f] for f in fams])
y=(p.y[sel]==b).astype(int); env=p.env[sel]; role=p.role[sel]; tr=np.isin(role,['train_original','train_supplemental'])
ok=np.zeros(len(sel),bool)
for e in sorted(set(env[tr])):
    te=tr&(env==e); trn=tr&(env!=e)
    sc=StandardScaler().fit(X[trn]); m=LDA(solver='lsqr',shrinkage='auto').fit(sc.transform(X[trn]),y[trn])
    ok[te]=m.predict(sc.transform(X[te]))==y[te]
sc=StandardScaler().fit(X[tr]); m=LDA(solver='lsqr',shrinkage='auto').fit(sc.transform(X[tr]),y[tr])
ok[~tr]=m.predict(sc.transform(X[~tr]))==y[~tr]
for k,e in enumerate(sorted(set(env[tr]))):
    cfg=ENVIRONMENTS[k]
    t=tr&(env==e); o=(~tr)&(env==e)
    print(f"{e} {cfg['transport']:6s} {cfg['style']:4s} prefix={cfg.get('prefix_words',0):5d} train-CV {ok[t].sum():2d}/{t.sum()}  outside(dev) {ok[o].sum():2d}/{o.sum()}")
# grouped by style
for key in ('style','transport'):
    for val in sorted(set(c[key] for c in ENVIRONMENTS)):
        es=[f'environment-{k+1:02d}' for k,c in enumerate(ENVIRONMENTS) if c[key]==val]
        t=tr&np.isin(env,es); o=(~tr)&np.isin(env,es)
        print(f'{key}={val}: train-CV {ok[t].mean():.3f} (n={t.sum()})  outside {ok[o].mean():.3f} (n={o.sum()})')
