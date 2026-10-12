import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from research.data import Panel
from research.features import extract
p=Panel()
a,b=p.classes.index('gpt-6-astra'),p.classes.index('gpt-6.1-sol')
sel=np.flatnonzero(np.isin(p.y,[a,b]))
rows=[p.rows[i] for i in sel]
y=(p.y[sel]==b).astype(int); env=p.env[sel]; role=p.role[sel]; tr=np.isin(role,['train_original','train_supplemental'])
def disc(r):
    over=len(r['numbers'])-r['requested_count']; dups=len(r['numbers'])-len(set(r['numbers']))
    ob=0 if over<0 else 1 if over==0 else 2 if over<=5 else 3 if over<=15 else 4
    db=min(dups,3)
    return ob, db
D=np.array([disc(r) for r in rows])
def nb_fit(trn):
    tabs=[]
    for k,card in ((0,5),(1,4)):
        t=np.ones((2,card))  # Laplace
        np.add.at(t,(y[trn],D[trn,k]),1); tabs.append(np.log(t/t.sum(1,keepdims=True)))
    return tabs
def nb_score(tabs,idx):
    return sum(tabs[k][:,D[idx,k]].T for k in range(2))  # (n,2)
for fams in (["value355","sdiff72","digits"],["value355","sdiff72","scalar","repeat","digits"],["value355","digits"],["value355"]):
    F=extract(rows,fams); X=np.hstack([F[f] for f in fams])
    def run(trn,te,use_nb,use_lda=True):
        s=np.zeros((te.sum(),2))
        if use_lda:
            sc=StandardScaler().fit(X[trn]); m=LDA(solver='lsqr',shrinkage='auto').fit(sc.transform(X[trn]),y[trn])
            s+=m.predict_log_proba(sc.transform(X[te]))
        if use_nb: s+=nb_score(nb_fit(trn),np.flatnonzero(te))
        return s
    for mode in ('lda','lda+nb','nb'):
        un='nb' in mode; ul='lda' in mode
        c=sum((run(tr&(env!=e),tr&(env==e),un,ul).argmax(1)==y[tr&(env==e)]).sum() for e in sorted(set(env[tr])))
        r=[c/tr.sum()]
        for s_,d_ in (('train_original','train_supplemental'),('train_supplemental','train_original')):
            r.append((run(role==s_,role==d_,un,ul).argmax(1)==y[role==d_]).mean())
        for o in ('old_outside','fresh_outside'):
            r.append((run(tr,role==o,un,ul).argmax(1)==y[role==o]).mean())
        print(f'{"+".join(fams):40s} {mode:7s} env {r[0]:.3f} o>s {r[1]:.3f} s>o {r[2]:.3f} | old {r[3]:.3f} fresh {r[4]:.3f}',flush=True)
