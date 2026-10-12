import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from research.data import Panel
from research.features import extract
p=Panel()
a,b=p.classes.index('gpt-6-astra'),p.classes.index('gpt-6.1-sol')
sel=np.flatnonzero(np.isin(p.y,[a,b])); rows=[p.rows[i] for i in sel]
fams=["value355","sdiff72","scalar","repeat","digits"]
F=extract(rows,fams); X=np.hstack([F[f] for f in fams])
y=(p.y[sel]==b).astype(int); env=p.env[sel]; role=p.role[sel]; tr=np.isin(role,['train_original','train_supplemental'])
def basis(Z, keys, rank):
    ks=sorted(set(keys)); off=np.stack([Z[keys==k].mean(0) for k in ks]); off-=off.mean(0)
    _,s,Vt=np.linalg.svd(off,full_matrices=False); return Vt[:rank]
def fp(trn,te,env_rank,batch_rank):
    sc=StandardScaler().fit(X[trn]); Z=sc.transform(X)
    Bs=[]
    if env_rank: Bs.append(basis(Z[trn],np.array([f'{y_}|{e}' for y_,e in zip(y[trn],env[trn])]) if False else env[trn],env_rank))
    if batch_rank and len(set(zip(y[trn],role[trn])))>2:
        keys=np.array([f'{y_}|{r}' for y_,r in zip(y[trn],role[trn])])
        # within-class batch offsets: center each class first
        Zc=Z[trn]-np.stack([Z[trn][y[trn]==k].mean(0) for k in y[trn]])
        Bs.append(basis(Zc,keys,batch_rank))
    if Bs:
        Bm=np.linalg.qr(np.vstack(Bs).T)[0].T; Z=Z-(Z@Bm.T)@Bm
    return LDA(solver='lsqr',shrinkage='auto').fit(Z[trn],y[trn]).predict(Z[te])
for er,br in ((0,0),(1,0),(2,0),(4,0),(0,1),(0,2),(2,1)):
    c=sum((fp(tr&(env!=e),tr&(env==e),er,br)==y[tr&(env==e)]).sum() for e in sorted(set(env[tr])))
    r=[c/tr.sum()]
    for s_,d_ in (('train_original','train_supplemental'),('train_supplemental','train_original')):
        r.append((fp(role==s_,role==d_,er,br)==y[role==d_]).mean())
    for o in ('old_outside','fresh_outside'): r.append((fp(tr,role==o,er,br)==y[role==o]).mean())
    print(f'env_rank={er} batch_rank={br}: env {r[0]:.3f} o>s {r[1]:.3f} s>o {r[2]:.3f} | old {r[3]:.3f} fresh {r[4]:.3f}',flush=True)
