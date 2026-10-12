import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from research.data import Panel
from research.features import extract
p=Panel()
a,b=p.classes.index('gpt-6-astra'),p.classes.index('gpt-6.1-sol')
fams=["value355","sdiff72","scalar","repeat","digits"]
F=extract(p.rows,fams); X=np.hstack([F[f] for f in fams])
TR=np.isin(p.role,['train_original','train_supplemental'])
pair=np.isin(p.y,[a,b]); yb=(p.y==b).astype(int)
def run(train_mask, test_mask, mode, k):
    # projection learned on ALL training rows (all classes) inside the fold
    sc=StandardScaler().fit(X[train_mask])
    Z=sc.transform(X)
    if mode=='pca':
        P=PCA(n_components=k,random_state=0).fit(Z[train_mask]); Z=P.transform(Z)
    elif mode=='lda17':
        L=LDA(solver='eigen',shrinkage='auto',n_components=k).fit(Z[train_mask],p.y[train_mask]); Z=L.transform(Z)
    elif mode=='gpt_within':
        # remove top-k directions of pooled within-class scatter (noise) estimated on all classes
        R=Z[train_mask]-np.stack([Z[train_mask&(p.y==c)].mean(0) for c in p.y[train_mask]])
        _,_,Vt=np.linalg.svd(R,full_matrices=False); Vn=Vt[:k]; Z=Z-(Z@Vn.T)@Vn
    trp=train_mask&pair
    m=LDA(solver='lsqr',shrinkage='auto').fit(Z[trp],yb[trp])
    tep=test_mask&pair
    return (m.predict(Z[tep])==yb[tep])
for mode,ks in (('none',[0]),('pca',[10,30,60,120]),('lda17',[8,16]),('gpt_within',[5,20,50])):
    for k in ks:
        ok=np.concatenate([run(TR&(p.env!=e),TR&(p.env==e),mode,k) for e in sorted(set(p.env[TR]))])
        r=[ok.mean()]
        for s_,d_ in (('train_original','train_supplemental'),('train_supplemental','train_original')):
            trm=TR&~(pair&(p.role==d_)); r.append(run(trm,pair&(p.role==d_),mode,k).mean())
        for o in ('old_outside','fresh_outside'):
            r.append(run(TR,p.role==o,mode,k).mean())
        print(f'{mode:10s} k={k:3d} env {r[0]:.3f} o>s {r[1]:.3f} s>o {r[2]:.3f} | old {r[3]:.3f} fresh {r[4]:.3f}',flush=True)
