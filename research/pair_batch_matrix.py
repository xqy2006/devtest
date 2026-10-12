"""Batch-to-batch transfer for the confusable pair, equal reference sizes (24 per class).

Batches = collection cohorts (role). Diagonal = env-CV inside the batch (subsampled to 36/class);
off-diagonal = fit on 24/class of batch X, score all of batch Y. Averaged over random draws.
Labels are used only for scoring."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import PAIR_ROLE, extended_panel
from research.features import extract
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

FAMS = ["value355", "sdiff72", "scalar", "repeat", "digits"]
p = extended_panel()
A, S = p.classes.index("gpt-6-astra"), p.classes.index("gpt-6.1-sol")
pair = np.flatnonzero(np.isin(p.y, [A, S]))
F = extract([p.rows[i] for i in pair], FAMS)
X = np.zeros((len(p.rows), sum(F[f].shape[1] for f in FAMS))); X[pair] = np.hstack([F[f] for f in FAMS])
yb = (p.y == S).astype(int)
dates = {r: sorted({p.rows[i]["collected_at"][:10] for i in pair if p.role[i] == r}) for r in set(p.role[pair])}
B = ["train_original", "train_supplemental", "old_outside", "fresh_outside", PAIR_ROLE]
rng = np.random.default_rng(3)

def fp(ref, te):
    sc = StandardScaler().fit(X[ref]); m = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(X[ref]), yb[ref])
    return m.decision_function(sc.transform(X[te]))

def sub(role, n=24):
    return np.concatenate([rng.choice(np.flatnonzero((p.y == c) & (p.role == role)), n, replace=False) for c in (A, S)])

print({r: dates[r] for r in B})
REPS = 20
M = np.zeros((len(B), len(B)))
for i, bx in enumerate(B):
    for j, by in enumerate(B):
        accs = []
        for _ in range(REPS if (i == j or bx in ("old_outside", PAIR_ROLE) or by == PAIR_ROLE) else 1):
            if i == j:
                idx = sub(bx); d = np.zeros(len(p.rows))
                for e in set(p.env[idx]):
                    te, tr = idx[p.env[idx] == e], idx[p.env[idx] != e]; d[te] = fp(tr, te)
                accs.append(((d[idx] > 0) == yb[idx]).mean())
            else:
                ref = sub(bx); te = np.flatnonzero(np.isin(p.y, [A, S]) & (p.role == by))
                accs.append(((fp(ref, te) > 0) == yb[te]).mean())
        M[i, j] = np.mean(accs)
print("rows = reference batch (24/class), cols = scored batch; diagonal = env-CV inside batch")
print(" " * 20 + "".join(f"{b[:14]:>16}" for b in B))
for i, b in enumerate(B):
    print(f"{b[:20]:20}" + "".join(f"{M[i, j]:16.3f}" for j in range(len(B))))
