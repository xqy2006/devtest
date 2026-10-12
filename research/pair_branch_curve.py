"""Does the opening-branch evidence grow with the number of references? (pooled training batches,
leave-one-environment-out; n references per class drawn from the other environments)

Components: branch = SeqLR prefix + prev2 depth sums (support >= 10), dense = v2 local LDA;
combined = equal-weight sum after standardizing each by its spread on out-of-fold reference scores
is approximated here by standardizing with the test-fold-free spread of the reference fit."""
import sys, warnings, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab2 as M
from research.seqlr import SeqLR, LEVELS
from research.gated_fingerprint import LOCAL_FAMILIES
L = M.L; p = L.p; yb = L.yb
tr = L.rows(L.TRB)
g = L.groups(L.TRB)
rng = np.random.default_rng(4)
res = {}
for n in (48, 96, 144, 190):
    acc = {"branch": [], "dense": [], "sum": [], "branch_k3": [], "sum_k3": [], "nodes": []}
    for rep in range(4):
        db, dd = np.zeros(len(p.rows)), np.zeros(len(p.rows))
        nodes = []
        for e in sorted(set(p.env[tr])):
            pool = tr[p.env[tr] != e]
            ref = np.concatenate([rng.choice(pool[yb[pool] == c], n, replace=False) for c in (0, 1)])
            te = tr[p.env[tr] == e]
            lr = SeqLR(5.0, 40, 10).fit([M.SEQ[int(i)] for i in ref], [int(yb[i]) for i in ref])
            Z = np.array([lr.level_scores(M.SEQ[int(i)]) for i in te])
            db[te] = Z[:, LEVELS.index("prefix")] + Z[:, LEVELS.index("prev2")]
            dd[te] = L.Model(list(LOCAL_FAMILIES), "lda").fit(ref).decision(te)
        sb, sd = db[tr].std(), dd[tr].std()
        ds = db / sb + dd / sd
        for k, d in (("branch", db), ("dense", dd), ("sum", ds)):
            acc[k].append(np.mean((d[tr] > 0) == yb[tr]))
        acc["branch_k3"].append(np.mean((db[g].sum(1) > 0) == yb[g[:, 0]]))
        acc["sum_k3"].append(np.mean((ds[g].sum(1) > 0) == yb[g[:, 0]]))
    res[n] = {k: float(np.mean(v)) for k, v in acc.items() if v}
    print(f"n={n}/class: " + "  ".join(f"{k} {v:.3f}" for k, v in res[n].items()), flush=True)
Path(__file__).with_name("out_pair_branch_curve.json").write_text(json.dumps(res, indent=1))
