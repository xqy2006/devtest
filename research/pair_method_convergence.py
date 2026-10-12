"""Do very different vector-mining methods read the same information? Out-of-fold decisions under
pooled leave-one-environment-out (three training batches) for: v3 cluster model, nearest shrunken
centroids, random-subspace LDA, PLS-DA. Reports accuracy, correlation of decisions, and the
equal-weight ensemble of standardized decisions (scales from reference-row spread only)."""
import sys, warnings, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_mining as M
import research.pair_seq_lab2 as M2
L = M.L; p = L.p; yb = L.yb
tr = L.rows(L.TRB); g = L.groups(L.TRB)
fams = M.DENSE + ["opening"]
makers = {"v3": lambda: M2.V3Local(), "pam": lambda: M.PAM(fams), "rslda": lambda: M.RSLDA(fams), "pls": lambda: M.PLS(fams)}
D = {k: np.zeros(len(p.rows)) for k in makers}
for e in sorted(set(p.env[tr])):
    ref, te = tr[p.env[tr] != e], tr[p.env[tr] == e]
    for k, mk in makers.items():
        m = mk().fit(ref)
        sc = np.std(m.decision(ref)) + 1e-9
        D[k][te] = m.decision(te) / sc
    print("fold", e, flush=True)
names = list(makers)
S = np.column_stack([D[k][tr] for k in names])
for k, n in enumerate(names):
    print(f"{n:6} env-CV k1 {np.mean((S[:, k] > 0) == yb[tr]):.3f} k3 {np.mean((D[n][g].sum(1) > 0) == yb[g[:, 0]]):.3f}")
print("correlation of out-of-fold decisions:\n", np.round(np.corrcoef(S.T), 2))
Zs = S / S.std(0)
ens = Zs.sum(1)
de = np.zeros(len(p.rows)); de[tr] = ens
print(f"equal-weight ensemble: k1 {np.mean((ens > 0) == yb[tr]):.3f} k3 {np.mean((de[g].sum(1) > 0) == yb[g[:, 0]]):.3f}")
json.dump({"names": names, "corr": np.corrcoef(S.T).tolist()}, open("research/out_pair_method_convergence.json", "w"), indent=1)
