"""Replication test of branch-node differences across deployments.

Discovery: nodes (anchored prefix for t < 40, or previous two numbers) in ONE batch where the two
labels' branch choices differ (G-test p < alpha, >= 5 visits per label). For each discovered node,
the label-preferred branches are fixed from the discovery batch. Replication: in every OTHER batch,
for each visit to that node, does the chosen branch favour the label it came from? Score per visit =
log-ratio of the discovery-batch branch frequencies (Dirichlet 0.5); replication accuracy and a
permutation null (labels shuffled within batch x environment in the replication batch)."""
import sys, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L

p, yb = L.p, L.yb
BATCHES = L.TRB + ("old_outside", "fresh_outside")
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05


def ctxs(x, t):
    out = []
    if t < 40:
        out.append(("P",) + tuple(x[:t]))
    if t >= 2:
        out.append(("B", x[t - 2], x[t - 1]))
    return out


visits = defaultdict(list)  # (batch) -> list of (row, ctx, value)
for b in BATCHES:
    for i in L.rows((b,)):
        x = list(map(int, p.rows[i]["numbers"]))
        for t, v in enumerate(x):
            for c in ctxs(x, t):
                visits[b].append((int(i), c, v))


def discover(b):
    cnt = defaultdict(lambda: [Counter(), Counter()])
    for i, c, v in visits[b]:
        cnt[c][int(yb[i])][v] += 1
    nodes = {}
    for c, (a, s) in cnt.items():
        na, ns = sum(a.values()), sum(s.values())
        if na < 5 or ns < 5:
            continue
        top = [v for v, _ in (a + s).most_common(4)]
        tab = np.array([[a[v] for v in top] + [na - sum(a[v] for v in top)],
                        [s[v] for v in top] + [ns - sum(s[v] for v in top)]], float) + 0.5
        e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        if chi2.sf(2 * (tab * np.log(tab / e)).sum(), 4) < ALPHA:
            nodes[c] = (a, s, na, ns)
    return nodes


def replicate(nodes, b, labels):
    score = defaultdict(float)
    for i, c, v in visits[b]:
        if c in nodes:
            a, s, na, ns = nodes[c]
            score[i] += np.log((s[v] + 0.5) / (ns + 2.5)) - np.log((a[v] + 0.5) / (na + 2.5))
    rows = [i for i in score]
    if not rows:
        return np.nan, 0
    acc = np.mean([(score[i] > 0) == labels[i] for i in rows if score[i] != 0])
    return acc, len(rows)


rng = np.random.default_rng(3)
print(f"alpha={ALPHA}")
for bd in (L.PAIR_ROLE, "train_original", "train_supplemental"):
    nodes = discover(bd)
    print(f"\ndiscovery batch {bd}: {len(nodes)} nodes")
    for br in BATCHES:
        if br == bd:
            continue
        lab = {int(i): int(yb[i]) for i in L.rows((br,))}
        acc, n = replicate(nodes, br, lab)
        null = []
        r = L.rows((br,))
        for _ in range(200):
            perm = {}
            for e in set(p.env[r]):
                m = r[p.env[r] == e]
                for i, y in zip(m, rng.permutation(yb[m])):
                    perm[int(i)] = int(y)
            null.append(replicate(nodes, br, perm)[0])
        null = np.array(null)
        print(f"   replicate in {br[-12:]:>12}: responses touching a node {n:3d}, accuracy {acc:.3f}, "
              f"null mean {np.nanmean(null):.3f}, p = {(1 + np.sum(null >= acc)) / 201:.3f}")
