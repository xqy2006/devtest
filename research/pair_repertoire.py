"""Repertoire test: does a response share more exact motifs with its nearest references of its
own label than with those of the other label? (numbers only; references never include the query's
environment / batch under the protocols used)

Per response and label c: overlap with each reference r of label c = number of shared exact
k-grams (k=3, positions ignored, opening path excluded: first 10 numbers dropped to avoid
re-measuring the branch evidence). Statistics per label: max, mean of top-5, mean over all.
Decision = statistic(Sol) - statistic(Astra)."""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import research.pair_seq_lab as L

p, yb = L.p, L.yb
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SKIP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
GR = {}
for i in L.pair:
    x = p.rows[i]["numbers"][SKIP:]
    GR[int(i)] = {tuple(x[t:t + K]) for t in range(len(x) - K + 1)}


class Repertoire:
    def __init__(self, stat="top5"):
        self.stat = stat

    def fit(self, ref):
        self.ref = ref
        # inverted index: k-gram -> reference ids
        self.inv = {}
        for r in ref:
            for g in GR[int(r)]:
                self.inv.setdefault(g, []).append(int(r))
        self.lab = {int(r): int(yb[r]) for r in ref}
        self.n = [int((yb[ref] == c).sum()) for c in (0, 1)]
        return self

    def decision(self, idx):
        out = []
        for i in idx:
            cnt = {}
            for g in GR[int(i)]:
                for r in self.inv.get(g, ()):
                    cnt[r] = cnt.get(r, 0) + 1
            per = [sorted([v for r, v in cnt.items() if self.lab[r] == c], reverse=True) for c in (0, 1)]
            if self.stat == "max":
                s = [pc[0] if pc else 0 for pc in per]
            elif self.stat == "top5":
                s = [np.mean((pc + [0] * 5)[:5]) for pc in per]
            else:
                s = [sum(pc) / self.n[c] for c, pc in enumerate(per)]
            out.append(s[1] - s[0])
        return np.array(out, dtype=float) + 1e-9 * np.random.default_rng(0).standard_normal(len(idx))


for stat in ("max", "top5", "mean"):
    print(L.line(f"repertoire k={K} skip={SKIP} {stat}", L.evaluate(lambda: Repertoire(stat))), flush=True)
