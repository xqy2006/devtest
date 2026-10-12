import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
from research.data import Panel
from research.harness import summarize
from research.methods_refine import GatedRefine
from research.methods_baseline import ProductionBaseline
p = Panel()
fams = ["value355", "sdiff72", "scalar", "repeat", "digits"]
class Logged(GatedRefine):
    def fit(self, p, train):
        m = super().fit(p, train)
        self.log.append([[p.classes[c] for c in K] for K in m["cluster_sets"]])
        return m
for tau in (0.05, 0.1, 0.2, 0.3):
    g = Logged(ProductionBaseline(), fams, tau=tau); g.log = []
    summarize(f"gated tau={tau}", g, p)
    print("   clusters per fit:", sorted(set(str(x) for x in g.log)))
