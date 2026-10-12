import sys, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import Panel
from research.harness import summarize
from research.methods_multi import MultiFamily
p = Panel()
base = ["value355", "sdiff72", "scalar", "repeat", "digits"]
configs = {
 "combo5": (base, "lda", {}),
 "combo5+value71": (base + ["value71"], "lda", {}),
 "combo5+bintrans6": (base + ["bintrans6"], "lda", {}),
 "combo5+autocorr+updown": (base + ["autocorr", "updown"], "lda", {}),
 "combo5+sdiff_lag2": (base + ["sdiff_lag2"], "lda", {}),
 "combo5+format": (base + ["format"], "lda", {}),
 "combo5 value x2": (base, "lda", {"value355": 2.0}),
 "combo5 scalar x2": (base, "lda", {"scalar": 2.0, "repeat": 2.0}),
 "lr C=1": (base, "logreg", {}, 1.0),
 "lr C=10": (base, "logreg", {}, 10.0),
}
for name, cfg in configs.items():
    fams, clf, w = cfg[:3]
    C = cfg[3] if len(cfg) > 3 else 0.01
    summarize(name, MultiFamily(fams, clf, C=C, family_weights=w), p)
