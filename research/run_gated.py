import sys, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import Panel
from research.harness import summarize, dump
from research.methods_refine import GatedRefine
from research.methods_baseline import ProductionBaseline
p = Panel()
fams = ["value355", "sdiff72", "scalar", "repeat", "digits"]
r = summarize("gated prod + local LDA", GatedRefine(ProductionBaseline(), fams, tau=0.1), p)
dump(r, "research/out_gated.json")
print(json.dumps(r["env"]["k1"]["per_class"]))
