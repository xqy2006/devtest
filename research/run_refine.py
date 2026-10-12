import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import Panel
from research.harness import summarize
from research.methods_refine import Refined
p = Panel()
fams = ["value355", "sdiff72", "scalar", "repeat", "digits"]
for tau in (0.1, 0.2):
    summarize(f"refine tau={tau}", Refined(fams, tau=tau), p)
