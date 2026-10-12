import sys, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import Panel
from research.harness import summarize
from research.methods_multi import MultiFamily
p = Panel()
configs = {
 "lda:value355": (["value355"], "lda"),
 "lda:value355+ordered": (["value355", "ordered"], "lda"),
 "lda:combo5": (["value355", "sdiff72", "scalar", "repeat", "digits"], "lda"),
 "lda:combo7": (["value355", "ordered", "sdiff72", "scalar", "repeat", "digits", "absdiff"], "lda"),
 "lr:combo5": (["value355", "sdiff72", "scalar", "repeat", "digits"], "logreg"),
 "lr:combo7": (["value355", "ordered", "sdiff72", "scalar", "repeat", "digits", "absdiff"], "logreg"),
}
for name, (fams, clf) in configs.items():
    summarize(name, MultiFamily(fams, clf), p)
