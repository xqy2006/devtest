import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import Panel
from research.harness import summarize
from research.methods_multi import MultiFamily
from research.methods_baseline import ProductionBaseline
from research.methods_fusion import Fusion
p = Panel()
lda = MultiFamily(["value355", "sdiff72", "scalar", "repeat", "digits"])
for w in (0.5, 1.0, 2.0):
    summarize(f"fuse prod+lda w={w}", Fusion([ProductionBaseline(), lda], [1.0, w]), p)
summarize("fuse nested", Fusion([ProductionBaseline(), lda], grid=[(1, w) for w in (0.25, 0.5, 1, 2, 4)], nested=True), p)
