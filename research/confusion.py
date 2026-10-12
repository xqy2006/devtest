import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
from research.data import Panel
from research.harness import env_holdout
from research.methods_multi import MultiFamily
from research.methods_baseline import ProductionBaseline
p = Panel()
short = [c.replace('claude-','c-').replace('-20251001','') for c in p.classes]
for name, m in (("production", ProductionBaseline()), ("lda combo5", MultiFamily(["value355", "sdiff72", "scalar", "repeat", "digits"]))):
    r = env_holdout(m, p)
    pool = r["_pool"]; pred = r["_s1"][pool].argmax(1); y = p.y[pool]
    print("==", name, "per class k1:", " ".join(f"{s}:{(pred[y==c]==c).sum()}" for c, s in enumerate(short)))
    conf = {}
    for t, q in zip(y, pred):
        if t != q: conf[(short[t], short[q])] = conf.get((short[t], short[q]), 0) + 1
    print("   top confusions:", sorted(conf.items(), key=lambda kv: -kv[1])[:14])
