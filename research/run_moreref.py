"""Development check of the 'more references help' hypothesis (outside panels are exposed)."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np
from research.data import Panel, TRAIN_ROLES
from research.harness import _k3, _tally
from research.methods_refine import GatedRefine
from research.methods_baseline import ProductionBaseline
p = Panel()
fams = ["value355", "sdiff72", "scalar", "repeat", "digits"]
for name, train_roles, test_role in (
    ("train -> fresh", TRAIN_ROLES, "fresh_outside"),
    ("train+old -> fresh", TRAIN_ROLES + ("old_outside",), "fresh_outside"),
    ("train -> old", TRAIN_ROLES, "old_outside"),
    ("train+fresh -> old", TRAIN_ROLES + ("fresh_outside",), "old_outside"),
):
    for mname, m in (("production", ProductionBaseline()), ("gated", GatedRefine(ProductionBaseline(), fams))):
        train = np.flatnonzero(np.isin(p.role, train_roles))
        model = m.fit(p, train)
        test = np.flatnonzero(p.role == test_role); gtest = np.flatnonzero(p.group_role == test_role)
        k1 = _tally(m.score(model, p, test).argmax(1), p.y[test], p.classes)
        k3 = _tally(_k3(m, model, p, p.group_members[gtest]).argmax(1), p.group_y[gtest], p.classes)
        print(f"{name:20s} {mname:10s} k1 {k1['correct']}/{k1['total']} k3 {k3['correct']}/{k3['total']}", flush=True)
