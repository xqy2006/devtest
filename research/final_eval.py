"""Final evaluation of the deliverable scorer on every protocol (historical 17-class panel)."""
import json, sys, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from research.data import Panel
from research.harness import summarize, dump
if "--v1" in sys.argv:  # reproduce earlier deliverable versions
    from research.gated_fingerprint_v1 import GatedDeliverable
elif "--v2" in sys.argv:
    from research.gated_fingerprint_v2 import GatedDeliverable
elif "--v3" in sys.argv:
    from research.gated_fingerprint_v3 import GatedDeliverable
elif "--v4" in sys.argv:
    from research.gated_fingerprint_v4 import GatedDeliverable
else:
    from research.gated_fingerprint import GatedDeliverable
TAG = ("" if "--v1" in sys.argv else "_v2" if "--v2" in sys.argv else "_v3" if "--v3" in sys.argv
       else "_v4" if "--v4" in sys.argv else "_v5")
from research.methods_baseline import ProductionBaseline
calibrate = "--calibrate" in sys.argv
p = Panel()
t = time.time()
out = {}
for name, m in (("production", ProductionBaseline()), ("gated_deliverable" + ("_calibrated" if calibrate else ""), GatedDeliverable(calibrate))):
    out[name] = summarize(name, m, p)
    print(f"  ({time.time()-t:.0f}s)", flush=True)
dump(out, f"research/final_results{'_calibrated' if calibrate else ''}{TAG}.json")
