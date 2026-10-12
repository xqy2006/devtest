"""Run a method unchanged on the current 18-class library and the evaluation-only recent panels.

Recent panels are evaluation-only; nothing here is used to select a method.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from research.generic import GenericPanel, current_library, env_holdout_library, holdout_eval, read_rows  # noqa: E402
from research.methods_baseline import ProductionBaseline  # noqa: E402
from research.methods_refine import GatedRefine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAMS = ["value355", "sdiff72", "scalar", "repeat", "digits"]


def methods():
    return {"production": ProductionBaseline(), "gated": GatedRefine(ProductionBaseline(), FAMS, tau=0.1)}


def main(which=("library", "sonnet", "haiku", "system")):
    datasets = json.loads((ROOT / "data/DATASETS.json").read_text(encoding="utf-8"))
    lib = current_library()
    classes18 = datasets["production_18"]["classes"]
    classes17 = [c for c in classes18 if c != "claude-haiku-5-5"]  # the bank the system/sonnet tests used
    lib17 = [r for r in lib if r["source"] in classes17]
    out = {}
    for name, method in methods().items():
        res = {}
        if "library" in which:
            p = GenericPanel(lib, classes18)
            k1, k3 = env_holdout_library(method, p)
            res["library18_envCV"] = {"k1": k1, "k3": k3}
            print(name, "library18 envCV k1", k1["correct"], "/", k1["total"], "k3", k3["correct"], "/", k3["total"], flush=True)
        if "sonnet" in which:
            ds = datasets["sonnet55_holdout"]
            rows = read_rows(ds["responses_file"])
            p = GenericPanel(lib17, classes17, [(r, "holdout") for r in rows])
            groups = [g["row_ids"] for g in ds["groups"]]
            res["sonnet55_holdout"] = holdout_eval(method, p, "holdout", groups)
            print(name, "sonnet55", {k: v for k, v in res["sonnet55_holdout"].items() if k != "predictions"}, flush=True)
        if "haiku" in which:
            ds = datasets["haiku55_holdout"]
            rows = read_rows(ds["responses_file"])
            p = GenericPanel(lib, classes18, [(r, "holdout") for r in rows])
            groups = [g["planned_row_ids"] for g in ds["groups"]]
            res["haiku55_holdout"] = holdout_eval(method, p, "holdout", groups)
            print(name, "haiku55", {k: v for k, v in res["haiku55_holdout"].items() if k != "predictions"}, flush=True)
        if "system" in which:
            ds = datasets["system_prompt_test"]
            rows = read_rows(ds["responses_file"])
            p = GenericPanel(lib17, classes17, [(r, "holdout") for r in rows])
            by_cond = defaultdict(list)
            for g in ds["groups"]:
                by_cond[g["condition"]].append(g["planned_row_ids"])
            res["system_prompt"] = {}
            for cond, groups in sorted(by_cond.items()):
                ids = {rid for g in groups for rid in g}
                sub = GenericPanel(lib17, classes17, [(r, "holdout") for r in rows if r["row_id"] in ids])
                res["system_prompt"][cond] = holdout_eval(method, sub, "holdout", groups)
                r = res["system_prompt"][cond]
                print(name, "system", cond, "k1", r["k1_valid"], "e2e", r["k1_end_to_end"], "k3", r["k3_strict"], flush=True)
        out[name] = res
    (ROOT / "research/out_generality.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("library", "sonnet", "haiku", "system"))
