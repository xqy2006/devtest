"""Production scorer wrapped for the harness (no changes to production code)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
import fingerprint as fp  # noqa: E402
from bank_builder import fit_robust_artifacts  # noqa: E402


class ProductionBaseline:
    def fit(self, p, train):
        rows = []
        for i in train:
            r = p.rows[i]
            rows.append({**r, "counts": fp.count_numbers(r["numbers"])})
        return {"robust": fit_robust_artifacts(rows, list(p.classes))}

    def score(self, model, p, idx):
        return np.array([fp.robust_score_numbers(p.rows[i]["numbers"], model)["fused"] for i in idx])
