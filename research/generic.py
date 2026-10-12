"""Generic panels for the current 18-class library and the evaluation-only recent panels.

The same methods run unchanged: a candidate set is defined only by its reference rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))
import fingerprint as fp  # noqa: E402


def read_rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8-sig").splitlines() if line]


class GenericPanel:
    """rows: reference rows (role 'library') + optional evaluation rows (role given)."""

    def __init__(self, library, classes, extra=(), groups=None):
        rows = []
        for r in library:
            rows.append({**r, "numbers": fp.parse_numbers(r["text"]), "role": "library"})
        for r, role in extra:
            rows.append({**r, "numbers": fp.parse_numbers(r["text"]), "role": role})
        self.rows = rows
        self.classes = list(classes)
        index = {c: i for i, c in enumerate(self.classes)}
        self.y = np.array([index.get(r["source"], -1) for r in rows])
        self.env = np.array([r["condition_id"] for r in rows])
        self.role = np.array([r["role"] for r in rows])
        self.row_index = {r["row_id"]: i for i, r in enumerate(rows)}
        self.groups = groups or []


def current_library():
    return read_rows("data/current/gpt_reference.jsonl") + read_rows("data/current/claude_reference.jsonl")


def library_groups(panel):
    cells = {}
    for i in np.flatnonzero(panel.role == "library"):
        cells.setdefault((panel.rows[i]["source"], panel.env[i]), []).append(i)
    assert all(len(v) == 3 for v in cells.values())
    return np.array(list(cells.values()))


def env_holdout_library(method, panel):
    """Leave-one-environment-out over the library rows (k1 and native library triples)."""
    pool = np.flatnonzero(panel.role == "library")
    members = library_groups(panel)
    C = len(panel.classes)
    s1 = np.zeros((len(panel.rows), C))
    s3 = np.zeros((len(members), C))
    genv = panel.env[members[:, 0]]
    for e in sorted(set(panel.env[pool])):
        test = pool[panel.env[pool] == e]
        model = method.fit(panel, pool[panel.env[pool] != e])
        s1[test] = method.score(model, panel, test)
        gi = np.flatnonzero(genv == e)
        s3[gi] = group_scores(method, model, panel, members[gi])
    y1 = panel.y[pool]
    y3 = panel.y[members[:, 0]]
    return per_class(s1[pool].argmax(1), y1, panel.classes), per_class(s3.argmax(1), y3, panel.classes)


def group_scores(method, model, panel, members):
    if hasattr(method, "score_group"):
        return method.score_group(model, panel, members)
    return method.score(model, panel, members.ravel()).reshape(len(members), members.shape[1], -1).mean(1)


def per_class(pred, truth, classes):
    out = {}
    for c in np.unique(truth):
        m = truth == c
        out[classes[c]] = [int((pred[m] == c).sum()), int(m.sum())]
    total = [sum(v[0] for v in out.values()), sum(v[1] for v in out.values())]
    return {"correct": total[0], "total": total[1], "per_class": out}


def holdout_eval(method, panel, role, planned_groups):
    """Fit on all library rows; score evaluation rows. ``planned_groups`` lists planned row-id
    triples; missing responses count as failures in end-to-end denominators."""
    lib = np.flatnonzero(panel.role == "library")
    model = method.fit(panel, lib)
    test = np.flatnonzero(panel.role == role)
    s1 = method.score(model, panel, test)
    k1_valid = int((s1.argmax(1) == panel.y[test]).sum())
    planned_rows = sum(len(g) for g in planned_groups)
    strict = []
    fallback = []
    for g in planned_groups:
        present = [panel.row_index[r] for r in g if r in panel.row_index]
        truth = panel.y[present[0]] if present else None
        if len(present) == 3:
            s = group_scores(method, model, panel, np.array([present]))
            strict.append(bool(s.argmax(1)[0] == truth))
        else:
            strict.append(False)
        if present:
            s = group_scores(method, model, panel, np.array([present]))
            fallback.append(bool(s.argmax(1)[0] == truth))
        else:
            fallback.append(False)
    return {"k1_valid": [k1_valid, len(test)], "k1_end_to_end": [k1_valid, planned_rows],
            "k3_strict": [int(sum(strict)), len(planned_groups)],
            "k3_budget_fallback": [int(sum(fallback)), len(planned_groups)],
            "predictions": [panel.classes[i] for i in s1.argmax(1)]}
