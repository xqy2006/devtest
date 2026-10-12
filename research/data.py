"""Historical 17-class panel loader with role/batch/group metadata (no labels used at inference)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROLES = ("train_original", "train_supplemental")


def load_historical():
    rows = [json.loads(line) for line in (ROOT / "data/historical/all_rows_834.jsonl").read_text(encoding="utf-8-sig").splitlines() if line]
    for row in rows:
        if isinstance(row["numbers"], str):
            row["numbers"] = json.loads(row["numbers"])
    classes = json.loads((ROOT / "data/historical/model_order.json").read_text())
    groups = json.loads((ROOT / "data/historical/native_groups_278.json").read_text())
    return rows, classes, groups


class Panel:
    def __init__(self):
        self.rows, self.classes, groups = load_historical()
        index = {c: i for i, c in enumerate(self.classes)}
        self.y = np.array([index[r["source"]] for r in self.rows])
        self.env = np.array([r["condition_id"] for r in self.rows])
        self.role = np.array([r["role"] for r in self.rows])
        # batch = role cohort; only meaningful for classes with several training batches
        self.batch = np.array([r["role"] for r in self.rows])
        self.train_roles = TRAIN_ROLES
        self.train = np.flatnonzero(np.isin(self.role, TRAIN_ROLES))
        self.group_members = np.array([g["canonical_indices"] for g in groups])
        self.group_role = np.array([g["role"] for g in groups])
        self.group_y = self.y[self.group_members[:, 0]]
        assert (self.y[self.group_members] == self.group_y[:, None]).all()


PAIR_DIR = ROOT / "data/codex_pair_20261011/data"
PAIR_ROLE = "train_codex_20261011"  # reference-only supplement; never a holdout


def extended_panel(p: Panel | None = None):
    """Historical panel + the 2026-10-11 paired Codex references (training rows of their own batch).

    Only complete native triples of the supplement become groups; incomplete triples' rows stay
    as single training references."""
    import copy
    p = p or Panel()
    new = [json.loads(line) for line in (PAIR_DIR / "codex_pair_references.jsonl").read_text(encoding="utf-8-sig").splitlines() if line]
    triples = json.loads((PAIR_DIR / "native_triples.json").read_text())
    q = copy.copy(p)
    n0 = len(p.rows)
    q.rows = p.rows + new
    index = {c: i for i, c in enumerate(p.classes)}
    q.y = np.concatenate([p.y, [index[r["source"]] for r in new]])
    q.env = np.concatenate([p.env, [r["condition_id"] for r in new]])
    q.role = np.concatenate([p.role, [PAIR_ROLE] * len(new)])
    q.batch = q.role
    q.train_roles = TRAIN_ROLES + (PAIR_ROLE,)
    q.train = np.flatnonzero(np.isin(q.role, q.train_roles))
    pos = {r["row_id"]: n0 + i for i, r in enumerate(new)}
    complete = [t for t in triples if t["complete"]]
    gm = np.array([[pos[rid] for rid in t["accepted_row_ids"]] for t in complete])
    q.group_members = np.vstack([p.group_members, gm])
    q.group_role = np.concatenate([p.group_role, [PAIR_ROLE] * len(gm)])
    q.group_y = q.y[q.group_members[:, 0]]
    assert (q.y[q.group_members] == q.group_y[:, None]).all()
    return q


def subset_panel(p: Panel, keep_classes):
    """Copy of the panel restricted to some classes (for enrollment experiments)."""
    import copy
    q = copy.copy(p)
    keep_rows = np.flatnonzero(np.isin(np.array([p.classes[c] for c in p.y]), keep_classes))
    remap = {int(old): new for new, old in enumerate(keep_rows)}
    q.rows = [p.rows[i] for i in keep_rows]
    q.classes = [c for c in p.classes if c in keep_classes]
    index = {c: i for i, c in enumerate(q.classes)}
    q.y = np.array([index[r["source"]] for r in q.rows])
    q.env, q.role, q.batch = p.env[keep_rows], p.role[keep_rows], p.batch[keep_rows]
    q.train = np.flatnonzero(np.isin(q.role, TRAIN_ROLES))
    gkeep = [g for g in range(len(p.group_members)) if all(int(i) in remap for i in p.group_members[g])]
    q.group_members = np.array([[remap[int(i)] for i in p.group_members[g]] for g in gkeep])
    q.group_role = p.group_role[gkeep]
    q.group_y = q.y[q.group_members[:, 0]]
    return q
