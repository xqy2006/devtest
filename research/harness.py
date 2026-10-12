"""Evaluation protocols for the historical 17-class panel.

A method implements ``fit(panel, train_idx) -> model`` and ``score(model, panel, idx) -> (n, C)``.
Optionally ``score_group(model, panel, members) -> (g, C)`` for k3 using only the group's own
three responses; otherwise k3 averages the three per-response scores.

Every fit sees only the rows in ``train_idx``. Test rows are scored one at a time (k1) or
as their own fixed native triple (k3). Labels/environments/batches of test rows are used only
for counting accuracy.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from research.data import Panel, TRAIN_ROLES



def _k3(method, model, p, members):
    if hasattr(method, "score_group"):
        return method.score_group(model, p, members)
    flat = method.score(model, p, members.ravel())
    return flat.reshape(len(members), 3, -1).mean(axis=1)


def _tally(pred, truth, classes):
    per = defaultdict(lambda: [0, 0])
    for a, t in zip(pred, truth):
        per[classes[t]][0] += int(a == t)
        per[classes[t]][1] += 1
    correct = int(sum(v[0] for v in per.values()))
    total = int(sum(v[1] for v in per.values()))
    bal = float(np.mean([v[0] / v[1] for v in per.values()]))
    return {"correct": correct, "total": total, "acc": correct / total, "balanced": bal,
            "per_class": {c: per[c] for c in classes if c in per}}


def env_holdout(method, p: Panel, roles=None):
    """Leave-one-environment-out over allowed training rows (all 17 classes)."""
    roles = roles or getattr(p, "train_roles", TRAIN_ROLES)
    pool = np.flatnonzero(np.isin(p.role, roles))
    gpool = np.flatnonzero(np.isin(p.group_role, roles))
    s1 = np.zeros((len(p.rows), len(p.classes)))
    s3 = np.zeros((len(p.group_members), len(p.classes)))
    for e in sorted(set(p.env[pool])):
        test = pool[p.env[pool] == e]
        train = pool[p.env[pool] != e]
        model = method.fit(p, train)
        s1[test] = method.score(model, p, test)
        gtest = gpool[p.env[p.group_members[gpool, 0]] == e]
        s3[gtest] = _k3(method, model, p, p.group_members[gtest])
    return {"k1": _tally(s1[pool].argmax(1), p.y[pool], p.classes),
            "k3": _tally(s3[gpool].argmax(1), p.group_y[gpool], p.classes),
            "_s1": s1, "_s3": s3, "_pool": pool, "_gpool": gpool}


def batch_cross(method, p: Panel):
    """For every class with several training batches, hold one batch out while the
    other batch stays in the reference library (all other classes keep all training rows)."""
    out = {}
    roles = getattr(p, "train_roles", TRAIN_ROLES)
    multi = [c for c in range(len(p.classes)) if len(set(p.role[(p.y == c) & np.isin(p.role, roles)])) > 1]
    held_roles = [r for r in roles if np.isin(p.y[p.role == r], multi).any()]
    for held in held_roles:
        test_mask = np.isin(p.y, multi) & (p.role == held)
        train = np.flatnonzero(np.isin(p.role, roles) & ~test_mask)
        test = np.flatnonzero(test_mask)
        gtest = np.flatnonzero(np.isin(p.group_y, multi) & (p.group_role == held))
        model = method.fit(p, train)
        s1 = method.score(model, p, test)
        s3 = _k3(method, model, p, p.group_members[gtest])
        out["hold_" + held] = {"k1": _tally(s1.argmax(1), p.y[test], p.classes),
                               "k3": _tally(s3.argmax(1), p.group_y[gtest], p.classes)}
    return out


def outside(method, p: Panel):
    train = np.flatnonzero(np.isin(p.role, getattr(p, "train_roles", TRAIN_ROLES)))
    model = method.fit(p, train)
    out = {}
    for role in ("old_outside", "fresh_outside"):
        test = np.flatnonzero(p.role == role)
        gtest = np.flatnonzero(p.group_role == role)
        s1 = method.score(model, p, test)
        s3 = _k3(method, model, p, p.group_members[gtest])
        out[role] = {"k1": _tally(s1.argmax(1), p.y[test], p.classes),
                     "k3": _tally(s3.argmax(1), p.group_y[gtest], p.classes)}
    return out


def summarize(name, method, p=None, protocols=("env", "batch", "outside"), verbose=True):
    p = p or Panel()
    res = {}
    if "env" in protocols:
        res["env"] = env_holdout(method, p)
    if "batch" in protocols:
        res["batch"] = batch_cross(method, p)
    if "outside" in protocols:
        res["outside"] = outside(method, p)
    if verbose:
        print(format_line(name, res), flush=True)
    return res


def _cc(t, c):
    v = t["per_class"].get(c)
    return f"{v[0]}/{v[1]}" if v else "-"


def format_line(name, res):
    parts = [f"[{name}]"]
    if "env" in res:
        e = res["env"]
        parts.append(f"envCV k1 {e['k1']['correct']}/{e['k1']['total']}={e['k1']['acc']:.3f} "
                     f"(A {_cc(e['k1'], 'gpt-6-astra')} S {_cc(e['k1'], 'gpt-6.1-sol')}) "
                     f"k3 {e['k3']['correct']}/{e['k3']['total']}")
    if "batch" in res:
        for k, v in res["batch"].items():
            parts.append(f"{k[5:]}-held k1 {v['k1']['correct']}/{v['k1']['total']} k3 {v['k3']['correct']}/{v['k3']['total']}")
    if "outside" in res:
        for k, v in res["outside"].items():
            parts.append(f"{k[:5]} k1 {v['k1']['correct']}/{v['k1']['total']} k3 {v['k3']['correct']}/{v['k3']['total']}")
    return " | ".join(parts)


def strip(res):
    if isinstance(res, dict):
        return {k: strip(v) for k, v in res.items() if not str(k).startswith("_")}
    return res


def dump(res, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(strip(res), fh, indent=1, ensure_ascii=False)
