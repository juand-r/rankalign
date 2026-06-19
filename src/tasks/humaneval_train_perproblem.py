"""Per-problem TRAIN-candidate tasks for HumanEval v2.1correct (additive; no existing logic touched).

Registers `humaneval-v2.1correct-{upper,multi}-train-<slug>` tasks, one per TRAIN problem,
each scoring that problem's OWN ~29 candidates (from
data/humaneval/v2.1correct-{ds}-train-perproblem/, built by scripts/build_perproblem_train_csvs.py).
This lets us average ROC over the 80 train problems -- a like-for-like comparison with the
per-problem TEST tasks (train/test are disjoint problem sets).

Reuses humaneval.py's per-problem factory + COMMON config unchanged. The factory returns
(global_train, this_csv_items); we pass the per-problem TRAIN csv, so the per-problem train
candidates land in L_test and are scored by a NORMAL eval (run WITHOUT --train).
"""
import os

from task_registry import register_task
from . import humaneval as he

_SPECS = [
    ("upper", he.create_load_data_v2_1_upper_for_problem, "_V2_1_UPPER_COMMON"),
    ("multi", he.create_load_data_v2_1_multi_for_problem, "_V2_1_MULTI_COMMON"),
]

_registered = []
for _ds, _factory, _common_attr in _SPECS:
    _common = getattr(he, _common_attr, None)
    _ddir = os.path.join(he.DATA_DIR, "humaneval", f"v2.1correct-{_ds}-train-perproblem")
    if _common is None or not os.path.isdir(_ddir):
        continue
    for _fn in sorted(os.listdir(_ddir)):
        if not _fn.endswith(".csv"):
            continue
        _slug = _fn[:-4]  # e.g. humaneval_101
        _csv = os.path.join(_ddir, _fn)
        try:
            register_task({
                "name": f"humaneval-v2.1correct-{_ds}-train-{_slug}",
                "load_data": _factory(_csv),
                "description": f"HumanEval v2.1correct-{_ds} per-problem TRAIN candidates: {_slug}",
                **_common,
            })
            _registered.append(f"{_ds}/{_slug}")
        except Exception as e:  # noqa: BLE001 — keep registry resilient; log loudly
            print(f"[humaneval-train-perproblem] could not register {_ds}/{_slug}: {e}")

if _registered:
    print(f"[humaneval-train-perproblem] Registered {len(_registered)} per-problem TRAIN tasks")
