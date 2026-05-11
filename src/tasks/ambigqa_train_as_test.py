"""
AmbigQA train-as-test memorization probe.

Loads the entire AmbigQA training set (data/ambigqa/with_negatives/train.csv)
and returns it as BOTH the train and test split. By construction this is an
overfit probe — same setup as src/tasks/rosch_combined.py for rosch — letting
us isolate train-time TC / online pair selection effects from the
generalization-vs-memorization confound.

DO NOT report cross-task generalization numbers from this task.

Reuses make_prompt / get_completion / get_label / load_csv_items from
src/tasks/ambigqa_v1.py so prompt formatting is identical to the per-question
ambigqa eval tasks.
"""

import os
import sys

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
from .ambigqa_v1 import (
    AQA_TRAIN_CSV,
    load_csv_items,
    make_prompt,
    get_completion,
    get_label,
)


TASK_NAME = "ambigqa-train-as-test"


def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
    """Load every item from ambigqa train.csv; return (items, items)."""
    if not os.path.exists(AQA_TRAIN_CSV):
        raise FileNotFoundError(f"AmbigQA train file not found: {AQA_TRAIN_CSV}")
    items = load_csv_items(AQA_TRAIN_CSV)
    return items, items


if os.path.exists(AQA_TRAIN_CSV):
    try:
        register_task({
            'name': TASK_NAME,
            'load_data': load_data,
            'make_prompt': make_prompt,
            'get_completion': get_completion,
            'get_label': get_label,
            'batch_size': {'with_ref': 1, 'without_ref': 4},
            'supports_split_types': ['random'],
            'description': 'AmbigQA: full training set used as BOTH train and test (memorization probe).',
        })
        print(f"[ambigqa_train_as_test] Registered {TASK_NAME} (train=test, full ambigqa train.csv)")
    except Exception as e:
        print(f"[ambigqa_train_as_test] Warning: Could not register {TASK_NAME}: {e}")
else:
    print(f"[ambigqa_train_as_test] Warning: AmbigQA data not found at {AQA_TRAIN_CSV}")
