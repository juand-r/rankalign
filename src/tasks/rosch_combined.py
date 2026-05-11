"""
Combined rosch task: rosch-furniture-and-bird.

Loads the union of furniture and bird test sets and returns it as BOTH
the train and test split. This is intentional — it's a quick-iteration
testbed for the IMPORTANT-RESEARCH-PLAN.md "in-domain memorization probe"
(train ≈ test). The model gets to overfit on purpose; the question we're
answering is whether train-time TC / online pair selection produce a
different signal than baseline RankAlign in the easiest possible setting.

DO NOT report cross-task generalization numbers from this task — by
construction, train and test are the same items.

Reuses the per-category load/make_prompt/get_completion/get_label from
src/tasks/rosch.py so prompt formatting is identical to the per-category
rosch evals.
"""

import os

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
from .rosch import load_rosch_csv, make_prompt, get_completion, get_label, DATA_DIR


COMBINED_SLUGS = ["furniture", "bird"]
TASK_NAME = "rosch-furniture-and-bird"


def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
    """Load furniture + bird test items, return (combined, combined).

    Train and test are the same list — this task is a memorization probe.
    """
    items = []
    for slug in COMBINED_SLUGS:
        test_path = os.path.join(DATA_DIR, f"rosch-{slug}_test.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Rosch test file not found: {test_path}")
        items.extend(load_rosch_csv(test_path))

    return items, items


try:
    register_task({
        'name': TASK_NAME,
        'load_data': load_data,
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'batch_size': {'with_ref': 1, 'without_ref': 8},
        'supports_split_types': ['random'],
    })
    print(f"[rosch_combined] Registered {TASK_NAME} (train=test, "
          f"{', '.join(COMBINED_SLUGS)})")
except Exception as e:
    print(f"[rosch_combined] Warning: Could not register {TASK_NAME}: {e}")
