"""
CodeContests tasks (deepmind/code_contests, AlphaCode paper).

Each item is a (problem, code solution, correct/incorrect) triple derived from
real competitive programming contest submissions.

Data layout:
    data/codecontests/descriptions.json       - {problem_name: {description, difficulty}}
    data/codecontests/train.jsonl             - compact train items (joined with descriptions at load)
    data/codecontests/split_manifest.json     - which slugs are test vs valid
    data/codecontests/test/<slug>.jsonl        - self-contained per-problem eval items

Training tasks:
    codecontests                  - ~400 problems, ~2000 items
    codecontests-double           - ~800 problems, ~4000 items
    codecontests-all              - all ~13K problems, ~70K items

Eval tasks (from the original dataset's held-out splits):
    TEST split  (162 problems):  codecontests-1575a .. codecontests-1623e
    VALID split (117 problems):  codecontests-1548c .. codecontests-1574f

    The split_manifest.json maps each slug to "test" or "valid".
    Each task's 'origin_split' field also records this.

Language codes: 0=unknown, 1=python2, 2=cpp, 3=python3, 4=java
"""

import json
import os
import random
from collections import defaultdict, namedtuple

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task

PromptCompletion = namedtuple("PromptCompletion", ["prompt", "completion"])

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'codecontests'
)
DESCRIPTIONS_PATH = os.path.join(DATA_DIR, 'descriptions.json')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.jsonl')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MANIFEST_PATH = os.path.join(DATA_DIR, 'split_manifest.json')

TRAIN_PROBLEMS_BASE = 400
TRAIN_PROBLEMS_DOUBLE = 800
SEED = 42

_descriptions_cache = None


def _load_descriptions():
    global _descriptions_cache
    if _descriptions_cache is None:
        with open(DESCRIPTIONS_PATH, 'r') as f:
            _descriptions_cache = json.load(f)
    return _descriptions_cache


def _load_train_items():
    """Load compact train items and join with descriptions."""
    descriptions = _load_descriptions()
    items = []
    with open(TRAIN_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            pname = item['problem_name']
            if pname in descriptions:
                item['description'] = descriptions[pname]['description']
                items.append(item)
    return items


def _load_test_items(jsonl_path):
    """Load self-contained test items from a per-problem JSONL file."""
    items = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            items.append(json.loads(line))
    return items


def _sample_by_problems(items, n_problems, rng):
    """Sample n_problems worth of items (all items for each sampled problem)."""
    by_problem = defaultdict(list)
    for item in items:
        by_problem[item['problem_name']].append(item)

    problem_names = sorted(by_problem.keys())
    rng.shuffle(problem_names)
    selected = problem_names[:n_problems]

    result = []
    for pname in selected:
        result.extend(by_problem[pname])
    rng.shuffle(result)
    return result


def create_load_data_train(n_problems=None):
    """Factory: create load_data for training variants.

    n_problems=None means use all problems.
    """
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        all_items = _load_train_items()
        rng = random.Random(seed)
        rng.shuffle(all_items)

        if n_problems is not None:
            L_train = _sample_by_problems(all_items, n_problems, random.Random(SEED))
        else:
            L_train = all_items

        return L_train, []
    return load_data


def create_load_data_test(test_jsonl_path):
    """Factory: create load_data for a per-problem eval-only task."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_test = _load_test_items(test_jsonl_path)
        rng = random.Random(seed)
        rng.shuffle(L_test)
        return [], L_test
    return load_data


def make_prompt(item, style='generator', shots='zero', gen_response=None,
                neg=False, variation=0, **kwargs):
    description = item['description']

    if style == 'generator':
        prompt = (
            f"Solve the following competitive programming problem.\n\n"
            f"{description}\n\n"
            f"Solution:"
        )
        completion = "\n" + item['solution']

    elif style == 'discriminator':
        code = gen_response if gen_response else item['solution']
        prompt = (
            f"Is the following code a correct solution to the programming problem?\n\n"
            f"Problem:\n{description}\n\n"
            f"Code:\n{code}\n\n"
            f"Answer:"
        )
        completion = " Yes" if item['correct'] == 'Yes' else " No"

    else:
        raise ValueError(f"Unknown style: {style}")

    return PromptCompletion(prompt.strip(), completion)


def get_completion(item):
    return "\n" + item['solution']


def get_label(item):
    return 'yes' if item['correct'] == 'Yes' else 'no'


# ============================================================================
# Registration
# ============================================================================

if os.path.exists(TRAIN_PATH) and os.path.exists(DESCRIPTIONS_PATH):
    register_task({
        'name': 'codecontests',
        'load_data': create_load_data_train(n_problems=TRAIN_PROBLEMS_BASE),
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'batch_size': {'with_ref': 1, 'without_ref': 2},
        'supports_split_types': ['random'],
        'description': f'CodeContests: ~{TRAIN_PROBLEMS_BASE} problems training set',
    })

    register_task({
        'name': 'codecontests-double',
        'load_data': create_load_data_train(n_problems=TRAIN_PROBLEMS_DOUBLE),
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'batch_size': {'with_ref': 1, 'without_ref': 2},
        'supports_split_types': ['random'],
        'description': f'CodeContests: ~{TRAIN_PROBLEMS_DOUBLE} problems training set',
    })

    register_task({
        'name': 'codecontests-all',
        'load_data': create_load_data_train(n_problems=None),
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'batch_size': {'with_ref': 1, 'without_ref': 2},
        'supports_split_types': ['random'],
        'description': 'CodeContests: full training set',
    })

if os.path.exists(TEST_DIR):
    # Load split manifest to tag each eval task with its origin (test vs valid)
    _split_lookup = {}  # slug -> "test" or "valid"
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            _manifest = json.load(f)
        for slug in _manifest.get('test', []):
            _split_lookup[slug] = 'test'
        for slug in _manifest.get('valid', []):
            _split_lookup[slug] = 'valid'

    _registered_test = []
    _registered_valid = []
    for filename in sorted(os.listdir(TEST_DIR)):
        if not filename.endswith('.jsonl'):
            continue
        slug = filename[:-6]  # strip .jsonl
        test_path = os.path.join(TEST_DIR, filename)
        task_name = f'codecontests-{slug}'
        origin = _split_lookup.get(slug, 'unknown')

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_test(test_path),
                'make_prompt': make_prompt,
                'get_completion': get_completion,
                'get_label': get_label,
                'batch_size': {'with_ref': 1, 'without_ref': 2},
                'supports_split_types': ['random'],
                'description': f'CodeContests eval ({origin}): {slug}',
                'origin_split': origin,
            })
            if origin == 'test':
                _registered_test.append(task_name)
            else:
                _registered_valid.append(task_name)
        except Exception as e:
            print(f"[codecontests] Warning: Could not register {task_name}: {e}")

    print(f"[codecontests] Registered {len(_registered_test)} TEST eval tasks "
          f"(codecontests-1575a .. codecontests-1623e)")
    print(f"[codecontests] Registered {len(_registered_valid)} VALID eval tasks "
          f"(codecontests-1548c .. codecontests-1574f)")
else:
    print(f"[codecontests] Warning: Test data not found at {TEST_DIR}")
