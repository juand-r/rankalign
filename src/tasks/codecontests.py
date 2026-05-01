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
    TEST split  (158 problems):  codecontests-1575a .. codecontests-1623e
    VALID split (114 problems):  codecontests-1548c .. codecontests-1574f

    7 problems were removed for having too few items or extreme class
    imbalance (see git log for the specific slugs removed).

    The split_manifest.json maps each slug to "test" or "valid", and lists
    the "short" subset (test problems with median completion < 300 tokens).
    Each task's registry entry has 'origin_split' and 'short' fields.

    Filter programmatically:
        short_tasks = [n for n, c in TASK_REGISTRY.items() if c.get('short')]

Language codes: 0=unknown, 1=python2, 2=cpp, 3=python3, 4=java
"""

import json
import os
import random
from collections import defaultdict

import sys
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
from tasks.common import PromptCompletion, normalize_yes_no, get_field

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'codecontests'
)
DESCRIPTIONS_PATH = os.path.join(DATA_DIR, 'descriptions.json')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.jsonl')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MANIFEST_PATH = os.path.join(DATA_DIR, 'split_manifest.json')

TRAIN_PROBLEMS_BASE = 90
TRAIN_PROBLEMS_DOUBLE = 180
SEED = 42
MAX_ITEM_CHARS = None  # set to e.g. 4096 to filter long items (prompt+completion chars)

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


def _filter_by_length(items, max_chars):
    """Drop items whose generator or discriminator prompt+completion exceeds max_chars."""
    kept = []
    for item in items:
        pc_gen = make_prompt(item, style='generator')
        pc_disc = make_prompt(item, style='discriminator')
        gen_len = len(pc_gen.prompt) + len(pc_gen.completion)
        disc_len = len(pc_disc.prompt) + len(pc_disc.completion)
        if max(gen_len, disc_len) <= max_chars:
            kept.append(item)
    return kept


def create_load_data_train(n_problems=None, max_item_chars=MAX_ITEM_CHARS):
    """Factory: create load_data for training variants.

    n_problems=None means use all problems.
    max_item_chars: if set, drop items whose prompt+completion exceeds this
        many characters (~4 chars/token, so 4096 chars ≈ 1024 tokens).
    """
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        all_items = _load_train_items()
        rng = random.Random(seed)
        rng.shuffle(all_items)

        if n_problems is not None:
            L_train = _sample_by_problems(all_items, n_problems, random.Random(SEED))
        else:
            L_train = all_items

        if max_item_chars is not None:
            before = len(L_train)
            L_train = _filter_by_length(L_train, max_item_chars)
            print(f"[codecontests] Length filter ({max_item_chars} chars): "
                  f"{before} -> {len(L_train)} items")

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
        completion = " " + item['solution']

    elif style == 'discriminator':
        if shots == 'few':
            raise NotImplementedError(
                "Few-shot discriminator is not implemented for codecontests. "
                "Prompts include full problem descriptions + code, making "
                "few-shot examples impractically long."
            )
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
    return " " + item['solution']


def get_label(item):
    return normalize_yes_no(item.get('correct', ''))


def make_negated_prompt(item, task, make_prompt, gen_shots='zero'):
    # NOTE: negated prompts for codecontests are not implemented yet.
    raise NotImplementedError(
        f"--neg-typicality is not implemented for task '{task}'. "
        "Competitive programming prompts are long and need a task-specific "
        "negation strategy."
    )


CSV_HEADER = [
    'problem_name',
    'solution_preview',
    'language',
    'num_tokens',
    'strategy',
    'correct',
    'val_score',
    'gen_score',
    'gen_score_typcorr',
    'gen_score_lenorm',
    'gen_score_typcorr_lenorm',
    'model_path',
]


def build_csv_row(item, task, strategy, num_toks, disc_score, gen_score_raw,
                  gen_score_typcorr_val, gen_score_lenorm,
                  gen_score_typcorr_lenorm, modelname):
    problem_name = get_field(item, 'problem_name', '')
    solution = get_field(item, 'solution', '')
    solution_preview = solution[:200].replace('\n', '\\n')
    language = get_field(item, 'language', '')
    item_strategy = get_field(item, 'strategy', strategy)
    correct_label = normalize_yes_no(get_field(item, 'correct', ''))
    return [
        problem_name,
        solution_preview,
        language,
        num_toks,
        item_strategy,
        correct_label,
        disc_score,
        gen_score_raw,
        gen_score_typcorr_val,
        gen_score_lenorm,
        gen_score_typcorr_lenorm,
        modelname,
    ]


# ============================================================================
# Registration
# ============================================================================

if os.path.exists(TRAIN_PATH) and os.path.exists(DESCRIPTIONS_PATH):
    # 1024 tokens ≈ 4096 chars; filters out long items to avoid OOM on 9B models
    TRAIN_MAX_CHARS = 4096
    _COMMON = {
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'make_negated_prompt': make_negated_prompt,
        'csv_header': CSV_HEADER,
        'csv_row_builder': build_csv_row,
        'batch_size': {'with_ref': 1, 'without_ref': 2},
        'supports_split_types': ['random'],
    }

    register_task({
        'name': 'codecontests',
        'load_data': create_load_data_train(n_problems=TRAIN_PROBLEMS_BASE, max_item_chars=TRAIN_MAX_CHARS),
        'description': f'CodeContests: ~{TRAIN_PROBLEMS_BASE} problems, <=1024 tok',
        **_COMMON,
    })

    register_task({
        'name': 'codecontests-double',
        'load_data': create_load_data_train(n_problems=TRAIN_PROBLEMS_DOUBLE, max_item_chars=TRAIN_MAX_CHARS),
        'description': f'CodeContests: ~{TRAIN_PROBLEMS_DOUBLE} problems, <=1024 tok',
        **_COMMON,
    })

    register_task({
        'name': 'codecontests-all',
        'load_data': create_load_data_train(n_problems=None, max_item_chars=TRAIN_MAX_CHARS),
        'description': 'CodeContests: all problems, <=1024 tok',
        **_COMMON,
    })

if os.path.exists(TEST_DIR):
    # Load split manifest to tag each eval task with its origin (test vs valid)
    # and whether it's in the "short" subset (median completion <= 200 tokens).
    # See data/codecontests/split_manifest.json.
    _split_lookup = {}  # slug -> "test" or "valid"
    _short_set = set()  # slugs with short completions
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            _manifest = json.load(f)
        for slug in _manifest.get('test', []):
            _split_lookup[slug] = 'test'
        for slug in _manifest.get('valid', []):
            _split_lookup[slug] = 'valid'
        _short_set = set(_manifest.get('short', []))

    _registered_test = []
    _registered_valid = []
    _registered_short = []
    for filename in sorted(os.listdir(TEST_DIR)):
        if not filename.endswith('.jsonl'):
            continue
        slug = filename[:-6]  # strip .jsonl
        test_path = os.path.join(TEST_DIR, filename)
        task_name = f'codecontests-{slug}'
        origin = _split_lookup.get(slug, 'unknown')
        is_short = slug in _short_set

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_test(test_path),
                'description': f'CodeContests eval ({origin}): {slug}',
                'origin_split': origin,
                'short': is_short,
                **_COMMON,
            })
            if origin == 'test':
                _registered_test.append(task_name)
            if is_short:
                _registered_short.append(task_name)
            if origin == 'valid':
                _registered_valid.append(task_name)
        except Exception as e:
            print(f"[codecontests] Warning: Could not register {task_name}: {e}")

    print(f"[codecontests] Registered {len(_registered_test)} TEST eval tasks, "
          f"{len(_registered_valid)} VALID eval tasks, "
          f"{len(_registered_short)} short (median completion < 300 tokens)")
else:
    print(f"[codecontests] Warning: Test data not found at {TEST_DIR}")
