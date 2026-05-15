"""
HumanEval code correctness tasks.

When running eval_by_claude.py on these tasks, pass --save-scores-csv: it is essential —
that script only writes the detailed per-example score CSV for HumanEval when this flag
is set (same CodeContests-style columns: problem_name, solution_preview, scores, etc.).

Reads from data/humaneval/with_solutions/:
    - train.csv: training data (~100 problems, solutions classified pass/fail)
    - humaneval_<N>.csv: one file per test problem (OOD by problem)

Columns: question, answer, correct, strategy

Registers:
    - humaneval: Full training set (train.csv), no test set
    - humaneval-humaneval_<N>: One task per test problem (train from train.csv, test from slug.csv)
"""

import os
import sys
import random

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
from tasks.common import PromptCompletion, load_csv_items, normalize_yes_no, get_field

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
HE_DIR = os.path.join(DATA_DIR, 'humaneval', 'with_solutions')       # v0
HE_V1_DIR = os.path.join(DATA_DIR, 'humaneval', 'v1')                # v1
HE_V2_DIR = os.path.join(DATA_DIR, 'humaneval', 'v2')                # v2: v1 with answer.strip()
HE_V2_1_DIR = os.path.join(DATA_DIR, 'humaneval', 'v2.1')            # v2.1: v2 filtered
HE_TRAIN_CSV = os.path.join(HE_DIR, 'train.csv')
HE_V1_TRAIN_CSV = os.path.join(HE_V1_DIR, 'train.csv')
HE_V2_TRAIN_CSV = os.path.join(HE_V2_DIR, 'train.csv')
HE_V2_1_TRAIN_CSV = os.path.join(HE_V2_1_DIR, 'train.csv')
HE_FIELDS = ('question', 'answer', 'correct', 'strategy')
HE_V1_FIELDS = ('question', 'answer', 'correct', 'strategy', 'model', 'temperature', 'task_id', 'error')
HE_V2_FIELDS = HE_V1_FIELDS
HE_V2_1_FIELDS = HE_V1_FIELDS


# ============================================================================
# Data loading
# ============================================================================

def _load_items(filepath):
    items = load_csv_items(filepath, fields=HE_FIELDS)
    for row in items:
        row['correct'] = str(row.get('correct', '')).strip()
        row['strategy'] = row.get('strategy', '')
    return items


def load_data_train_only(seed=0, split_type='random', sample_negative=False, **kwargs):
    """Load full training set. Test set is empty (use per-problem tasks for eval)."""
    L_train = _load_items(HE_TRAIN_CSV)
    random.Random(seed).shuffle(L_train)
    return L_train, []


def create_load_data_for_problem(test_csv_path):
    """Factory: create load_data that uses train.csv for train and a specific test CSV for test."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = _load_items(HE_TRAIN_CSV)
        L_test = _load_items(test_csv_path)
        rng = random.Random(seed)
        rng.shuffle(L_train)
        rng.shuffle(L_test)
        return L_train, L_test
    return load_data


# ============================================================================
# Prompt generation
# ============================================================================

DISC_FEW_SHOT_EXAMPLES = [
    {
        'question': 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
        'answer': '    return a + b',
        'label': 'Yes',
    },
    {
        'question': 'def is_even(n: int) -> bool:\n    """Return True if n is even."""\n',
        'answer': '    return n % 2 == 1',
        'label': 'No',
    },
    {
        'question': 'def factorial(n: int) -> int:\n    """Return the factorial of n."""\n',
        'answer': '    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
        'label': 'Yes',
    },
]


def _disc_query(question, answer):
    """Format a single discriminator query."""
    return (
        f'Is this a correct solution to the programming problem?\n\n'
        f'Problem:\n{question}\n'
        f'Solution:\n{answer}\n\n'
        f'Answer Yes or No:'
    )


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, **kwargs):
    """Create prompt for generator or discriminator."""
    question = item['question']
    answer = item['answer']

    if style == 'generator':
        prompt = f"Complete the following Python function:\n\n{question}\nSolution:"
        completion = " " + answer

    elif style == 'discriminator':
        cur_answer = gen_response if gen_response else answer
        query = _disc_query(question, cur_answer)

        if shots == 'few':
            examples = ""
            for ex in DISC_FEW_SHOT_EXAMPLES:
                examples += _disc_query(ex['question'], ex['answer']) + f" {ex['label']}\n\n"
            prompt = examples + query
        else:
            prompt = query

        correct = item['correct'].strip().capitalize()
        completion = " Yes" if correct == 'Yes' else " No"

    else:
        raise ValueError(f"Unknown style: {style}. Must be 'generator' or 'discriminator'.")

    return PromptCompletion(prompt.strip(), completion)


def get_completion(item):
    """Extract generator completion text."""
    return " " + item['answer']


def get_label(item):
    """Extract binary label ('yes' or 'no')."""
    return normalize_yes_no(item.get('correct', ''))


def make_negated_prompt(item, task, make_prompt, gen_shots='zero'):
    """Task-local negated prompt for --neg-typicality."""
    gen_obj = make_prompt(item, style='generator', shots='zero')
    neg_prompt = gen_obj.prompt.replace(
        "Complete the following Python function:",
        "Write an incorrect implementation of the following Python function:"
    )
    neg_prompt = neg_prompt.replace("\nSolution:", "\nIncorrect solution:")
    if neg_prompt == gen_obj.prompt:
        raise ValueError(
            f"Negated prompt unchanged for humaneval task '{task}'. "
            f"Prompt '{gen_obj.prompt[:80]}' doesn't match expected format."
        )
    return neg_prompt, gen_obj.completion


CSV_HEADER = [
    "problem_name",
    "solution_preview",
    "language",
    "num_tokens",
    "strategy",
    "correct",
    "val_score",
    "gen_score",
    "gen_score_typcorr",
    "gen_score_lenorm",
    "gen_score_typcorr_lenorm",
    "model_path",
]


def build_csv_row(item, task, strategy, num_toks, disc_score, gen_score_raw,
                  gen_score_typcorr_val, gen_score_lenorm,
                  gen_score_typcorr_lenorm, modelname):
    problem_name = task[len("humaneval-"):] if task.startswith("humaneval-") and task != "humaneval" else ""
    solution = get_field(item, "answer", "")
    solution_preview = solution[:200].replace("\n", "\\n")
    item_strategy = get_field(item, "strategy", strategy)
    correct_label = normalize_yes_no(get_field(item, "correct", ""))
    return [
        problem_name,
        solution_preview,
        "python",
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
# Task registration
# ============================================================================

if os.path.exists(HE_TRAIN_CSV):
    _COMMON = {
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'make_negated_prompt': make_negated_prompt,
        'csv_header': CSV_HEADER,
        'csv_row_builder': build_csv_row,
        'batch_size': {'with_ref': 1, 'without_ref': 4},
        'supports_split_types': ['random'],
    }

    register_task({
        'name': 'humaneval',
        'load_data': load_data_train_only,
        'description': 'HumanEval: full training set',
        **_COMMON,
    })

    _registered = []
    for filename in sorted(os.listdir(HE_DIR)):
        if not filename.endswith('.csv') or filename == 'train.csv':
            continue
        slug = filename[:-4]  # strip .csv
        test_csv_path = os.path.join(HE_DIR, filename)
        task_name = f'humaneval-{slug}'

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_for_problem(test_csv_path),
                'description': f'HumanEval: {slug}',
                **_COMMON,
            })
            _registered.append(task_name)
        except Exception as e:
            print(f"[humaneval] Warning: Could not register {task_name}: {e}")

    if _registered:
        print(f"[humaneval] Registered {len(_registered)} tasks")
else:
    print(f"[humaneval] Data not found at {HE_DIR} — skipping registration")


# --- v1 registration ---

def _load_v1_items(filepath):
    items = load_csv_items(filepath, fields=HE_V1_FIELDS)
    for row in items:
        row['correct'] = str(row.get('correct', '')).strip()
        row['strategy'] = row.get('strategy', '')
    return items


def load_data_v1_train_only(seed=0, split_type='random', sample_negative=False, **kwargs):
    """Load v1 full training set."""
    L_train = _load_v1_items(HE_V1_TRAIN_CSV)
    random.Random(seed).shuffle(L_train)
    return L_train, []


def create_load_data_v1_for_problem(test_csv_path):
    """Factory: v1 train.csv for train, specific test CSV for test."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = _load_v1_items(HE_V1_TRAIN_CSV)
        L_test = _load_v1_items(test_csv_path)
        rng = random.Random(seed)
        rng.shuffle(L_train)
        rng.shuffle(L_test)
        return L_train, L_test
    return load_data


if os.path.exists(HE_V1_TRAIN_CSV):
    _V1_COMMON = {
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'make_negated_prompt': make_negated_prompt,
        'csv_header': CSV_HEADER,
        'csv_row_builder': build_csv_row,
        'batch_size': {'with_ref': 1, 'without_ref': 4},
        'supports_split_types': ['random'],
    }

    register_task({
        'name': 'humaneval-v1',
        'load_data': load_data_v1_train_only,
        'description': 'HumanEval v1: full training set (19 models, 6 strategies)',
        **_V1_COMMON,
    })

    _v1_registered = []
    for filename in sorted(os.listdir(HE_V1_DIR)):
        if not filename.endswith('.csv') or filename == 'train.csv':
            continue
        slug = filename[:-4]  # strip .csv
        test_csv_path = os.path.join(HE_V1_DIR, filename)
        task_name = f'humaneval-v1-{slug}'

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_v1_for_problem(test_csv_path),
                'description': f'HumanEval v1: {slug}',
                **_V1_COMMON,
            })
            _v1_registered.append(task_name)
        except Exception as e:
            print(f"[humaneval-v1] Warning: Could not register {task_name}: {e}")

    if _v1_registered:
        print(f"[humaneval-v1] Registered {len(_v1_registered)} tasks")
else:
    print(f"[humaneval-v1] Data not found at {HE_V1_DIR} — skipping registration")


# --- v2 registration ---
#
# v2 = v1 with answer.strip() applied + a reworked prompt ("format C"):
#
#   1. Instruction explicitly tells the model to return body-only code, no markdown,
#      and to start "inside the function".
#   2. The target function's def-signature line is appended at the end of the
#      user content, so the model's response semantically picks up "inside the
#      function" instead of wanting to emit ```python\ndef ...``` markdown.
#
# This fixes the first-token noise that dominated v1 scoring on gemma-4-31B-it
# (chat-template wrapping pushed position-0 log P to ≈ -17 because the model
# wanted to produce a markdown-fenced full function, not just the body).
# See notes/log_P_diff_plots/humaneval-v1/V2_PROMPT_DESIGN.md for details.

_V2_INSTRUCTION = (
    "Complete the following Python function. "
    "Return ONLY the solution code, no markdown, starting from inside the function:"
)
_V2_NEG_INSTRUCTION = (
    "Write an incorrect implementation of the following Python function. "
    "Return ONLY the solution code, no markdown, starting from inside the function:"
)

# Cache last-resort regex for extracting the target def signature from a question.
import re as _re
_V2_DEF_SIG_RE = _re.compile(r'^def [^\n]+:\s*$', _re.MULTILINE)


def _v2_extract_signature(question: str) -> str:
    """Return the target function's def-signature line (last def in the question)."""
    matches = _V2_DEF_SIG_RE.findall(question)
    if not matches:
        raise ValueError(
            f"No 'def ...:' signature line found in humaneval-v2 question. "
            f"Question starts: {question[:120]!r}"
        )
    return matches[-1]


def _load_v2_items(filepath):
    items = load_csv_items(filepath, fields=HE_V2_FIELDS)
    for row in items:
        row['correct'] = str(row.get('correct', '')).strip()
        row['strategy'] = row.get('strategy', '')
    return items


def load_data_v2_train_only(seed=0, split_type='random', sample_negative=False, **kwargs):
    L_train = _load_v2_items(HE_V2_TRAIN_CSV)
    random.Random(seed).shuffle(L_train)
    return L_train, []


def create_load_data_v2_for_problem(test_csv_path):
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = _load_v2_items(HE_V2_TRAIN_CSV)
        L_test = _load_v2_items(test_csv_path)
        rng = random.Random(seed)
        rng.shuffle(L_train)
        rng.shuffle(L_test)
        return L_train, L_test
    return load_data


def make_prompt_v2(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, **kwargs):
    """v2 (format C): amended instruction + def signature at end of user content.

    answer is pre-stripped in the v2 CSVs (leading/trailing whitespace removed).
    """
    question = item['question']
    answer = item['answer']

    if style == 'generator':
        sig = _v2_extract_signature(question)
        prompt = (
            f"{_V2_INSTRUCTION}\n\n"
            f"{question}\n"
            f"Solution:\n"
            f"{sig}"
        )
        completion = answer
        return PromptCompletion(prompt, completion)

    elif style == 'discriminator':
        cur_answer = gen_response if gen_response else answer
        query = _disc_query(question, cur_answer)

        if shots == 'few':
            examples = ""
            for ex in DISC_FEW_SHOT_EXAMPLES:
                examples += _disc_query(ex['question'], ex['answer']) + f" {ex['label']}\n\n"
            prompt = examples + query
        else:
            prompt = query

        correct = item['correct'].strip().capitalize()
        completion = " Yes" if correct == 'Yes' else " No"
        return PromptCompletion(prompt.strip(), completion)

    else:
        raise ValueError(f"Unknown style: {style}. Must be 'generator' or 'discriminator'.")


def get_completion_v2(item):
    """v2 generator completion is the stripped answer (no leading space)."""
    return item['answer']


def make_negated_prompt_v2(item, task, make_prompt, gen_shots='zero'):
    """Task-local negated prompt for --neg-typicality (v2).

    Uses the V1 wording ("Write an incorrect implementation of the following
    Python function. Return ONLY ... Incorrect solution:"). Validated against
    gemma-4-31B-it via generative probe (all 4 tasks produced body-only output,
    no refusals, no markdown) AND scoring AUROC on 82-task v2/v2.1 (tc_neg_v1
    is the best of three tested neg wordings). See V2_PROMPT_DESIGN.md and
    notes/log_P_diff_plots/humaneval-v2.1/V2_1_ANALYSIS_REPORT.md.
    """
    gen_obj = make_prompt(item, style='generator', shots='zero')
    neg_prompt = gen_obj.prompt.replace(_V2_INSTRUCTION, _V2_NEG_INSTRUCTION)
    neg_prompt = neg_prompt.replace("\nSolution:\n", "\nIncorrect solution:\n")
    if neg_prompt == gen_obj.prompt:
        raise ValueError(
            f"Negated prompt unchanged for humaneval-v2 task '{task}'. "
            f"Prompt '{gen_obj.prompt[:80]}' doesn't match expected format."
        )
    return neg_prompt, gen_obj.completion


if os.path.exists(HE_V2_TRAIN_CSV):
    _V2_COMMON = {
        'make_prompt': make_prompt_v2,
        'get_completion': get_completion_v2,
        'get_label': get_label,
        'make_negated_prompt': make_negated_prompt_v2,
        'csv_header': CSV_HEADER,
        'csv_row_builder': build_csv_row,
        'batch_size': {'with_ref': 1, 'without_ref': 4},
        'supports_split_types': ['random'],
    }

    register_task({
        'name': 'humaneval-v2',
        'load_data': load_data_v2_train_only,
        'description': 'HumanEval v2: v1 with answer.strip() + 4-space indent moved into prompt',
        **_V2_COMMON,
    })

    _v2_registered = []
    for filename in sorted(os.listdir(HE_V2_DIR)):
        if not filename.endswith('.csv') or filename == 'train.csv':
            continue
        slug = filename[:-4]
        test_csv_path = os.path.join(HE_V2_DIR, filename)
        task_name = f'humaneval-v2-{slug}'

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_v2_for_problem(test_csv_path),
                'description': f'HumanEval v2: {slug}',
                **_V2_COMMON,
            })
            _v2_registered.append(task_name)
        except Exception as e:
            print(f"[humaneval-v2] Warning: Could not register {task_name}: {e}")

    if _v2_registered:
        print(f"[humaneval-v2] Registered {len(_v2_registered)} tasks")
else:
    print(f"[humaneval-v2] Data not found at {HE_V2_DIR} — skipping registration")


# --- v2.1 registration ---
#
# v2.1 = v2 with retroactive garbage filter applied to test set:
#   - chars ∈ [10, 900]
#   - mean log P(y | x) ≥ −5 (under gemma-4-31B-it, format C, chat template)
#   - total log P(y | x) ≥ −500
#
# Filter drops 148/2367 rows (6.3%): 46 by chars, 81 by mean log P, 21 by raw sum.
# Per-class loss: 0.2% correct, 12.5% wrong (the asymmetry catches garbage outputs
# like 'pengow', 'Sure' that came primarily from low-capability models on creative
# strategies). 4 tasks fall below 10 wrong (humaneval_10, _23, _35, _63 at 8-9).
#
# Same prompt-construction code as v2 (make_prompt_v2 + make_negated_prompt_v2);
# only the data source changes.

def _load_v2_1_items(filepath):
    items = load_csv_items(filepath, fields=HE_V2_1_FIELDS)
    for row in items:
        row['correct'] = str(row.get('correct', '')).strip()
        row['strategy'] = row.get('strategy', '')
    return items


def load_data_v2_1_train_only(seed=0, split_type='random', sample_negative=False, **kwargs):
    L_train = _load_v2_1_items(HE_V2_1_TRAIN_CSV)
    random.Random(seed).shuffle(L_train)
    return L_train, []


def create_load_data_v2_1_for_problem(test_csv_path):
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = _load_v2_1_items(HE_V2_1_TRAIN_CSV)
        L_test = _load_v2_1_items(test_csv_path)
        rng = random.Random(seed)
        rng.shuffle(L_train)
        rng.shuffle(L_test)
        return L_train, L_test
    return load_data


if os.path.exists(HE_V2_1_TRAIN_CSV):
    # Reuse v2's prompt code (make_prompt_v2, make_negated_prompt_v2, get_completion_v2)
    _V2_1_COMMON = {
        'make_prompt': make_prompt_v2,
        'get_completion': get_completion_v2,
        'get_label': get_label,
        'make_negated_prompt': make_negated_prompt_v2,
        'csv_header': CSV_HEADER,
        'csv_row_builder': build_csv_row,
        'batch_size': {'with_ref': 1, 'without_ref': 4},
        'supports_split_types': ['random'],
    }

    register_task({
        'name': 'humaneval-v2.1',
        'load_data': load_data_v2_1_train_only,
        'description': 'HumanEval v2.1: v2 with retroactive garbage filter (chars [10,900], mean l ≥ -5, raw ≥ -500)',
        **_V2_1_COMMON,
    })

    _v2_1_registered = []
    for filename in sorted(os.listdir(HE_V2_1_DIR)):
        if not filename.endswith('.csv') or filename == 'train.csv':
            continue
        slug = filename[:-4]
        test_csv_path = os.path.join(HE_V2_1_DIR, filename)
        task_name = f'humaneval-v2.1-{slug}'

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_v2_1_for_problem(test_csv_path),
                'description': f'HumanEval v2.1: {slug}',
                **_V2_1_COMMON,
            })
            _v2_1_registered.append(task_name)
        except Exception as e:
            print(f"[humaneval-v2.1] Warning: Could not register {task_name}: {e}")

    if _v2_1_registered:
        print(f"[humaneval-v2.1] Registered {len(_v2_1_registered)} tasks")
else:
    print(f"[humaneval-v2.1] Data not found at {HE_V2_1_DIR} — skipping registration")
