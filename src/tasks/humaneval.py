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
HE_DIR = os.path.join(DATA_DIR, 'humaneval', 'with_solutions')
HE_TRAIN_CSV = os.path.join(HE_DIR, 'train.csv')
HE_FIELDS = ('question', 'answer', 'correct', 'strategy')


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
