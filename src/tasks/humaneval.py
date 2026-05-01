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
import csv
import random
from collections import namedtuple

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task

PromptCompletion = namedtuple("PromptCompletion", ["prompt", "completion"])

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
HE_DIR = os.path.join(DATA_DIR, 'humaneval', 'with_solutions')
HE_TRAIN_CSV = os.path.join(HE_DIR, 'train.csv')


# ============================================================================
# Data loading
# ============================================================================

def load_csv_items(filepath):
    """Load rows from a humaneval CSV as dicts."""
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                'question': row['question'],
                'answer': row['answer'],
                'correct': row['correct'].strip(),
                'strategy': row['strategy'],
            })
    return items


def load_data_train_only(seed=0, split_type='random', sample_negative=False, **kwargs):
    """Load full training set. Test set is empty (use per-problem tasks for eval)."""
    L_train = load_csv_items(HE_TRAIN_CSV)
    random.seed(seed)
    random.shuffle(L_train)
    return L_train, []


def create_load_data_for_problem(test_csv_path):
    """Factory: create load_data that uses train.csv for train and a specific test CSV for test."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = load_csv_items(HE_TRAIN_CSV)
        L_test = load_csv_items(test_csv_path)
        random.seed(seed)
        random.shuffle(L_train)
        random.shuffle(L_test)
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
    return 'yes' if item['correct'].strip().capitalize() == 'Yes' else 'no'


# ============================================================================
# Task registration
# ============================================================================

if os.path.exists(HE_TRAIN_CSV):
    register_task({
        'name': 'humaneval',
        'load_data': load_data_train_only,
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'batch_size': {'with_ref': 1, 'without_ref': 4},
        'supports_split_types': ['random'],
        'description': 'HumanEval: full training set',
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
                'make_prompt': make_prompt,
                'get_completion': get_completion,
                'get_label': get_label,
                'batch_size': {'with_ref': 1, 'without_ref': 4},
                'supports_split_types': ['random'],
                'description': f'HumanEval: {slug}',
            })
            _registered.append(task_name)
        except Exception as e:
            print(f"[humaneval] Warning: Could not register {task_name}: {e}")

    if _registered:
        print(f"[humaneval] Registered {len(_registered)} tasks")
else:
    print(f"[humaneval] Data not found at {HE_DIR} — skipping registration")
