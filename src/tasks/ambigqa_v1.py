"""
AmbigQA v1 tasks.

Reads from data/ambigqa/with_negatives/:
    - train.csv: training data (160 questions, min 10 positive answers each)
    - <slug>.csv: one file per test question (17 questions, min 15 positive answers)

Columns: question, answer, correct, strategy

Registers:
    - ambigqa: Full training set (train.csv), no test set
    - ambigqa-<slug>: One task per test question (train from train.csv, test from <slug>.csv)
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
AQA_DIR = os.path.join(DATA_DIR, 'ambigqa', 'with_negatives')
AQA_TRAIN_CSV = os.path.join(AQA_DIR, 'train.csv')


# ============================================================================
# Data loading
# ============================================================================

def load_csv_items(filepath):
    """Load rows from an ambigqa CSV as dicts."""
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
    """Load full training set. Test set is empty (use per-question tasks for eval)."""
    L_train = load_csv_items(AQA_TRAIN_CSV)
    random.seed(seed)
    random.shuffle(L_train)
    return L_train, []


def create_load_data_for_question(test_csv_path):
    """Factory: create load_data that uses train.csv for train and a specific test CSV for test."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = load_csv_items(AQA_TRAIN_CSV)
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
        'question': 'When was the last time pittsburgh steelers won the superbowl?',
        'answer': '2008',
        'label': 'Yes',
    },
    {
        'question': "Who sings the song to orange is the new black?",
        'answer': "Regina Spektor's contemporary, Ingrid Michaelson",
        'label': 'No',
    },
    {
        'question': 'What states were hit the hardest by the dust bowl?',
        'answer': 'Colorado',
        'label': 'Yes',
    },
]


def _disc_query(question, answer):
    """Format a single discriminator query."""
    return (
        f'Is the correct answer to the question "{question}" '
        f'given by "{answer}"? Answer Yes or No:'
    )


def make_prompt(item, style='generator', shots='zero', gen_response=None, neg=False, variation=0, **kwargs):
    """Create prompt for generator or discriminator."""
    question = item['question']
    answer = item['answer']

    if style == 'generator':
        prompt = f"Answer the question: {question}\nAnswer:"
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

if os.path.exists(AQA_TRAIN_CSV):
    register_task({
        'name': 'ambigqa',
        'load_data': load_data_train_only,
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'batch_size': {'with_ref': 1, 'without_ref': 4},
        'supports_split_types': ['random'],
        'description': 'AmbigQA: full training set',
    })

    _registered = []
    for filename in sorted(os.listdir(AQA_DIR)):
        if not filename.endswith('.csv') or filename == 'train.csv':
            continue
        slug = filename[:-4]  # strip .csv
        test_csv_path = os.path.join(AQA_DIR, filename)
        task_name = f'ambigqa-{slug}'

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_for_question(test_csv_path),
                'make_prompt': make_prompt,
                'get_completion': get_completion,
                'get_label': get_label,
                'batch_size': {'with_ref': 1, 'without_ref': 4},
                'supports_split_types': ['random'],
                'description': f'AmbigQA: {slug}',
            })
            _registered.append(task_name)
        except Exception as e:
            print(f"[ambigqa] Warning: Could not register {task_name}: {e}")

    if _registered:
        print(f"[ambigqa] Registered {len(_registered)} tasks")
else:
    print(f"[ambigqa] Warning: Data not found at {AQA_DIR}")
