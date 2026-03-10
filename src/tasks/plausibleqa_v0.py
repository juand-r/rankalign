"""
PlausibleQA tasks.

Reads from data/plausibleqa/:
    - train.csv: training data (all questions combined)
    - test/<id>.csv: one file per test question

Registers:
    - plausibleqa: Full training set (train.csv), no test set
    - plausibleqa-<id>: One task per test question (train from train.csv, test from test/<id>.csv)
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
# Original paths (v0):
# PQA_DIR = os.path.join(DATA_DIR, 'plausibleqa')
# PQA_TRAIN_CSV = os.path.join(PQA_DIR, 'train.csv')
# PQA_TEST_DIR = os.path.join(PQA_DIR, 'test')
# Fixed paths (v1):
# NOTE: This is the path with the version with more cleaned data with GPT codex + web search.
PQA_DIR = os.path.join(DATA_DIR, 'plausibleqa', 'fixed-plausibleqa')
PQA_TRAIN_CSV = os.path.join(PQA_DIR, 'train.csv')
PQA_TEST_DIR = os.path.join(PQA_DIR, 'test')
PQA_TRAIN_PER_Q_DIR = os.path.join(PQA_DIR, 'train-per-question')


# ============================================================================
# Data loading
# ============================================================================

def load_csv_items(filepath):
    """Load rows from a plausibleqa CSV as dicts."""
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row['gpt4_ground_truth'].strip()
            items.append({
                'id': row['id'],
                'question': row['question'],
                'answer': row['answer'],
                'correct': gt,
                'strategy': row['strategy'],
            })
    return items


def load_data_train_only(seed=0, split_type='random', sample_negative=False, **kwargs):
    """Load full training set. Test set is empty (use per-question tasks for eval)."""
    L_train = load_csv_items(PQA_TRAIN_CSV)
    random.seed(seed)
    random.shuffle(L_train)
    return L_train, []


def create_load_data_for_question(test_csv_path):
    """Factory: create load_data that uses train.csv for train and a specific test CSV for test."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = load_csv_items(PQA_TRAIN_CSV)
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
        'question': 'In the Hindu religion what name is given to the triad of chief gods Brahma, Vishnu and Siva?',
        'answer': 'Trimurti',
        'label': 'Yes',
    },
    {
        'question': 'when did the golden state warriors win the finals?',
        'answer': '1994',
        'label': 'No',
    },
    {
        'question': 'what is the zip code for midland tx?',
        'answer': '79702',
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

if os.path.exists(PQA_TRAIN_CSV):
    register_task({
        'name': 'plausibleqa',
        'load_data': load_data_train_only,
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'batch_size': {'with_ref': 1, 'without_ref': 4},
        'supports_split_types': ['random'],
        'description': 'PlausibleQA: full training set',
    })

if os.path.exists(PQA_TEST_DIR) and os.path.exists(PQA_TRAIN_CSV):
    _registered = []
    for filename in sorted(os.listdir(PQA_TEST_DIR)):
        if not filename.endswith('.csv'):
            continue
        qid = filename[:-4]  # strip .csv
        test_csv_path = os.path.join(PQA_TEST_DIR, filename)
        task_name = f'plausibleqa-{qid}'

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_for_question(test_csv_path),
                'make_prompt': make_prompt,
                'get_completion': get_completion,
                'get_label': get_label,
                'batch_size': {'with_ref': 1, 'without_ref': 4},
                'supports_split_types': ['random'],
                'description': f'PlausibleQA: {qid}',
            })
            _registered.append(task_name)
        except Exception as e:
            print(f"[plausibleqa] Warning: Could not register {task_name}: {e}")

    if _registered:
        print(f"[plausibleqa] Registered {len(_registered)} test tasks")
else:
    print(f"[plausibleqa] Warning: Data not found at {PQA_DIR}")

if os.path.exists(PQA_TRAIN_PER_Q_DIR):
    _registered_train = []
    for filename in sorted(os.listdir(PQA_TRAIN_PER_Q_DIR)):
        if not filename.endswith('.csv'):
            continue
        qid = filename[:-4]
        csv_path = os.path.join(PQA_TRAIN_PER_Q_DIR, filename)
        task_name = f'plausibleqa-train-{qid}'

        def create_load_eval_only(eval_csv_path):
            def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
                L_test = load_csv_items(eval_csv_path)
                random.seed(seed)
                random.shuffle(L_test)
                return [], L_test
            return load_data

        try:
            register_task({
                'name': task_name,
                'load_data': create_load_eval_only(csv_path),
                'make_prompt': make_prompt,
                'get_completion': get_completion,
                'get_label': get_label,
                'batch_size': {'with_ref': 1, 'without_ref': 4},
                'supports_split_types': ['random'],
                'description': f'PlausibleQA train eval: {qid}',
            })
            _registered_train.append(task_name)
        except Exception as e:
            print(f"[plausibleqa] Warning: Could not register {task_name}: {e}")

    if _registered_train:
        print(f"[plausibleqa] Registered {len(_registered_train)} train-eval tasks")
