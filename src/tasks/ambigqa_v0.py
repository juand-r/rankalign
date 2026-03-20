"""
AmbigQA + PlausibleQA combined v0 tasks.

Reads from data/ambigqa/v0/combined.csv which contains questions with multiple
plausible answers (from AmbigQA) and plausible-but-wrong candidates (from PlausibleQA/GPT).

Registers:
    - ambigqa-plausibleqa-combined-v0: All questions combined
    - ambigqa-v0-{slug}: One task per unique question (slug = first N non-stopwords)
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

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
AMBIGQA_V0_CSV = os.path.join(DATA_DIR, 'ambigqa', 'v0', 'combined.csv')

STOPWORDS = {
    'who', 'what', 'when', 'where', 'which', 'how', 'is', 'are', 'was', 'were',
    'do', 'does', 'did', 'has', 'have', 'had', 'the', 'a', 'an', 'of', 'in',
    'on', 'at', 'to', 'for', 'with', 'from', 'by', 'about', 'and', 'or', 'not',
    'it', 'its', 'that', 'this', 'be', 'been', 'being', 'many', 'much', 'more',
    'most', 'some',
}


# ============================================================================
# Data loading
# ============================================================================

def load_csv_items(filepath):
    """Load all rows from the combined CSV as dicts."""
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                'id': row['id'],
                'question': row['question'],
                'answer': row['answer'],
                'correct': row['correct'].strip().upper(),
                'strategy': row['strategy'],
            })
    return items


def load_data_combined(seed=0, split_type='random', sample_negative=False, **kwargs):
    """Load all data. Everything goes to test; train is empty."""
    items = load_csv_items(AMBIGQA_V0_CSV)
    random.seed(seed)
    random.shuffle(items)
    print(f"[ambigqa-v0] Loaded {len(items)} items (all test)")
    return [], items


def create_load_data_for_question(question_text):
    """Factory: create load_data that filters to a single question."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        items = load_csv_items(AMBIGQA_V0_CSV)
        filtered = [item for item in items if item['question'] == question_text]
        random.seed(seed)
        random.shuffle(filtered)
        return [], filtered
    return load_data


# ============================================================================
# Prompt generation
# ============================================================================

# Few-shot examples for discriminator (fixed from the dataset, seed=42)
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
    """Format a single discriminator query (without the answer label)."""
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

        label = item['correct'].strip().upper()
        completion = " Yes" if label == 'YES' else " No"

    else:
        raise ValueError(f"Unknown style: {style}. Must be 'generator' or 'discriminator'.")

    return PromptCompletion(prompt.strip(), completion)


def get_completion(item):
    """Extract generator completion text."""
    return " " + item['answer']


def get_label(item):
    """Extract binary label ('yes' or 'no')."""
    return 'yes' if item['correct'].strip().upper() == 'YES' else 'no'


# ============================================================================
# Question slug generation
# ============================================================================

def get_nonstop_words(question, n):
    """Extract first n non-stopwords from a question, shell-safe (no apostrophes)."""
    words = question.lower().rstrip('?').split()
    nonstop = []
    for w in words:
        clean = w.strip("'\".,!?").replace("'", "")
        if clean and clean not in STOPWORDS:
            nonstop.append(clean)
            if len(nonstop) == n:
                break
    return nonstop


def build_unique_slugs(questions):
    """
    Build a unique slug for each question using the minimum number of
    non-stopwords needed. Starts at 2 and adds words for collisions.
    """
    slug_map = {}
    remaining = set(questions)
    n = 2

    while remaining and n <= 10:
        # Build candidate slugs for remaining questions
        candidates = {}
        for q in remaining:
            words = get_nonstop_words(q, n)
            candidates[q] = '-'.join(words)

        # Find which slugs are unique (among remaining questions only)
        from collections import Counter
        counts = Counter(candidates.values())
        for q in list(remaining):
            slug = candidates[q]
            # Also check against already-assigned slugs
            if counts[slug] == 1 and slug not in slug_map.values():
                slug_map[q] = slug
                remaining.remove(q)

        n += 1

    # Fallback: if still not unique, append question id hash
    for q in remaining:
        words = get_nonstop_words(q, 5)
        base = '-'.join(words)
        slug_map[q] = base

    return slug_map


# ============================================================================
# Task registration
# ============================================================================

if os.path.exists(AMBIGQA_V0_CSV):
    # Register combined task
    register_task({
        'name': 'ambigqa-plausibleqa-combined-v0',
        'load_data': load_data_combined,
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
        'batch_size': {'with_ref': 1, 'without_ref': 4},
        'supports_split_types': ['random'],
        'description': 'AmbigQA + PlausibleQA combined v0 (all questions)',
    })

    # Discover unique questions and register per-question tasks
    all_items = load_csv_items(AMBIGQA_V0_CSV)
    unique_questions = sorted(set(item['question'] for item in all_items))
    slug_map = build_unique_slugs(unique_questions)

    _registered_per_question = []
    for question_text in unique_questions:
        slug = slug_map[question_text]
        task_name = f'ambigqa-v0-{slug}'
        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_for_question(question_text),
                'make_prompt': make_prompt,
                'get_completion': get_completion,
                'get_label': get_label,
                'batch_size': {'with_ref': 1, 'without_ref': 4},
                'supports_split_types': ['random'],
                'description': f'AmbigQA v0: {question_text[:60]}',
            })
            _registered_per_question.append(task_name)
        except Exception as e:
            print(f"[ambigqa_v0] Warning: Could not register {task_name}: {e}")

    if _registered_per_question:
        print(f"[ambigqa_v0] Registered {len(_registered_per_question)} per-question tasks")
else:
    print(f"[ambigqa_v0] Warning: Data file not found: {AMBIGQA_V0_CSV}")
