"""
k-SAT satisfiability tasks (modern pattern, mirrors humaneval.py / codecontests.py).

Generator:     given a CNF formula, produce a satisfying assignment.
Discriminator: given a (formula, assignment) pair, decide if the assignment
               satisfies the formula (Yes/No).

Each row of the source CSVs is a (formula, assignment, label) triple plus
a known satisfying_assignment field used as the generator target. CSVs are
balanced 50/50 (yes/no) by `scripts/ksat/generate_ksat_data.py`.

Tasks registered (only when the corresponding CSVs are present):

    2sat      — 2-SAT,  4 variables (x0..x3),  3000 train + 1000 test
    2sat-10   — 2-SAT, 10 variables (x0..x9),  3000 train + 1000 test
    3sat      — 3-SAT,  4 variables            (registered if data appears)
    3sat-10   — 3-SAT, 10 variables            (registered if data appears)

The "-10" suffix mirrors the data filename convention (data/2sat_train-10.csv
holds the 10-variable variant). Few-shot examples are selected automatically
based on the (k, n_vars) combination — n_vars is inferred per item from the
assignment string at prompt time, so the same task module covers both var
counts cleanly.

Build details: scripts/ksat/KSAT_BUILD_LOG.md.
"""

import os
import sys
import random
from collections import namedtuple

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
from tasks.common import PromptCompletion, load_csv_items, normalize_yes_no, get_field

# ============================================================================
# Paths
# ============================================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data',
)

CSV_FIELDS = ('formula', 'assignment', 'label', 'satisfying_assignment')


# Mapping from task_name -> (train_csv, test_csv, k, n_vars). Only those whose
# CSVs actually exist on disk get registered. To enable a missing variant,
# generate the data with:
#
#   python scripts/ksat/generate_ksat_data.py --k {k} --n_vars {n_vars} \
#       --n_train 3000 --n_test 1000
#
# (the script writes data/{k}sat_{train,test}.csv for the default n_vars=10
# case and data/{k}sat_{train,test}-10.csv for the 10-var variant if --n_vars
# is set to 10; for 4-var see scripts/ksat/KSAT_BUILD_LOG.md.)
TASK_VARIANTS = [
    # (task_name, train_csv,        test_csv,        k, n_vars)
    ('2sat',     '2sat_train.csv',     '2sat_test.csv',     2, 4),
    ('2sat-10',  '2sat_train-10.csv',  '2sat_test-10.csv',  2, 10),
    ('3sat',     '3sat_train.csv',     '3sat_test.csv',     3, 4),
    ('3sat-10',  '3sat_train-10.csv',  '3sat_test-10.csv',  3, 10),
]


# ============================================================================
# Few-shot examples
# ============================================================================

# All few-shot examples are listed (correctness verifiable by hand). The
# generator uses only positive examples; the discriminator uses 4 mixed.
# Padding with x_i=0 keeps the assignment string at the right n_vars width
# without changing meaning (those vars don't appear in the formula).

_FEW_SHOT = {
    (2, 4): [
        # (x0 ∨ x1) ∧ (¬x1 ∨ x2) — satisfied by x0=1, x1=0
        {'formula': '(x0 ∨ x1) ∧ (¬x1 ∨ x2)',
         'assignment': 'x0=1, x1=0, x2=0, x3=0', 'label': 'yes'},
        # (x0 ∨ x1) ∧ (¬x0 ∨ ¬x1) — NOT satisfied by x0=0, x1=0
        {'formula': '(x0 ∨ x1) ∧ (¬x0 ∨ ¬x1)',
         'assignment': 'x0=0, x1=0, x2=0, x3=0', 'label': 'no'},
        # (¬x0 ∨ x2) ∧ (x1 ∨ ¬x2) — satisfied by x0=0, x1=1, x2=1
        {'formula': '(¬x0 ∨ x2) ∧ (x1 ∨ ¬x2)',
         'assignment': 'x0=0, x1=1, x2=1, x3=0', 'label': 'yes'},
        # (x0 ∨ x3) ∧ (¬x0 ∨ ¬x3) — NOT satisfied by x0=0, x3=0
        {'formula': '(x0 ∨ x3) ∧ (¬x0 ∨ ¬x3)',
         'assignment': 'x0=0, x1=0, x2=0, x3=0', 'label': 'no'},
    ],
    (2, 10): [
        {'formula': '(x0 ∨ x1) ∧ (¬x1 ∨ x2)',
         'assignment': 'x0=1, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
         'label': 'yes'},
        {'formula': '(x0 ∨ x1) ∧ (¬x0 ∨ ¬x1)',
         'assignment': 'x0=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
         'label': 'no'},
        {'formula': '(¬x0 ∨ x2) ∧ (x1 ∨ ¬x2)',
         'assignment': 'x0=0, x1=1, x2=1, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
         'label': 'yes'},
        {'formula': '(x0 ∨ x3) ∧ (¬x0 ∨ ¬x3)',
         'assignment': 'x0=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
         'label': 'no'},
    ],
    (3, 4): [
        {'formula': '(x0 ∨ x1 ∨ x2) ∧ (¬x0 ∨ x1 ∨ ¬x2)',
         'assignment': 'x0=1, x1=1, x2=0, x3=0', 'label': 'yes'},
        {'formula': '(x0 ∨ x1 ∨ x2) ∧ (¬x0 ∨ ¬x1 ∨ ¬x2)',
         'assignment': 'x0=0, x1=0, x2=0, x3=0', 'label': 'no'},
        {'formula': '(¬x0 ∨ x1 ∨ x2) ∧ (x0 ∨ ¬x1 ∨ x2)',
         'assignment': 'x0=0, x1=0, x2=1, x3=0', 'label': 'yes'},
        {'formula': '(x0 ∨ x1 ∨ x2) ∧ (¬x0 ∨ x1 ∨ x2)',
         'assignment': 'x0=0, x1=0, x2=0, x3=0', 'label': 'no'},
    ],
    (3, 10): [
        {'formula': '(x0 ∨ x1 ∨ x2) ∧ (¬x0 ∨ x1 ∨ ¬x2)',
         'assignment': 'x0=1, x1=1, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
         'label': 'yes'},
        {'formula': '(x0 ∨ x1 ∨ x2) ∧ (¬x0 ∨ ¬x1 ∨ ¬x2)',
         'assignment': 'x0=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
         'label': 'no'},
        {'formula': '(¬x0 ∨ x1 ∨ x2) ∧ (x0 ∨ ¬x1 ∨ x2)',
         'assignment': 'x0=0, x1=0, x2=1, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
         'label': 'yes'},
        {'formula': '(x0 ∨ x1 ∨ x2) ∧ (¬x0 ∨ x1 ∨ x2) ∧ (x0 ∨ ¬x1 ∨ x2) ∧ (x0 ∨ x1 ∨ ¬x2)',
         'assignment': 'x0=0, x1=0, x2=0, x3=0, x4=0, x5=0, x6=0, x7=0, x8=0, x9=0',
         'label': 'no'},
    ],
}


def _detect_n_vars(assignment_str):
    """Detect n_vars from an assignment like 'x0=1, x1=0, ...'."""
    return len(assignment_str.split(', '))


def _few_shot_for(k, n_vars):
    """Pick the closest few-shot bank (defaults to 4-var)."""
    if (k, n_vars) in _FEW_SHOT:
        return _FEW_SHOT[(k, n_vars)]
    # Sensible default for unexpected n_vars: the 4-var bank for this k,
    # or 2-SAT 4-var as a last resort.
    return _FEW_SHOT.get((k, 4), _FEW_SHOT[(2, 4)])


# ============================================================================
# Data loading
# ============================================================================

def _make_load_data(train_csv_basename, test_csv_basename, k):
    """Factory for load_data closures (one per task variant).

    `k` is captured for downstream prompt construction (passed via item dict
    so make_prompt doesn't have to re-detect it from filenames).
    """
    train_csv = os.path.join(DATA_DIR, train_csv_basename)
    test_csv = os.path.join(DATA_DIR, test_csv_basename)

    def _load_csv(path):
        items = load_csv_items(path, fields=CSV_FIELDS)
        for row in items:
            row['label'] = normalize_yes_no(row.get('label', ''))
            row['satisfying_assignment'] = row.get('satisfying_assignment', '') or ''
            row['k'] = k
        return items

    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        if not os.path.exists(train_csv):
            raise FileNotFoundError(
                f"Training data not found: {train_csv}\n"
                f"Generate it with:\n"
                f"  python scripts/ksat/generate_ksat_data.py --k {k} "
                f"--n_train 3000 --n_test 1000"
            )
        if not os.path.exists(test_csv):
            raise FileNotFoundError(
                f"Test data not found: {test_csv}\n"
                f"Generate it with:\n"
                f"  python scripts/ksat/generate_ksat_data.py --k {k} "
                f"--n_train 3000 --n_test 1000"
            )
        L_train = _load_csv(train_csv)
        L_test = _load_csv(test_csv)
        rng = random.Random(seed)
        rng.shuffle(L_train)
        rng.shuffle(L_test)
        return L_train, L_test

    return load_data


# ============================================================================
# Prompts
# ============================================================================

def _gen_prompt_zero(item):
    return (
        f"Given the following Boolean formula in CNF (conjunctive normal form):\n"
        f"{item['formula']}\n\n"
        f"Provide a variable assignment that satisfies this formula.\n"
        f"Assignment:"
    )


def _gen_prompt_few(item, k, n_vars):
    examples = [ex for ex in _few_shot_for(k, n_vars) if ex['label'] == 'yes'][:2]
    body = "Given a Boolean formula in CNF, provide a variable assignment that satisfies it.\n\n"
    for ex in examples:
        body += f"Formula: {ex['formula']}\nAssignment: {ex['assignment']}\n\n"
    body += f"Formula: {item['formula']}\nAssignment:"
    return body


def _disc_prompt_zero(item, assignment):
    return (
        f"Given the Boolean formula in CNF:\n"
        f"{item['formula']}\n\n"
        f"Does the variable assignment \"{assignment}\" satisfy this formula?\n"
        f"Answer with Yes or No.\n"
        f"Answer:"
    )


def _disc_prompt_few(item, assignment, k, n_vars):
    examples = _few_shot_for(k, n_vars)[:4]
    body = (
        "Determine if a variable assignment satisfies a Boolean formula in CNF.\n"
        "Answer with Yes or No.\n\n"
    )
    for ex in examples:
        ans = "Yes" if ex['label'] == 'yes' else "No"
        body += (f"Formula: {ex['formula']}\n"
                 f"Assignment: {ex['assignment']}\n"
                 f"Answer: {ans}\n\n")
    body += (f"Formula: {item['formula']}\n"
             f"Assignment: {assignment}\n"
             f"Answer:")
    return body


def make_prompt(item, style='generator', shots='zero',
                gen_response=None, neg=False, variation=0, **kwargs):
    """Build the generator or discriminator prompt + completion.

    The generator target is the satisfying_assignment column (which is the
    canonical correct answer regardless of label). The discriminator target
    is just Yes/No reflecting whether item['assignment'] satisfies the
    formula. `gen_response` overrides which assignment the discriminator
    evaluates (for use during eval-time prompt-pair construction).
    """
    k = item.get('k', 2)
    n_vars = _detect_n_vars(item['assignment'])

    if style == 'generator':
        prompt = _gen_prompt_zero(item) if shots == 'zero' \
            else _gen_prompt_few(item, k=k, n_vars=n_vars)
        # Use satisfying_assignment when present (canonical correct answer);
        # fall back to the row's assignment (already correct for label='yes').
        sat = item.get('satisfying_assignment') or item['assignment']
        completion = " " + sat

    elif style == 'discriminator':
        assignment = gen_response if gen_response else item['assignment']
        prompt = _disc_prompt_zero(item, assignment) if shots == 'zero' \
            else _disc_prompt_few(item, assignment, k=k, n_vars=n_vars)
        label = normalize_yes_no(item.get('label', ''))
        completion = " Yes" if label == 'yes' else " No"

    else:
        raise ValueError(f"Unknown style: {style}. Must be 'generator' or 'discriminator'.")

    return PromptCompletion(prompt.strip(), completion)


def get_completion(item):
    """Generator target: prefer the canonical satisfying_assignment."""
    sat = item.get('satisfying_assignment') or item.get('assignment', '')
    return " " + sat


def get_label(item):
    return normalize_yes_no(item.get('label', ''))


def make_negated_prompt(item, task, make_prompt, gen_shots='zero'):
    """Negated generator prompt for --neg-typicality.

    Asks for an UNSATISFYING assignment instead of a satisfying one. Always
    zero-shot to avoid negating few-shot examples.
    """
    gen_obj = make_prompt(item, style='generator', shots='zero')
    neg_prompt = gen_obj.prompt.replace(
        "Provide a variable assignment that satisfies this formula.",
        "Provide a variable assignment that does NOT satisfy this formula.",
    )
    if neg_prompt == gen_obj.prompt:
        raise ValueError(
            f"Negated prompt unchanged for ksat task '{task}'. "
            f"Prompt '{gen_obj.prompt[:80]}' doesn't match expected format."
        )
    return neg_prompt, gen_obj.completion


# ============================================================================
# CSV schema for --save-scores-csv
# ============================================================================

CSV_HEADER = [
    "formula",
    "assignment",
    "satisfying_assignment",
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
    formula = get_field(item, 'formula', '')
    assignment = get_field(item, 'assignment', '')
    sat_assignment = get_field(item, 'satisfying_assignment', '')
    correct_label = normalize_yes_no(get_field(item, 'label', ''))
    item_strategy = get_field(item, 'strategy', strategy)
    return [
        formula,
        assignment,
        sat_assignment,
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


_registered = []
_skipped = []
for task_name, train_basename, test_basename, k, n_vars in TASK_VARIANTS:
    train_path = os.path.join(DATA_DIR, train_basename)
    test_path = os.path.join(DATA_DIR, test_basename)
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        _skipped.append(task_name)
        continue
    register_task({
        'name': task_name,
        'load_data': _make_load_data(train_basename, test_basename, k=k),
        'description': f'k-SAT (k={k}, {n_vars} vars): {task_name}',
        **_COMMON,
    })
    _registered.append(task_name)

if _registered:
    print(f"[ksat] Registered {len(_registered)} task(s): {', '.join(_registered)}")
if _skipped:
    print(f"[ksat] Skipped (data not on disk): {', '.join(_skipped)}")
