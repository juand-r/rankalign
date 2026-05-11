"""
GSM8K math word problem tasks (modern pattern, mirrors humaneval.py / codecontests.py).

Two parallel families differing only in whether the candidate solution
includes its final numeric answer:

    gsm8k-full       — solution ends with "The answer is: <N>"
    gsm8k-truncated  — solution truncated to "The answer is" (no number)

Each family registers three train variants and one auto-discovered eval
task per held-out test problem.

Training variants (sample N problems from the shared train.csv at load):

    gsm8k-full              ~45 problems (~2.5k rows)         [SMALL]
    gsm8k-full-double       ~90 problems (~5k rows)
    gsm8k-full-all          all 639 train problems (~36k rows)

    gsm8k-truncated, gsm8k-truncated-double, gsm8k-truncated-all
        identical splits; same underlying solutions, just with the final
        answer stripped from each completion.

Eval tasks (auto-discovered from the per-problem CSVs in
data/gsm8k/with_solutions/{full_response,truncated_response}/):

    gsm8k-full-gsm8k_test_<N>
    gsm8k-truncated-gsm8k_test_<N>

The eval split holds out 100 problems chosen by --split-seed 42 inside
build_gsm8k_dataset.py. The full and truncated CSVs share the same
underlying solutions so paired comparisons across the two families are
meaningful (see scripts/dataset_builder/GSM8K_BUILD_LOG.md).

Source data: RLHFlow/Mistral-GSM8K-Test (1.32K GSM8K test problems,
1024 Mistral-7B generations each, binary-labeled by gold-answer match).
"""

import os
import sys
import random
from collections import defaultdict

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
    'data', 'gsm8k', 'with_solutions',
)
FULL_DIR = os.path.join(DATA_DIR, 'full_response')
TRUNC_DIR = os.path.join(DATA_DIR, 'truncated_response')

# v1 family (4 models × 3 strategies cycle build, May 2026). Output of
# scripts/dataset_builder/build_gsm8k_v1.py with --test-ids-file mode.
V1_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'gsm8k', 'v1',
)
V1_TRAIN_CSV = os.path.join(V1_DIR, 'train.csv')
V1_FIELDS = ('question', 'answer', 'correct', 'strategy',
             'model', 'temperature', 'task_id', 'error')

# Train problem caps for the three sized variants. Each problem contributes
# ~56 rows on average at --balance --max-per-side 30, so:
#   45  →  ~2.5k rows  (matches humaneval / codecontests-base / ifeval-concat)
#   90  →  ~5k   rows
#   639 →  ~36k rows   (use all available training problems)
TRAIN_PROBLEMS_BASE = 45
TRAIN_PROBLEMS_DOUBLE = 90
SAMPLE_SEED = 42  # used by sized variants to pick which problems to keep

CSV_FIELDS = ('question', 'answer', 'correct', 'strategy')


# ============================================================================
# Data loading
# ============================================================================

def _load_items(filepath):
    items = load_csv_items(filepath, fields=CSV_FIELDS)
    for row in items:
        row['correct'] = str(row.get('correct', '')).strip()
        row['strategy'] = row.get('strategy', '')
    return items


def _sample_by_problems(items, n_problems, rng):
    """Pick all items belonging to n_problems randomly-chosen questions.

    Mirrors codecontests._sample_by_problems(). Independent of dict
    iteration order: groups by question text, sorts the question keys,
    then shuffles with the supplied RNG.
    """
    by_question = defaultdict(list)
    for item in items:
        by_question[item['question']].append(item)

    questions = sorted(by_question.keys())
    rng.shuffle(questions)
    selected = questions[:n_problems]

    result = []
    for q in selected:
        result.extend(by_question[q])
    rng.shuffle(result)
    return result


def _make_train_loader(version_dir, n_problems):
    """Factory for load_data on a sized train variant.

    n_problems = None means "use all train problems".
    Test set is empty for train variants — use the per-problem
    auto-registered tasks below for eval.
    """
    train_csv = os.path.join(version_dir, 'train.csv')

    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        all_items = _load_items(train_csv)
        if n_problems is None:
            rng = random.Random(seed)
            rng.shuffle(all_items)
            return all_items, []
        # Use a fixed sampling seed (independent of `seed`) so the small/
        # double variants deterministically pick the same problem subset
        # across runs — matching the codecontests pattern.
        rng = random.Random(SAMPLE_SEED)
        return _sample_by_problems(all_items, n_problems, rng), []

    return load_data


def _make_eval_loader(version_dir, test_csv_basename):
    """Factory for load_data on a per-problem eval task.

    Train comes from the shared train.csv (so eval-time prompt-pair
    construction has access to the same training pool the trained model
    saw); test comes from the held-out per-problem CSV.
    """
    train_csv = os.path.join(version_dir, 'train.csv')
    test_csv = os.path.join(version_dir, test_csv_basename)

    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = _load_items(train_csv)
        L_test = _load_items(test_csv)
        rng = random.Random(seed)
        rng.shuffle(L_train)
        rng.shuffle(L_test)
        return L_train, L_test

    return load_data


# ============================================================================
# Prompts
# ============================================================================

# Few-shot examples kept short to avoid blowing up context: short numeric
# problems with single-step reasoning. Used only when shots='few'.
DISC_FEW_SHOT_EXAMPLES = [
    {
        'question': 'Apples cost $2 each. How much do 3 apples cost in dollars?',
        'answer': 'Each apple is $2, so 3 apples cost 3 * $2 = $6. The answer is: 6',
        'label': 'Yes',
    },
    {
        'question': 'Sarah has 12 cookies. She eats 5 of them. How many cookies are left?',
        'answer': 'Sarah eats 5, so she has 12 + 5 = 17 cookies left. The answer is: 17',
        'label': 'No',
    },
    {
        'question': 'A train travels at 60 mph for 2 hours. How many miles does it travel?',
        'answer': 'Distance = speed * time = 60 * 2 = 120. The answer is: 120',
        'label': 'Yes',
    },
]


def _disc_query(question, answer):
    return (
        f'Is this a correct solution to the math problem?\n\n'
        f'Problem:\n{question}\n\n'
        f'Solution:\n{answer}\n\n'
        f'Answer Yes or No:'
    )


def make_prompt(item, style='generator', shots='zero',
                gen_response=None, neg=False, variation=0, **kwargs):
    question = item['question']
    answer = item['answer']

    if style == 'generator':
        prompt = f"Solve the following math problem step by step.\n\nProblem: {question}\nSolution:"
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
        correct = normalize_yes_no(item.get('correct', ''))
        completion = " Yes" if correct == 'yes' else " No"

    else:
        raise ValueError(f"Unknown style: {style}. Must be 'generator' or 'discriminator'.")

    return PromptCompletion(prompt.strip(), completion)


def get_completion(item):
    return " " + item['answer']


def get_label(item):
    return normalize_yes_no(item.get('correct', ''))


def make_negated_prompt(item, task, make_prompt, gen_shots='zero'):
    """Negated generator prompt for --neg-typicality.

    Replaces "Solve the following math problem step by step." with an
    explicit request for a wrong step-by-step solution. Always zero-shot
    to avoid negating few-shot examples.
    """
    gen_obj = make_prompt(item, style='generator', shots='zero')
    neg_prompt = gen_obj.prompt.replace(
        "Solve the following math problem step by step.",
        "Write an incorrect step-by-step solution to the following math problem.",
    )
    if neg_prompt == gen_obj.prompt:
        raise ValueError(
            f"Negated prompt unchanged for gsm8k task '{task}'. "
            f"Prompt '{gen_obj.prompt[:80]}' doesn't match expected format."
        )
    return neg_prompt, gen_obj.completion


# ============================================================================
# CSV schema for --save-scores-csv
# ============================================================================

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
    # Per-problem eval tasks look like 'gsm8k-full-gsm8k_test_42';
    # train variants look like 'gsm8k-full' / 'gsm8k-full-double' / 'gsm8k-full-all'.
    # In the latter case there is no per-problem id, so leave problem_name blank.
    problem_name = ""
    for prefix in ("gsm8k-full-", "gsm8k-truncated-"):
        if task.startswith(prefix) and task != prefix.rstrip('-'):
            tail = task[len(prefix):]
            if tail not in ("double", "all"):
                problem_name = tail
            break
    solution = get_field(item, "answer", "")
    solution_preview = solution[:200].replace("\n", "\\n")
    item_strategy = get_field(item, "strategy", strategy)
    correct_label = normalize_yes_no(get_field(item, "correct", ""))
    return [
        problem_name,
        solution_preview,
        "math",
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


def _register_family(family_name, version_dir):
    """Register the three train variants + per-problem eval tasks for one
    response-style family ('full' or 'truncated')."""
    if not os.path.exists(version_dir):
        print(f"[gsm8k] Data not found at {version_dir} — skipping {family_name} family")
        return 0

    train_csv = os.path.join(version_dir, 'train.csv')
    if os.path.exists(train_csv):
        register_task({
            'name': family_name,
            'load_data': _make_train_loader(version_dir, n_problems=TRAIN_PROBLEMS_BASE),
            'description': f'GSM8K {family_name}: ~{TRAIN_PROBLEMS_BASE} train problems',
            **_COMMON,
        })
        register_task({
            'name': f'{family_name}-double',
            'load_data': _make_train_loader(version_dir, n_problems=TRAIN_PROBLEMS_DOUBLE),
            'description': f'GSM8K {family_name}: ~{TRAIN_PROBLEMS_DOUBLE} train problems',
            **_COMMON,
        })
        register_task({
            'name': f'{family_name}-all',
            'load_data': _make_train_loader(version_dir, n_problems=None),
            'description': f'GSM8K {family_name}: all train problems',
            **_COMMON,
        })

    n_eval = 0
    for filename in sorted(os.listdir(version_dir)):
        if not filename.endswith('.csv') or filename == 'train.csv':
            continue
        slug = filename[:-4]
        task_name = f'{family_name}-{slug}'
        try:
            register_task({
                'name': task_name,
                'load_data': _make_eval_loader(version_dir, filename),
                'description': f'GSM8K {family_name}: {slug}',
                **_COMMON,
            })
            n_eval += 1
        except Exception as e:
            print(f"[gsm8k] Warning: Could not register {task_name}: {e}")
    return n_eval


_n_full = _register_family('gsm8k-full', FULL_DIR)
_n_trunc = _register_family('gsm8k-truncated', TRUNC_DIR)
if _n_full or _n_trunc:
    print(f"[gsm8k] Registered {_n_full} eval tasks under gsm8k-full and "
          f"{_n_trunc} under gsm8k-truncated (plus 3 train variants per family)")


# ============================================================================
# v1 family (modern build, May 2026 — 4 cycle models × 3 cycle strategies)
# ============================================================================

def _load_v1_items(filepath):
    items = load_csv_items(filepath, fields=V1_FIELDS)
    for row in items:
        row['correct'] = str(row.get('correct', '')).strip()
        row['strategy'] = row.get('strategy', '')
    return items


def load_data_v1_train_only(seed=0, split_type='random', sample_negative=False, **kwargs):
    """v1 full training set; test set is empty (use per-problem tasks for eval)."""
    L_train = _load_v1_items(V1_TRAIN_CSV)
    random.Random(seed).shuffle(L_train)
    return L_train, []


def create_load_data_v1_for_problem(test_csv_path):
    """Factory: v1 train.csv for train, specific per-problem CSV for test."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = _load_v1_items(V1_TRAIN_CSV)
        L_test = _load_v1_items(test_csv_path)
        rng = random.Random(seed)
        rng.shuffle(L_train)
        rng.shuffle(L_test)
        return L_train, L_test
    return load_data


if os.path.exists(V1_TRAIN_CSV):
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
        'name': 'gsm8k-v1',
        'load_data': load_data_v1_train_only,
        'description': 'GSM8K v1: full training set (4 models × 3 strategies)',
        **_V1_COMMON,
    })

    _v1_registered = []
    for filename in sorted(os.listdir(V1_DIR)):
        if not filename.endswith('.csv') or filename == 'train.csv' or filename == 'train_old.csv':
            continue
        slug = filename[:-4]  # strip .csv
        test_csv_path = os.path.join(V1_DIR, filename)
        task_name = f'gsm8k-v1-{slug}'
        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_v1_for_problem(test_csv_path),
                'description': f'GSM8K v1: {slug}',
                **_V1_COMMON,
            })
            _v1_registered.append(task_name)
        except Exception as e:
            print(f"[gsm8k-v1] Warning: Could not register {task_name}: {e}")

    if _v1_registered:
        print(f"[gsm8k-v1] Registered {len(_v1_registered)} eval tasks (plus gsm8k-v1 train)")
else:
    print(f"[gsm8k-v1] Data not found at {V1_DIR} — skipping v1 registration")


# ============================================================================
# v1.1 family — v1 rows with retroactive tweaks applied:
#   (1) dual-metric extractor relabel
#   (2) length filter [60, 3000]
#   (3) gpt-4.1-mini truncation judge filter
#   (4) gemma-2-9b-it logP/token >= -2 filter
# Built by notes/gsm8k-v1.1/build_v1_1_step{1,2_prep,3_finalize}.py.
# Clean separation: v1.1 reuses the same prompt/completion/label helpers
# as v1 (so eval pipelines work unchanged), only the per-problem CSV
# contents differ. There is no v1.1 train.csv — eval tasks load v1's
# train.csv as their train set (we are only changing the test data).
# ============================================================================

V1_1_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'gsm8k', 'v1.1',
)

# v1.1 CSVs have extra columns beyond V1_FIELDS, but load_csv_items only
# pulls the requested fields and tolerates extras — so the same V1_FIELDS
# spec works. The only column-level semantic change is that `correct` in
# v1.1 is the new flexible_correct (not v1's buggy-extractor label).
V1_1_FIELDS = V1_FIELDS


def _load_v1_1_items(filepath):
    items = load_csv_items(filepath, fields=V1_1_FIELDS)
    for row in items:
        row['correct'] = str(row.get('correct', '')).strip()
        row['strategy'] = row.get('strategy', '')
    return items


def create_load_data_v1_1_for_problem(test_csv_path):
    """Factory: v1 train.csv for train, v1.1 per-problem CSV for test."""
    def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
        L_train = _load_v1_items(V1_TRAIN_CSV)         # v1 train (unchanged)
        L_test = _load_v1_1_items(test_csv_path)       # v1.1 test (filtered)
        rng = random.Random(seed)
        rng.shuffle(L_train)
        rng.shuffle(L_test)
        return L_train, L_test
    return load_data


if os.path.exists(V1_1_DIR):
    _v1_1_registered = []
    for filename in sorted(os.listdir(V1_1_DIR)):
        if not filename.endswith('.csv'):
            continue
        if filename in ('train.csv', 'train_old.csv'):
            continue
        slug = filename[:-4]
        test_csv_path = os.path.join(V1_1_DIR, filename)
        task_name = f'gsm8k-v1.1-{slug}'
        try:
            register_task({
                'name': task_name,
                'load_data': create_load_data_v1_1_for_problem(test_csv_path),
                'description': f'GSM8K v1.1: {slug}',
                **_V1_COMMON,
            })
            _v1_1_registered.append(task_name)
        except Exception as e:
            print(f"[gsm8k-v1.1] Warning: Could not register {task_name}: {e}")

    if _v1_1_registered:
        print(f"[gsm8k-v1.1] Registered {len(_v1_1_registered)} eval tasks")
else:
    print(f"[gsm8k-v1.1] Data not found at {V1_1_DIR} — skipping v1.1 registration")
