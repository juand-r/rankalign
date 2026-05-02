"""
Example task template (MODERN PATTERN).

Mirrors `humaneval.py` and `codecontests.py` — these are the canonical
references for adding a new task family. Copy this file, rename, fill
in the four required functions plus the modern callbacks, and register.

Step-by-step:
    1. cp example_task.py my_task.py
    2. Pick a task name that does NOT start with a legacy prefix:
         hypernym- ifeval- ambigqa- plausibleqa- membership-sans-rosch- rosch-
       (those route through hardcoded branches in eval_by_claude.py)
    3. Implement load_data, make_prompt, get_completion, get_label
    4. Implement make_negated_prompt (required for --neg-typicality)
    5. Implement CSV_HEADER + build_csv_row (required for --save-scores-csv)
    6. Update the register_task() call(s) at the bottom
    7. Add `from . import my_task` to src/tasks/__init__.py
    8. Smoke-test: registration + a printed generator+discriminator example

NOTE: This template intentionally raises NotImplementedError in the
function bodies, so it is NOT imported in src/tasks/__init__.py. After
you copy and fill it in, remember to add the import.

See docs/adding_new_data_and_tasks.md for the full guide.
"""

import os
import sys
import random

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
from tasks.common import PromptCompletion, load_csv_items, normalize_yes_no, get_field


# Build paths from __file__, not CWD.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'my_task',
)


# =============================================================================
# Required: load_data, make_prompt, get_completion, get_label
# =============================================================================


def load_data(seed=0, split_type='random', sample_negative=False, **kwargs):
    """
    Return (L_train, L_test). Items can be dicts, namedtuples, or anything
    your other functions accept consistently.

    Use a LOCAL RNG (`random.Random(seed)`), not `random.seed(seed)` —
    seeding the global random module pollutes state for everything else.

    Example:
        items = load_csv_items(os.path.join(DATA_DIR, 'data.csv'))
        rng = random.Random(seed)
        rng.shuffle(items)
        split_idx = int(len(items) * 0.8)
        return items[:split_idx], items[split_idx:]
    """
    raise NotImplementedError("Implement load_data().")


def make_prompt(item, style='generator', shots='zero',
                gen_response=None, neg=False, variation=0, **kwargs):
    """
    Build the generator or discriminator prompt + completion.

    style:
        'generator'     — the model produces the answer.
        'discriminator' — the model answers Yes/No about a proposed answer.
    shots:
        'zero' — no in-context examples.
        'few'  — include in-context examples (define DISC_FEW_SHOT_EXAMPLES).
    gen_response:
        If given, the discriminator asks about THIS string (typically a
        model generation) instead of item['answer']. Used by some eval
        flows; safe to ignore for vanilla item-based scoring.
    neg:
        Used for negative-prompt training variants. Most new tasks can
        ignore this (no-op).
    variation:
        Used by hypernym for prompt-variation studies. Most new tasks can
        ignore.

    Returns PromptCompletion(prompt=str, completion=str). The completion
    must include a leading space (" Yes", " Paris", ...) so the leading
    token is tokenized as a separate word.

    Example:
        if style == 'generator':
            prompt = f"Answer the question: {item['question']}\\nAnswer:"
            completion = " " + item['answer']
        elif style == 'discriminator':
            ans = gen_response if gen_response else item['answer']
            prompt = f'Is "{ans}" the answer to "{item["question"]}"? Answer:'
            completion = " Yes" if normalize_yes_no(item['correct']) == 'yes' else " No"
        return PromptCompletion(prompt.strip(), completion)
    """
    raise NotImplementedError("Implement make_prompt().")


def get_completion(item):
    """
    Return the generator's target completion as a string with a leading
    space. Should match `make_prompt(item, style='generator').completion`.

    Example: `return " " + item['answer']`
    """
    raise NotImplementedError("Implement get_completion().")


def get_label(item):
    """
    Return 'yes' or 'no' (lowercase, exact). Use normalize_yes_no(...)
    from tasks.common — it handles 'Yes', 'YES', 'true', '1', etc.

    Example: `return normalize_yes_no(item.get('correct', ''))`
    """
    raise NotImplementedError("Implement get_label().")


# =============================================================================
# Required for the modern pattern: make_negated_prompt
# (Used by eval_by_claude.py when --neg-typicality is passed.)
# =============================================================================


def make_negated_prompt(item, task, make_prompt, gen_shots='zero'):
    """
    Build a "negated" generator prompt — the same task framed to ask for
    an INCORRECT answer. Used by --neg-typicality to compute
    log P(completion | negated_prompt) as the typicality denominator.

    Returns (neg_prompt: str, completion: str).

    Strategy: take the normal generator prompt and replace the framing
    so it elicits a wrong answer. Always uses zero-shot to avoid
    negating in-context examples. Raise if the substitution didn't
    change anything (silent failure mode).

    Example (humaneval style):
        gen_obj = make_prompt(item, style='generator', shots='zero')
        neg_prompt = gen_obj.prompt.replace(
            "Complete the following Python function:",
            "Write an incorrect implementation of the following Python function:",
        ).replace("\\nSolution:", "\\nIncorrect solution:")
        if neg_prompt == gen_obj.prompt:
            raise ValueError(f"Negated prompt unchanged for task '{task}'.")
        return neg_prompt, gen_obj.completion
    """
    raise NotImplementedError("Implement make_negated_prompt().")


# =============================================================================
# Required for the modern pattern: CSV schema for --save-scores-csv
# =============================================================================


CSV_HEADER = [
    # Task-specific identifying columns first (rename freely):
    "item_id",
    "answer_preview",
    # Standard columns (keep these names — dashboard tools expect them):
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
    """
    Build one row matching CSV_HEADER. The standard score columns are
    passed in pre-computed; you only need to fill the identifying
    columns at the front and pass the rest through.

    Use get_field(item, 'foo', '') to read fields from either a dict or
    namedtuple uniformly.
    """
    item_id = get_field(item, 'id', '') or get_field(item, 'question', '')[:60]
    answer = get_field(item, 'answer', '')
    return [
        item_id,
        answer[:200].replace("\n", "\\n"),
        num_toks,
        get_field(item, 'strategy', strategy),
        normalize_yes_no(get_field(item, 'correct', '')),
        disc_score,
        gen_score_raw,
        gen_score_typcorr_val,
        gen_score_lenorm,
        gen_score_typcorr_lenorm,
        modelname,
    ]


# =============================================================================
# Registration
# =============================================================================
#
# Uncomment and customize after filling in the functions above.
#
# _COMMON = {
#     'make_prompt': make_prompt,
#     'get_completion': get_completion,
#     'get_label': get_label,
#     'make_negated_prompt': make_negated_prompt,
#     'csv_header': CSV_HEADER,
#     'csv_row_builder': build_csv_row,
#     'batch_size': {'with_ref': 1, 'without_ref': 4},
#     'supports_split_types': ['random'],
# }
#
# register_task({
#     'name': 'my-task',
#     'load_data': load_data,
#     'description': 'My new task',
#     **_COMMON,
# })
#
# For multi-variant or per-file auto-discovered families, see
# humaneval.py / codecontests.py for the factory + loop pattern.
