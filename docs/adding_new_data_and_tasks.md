# Adding New Data and Tasks

This guide tells you exactly what to do when you want to add a new dataset
or a brand-new task family. There is one supported path for new tasks
(the "modern" pattern used by `humaneval` and `codecontests`); follow it.

---

## 1) TL;DR

- **Adding more data to an existing auto-discovery family?** Drop the file
  in the right directory. No Python edits. See §3.
- **Adding a brand-new task family?** Copy [`src/tasks/humaneval.py`](../src/tasks/humaneval.py)
  (or [`codecontests.py`](../src/tasks/codecontests.py)) as a starting point.
  Edit four functions, register, add one import line. See §4.
- **Do not** put new code on the legacy path or follow legacy modules
  (`hypernym_*`, `ifeval_*`, `ambigqa_*`, `plausibleqa_*`, `membership`,
  `rosch`) as templates. See §2.
- **After implementing**: load your data through `get_L_prompt(...)` and
  print a generator and discriminator example to confirm the prompts look
  right.

---

## 2) Two patterns: modern (use this) vs. legacy (frozen)

Two patterns coexist in this repo, **by design**:

| Pattern | Tasks using it | What it uses for `--save-scores-csv` and `--neg-typicality` |
|---|---|---|
| **Modern** (recommended for all new tasks) | `humaneval`, `codecontests` | Generic registry callbacks: `csv_header` + `csv_row_builder` and `make_negated_prompt` |
| **Legacy** (frozen — do not extend) | `hypernym-*`, `ifeval-*`, `ambigqa*`, `plausibleqa*`, `membership-sans-rosch-*`, `rosch-*`, `trivia-qa`, `swords`, `lambada`, `collie`, `ksat`, `hyponym` | Hardcoded `if is_<family>_task(task)` branches in `scripts/eval_by_claude.py` |

**Rule for new work**: write tasks in the modern pattern. Don't migrate
legacy tasks (it's a known consistency hazard but stable; we don't want
to introduce regressions). Don't reuse legacy task name prefixes for new
tasks — see Gotcha 1 in §7.

The canonical references for the modern pattern are
[`src/tasks/humaneval.py`](../src/tasks/humaneval.py) and
[`src/tasks/codecontests.py`](../src/tasks/codecontests.py). When in
doubt, copy from those.

---

## 3) Adding more data to an existing auto-discovery family

These families auto-register a new task whenever a matching data file
appears. **No Python edits needed.**

### Hypernym per-hyponym (legacy family — extend only if you must)

Drop CSVs matching:

```
data/hypernym_<hyponym>_google-gemma-2-2b_train.csv
data/hypernym_<hyponym>_google-gemma-2-2b_test.csv
```

→ auto-registers `hypernym-<hyponym>`. For v2/grammar-corrected prompts
also maintain the `-fixed.csv` files in `data/fixed-hypernyms/`.

### IFEval per-prompt (legacy family)

Drop:

```
data/fixed-prompts-ifeval/gpt_ifeval_results_<prompt_name>.jsonl
```

→ auto-registers `ifeval-<prompt_name>`.

### AmbigQA v1 per-question (legacy family)

Drop per-question CSVs in `data/ambigqa/with_negatives/<slug>.csv`.
`train.csv` is the shared train set; every other CSV becomes
`ambigqa-<slug>`.

### PlausibleQA per-question (legacy family)

The active data lives under `data/plausibleqa/fixed-plausibleqa/`
(historical filename `plausibleqa_v0.py`). Drop:

- test task files in `data/plausibleqa/fixed-plausibleqa/test/*.csv`
  → `plausibleqa-<id>`
- optional train-eval task files in
  `data/plausibleqa/fixed-plausibleqa/train-per-question/*.csv`
  → `plausibleqa-train-<id>`

### Rosch per-category (legacy family, eval-only)

Drop `data/rosch/rosch-<slug>_test.csv` → auto-registers `rosch-<slug>`.

### CodeContests per-problem (modern family)

Drop `data/codecontests/test/<slug>.jsonl` → auto-registers
`codecontests-<slug>`. To regenerate from the HuggingFace source, run
`python scripts/preprocess_codecontests.py`.

### HumanEval per-problem (modern family)

Drop `data/humaneval/with_solutions/humaneval_<N>.csv` →
auto-registers `humaneval-humaneval_<N>`. The shared train file is
`data/humaneval/with_solutions/train.csv`.

---

## 4) Adding a brand-new task family (modern pattern)

Copy `humaneval.py` or `codecontests.py` as your starting point —
they're the most up-to-date examples. The boilerplate version of these
steps lives in [`src/tasks/example_task.py`](../src/tasks/example_task.py),
which mirrors the modern pattern.

### Step A: Pick a task name

The name appears in `--task <name>`. **Do not start it with any of
these legacy prefixes** unless you intend the legacy CSV/neg-typicality
behavior:

```
hypernym- ifeval- ambigqa- plausibleqa- membership-sans-rosch- rosch-
```

Modern-pattern tasks need a name that doesn't match any of those
prefixes. `eval_by_claude.py:is_legacy_csv_task(...)` is the source of
truth for which prefixes route through legacy code.

### Step B: Lay out your data

Put data files in `data/<your_task>/`. Use stable, deterministic file
names. Build paths in your task module from `__file__`, not from CWD.

### Step C: Create `src/tasks/<your_task>.py`

Required imports and structure (mirror `humaneval.py`):

```python
import os, sys, random
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from task_registry import register_task
from tasks.common import PromptCompletion, load_csv_items, normalize_yes_no, get_field
```

Implement four required functions and (for the modern pattern) two
optional ones:

| Function | Required? | What it does |
|---|---|---|
| `load_data(seed, split_type, sample_negative=False, **kwargs) -> (L_train, L_test)` | yes | Load + split data. Use `random.Random(seed)`, never `random.seed(seed)`. |
| `make_prompt(item, style, shots, gen_response=None, neg=False, **kwargs) -> PromptCompletion` | yes | Build generator/discriminator prompts. |
| `get_completion(item) -> str` | yes | Return `" " + answer` (leading space). |
| `get_label(item) -> "yes" \| "no"` | yes | Use `normalize_yes_no(...)` from `tasks.common`. |
| `make_negated_prompt(item, task, make_prompt, gen_shots) -> (neg_prompt, completion)` | yes if you'll use `--neg-typicality` | Build the negated generator prompt. |
| `csv_row_builder(...) -> list` and `CSV_HEADER: list[str]` | yes if you'll use `--save-scores-csv` | Schema for the per-example score CSV. |

The two "yes if you'll use…" callbacks are **functionally required for
the modern pattern**. Without `make_negated_prompt`, `--neg-typicality`
raises `NotImplementedError` for your task. Without
`csv_header`/`csv_row_builder`, `--save-scores-csv` raises `ValueError`.
Add them.

### Step D: Register

Use a `_COMMON` dict so multi-variant registration stays clean (see
`humaneval.py` for the pattern):

```python
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

register_task({'name': 'my-task', 'load_data': load_data, **_COMMON})
```

### Step E: Import in `src/tasks/__init__.py`

```python
from . import my_task
```

If you skip this, your task simply won't show up — there is no warning.
This is the most common mistake.

### Step F: Smoke-test registration

```bash
source /u/jdr/venvs/venv_lexcons/bin/activate
python - <<'PY'
import sys; sys.path.insert(0, "src")
import tasks
from task_registry import is_registered
assert is_registered("my-task"), "Not registered — did you import in __init__.py?"
print("OK")
PY
```

### Step G: Smoke-test prompts

This is non-optional. Load the data through the same path the training
and eval code use, and print one generator and one discriminator
example. Bad prompts only show up here.

```bash
python - <<'PY'
import sys; sys.path.insert(0, "src")
import tasks
from utils import get_L_prompt
L_train, L_test, make_prompt = get_L_prompt("my-task", "random", seed=0)
g = make_prompt(L_train[0], style='generator',     shots='zero')
d = make_prompt(L_train[0], style='discriminator', shots='zero')
print("GENERATOR PROMPT:\n", g.prompt)
print("GENERATOR COMPLETION:", repr(g.completion))
print("---")
print("DISCRIMINATOR PROMPT:\n", d.prompt)
print("DISCRIMINATOR COMPLETION:", repr(d.completion))
PY
```

### Step H: Smoke-test train/eval

Use the standard wrappers:

- training: `scripts/run_train_semi.sh`
- evaluation: `scripts/run_eval_semi.sh`

These wrappers `cd` to the right working directory and pass the right
flags. For modern tasks no edits to `eval_by_claude.py` should be needed.

---

## 5) Registry contract (full reference)

`register_task(config: dict)` accepts these fields. See
[`src/task_registry.py`](../src/task_registry.py) for the source of truth.

### Required (all tasks)

- `name: str` — task identifier used in `--task <name>`.
- `load_data: Callable(seed, split_type, **kwargs) -> (L_train, L_test)`.
- `make_prompt: Callable(item, style, shots, **kwargs) -> PromptCompletion`.
- `get_completion: Callable(item) -> str` — should return `" " + answer`.
- `get_label: Callable(item) -> "yes" | "no"`.

### Required for the modern pattern (any new task family)

- `make_negated_prompt: Callable(item, task, make_prompt, gen_shots) -> (neg_prompt, completion)`
  — used by `eval_by_claude.py:make_negated_gen_prompt` for
  `--neg-typicality`.
- `csv_header: list[str]` and
  `csv_row_builder: Callable(...) -> list` — used by
  `eval_by_claude.py` for `--save-scores-csv`.

### Optional

- `get_indicator(item) -> 0|1` — defaults to `1 if get_label(item)=='yes' else 0`.
- `batch_size: {"with_ref": int, "without_ref": int}` —
  consumed by `ranking_loss_ref.py`. Default `{"with_ref": 1, "without_ref": 2}`.
- `description: str` — free-form, displayed in registration log only.
- `origin_split: "test" | "valid"` — used by `codecontests` to mark which
  HuggingFace split a per-problem eval came from; queryable via
  `cfg.get('origin_split')`.
- `short: bool` — used by `codecontests` to mark short-completion tasks.

### Currently not enforced (informational only)

- `supports_split_types: list[str]`
- `supports_negative_sampling: bool`
- `filter_positive: Callable(item) -> bool`

These are accepted and stored, but no script reads them. Don't rely on
them for behavior.

---

## 6) Where existing data lives (reference)

| Family | Path | Pattern |
|---|---|---|
| Hypernym (legacy base) | `data/ranks.txt` | legacy `hypernym` task |
| Hypernym per-hyponym | `data/hypernym_<X>_google-gemma-2-2b_{train,test}.csv` | legacy auto-discovery |
| Hypernym v2/fixed | `data/fixed-hypernyms/hypernym_<X>_google-gemma-2-2b_{train,test}-fixed.csv` | legacy |
| IFEval per-prompt | `data/fixed-prompts-ifeval/gpt_ifeval_results_<prompt>.jsonl` | legacy |
| AmbigQA v1 | `data/ambigqa/with_negatives/{train,<slug>}.csv` | legacy |
| PlausibleQA | `data/plausibleqa/fixed-plausibleqa/{train.csv, test/*.csv, train-per-question/*.csv}` | legacy |
| Rosch | `data/rosch/rosch-<slug>_test.csv` | legacy (eval-only) |
| Membership | `data/membership/{combined_train_categories_final*.json, excluded_categories.json}` | legacy |
| k-SAT | `data/{2,3}sat_{train,test}.csv` | legacy |
| CodeContests | `data/codecontests/{descriptions.json, train.jsonl, test/<slug>.jsonl, split_manifest.json}` | **modern** |
| HumanEval | `data/humaneval/with_solutions/{train.csv, humaneval_<N>.csv}` | **modern** |

Codecontests-specific notes: each per-problem eval task has
`origin_split` ("test" or "valid") and a `short` flag (median completion
< 300 tokens). To filter:

```python
from task_registry import TASK_REGISTRY
test_tasks  = [n for n, c in TASK_REGISTRY.items() if c.get('origin_split') == 'test']
short_tasks = [n for n, c in TASK_REGISTRY.items() if c.get('short')]
```

---

## 7) Gotchas

1. **Legacy prefixes route through legacy code.** A task name starting
   with `hypernym-`, `ifeval-`, `ambigqa-`, `plausibleqa-`,
   `membership-sans-rosch-`, or `rosch-` will be detected as legacy by
   `eval_by_claude.py` and use hardcoded CSV-writing and neg-typicality
   branches, **even if your registry entry provides
   `csv_header`/`csv_row_builder`/`make_negated_prompt`**. The registry
   callbacks for legacy-prefixed tasks are silently ignored. Pick a
   non-legacy prefix.

2. **A task module must be imported to register.** Adding a file to
   `src/tasks/` is not enough — `src/tasks/__init__.py` must contain
   `from . import <module>`. If you skip the import the task will not
   appear and `--task <name>` will reject it as an unknown choice with
   no clue why.

3. **`make_negated_prompt` is required for `--neg-typicality`** on
   modern tasks. Without it, `make_negated_gen_prompt` in
   `eval_by_claude.py` raises `NotImplementedError`.

4. **`csv_header` + `csv_row_builder` are required for
   `--save-scores-csv`** on modern tasks. Without both,
   `eval_by_claude.py` raises `ValueError`.

5. **`ranking_loss_ref.py` only exposes `--split_type
   {random,hyper,both}`.** If your task advertises additional
   `supports_split_types`, the trainer's argparse won't let users select
   them. `random` is the default and what almost every task uses.

6. **Large dynamic families inflate startup output.** Each registration
   prints a line. PlausibleQA + CodeContests register ~800 tasks
   between them.

7. **Working directory matters for legacy loaders.** Some functions in
   `src/utils.py` use `../data/...` relative paths. Wrapper scripts
   `cd scripts/` before calling Python. New modern tasks should always
   build paths from `__file__`.

8. **Known broken: `ambigqa_v0.py`.** It's still imported in
   `src/tasks/__init__.py` but its expected path
   `data/ambigqa/v0/combined.csv` does not exist (data was moved to
   `data/ambigqa/v0-combined-ambigqa-plausibleqa/`). Registration
   silently skips. Either fix the path or remove the import; not a
   blocker for new work.

9. **`register_task` will warn on duplicate names** but still
   overwrites. The last registration wins.

---

## 8) Script compatibility (honest version)

| Script | Behavior |
|---|---|
| `scripts/ranking_loss_ref.py` | Registry-first, with legacy `if/elif` chains for `hypernym`, `hypernym-car`, `trivia-qa`, `swords`, `lambada`, `ifeval`, `collie` |
| `scripts/eval_by_claude.py` | Registry-first, with hardcoded family branches (`is_<family>_task(...)`) for hypernym, ifeval, ambigqa, plausibleqa, membership, rosch (CSV writing + neg-typicality) |
| `scripts/eval.py` | Older eval script. Registry-first; small legacy fallback. Still used in some workflows. |
| `src/utils.py:get_L_prompt` | Registry-first. |
| `src/logitlens.py` | Registry-first. |
| `scripts/consistency_ft.py` | Has task-specific output formatting branches. Don't rely on for new families without checking. |
| Orchestration shell scripts (`run_*.sh` per-task wrappers) | Mostly hardcoded `TASKS=(...)` lists. Update if your workflow uses one. |

For modern tasks, you should not need to edit `ranking_loss_ref.py`,
`eval_by_claude.py`, or `eval.py`.

---

## 9) Practical checklist

- [ ] Data files are in stable locations under `data/<family>/...`.
- [ ] Task name does not start with a legacy prefix (§2 table).
- [ ] Module copied from `humaneval.py` or `codecontests.py`.
- [ ] Imports `PromptCompletion` and helpers from `tasks.common`.
- [ ] Implements `load_data`, `make_prompt`, `get_completion`, `get_label`.
- [ ] Implements `make_negated_prompt`, `CSV_HEADER`, `build_csv_row`.
- [ ] Uses `random.Random(seed)`, not `random.seed(seed)`.
- [ ] `register_task(...)` uses the `_COMMON` dict pattern.
- [ ] `from . import <module>` added to `src/tasks/__init__.py`.
- [ ] Registration smoke test passes (§4 step F).
- [ ] **Prompts smoke test passes — printed gen + disc examples look correct (§4 step G).**
- [ ] Train smoke test runs via `scripts/run_train_semi.sh`.
- [ ] Eval smoke test runs via `scripts/run_eval_semi.sh`.
