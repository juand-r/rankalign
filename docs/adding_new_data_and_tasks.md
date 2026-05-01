# Adding New Data and Tasks

This guide explains how tasks and datasets are wired in this repo, and what to do when you want to add a new one.

It focuses on:

- where tasks are defined
- how they are discovered and loaded
- how train/eval scripts consume them
- pitfalls that commonly break new task integrations

---

## 1) Mental Model

At runtime, task loading works like this:

1. A script adds `src/` to `sys.path`.
2. The script imports `tasks` (from `src/tasks/__init__.py`).
3. Importing `tasks` imports each task module, and each module calls `register_task(...)`.
4. Scripts query `get_task(task_name)` from `src/task_registry.py`.
5. If a task is registered, scripts use that config.
6. If a task is not registered, scripts fall back to legacy `if/elif` paths.

Core files:

- `src/task_registry.py`: task registry API and defaults
- `src/tasks/__init__.py`: task module import list (registration trigger)
- `src/tasks/example_task.py`: template for new task modules
- `src/utils.py`: shared loader/prompt path via `get_L_prompt(...)`
- `scripts/ranking_loss_ref.py`: main training entrypoint
- `scripts/eval_by_claude.py`: main evaluation entrypoint
- `scripts/eval.py`: older evaluation entrypoint (still used in some workflows)

---

## 2) Registry Contract

In `src/task_registry.py`, each task config requires:

- `name`: CLI task identifier
- `load_data(seed, split_type, **kwargs) -> (L_train, L_test)`
- `make_prompt(item, style, shots, **kwargs) -> PromptCompletion`
- `get_completion(item) -> str`
- `get_label(item) -> "yes" | "no"`

Optional fields:

- `get_indicator(item) -> 1/0` (default derived from label)
- `batch_size`: `{"with_ref": int, "without_ref": int}`
- `supports_negative_sampling` (metadata only right now)
- `filter_positive` (metadata only right now)
- `supports_split_types` (metadata only right now)

Important current behavior:

- `batch_size` and `get_indicator` are used by `scripts/ranking_loss_ref.py`.
- `supports_split_types`, `supports_negative_sampling`, and `filter_positive` are currently not enforced by train/eval scripts.
- Duplicate task names overwrite earlier registration silently (last one wins).

---

## 3) Where Existing Task Data Lives

Task families are mostly convention-driven:

- **Legacy hypernym base task**: `data/ranks.txt`
- **Hypernym per-hyponym CSVs**:
  - raw: `data/hypernym_<hyponym>_google-gemma-2-2b_{train|test}.csv`
  - fixed/v2: `data/fixed-hypernyms/hypernym_<hyponym>_google-gemma-2-2b_{train|test}-fixed.csv`
- **IFEval per-prompt**: `data/fixed-prompts-ifeval/gpt_ifeval_results_<prompt>.jsonl`
- **AmbigQA v1**: `data/ambigqa/with_negatives/` (`train.csv` + per-question CSVs)
- **PlausibleQA fixed**: `data/plausibleqa/fixed-plausibleqa/` (`train.csv`, `test/*.csv`, `train-per-question/*.csv`)
- **k-SAT**: `data/2sat_{train|test}.csv`, `data/3sat_{train|test}.csv`
- **CodeContests**: `data/codecontests/` (see below)

Representative task modules:

- `src/tasks/hypernym_hyponyms.py`
- `src/tasks/hypernym_concat.py`
- `src/tasks/hypernym_concat_subset_v2.py`
- `src/tasks/ifeval_per_prompt.py`
- `src/tasks/ifeval_concat.py`
- `src/tasks/ambigqa_v1.py`
- `src/tasks/plausibleqa_v0.py` (historical module name; it currently loads the newer cleaned data under `data/plausibleqa/fixed-plausibleqa/`)
- `src/tasks/ksat.py`
- `src/tasks/codecontests.py`

---

## 4) Adding a Brand-New Task (Recommended Flow)

### Step A: Pick the task name carefully

Name choice has downstream effects.

In eval scripts, these prefixes trigger family-specific behavior:

- `hypernym-...`
- `ifeval-...`
- `ambigqa...`
- `plausibleqa...`

If your new task is not one of these families, avoid these prefixes unless you also want those behaviors.

### Step B: Add your data files

Prefer a dedicated directory like `data/<your_task>/...`, and keep file naming stable.

Use paths derived from `__file__` in task modules (more robust than CWD-relative paths).

### Step C: Create a task module

Copy `src/tasks/example_task.py` to a new file (for example `src/tasks/my_task.py`) and implement:

- `load_data(...)`
- `make_prompt(...)`
- `get_completion(...)`
- `get_label(...)`

For item structure, use either:

- dicts (common in QA tasks), or
- namedtuples (common in hypernym/sat-style tasks)

Just stay consistent across all four functions.

### Step D: Register the task

At module scope:

```python
register_task({
    "name": "my-task",
    "load_data": load_data,
    "make_prompt": make_prompt,
    "get_completion": get_completion,
    "get_label": get_label,
    "batch_size": {"with_ref": 1, "without_ref": 4},
    "supports_split_types": ["random"],
})
```

### Step E: Import it in `src/tasks/__init__.py`

Add:

```python
from . import my_task
```

No import means no registration.

### Step F: Smoke-test registration

From repo root:

```bash
source /u/jdr/venvs/venv_lexcons/bin/activate
python - <<'PY'
import sys
sys.path.append("src")
import tasks
from task_registry import is_registered
print(is_registered("my-task"))
PY
```

### Step G: Smoke-test train/eval

Use the standard wrappers first:

- train wrapper: `scripts/run_train_semi.sh`
- eval wrapper: `scripts/run_eval_semi.sh`

These wrappers also set working directory/flags in a way that matches existing assumptions.

---

## 5) Adding Data to Existing Auto-Discovery Families

If your use case fits an existing family, you may not need new Python code.

### Hypernym per-hyponym tasks

Used by `src/tasks/hypernym_hyponyms.py`.

Add files matching:

- `data/hypernym_<hyponym>_google-gemma-2-2b_train.csv`
- `data/hypernym_<hyponym>_google-gemma-2-2b_test.csv`

This auto-registers `hypernym-<hyponym>`.

If using v2/fixed prompts, also maintain fixed files in `data/fixed-hypernyms/` with the `-fixed.csv` suffix.

### IFEval per-prompt tasks

Used by `src/tasks/ifeval_per_prompt.py`.

Add:

- `data/fixed-prompts-ifeval/gpt_ifeval_results_<prompt_name>.jsonl`

This auto-registers `ifeval-<prompt_name>`.

### AmbigQA v1 per-question tasks

Used by `src/tasks/ambigqa_v1.py`.

Add per-question CSVs in:

- `data/ambigqa/with_negatives/<slug>.csv`

`train.csv` is the shared train set; every other CSV becomes `ambigqa-<slug>`.

### PlausibleQA per-question tasks

Used by `src/tasks/plausibleqa_v0.py`.

Naming note: `plausibleqa_v0.py` is a legacy filename, but its active data paths point to the newer cleaned dataset in `data/plausibleqa/fixed-plausibleqa/`.

Add:

- test task files in `data/plausibleqa/fixed-plausibleqa/test/*.csv` -> `plausibleqa-<id>`
- optional train-eval task files in `.../train-per-question/*.csv` -> `plausibleqa-train-<id>`

### CodeContests tasks

Used by `src/tasks/codecontests.py`. Source: `deepmind/code_contests` on HuggingFace.

Data layout:

```
data/codecontests/
    descriptions.json           # {problem_name: {description, difficulty}} for all splits
    train.jsonl                 # compact train items (no description; joined at load time)
    split_manifest.json         # maps each slug -> "test" or "valid"
    test/<slug>.jsonl           # self-contained per-problem eval items
```

Training variants (sample different numbers of problems from train.jsonl):

- `codecontests` — ~400 problems, ~2000 items
- `codecontests-double` — ~800 problems, ~4000 items
- `codecontests-all` — all ~13K problems, ~70K items

Eval tasks (one per problem, auto-discovered from `test/*.jsonl`):

- **TEST split** (162 problems): `codecontests-1575a` .. `codecontests-1623e`
- **VALID split** (117 problems): `codecontests-1548c` .. `codecontests-1574f`

Each eval task's registry entry has an `origin_split` field ("test" or "valid").
To filter programmatically:

```python
from task_registry import TASK_REGISTRY
test_tasks = [name for name, cfg in TASK_REGISTRY.items()
              if cfg.get('origin_split') == 'test']
```

To regenerate the data files from scratch, run:

```bash
python scripts/preprocess_codecontests.py
```

---

## 6) Script Compatibility Matrix

### Fully registry-aware path (recommended)

- `scripts/ranking_loss_ref.py`
- `scripts/eval_by_claude.py`
- `scripts/eval.py`
- `src/utils.py` (`get_L_prompt`)
- `src/logitlens.py`

### Partially registry-aware or task-hardcoded scripts

Some older scripts still have hardcoded assumptions or output schemas. For new families, review before relying on them:

- `scripts/consistency_ft.py` (contains task-specific output formatting branches)
- many orchestration scripts under `scripts/` with hardcoded task lists

If your workflow depends on one of those scripts, grep for your task family name or hardcoded `TASKS=(...)`.

---

## 7) Important Gotchas

1. **Task naming affects eval behavior**  
   `eval_by_claude.py` and `eval.py` treat `hypernym-*`, `ifeval-*`, `ambigqa*`, and `plausibleqa*` specially for CSV writing and some prompt transforms.

2. **Neg-typicality is not generic**  
   In `eval_by_claude.py`, `--neg-typicality` is implemented via task-family-specific prompt negation in `make_negated_gen_prompt(...)`. New families require code there.

3. **`--save-scores-csv` schemas are family-specific**  
   New families may run metrics fine but not emit your desired detailed CSV format unless you add a branch.

4. **`ranking_loss_ref.py` uses limited split types**  
   CLI choices are currently `random|hyper|both`. Even if your task advertises more split types, the training CLI will not expose them unless you extend the parser.

5. **Some registry metadata is informational only**  
   `supports_split_types`, `supports_negative_sampling`, and `filter_positive` are not currently consumed in the main scripts.

6. **Large dynamic families expand parser choices**  
   `ranking_loss_ref.py` builds `--task` choices from all registered tasks. Families like plausibleqa can register hundreds of tasks, which increases startup verbosity.

7. **Working directory assumptions still exist in shared utils**  
   Some legacy loaders in `src/utils.py` use paths like `../data/...` relative to CWD. Wrapper scripts typically `cd scripts/` to avoid path issues.

8. **Known AmbigQA v0 path mismatch**  
   `src/tasks/ambigqa_v0.py` expects `data/ambigqa/v0/combined.csv`, while existing data is under `data/ambigqa/v0-combined-ambigqa-plausibleqa/`. As-is, v0 tasks may not register.

---

## 8) Practical Checklist

- [ ] Data files are in stable locations with deterministic names.
- [ ] Task module implements `load_data`, `make_prompt`, `get_completion`, `get_label`.
- [ ] `register_task(...)` is called with a unique `name`.
- [ ] Module is imported in `src/tasks/__init__.py`.
- [ ] Registration smoke test passes (`is_registered(...) == True`).
- [ ] Eval smoke test works with `scripts/run_eval_semi.sh`.
- [ ] Train smoke test works with `scripts/run_train_semi.sh`.
- [ ] If needed, added support in `eval_by_claude.py` for neg-typicality and detailed CSV output for your new family.
- [ ] If using hardcoded orchestration scripts, updated their task lists.

---

## 9) Suggested Minimal Template

Use this shape for new task items and prompts:

- item schema: plain dict with `question`, `answer`, `correct`
- generator completion: `" " + answer`
- label normalization: map to lowercase `yes`/`no` at load time

Keeping this pattern makes your task compatible with both rank training and eval tooling with minimal custom code.

## 10) After writing prompt templates and data loaders, LOAD THE DATA using our data/task loader (same ones used by the training and eval code) and print out a few examples of generator and validator!