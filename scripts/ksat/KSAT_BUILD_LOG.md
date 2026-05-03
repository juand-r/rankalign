# k-SAT Dataset Build Log

How the k-SAT rankalign datasets were built. v0 generated 2026-12-26; task
module modernised 2026-05-03 (see "Task Module Update" section below).

## Goal

Create satisfiability tasks for rankalign:

- **Generator:** given a CNF formula, produce a satisfying assignment.
- **Discriminator:** given (formula, assignment), output Yes/No on whether
  the assignment satisfies the formula.

## Data Source

Synthetic. Random k-CNF formulas + variable assignments labelled by direct
SAT evaluation (and by `pysat` when available). No external dataset
download — fully reproducible from `scripts/ksat/generate_ksat_data.py`.

## Build Script

[`scripts/ksat/generate_ksat_data.py`](generate_ksat_data.py) is the source
of truth. Key design choices:

- **Formula generation:** for each row, sample `n_clauses` random k-clauses
  uniformly over `n_vars` variables (`x_0..x_{n_vars-1}`). Each literal is
  randomly negated.
- **Auto-balanced clause count:** if `--n_clauses` isn't passed, the script
  computes `m ≈ log(0.5) / log(1 - 1/2^k)` so a uniformly random assignment
  satisfies the formula with ≈ 50% probability. For k=2 that's `m=2`
  clauses; for k=3 it's `m=5`.
- **Balanced labels:** by default the script generates 50% positive (the
  assignment satisfies the formula) and 50% negative (it doesn't) rows.
  - Positive rows: solve with `pysat`, return that assignment.
  - Negative rows: take the satisfying assignment and flip 1–3 bits, verify
    the result is unsatisfying.
- **Canonical satisfying assignment:** every row also stores a known
  satisfying assignment in the `satisfying_assignment` column (when one
  exists). The task module uses this as the **generator target** so the
  generator is always asked to produce a *correct* answer regardless of
  the row's `assignment`/`label` (which describes the discriminator's
  candidate, not necessarily the right answer).
- **Train/test no-overlap:** test generation is seeded differently and
  excludes any (formula, assignment) pair seen in training, so the splits
  are disjoint by construction.

## Generation commands used (v0 — 2026-12-26)

```bash
# 4-variable variants (default n_vars=10, but historically called with n_vars=4
# for the canonical files; see CSVs on disk for ground truth):
python scripts/ksat/generate_ksat_data.py --k 2 --n_vars 4 \
    --n_train 3000 --n_test 1000  --seed 42
python scripts/ksat/generate_ksat_data.py --k 3 --n_vars 4 \
    --n_train 3000 --n_test 1000  --seed 42

# 10-variable variants:
python scripts/ksat/generate_ksat_data.py --k 2 --n_vars 10 \
    --n_train 3000 --n_test 1000  --seed 42
python scripts/ksat/generate_ksat_data.py --k 3 --n_vars 10 \
    --n_train 3000 --n_test 1000  --seed 42
```

`scripts/ksat/generate_ksat_data.py` writes to `../data/{k}sat_{split}.csv`
relative to the script directory. The 4-var files are unsuffixed; the
10-var files have a `-10` filename suffix that's been added by-hand to
distinguish them from the 4-var files (the script itself just overwrites
based on `k`, so the convention is: rename to add `-10` after generating
the 10-var version, or generate to a different output dir and move).

## Resulting CSVs (currently on disk)

| File | Rows | k | n_vars | Pos/Neg balance |
|---|---|---|---|---|
| `data/2sat_train.csv`     | 3,000 | 2 | 4  | 50/50 |
| `data/2sat_test.csv`      | 1,000 | 2 | 4  | 50/50 |
| `data/2sat_train-10.csv`  | 3,000 | 2 | 10 | 50/50 |
| `data/2sat_test-10.csv`   | 1,000 | 2 | 10 | 50/50 |
| `data/3sat_train.csv`     | (not generated) |
| `data/3sat_test.csv`      | (not generated) |
| `data/3sat_train-10.csv`  | (not generated) |
| `data/3sat_test-10.csv`   | (not generated) |

CSV schema: `formula, assignment, label, clause1, clause2[, clause3 ...], satisfying_assignment`.
Assignments use `x0=0, x1=1, ...` format. Formulas use Unicode `∨` (OR), `∧`
(AND), `¬` (NOT).

## Task Module Update (2026-05-03)

The original `src/tasks/ksat.py` registered `2sat` and `3sat` but used the
pre-modern pattern: locally-defined `PromptCompletion`, `random.seed(...)`
global pollution, no `make_negated_prompt`/`csv_header`/`csv_row_builder`,
and registered `3sat` even though no `data/3sat_*.csv` files exist.

The 2026-05-03 rewrite:

1. **Conforms to the modern pattern** (mirrors `humaneval.py` /
   `codecontests.py` / `gsm8k.py`):
   - Imports `PromptCompletion`, `load_csv_items`, `normalize_yes_no`,
     `get_field` from `tasks.common`.
   - Uses local `random.Random(seed)` per load.
   - Adds `make_negated_prompt` ("Provide a variable assignment that does
     NOT satisfy this formula"), so `--neg-typicality` works.
   - Adds `csv_header` + `csv_row_builder`, so `--save-scores-csv` works.
   - Registers via the `_COMMON` dict pattern.
2. **Auto-skips variants whose CSVs are missing.** The four canonical
   variants (`2sat`, `2sat-10`, `3sat`, `3sat-10`) are listed in
   `TASK_VARIANTS`; only those whose train/test files exist on disk get
   registered. `3sat` and `3sat-10` will start registering automatically
   the moment the corresponding CSVs appear.
3. **Adds `2sat-10`** as a sibling task — the 10-variable CSVs already
   existed but were never wired up.
4. **Few-shot examples adapt to the row's `n_vars` automatically.** Both
   the (k, 4) and (k, 10) banks are kept in the module; `make_prompt`
   detects the variable count from the row's assignment string and picks
   the right bank. So the same module covers both var counts cleanly.

Resulting registered tasks (current state of disk):

```
2sat       3000 train, 1000 test, k=2, 4 vars
2sat-10    3000 train, 1000 test, k=2, 10 vars
```

## Files

- [`scripts/ksat/generate_ksat_data.py`](generate_ksat_data.py) — dataset
  builder.
- [`scripts/ksat/check_sat.py`](check_sat.py),
  [`scripts/ksat/generate_ksat.py`](generate_ksat.py),
  [`scripts/ksat/dspy_optimize_ksat.py`](dspy_optimize_ksat.py),
  [`scripts/ksat/prompt_search_ksat.py`](prompt_search_ksat.py) —
  exploratory tooling (standalone, not part of the rankalign pipeline).
- `data/{2,3}sat_{train,test}{,-10}.csv` — the generated CSVs.
- [`src/tasks/ksat.py`](../../src/tasks/ksat.py) — task module (modern
  pattern; auto-skips missing variants).

## What's Missing

- **3-SAT data.** Generate with `--k 3 --n_vars 4` and `--k 3 --n_vars 10`
  (above) when needed; the task module will auto-register both `3sat` and
  `3sat-10` once the CSVs land.
- **Larger var counts** (e.g. 20-var, 50-var). Trivially obtainable by
  rerunning the builder with `--n_vars N` and adding the file naming
  convention to the `TASK_VARIANTS` table in `src/tasks/ksat.py`.
- **Real-world SAT instances** (e.g. SATLIB benchmarks). Would need a
  separate ingestion path — the random-CNF generator above produces
  formulas that are uncorrelated with practical SAT-solving distributions.
