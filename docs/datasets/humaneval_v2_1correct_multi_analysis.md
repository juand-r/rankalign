# humaneval-v2.1correct-multi — Build & Analysis Report

## What it is

A variant of `humaneval-v2.1` where every **correct** answer has been stylized
with a diverse set of code transforms. Wrong answers are byte-identical to v2.1.

The purpose: `v2.1correct-upper` (the previous variant) made all correct
answers atypical in one specific, obvious way — rename every local variable to
`UPPER_CASE`. That creates a near-perfect surface shortcut: any model that
notices "is this code shouted?" can predict the label without thinking about
typicality. `v2.1correct-multi` uses six different rename styles plus four
additional stacked transforms, assigned randomly per row, so there is no single
surface signal that separates the classes.

---

## Which transforms were included

All transforms were **selected empirically** by a canary run before building
the dataset (see *Canary* section below). A transform was only included if it
measurably lowered the model's per-token log-probability on the transformed
answer — i.e., it genuinely makes the answer more atypical, not just different.

### Rename schemes (one applied per correct row)

Each correct answer has all its local variable names renamed. Exactly one scheme
is applied per row — they are mutually exclusive.

| Scheme | Example: `result`, `count` become... |
|---|---|
| `upper` | `RESULT`, `COUNT` |
| `camel` | `theResult`, `theCount` |
| `verbose` | `the_result_value`, `the_count_value` |
| `cryptic` | `tc1`, `tc2` |
| `hungarian` | `any_result`, `int_count` |
| `numbered` | `result1`, `count1` |

Only local variables and function parameters are renamed. Built-ins, stdlib
names, and names starting with `_` are never touched. The rename set is derived
from the trusted AST scope analysis in `build_humaneval_v2_1_correct_upper.py`
and applied via libcst (which preserves comments, unlike `ast.unparse`).

### Composable transforms (stacked on top of the rename)

These four transforms are applied **in addition to** the rename scheme. Each
one is toggled independently per row with ~45% probability, so a row typically
gets 1–3 of them stacked together.

| Transform | What it does |
|---|---|
| `redundant_temp` | `return f(x)` → `_T0 = f(x)\nreturn _T0` — introduces a named intermediate variable |
| `dead_cruft` | Inserts `assert True` at the top of the function body — a syntactically valid no-op |
| `inject_comment` | Inserts an inline `# ...` comment — no semantic effect |
| `boolean_expand` | `return x > 0` → `if x > 0:\n    return True\nelse:\n    return False` — only applies when the return value is a simple comparison |

All four are **semantics-preserving by construction**, and every transformed
answer is re-validated by running the HumanEval test suite before being kept.

---

## Canary: how transforms were selected

Before building the full dataset, a small canary run measured the effect of
each transform on model log-probability. We sampled 36 correct answers from
v2.1 (across ≥10 tasks), applied each transform, and scored both the original
and the transformed version with gemma-4-31B-it. The metric is the **per-token
change in conditional log-probability** ΔlogP(y|x) — i.e., how much less
probable (per token) the transformed answer is compared to a libcst no-op
baseline.

**Keep rule:** mean ΔlogP/token ≤ −0.05 AND at least 70% of individual rows
negative AND revert rate < 10%.

The 70%-of-rows condition prevents a transform from passing because a few big
outliers drag the mean down while most rows are unaffected.

### Canary results (gemma-4-31B-it, 36 sampled correct rows)

| Transform | Rows scored | Mean ΔlogP/tok | % rows negative | Decision |
|---|---|---|---|---|
| `redundant_temp` | 24 | −0.471 | 100% | **KEEP** |
| `dead_cruft` | 36 | −0.324 | 100% | **KEEP** |
| `inject_comment` | 36 | −0.300 | 100% | **KEEP** |
| `boolean_expand` | 4 | −0.226 | 75% | **KEEP** (low-N: only 4/36 rows had a bare comparison return) |
| `cryptic` | 31 | −0.395 | 97% | **KEEP** |
| `numbered` | 31 | −0.416 | 100% | **KEEP** |
| `hungarian` | 31 | −0.294 | 90% | **KEEP** |
| `verbose` | 31 | −0.257 | 87% | **KEEP** |
| `upper` | 31 | −0.269 | 94% | **KEEP** |
| `camel` | 12 | −0.187 | 100% | **KEEP** (low-N: only 12/36 rows had renameable locals in camel style) |

All 10 transforms kept. Revert rate 0% for all. Negative control (a transform
that injects `raise AssertionError` — intentionally broken): 36/36 correctly
rejected by the HumanEval test suite.

*Full canary numbers: `docs/datasets/humaneval_v2_1correct_multi_canary_results.md`*

---

## How each row is assigned its transforms

Assignment is **deterministic and seeded** (seed=42) via a hash of
`(task_id, row_idx)`. This means re-running the builder always produces
identical output.

1. **Rename scheme:** one scheme is selected uniformly from the 6 kept schemes
   using the seeded hash.
2. **Composable transforms:** each of the 4 kept composables is independently
   included with probability 0.45, using the same hash (different salt per
   transform).
3. The resulting combination is applied via libcst and re-validated by running
   HumanEval. If validation fails, composables are dropped one at a time until
   it passes. If it still fails with rename-only, the original answer is kept
   unchanged.

---

## Full dataset stats

**Source:** `data/humaneval/v2.1` (82 task CSVs + train.csv)
**Output:** `data/humaneval/v2.1correct-multi/` (same file structure)

| | Count |
|---|---|
| Total rows | 4502 |
| Correct rows | 2385 |
| Wrong rows (unchanged) | 2117 |
| Reverts (see below) | 35 (1.5% of correct) |
| Transformed-correct failures | **0** |

### Rename scheme distribution (2385 correct rows)

| Scheme | Count | Share |
|---|---|---|
| upper | 416 | 17.4% |
| camel | 410 | 17.2% |
| numbered | 395 | 16.6% |
| verbose | 395 | 16.6% |
| cryptic | 387 | 16.2% |
| hungarian | 382 | 16.0% |

### Composable transform selection (independent, each ~45%)

| Transform | Times selected | Share of correct rows |
|---|---|---|
| dead_cruft | 1089 | 45.7% |
| inject_comment | 1084 | 45.5% |
| redundant_temp | 1063 | 44.6% |
| boolean_expand | 1060 | 44.4% |

### How many composables are stacked per correct row

| # composables on top of rename | Count | Share |
|---|---|---|
| 0 — rename scheme only | 235 | 9.9% |
| 1 | 700 | 29.4% |
| 2 | 848 | 35.6% |
| 3 | 508 | 21.3% |
| 4 — all composables stacked | 94 | 3.9% |

About 90% of correct rows receive at least one composable transform on top of
the rename. Most rows are a genuine combination of 2–3 transforms.

### Combination coverage

There are 6 rename schemes × 2⁴=16 possible composable subsets = **96 distinct
transformation recipes**. All 96 are present in the dataset. No combination
was missed. The most common single combination appears 50 times.

---

## Reverts and pre-existing validation failures

**35 correct rows "reverted"** (transform was partially or fully dropped):

- **34 rows:** the stored v2.1 answer fails strict `validate()` regardless of
  what we do to it — even the unmodified v2.1 original fails. These rows come
  from HumanEval problems whose prompt defines a helper function before the
  target (e.g. HumanEval/10 defines `is_palindrome` before `make_palindrome`).
  The model wrote a complete two-function solution, but the v2.1 answer storage
  format assumes a single-function completion; re-assembling a two-function
  answer by the standard "+4 indent" contract produces an `IndentationError`.
  The code is correct — the validator just cannot reconstruct it. These rows
  are kept byte-identical to v2.1 with label `correct: yes` unchanged.
  `v2.1correct-upper` handles these identically (`skip_orig_fail`), so the
  two datasets treat the same rows the same way.

- **1 row:** the full assigned transform produced an invalid answer, but
  dropping one composable yielded a valid (still-transformed) answer. Kept
  with the reduced transform.

**`transformed_correct_failures: 0`** — every row we successfully transformed
passes the HumanEval test suite. This is the key correctness guarantee.

The 34 pre-existing rows span 22 tasks:
`HumanEval/10, /106, /107, /127, /128, /139, /14, /149, /150, /153, /23, /24,
/30, /32, /38, /39, /5, /50, /55, /58, /63, /82`

---

## Reproducibility

```bash
cd private_projects/rankalign
/path/to/rankalign/.tools-venv/bin/python \
    scripts-more/correct_multi/build_v2_1_correct_multi.py \
    --menu ../../../../notes/log_P_diff_plots/humaneval-v2.1correct-multi/kept_menu.json \
    --seed 42
```

Dependencies: pandas 3.0.2, libcst 1.8.6 (in `.tools-venv`). The `kept_menu.json`
is the canary output file and is committed at
`notes/log_P_diff_plots/humaneval-v2.1correct-multi/kept_menu.json`.

The build is deterministic: same seed + same `kept_menu.json` always produces
identical output.

---

## File locations

| File | What it is |
|---|---|
| `data/humaneval/v2.1correct-multi/` | The dataset (82 task CSVs + train.csv) |
| `data/humaneval/v2.1correct-multi/_BUILD_REPORT.json` | Machine-readable build summary |
| `scripts-more/correct_multi/transforms.py` | All transform implementations |
| `scripts-more/correct_multi/build_v2_1_correct_multi.py` | The dataset builder |
| `scripts-more/correct_multi/run_full_canary.sh` | The canary pipeline (build variants → score → analyze) |
| `scripts-more/correct_multi/canary_score.py` | Scores variants with gemma-4-31B-it |
| `scripts-more/correct_multi/analyze_canary.py` | Computes ΔlogP keep/cut decision |
| `notes/…/humaneval-v2.1correct-multi/kept_menu.json` | Canary output: kept transforms (input to builder) |
| `notes/…/humaneval-v2.1correct-multi/CANARY_DELTA_LOGP.md` | Full per-transform canary results |
| `src/tasks/humaneval.py` | Task registration (`humaneval-v2.1correct-multi`) |
