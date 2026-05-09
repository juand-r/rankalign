# GSM8K Dataset Build Log

How the GSM8K rankalign dataset was built. v0 written 2026-05-01 (test-only,
no train/test split). v1 written 2026-05-02 with the single-pool 100/639 split
and paired full/truncated solutions (see "Dataset Build (v1, 2026-05-02)" below).

## Goal

Create a GSM8K-based rankalign task with positive (correct) and negative (incorrect) math solutions. Two versions: one with the full response including the final numerical answer, one with the response truncated just before the answer is revealed.

## Source

[OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k) — 7,473 train + 1,319 test grade-school math problems. Each problem has a gold step-by-step solution ending with `#### <answer>`.

## Existing Datasets Investigated

We searched HuggingFace for datasets with multiple labeled (correct/incorrect) solutions per GSM8K problem:

| Dataset | Problems | Solutions/problem | Has negatives? | Useful? |
|---|---|---|---|---|
| [RLHFlow/Mistral-GSM8K-Test](https://huggingface.co/datasets/RLHFlow/Mistral-GSM8K-Test) | 1,319 (test) | 1,024 | Yes (binary labels) | **Yes — used** |
| [hkust-nlp/dart-math-pool-gsm8k](https://huggingface.co/datasets/hkust-nlp/dart-math-pool-gsm8k) | 7,468 (train) | ~263 median | No (100% correct, pre-filtered) | No |
| [peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd) | Mixed GSM8K+MATH | 1 per row | Step-level +/- labels | Not directly |
| [Randolphzeng/Mr-GSM8K](https://huggingface.co/datasets/Randolphzeng/Mr-GSM8K) | 3,000 | 1 | Yes (error detection) | Too few per problem |
| [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | ~395K rows | Augmented | No (all correct) | No |

**dart-math-pool-gsm8k** looked promising (2.7M rows, train problems) but turned out to be 100% correct solutions — pre-filtered for correctness. Useless for negatives.

**RLHFlow/Mistral-GSM8K-Test** was the winner: 1,319 test problems × 1,024 Mistral-7B generations each, with binary correctness labels from answer matching.

## Data Source: RLHFlow/Mistral-GSM8K-Test

- **Model:** Mistral-7B variant (from the RLHFlow RLHF reward modeling project)
- **Format:** Step-by-step solutions with `ки` (Cyrillic U+043A U+0438) step boundary markers between steps, ending with `The answer is: <number> ки`
- **Correctness labels:** Binary (1=correct, 0=incorrect), determined by extracting the final number and comparing to the GSM8K gold answer
- **Coverage:** GSM8K test set only (1,319 problems). No train set coverage.

### Raw stats

```
Total solutions: 1,350,656
Total questions: 1,319
Correct: 1,046,657 (77.5%)
Incorrect: 303,999 (22.5%)

Per-question correct:   min=0, median=989, max=1,024
Per-question incorrect: min=0, median=35, max=1,024

Qualified (>=10 correct AND >=10 incorrect): 739/1,319
```

580 problems don't qualify — mostly because they're easy (Mistral gets nearly all 1,024 attempts right, <10 failures).

### Nature of the incorrect solutions

The errors are **naturally occurring model mistakes**, not intentionally generated. Examples of failure modes:

- **Arithmetic errors:** `204 + 160 + 330 = 794` instead of `694`
- **Reasoning errors:** Adding distances instead of subtracting when driving back home (got 315 instead of 45)
- **Misreading the problem:** Forgetting to count a step (download before restart, got 120 instead of 160)
- **Wrong operations:** Multiplying where division was needed, or vice versa

These are more realistic than intentionally-prompted bugs since they reflect actual model failure modes.

## Download and Conversion

```python
# Downloaded via HuggingFace datasets library
ds = load_dataset('RLHFlow/Mistral-GSM8K-Test', split='train')
gsm = load_dataset('openai/gsm8k', 'main', split='test')  # for gold answers
```

Converted to standard JSONL format at `data/gsm8k/rlhflow_mistral_solutions.jsonl`:

```json
{
    "question_id": "gsm8k_test_0",
    "question": "Janet's ducks lay 16 eggs per day...",
    "response": " Step 1: Janet's ducks lay 16 eggs... The answer is: 18 ки",
    "correct": true,
    "gold_answer": "18",
    "source": "RLHFlow/Mistral-GSM8K-Test",
    "model": "mistral"
}
```

## Two Response Versions

### full_response

The complete model output including the final numerical answer:

```
Step 1: Janet's ducks lay 16 eggs per day. ки
Step 2: She eats 3 eggs for breakfast and bakes muffins with 4 eggs, so she uses 3 + 4 = 7 eggs. ки
Step 3: The remaining eggs are 16 - 7 = 9 eggs. ки
Step 4: She sells each egg for $2, so she makes 9 * $2 = $18. The answer is: 18 ки
```

### truncated_response

Cut off at "The answer is" — the reasoning chain is intact but the final number is removed:

```
Step 1: Janet's ducks lay 16 eggs per day. ки
Step 2: She eats 3 eggs for breakfast and bakes muffins with 4 eggs, so she uses 3 + 4 = 7 eggs. ки
Step 3: The remaining eggs are 16 - 7 = 9 eggs. ки
Step 4: She sells each egg for $2, so she makes 9 * $2 = $18. The answer is
```

The truncation regex handles multiple ending patterns:
- `The answer is: 18 ки` → `The answer is`
- `The final answer is 42.` → `The final answer is`
- `#### 18` → everything before `####`
- `\boxed{42}` → everything before `\boxed`
- Bare trailing number on last line → remove that line

**Script:** `scripts/dataset_builder/build_gsm8k_dataset.py`

## Dataset Build (v1, 2026-05-02)

Two-step pipeline:

```bash
# Step 1: strip ки PRM step markers (one-time, writes a new clean JSONL)
python scripts/dataset_builder/strip_step_markers.py \
    data/gsm8k/rlhflow_mistral_solutions.jsonl \
    data/gsm8k/rlhflow_mistral_solutions_clean.jsonl

# Step 2: build the per-version CSVs (single-pool train/test split)
python scripts/dataset_builder/build_gsm8k_dataset.py \
    --input data/gsm8k/rlhflow_mistral_solutions_clean.jsonl \
    --output-dir data/gsm8k/with_solutions \
    --balance --max-per-side 30 \
    --num-test-problems 100 --split-seed 42 --seed 42
```

Notes on the build:

- **`--balance --max-per-side 30`**: balanced 30 correct + 30 incorrect per problem. Some problems have fewer than 30 on one side, so per-problem rows range from ~24 up to 60 (mean ≈ 56).
- **`--num-test-problems 100 --split-seed 42`** (new flag): the source dataset only labels GSM8K *test* problems, so the legacy "classify by question_id substring" mode would put every problem in the test pile and leave train empty. The new flag pools all qualified problems together, then deterministically holds out 100 of them for per-problem eval CSVs and writes the remaining 639 into a shared `train.csv`. Use `--split-seed` to control which 100 are held out.
- **Paired full/truncated versions**: solutions are selected **once per question** (with a per-question deterministic RNG seeded by `args.seed + hash(qid)`) and then both `full_response` and `truncated_response` are written from the same selection. So for each (question, row index) the truncated answer is exactly the prefix of the full answer up through "The answer is" — not an independent re-sample. (Earlier versions of `build_gsm8k_dataset.py` shuffled inside the version loop and produced different traces for the two versions.)

### Final dataset (v1 — single-pool 100/639 holdout)

| Split | Problems | Total rows | Pos | Neg |
|---|---|---|---|---|
| Train | 639 | 35,720 | 17,860 | 17,860 |
| Test | 100 | 5,460 | 2,730 | 2,730 |

Per problem: 60 rows where both pos and neg pools have ≥30 samples; fewer when a side is short.

Outputs:

```
data/gsm8k/with_solutions/full_response/
    train.csv                    # 35,720 rows, 639 questions
    gsm8k_test_<N>.csv           # 100 per-problem eval CSVs (~60 rows each)

data/gsm8k/with_solutions/truncated_response/
    train.csv                    # paired with full's train.csv (same selections)
    gsm8k_test_<N>.csv           # paired with full's per-problem CSVs
```

## Task Registration

Wired up in `src/tasks/gsm8k.py` following the modern pattern (mirrors
`humaneval.py` / `codecontests.py`). Two parallel families:

| Train task | Problems sampled at load | Approx rows |
|---|---|---|
| `gsm8k-full` | 45 of 639 | ~2,500 |
| `gsm8k-full-double` | 90 of 639 | ~5,000 |
| `gsm8k-full-all` | all 639 | 35,720 |
| `gsm8k-truncated` | 45 of 639 (same problems as `gsm8k-full`) | ~2,500 |
| `gsm8k-truncated-double` | 90 of 639 | ~5,000 |
| `gsm8k-truncated-all` | all 639 | 35,720 |

Sampling for the small / double variants uses `random.Random(SAMPLE_SEED)`
with `SAMPLE_SEED = 42` so the chosen subsets are deterministic and identical
across train families. Train sizes match the existing modern families
(`humaneval` 2,744 / `codecontests` 2,522 / `ifeval-concat` 3,160).

Per-problem eval tasks auto-discovered from the per-problem CSVs:

```
gsm8k-full-gsm8k_test_<N>
gsm8k-truncated-gsm8k_test_<N>
```

(100 tasks per family, paired across families — same 100 question_ids in
both `gsm8k-full-*` and `gsm8k-truncated-*`.)

The task module also registers `make_negated_prompt`, `csv_header`, and
`csv_row_builder` so `--neg-typicality` and `--save-scores-csv` work
without any edits to `eval_by_claude.py`.

## What's Missing for a Full Dataset

1. **Train solutions.** The RLHFlow dataset only covers GSM8K test problems,
   so the 639 "train" problems above are a sub-split of the same source pool
   (different problems from the same model, not problems from GSM8K-train).
   To get true GSM8K-train coverage either:
   - Generate via `generate_solutions.py gsm8k` using OpenAI API (~150K calls)
   - Find another labeled dataset on GSM8K train
   - Use dart-math-pool for positives + intentionally generate negatives

## Files

- `data/gsm8k/rlhflow_mistral_solutions.jsonl` — 1,350,656 solutions from RLHFlow (LFS-tracked, ~1.2 GB)
- `data/gsm8k/rlhflow_mistral_solutions_clean.jsonl` — same JSONL with `ки` step markers stripped (regenerated locally; not committed)
- `data/gsm8k/with_solutions/full_response/{train.csv, gsm8k_test_<N>.csv}` — complete responses
- `data/gsm8k/with_solutions/truncated_response/{train.csv, gsm8k_test_<N>.csv}` — same solutions, answer line truncated
- `scripts/dataset_builder/build_gsm8k_dataset.py` — dataset builder (supports `--num-test-problems`)
- `scripts/dataset_builder/strip_step_markers.py` — pre-processing: strip `ки` PRM markers
- `scripts/dataset_builder/solution_generator/gsm8k_config.py` — generation config (for future train-set generation)
- `src/tasks/gsm8k.py` — task module (registers `gsm8k-full[-double|-all]`, `gsm8k-truncated[-double|-all]`, plus 100 per-problem eval tasks per family)

---

# GSM8K v1 build, parallel vLLM + multi-strategy (started 2026-05-03)

A second v1 pipeline, built to mirror the **HumanEval v1** pattern (parallel
generation of fresh solutions across multiple models and prompt strategies)
rather than reusing RLHFlow's pre-generated Mistral solutions. Output dir is
`data/gsm8k/v1/` so it does not collide with the RLHFlow-based v1 above (which
lives at `data/gsm8k/with_solutions/`). Both v1s coexist; this section
describes the new one.

## Source: GSM8K canonical *test* split only (1,319 problems)

The full problem pool for this pipeline is `data/gsm8k/gsm8k_test_problems.jsonl`
— a JSONL dump of `load_dataset('openai/gsm8k', 'main', split='test')`,
reformatted as one record per line with fields `{question_id, question,
gold_solution, gold_answer}`. 1,319 problems, ~830 KB.

The `build_gsm8k_v1.py` builder partitions these 1,319 problems into
**train + held-out test** for the rankalign experiment (default
`--n-test 100` → 1,219 train + 100 test). Per-problem eval CSVs in
`data/gsm8k/v1/` are named `gsm8k_test_<N>.csv` to preserve the canonical
GSM8K task_id (the `_test_` substring refers to GSM8K's split, not to the
rankalign train/test partition).

### Why test-only (vs. GSM8K's 7,473-problem train split)?

1. **Contamination risk.** GSM8K's official train split has very likely been
   seen during the underlying LLMs' pretraining or instruction-tuning data.
   Using only the test split keeps the problems "fresh" w.r.t. the LLMs whose
   typicality / discriminator behavior we are studying.
2. **Sufficiency.** 1,319 problems is plenty for the rankalign discriminator/
   generator setup — partitioning into 1,219 train + 100 held-out test gives
   the same "lots of train problems, modest OOD test set" shape that
   humaneval-v1 has (164 problems → 80 train + 82 test).
3. **Symmetry with humaneval-v1.** The two v1 pipelines now have parallel
   contracts: both consume a fixed canonical problem set, both partition
   into train/test at build time, both emit per-problem CSVs.

This is a deliberate departure from the RLHFlow-based v1 above (which used
RLHFlow's solutions for the same 1,319 test problems and split them
100/639). The new pipeline produces *its own* solutions via vLLM/API.

## Pipeline

```
                                    ┌────────────────────────┐
data/gsm8k/gsm8k_test_problems.jsonl│  1,319 GSM8K problems  │
                                    └────────────┬───────────┘
                                                 │
            generate_solutions_parallel_gsm8k.py │  parallel vLLM/API,
                                                 │  multi-strategy prompts
                                                 ▼
                                    ┌────────────────────────┐
                                    │ data/gsm8k/v1/         │
                                    │   solutions.jsonl      │  append-only pool
                                    └────────────┬───────────┘
                                                 │
            reclean_and_merge_gsm8k.py           │  re-clean, re-validate,
                                                 │  filter empty, dedup
                                                 ▼
                                    ┌────────────────────────┐
                                    │ data/gsm8k/v1/         │
                                    │   solutions_merged.jsonl│
                                    └────────────┬───────────┘
                                                 │
            build_gsm8k_v1.py                    │  stratified per-problem
                                                 │  sampling, train/test split
                                                 ▼
                                    ┌────────────────────────┐
                                    │ data/gsm8k/v1/         │
                                    │   train.csv            │
                                    │   gsm8k_test_<N>.csv   │  (one per held-out
                                    │   …                    │   test problem)
                                    └────────────────────────┘
```

### Step 1 — generate

```bash
.tools-venv/bin/python scripts/dataset_builder/generate_solutions_parallel_gsm8k.py \
    --problems data/gsm8k/gsm8k_test_problems.jsonl \
    --output data/gsm8k/v1/solutions.jsonl \
    --model <vllm-or-openai-model> \
    --strategy <name-from-gsm8k_strategies.json> \
    [--filter-ids <id_csv>] [--target-pass-fail N]
```

`gsm8k_strategies.json` (parallel to humaneval's `prompt_strategies.json`)
defines the named prompt strategies. New strategies can be added without
touching code; `gsm8k_config.py` automatically picks them up via
`_load_gsm8k_strategies()`.

### Step 2 — reclean + dedup

```bash
.tools-venv/bin/python scripts/dataset_builder/reclean_and_merge_gsm8k.py
```

Reads `data/gsm8k/v1/solutions.jsonl`, re-cleans with the latest
`clean_solution()`, re-validates rows whose cleaned form changed, removes
empties, dedups, writes `data/gsm8k/v1/solutions_merged.jsonl`.

### Step 3 — stratified build

```bash
.tools-venv/bin/python scripts/dataset_builder/build_gsm8k_v1.py \
    --input data/gsm8k/v1/solutions_merged.jsonl \
    --output-dir data/gsm8k/v1 \
    --samples-per-problem 30 \
    --n-test 100 --seed 42
```

Filters out `intentional_bug` rows and any task_ids in the
`EXCLUDED` set, requires ≥10 pass and ≥10 fail per problem, splits the
qualified problems into train + held-out test, and samples `--samples-per-problem`
rows per problem balanced 50/50 pass/fail across (model, strategy) groups
via round-robin.

## Status as of 2026-05-04 (Mac death) and 2026-05-09 recovery

- **`data/gsm8k/gsm8k_test_problems.jsonl`** — committed (this PR / commit).
  1,319 records.
- **`data/gsm8k/v1/solutions.jsonl`** — 328 MB, 133,525 rows, **on RunPod's
  `general-eval` volume only** (pod id `tskvcwypxspgku`). Not pushed to git
  because of GitHub's 100 MB hard limit; also still being added to. Local
  copy at `~/Downloads/gsm9k-solutions-tmp/solutions.jsonl` on the user's
  laptop for safekeeping.
- **`data/gsm8k/v1/solutions_merged.jsonl`** — not yet produced (the merge
  step had not been run before the Mac died).
- **`data/gsm8k/v1/{train,gsm8k_test_<N>}.csv`** — present on mll
  (`/datastor1/jdr/gv-gap/rankalign/data/gsm8k/v1/`), built 2026-05-04 from
  a predecessor solutions.jsonl. Not yet committed.

## Files (this pipeline only)

- `data/gsm8k/gsm8k_test_problems.jsonl` — canonical 1,319-problem pool
- `scripts/dataset_builder/generate_solutions_parallel_gsm8k.py` — parallel
  vLLM/API generator with per-strategy prompts
- `scripts/dataset_builder/reclean_and_merge_gsm8k.py` — reclean + dedup
- `scripts/dataset_builder/build_gsm8k_v1.py` — stratified per-problem
  sampler / writer
- `scripts/dataset_builder/solution_generator/gsm8k_config.py` — task
  config; loads `gsm8k_strategies.json` for multi-strategy prompting
- `scripts/dataset_builder/solution_generator/gsm8k_strategies.json` —
  named prompt strategies (parallel to `prompt_strategies.json` for
  humaneval)

