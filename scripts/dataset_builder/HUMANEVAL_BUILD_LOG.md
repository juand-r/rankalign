# HumanEval Dataset Build Log

How the HumanEval rankalign dataset was built (2026-05-01).

## Goal

Create a HumanEval-based rankalign task with positive (passing) and negative (plausible but failing) code solutions. Each problem needs >=10 correct and >=10 incorrect solutions for the discriminator to have signal.

## Source

[OpenAI HumanEval](https://github.com/openai/human-eval) — 164 Python function completion problems with unit tests.

## The Problem

HumanEval is too easy for strong LLMs. GPT-4o solves most problems correctly even at high temperature, so getting enough *incorrect* solutions is the main challenge.

## Generation Rounds

All solutions saved to `data/humaneval/solutions.jsonl` (append-only JSONL, 16,070 total solutions).

### Round 1: GPT-4o at varied temperatures

**Script:** `scripts/generate_humaneval_solutions.py`

- 40 solutions per problem (164 problems)
- Temperature schedule: 5×0.2, 5×0.6, 15×1.0, 10×1.2, 5×1.4
- Model: `gpt-4o`

**Result:** 6,560 solutions. 76% pass / 24% fail. Only **20/164** problems qualified. GPT-4o is too good — 67 problems had zero failures even at temp 1.4.

### Round 2: Weaker model at high temperature

**Script:** `scripts/generate_humaneval_supplemental.py`

- 40 additional solutions for 117 problems with <10 negatives (using `gpt-4o-mini`, temps 1.2-1.6)
- 20 additional solutions for 27 problems with <10 positives (using `gpt-4o`, temps 0.0-0.2)

**Result:** +5,220 solutions. Only bumped to **37 qualified**. GPT-4o-mini is also too good on simple HumanEval problems — 8.6% failure rate even at temp 1.6.

### Round 3: Intentional bug prompting with GPT-4o-mini

**Script:** `scripts/generate_humaneval_wrong.py`

System prompt: *"You are a Python programmer who makes subtle mistakes. Complete the given function, but introduce a subtle bug that would cause it to fail on some inputs. The bug should be plausible — the kind of mistake a real programmer might make (off-by-one errors, wrong comparison operators, missing edge cases, wrong variable, etc.)."*

- 20 solutions per problem for 101 remaining problems needing negatives
- Model: `gpt-4o-mini`, temps 0.6-1.2

**Result:** +2,020 solutions. 37.2% failure rate (vs 8.6% from normal sampling). **80 qualified.**

### Round 4: Intentional bug prompting with GPT-4o

Same "intentional bug" prompt but with `gpt-4o` (better at following the "write wrong code" instruction).

- 25 solutions for 58 remaining problems
- Model: `gpt-4o`, temps 0.6-1.2

**Result:** +1,450 solutions. **112 qualified.**

### Round 5: Final push with GPT-4o

- 30 solutions for 26 remaining problems
- Model: `gpt-4o`

**Result:** +780 solutions. **120 qualified.**

### Remaining 44 unqualified problems

- 18 too easy: even intentional bugs pass all tests (trivial problems like "return sum of list")
- 26 too hard: GPT-4o can't solve them (0-9 correct out of 60-100 attempts)

## Final Solution Breakdown

### All 16,070 solutions by strategy

| Strategy | Total | Pass | Fail | Fail% |
|---|---|---|---|---|
| gpt-4o / normal | 7,100 | 5,009 | 2,091 | 29.5% |
| gpt-4o-mini / normal | 4,720 | 4,314 | 406 | 8.6% |
| gpt-4o-mini / intentional_bug | 2,020 | 1,269 | 751 | 37.2% |
| gpt-4o / intentional_bug | 2,230 | 1,653 | 577 | 25.9% |

### In the final dataset (11,680 solutions across 120 qualified problems)

| Strategy | Total | Pass | Fail | Fail% |
|---|---|---|---|---|
| gpt-4o / normal | 4,820 | 4,217 | 603 | 12.5% |
| gpt-4o-mini / normal | 3,960 | 3,563 | 397 | 10.0% |
| gpt-4o-mini / intentional_bug | 1,660 | 923 | 737 | 44.4% |
| gpt-4o / intentional_bug | 1,240 | 728 | 512 | 41.3% |

### Source of failures in the dataset

- 33% from gpt-4o-mini intentional bugs
- 27% from gpt-4o normal sampling (naturally hard problems)
- 23% from gpt-4o intentional bugs
- 18% from gpt-4o-mini normal sampling

The "intentional bug" strategy contributed **55% of all failures** despite being only 25% of total solutions.

## Dataset Build

**Script:** `scripts/dataset_builder/build_humaneval_dataset.py`

```bash
python scripts/dataset_builder/build_humaneval_dataset.py \
    --input data/humaneval/solutions.jsonl \
    --train-count 80 --balance
```

- 120 qualified problems → random shuffle (seed=42) → 80 train + 40 test
- `--balance`: downsample positives to match negatives per problem (50/50)
- Output: `data/humaneval/with_solutions/`

### Final balanced dataset

| Split | Problems | Total rows | Pos | Neg | Per-problem (each side) |
|---|---|---|---|---|---|
| Train | 80 | 2,744 | 1,372 | 1,372 | min=10, median=15, max=39 |
| Test | 40 | 1,394 | 697 | 697 | min=10, median=18, max=39 |

## Key Lessons

1. **Strong models are too good for easy benchmarks.** Normal sampling from GPT-4o/4o-mini produces almost no failures on HumanEval. You need intentional bug prompting or much weaker models.
2. **"Write wrong code" prompting works.** 37-44% failure rate vs 8-12% from normal sampling. The model is good at introducing plausible bugs when asked.
3. **Some problems are unsolvable either way.** 18 problems are so trivial that even intentional bugs still pass. 26 are so hard that GPT-4o can't solve them. These get excluded.
4. **Multi-round targeted generation is necessary.** Each round only targets problems that still need more solutions on the deficient side, avoiding wasted API calls.

## Validation

The HumanEval validation runs each solution against the original unit tests in a subprocess with a 10-second timeout. A solution passes if and only if all test assertions succeed with exit code 0. No string matching or heuristics — it's ground truth from execution.

## Files

- `data/humaneval/solutions.jsonl` — 16,070 raw solutions (all rounds)
- `data/humaneval/with_solutions/train.csv` — 2,744 balanced training rows
- `data/humaneval/with_solutions/humaneval_*.csv` — 40 per-problem test CSVs
- `scripts/generate_humaneval_solutions.py` — round 1 generation (standalone, predates modular framework)
- `scripts/generate_humaneval_supplemental.py` — round 2 generation (standalone)
- `scripts/generate_humaneval_wrong.py` �� rounds 3-5 intentional bug generation (standalone)
- `scripts/dataset_builder/solution_generator/humaneval_config.py` — modular version (for future use)

---

# HumanEval v1 Build (2026-05-02)

## Goal

Recover all 164 HumanEval problems (v0 only had 120 qualified) by:
1. Generating solutions from diverse models (not just GPT-4o/4o-mini)
2. Importing external solutions from public repos
3. **Excluding `intentional_bug` strategy** from the final dataset build (filter at build time, not by mutating solutions.jsonl)

The `intentional_bug` exclusion is motivated by concern that bug-prompted solutions have a systematically different distribution from natural failures, which could make negative prompting trivially effective as a discriminator strategy.

## New Solution Sources

### External imports (`import_external_solutions.py`)

| Source | Models | Solutions | Notes |
|---|---|---|---|
| jamesmurdza/humaneval-results | CodeLlama-34b-Instruct, gpt-3.5-turbo, gpt-4 | 10 runs each per problem, markdown format | Validated against our unit tests |
| breath24/FailureBench | Claude Sonnet-4, DeepSeek-V3, GPT-4o, Llama-3.3-70B, Mistral-3.2-24B, Qwen3-Coder | 1 solution each per problem, JSON format | Validated against our unit tests |

### API generation (hard problems needing passes)

| Model | Strategy | Targets | Notes |
|---|---|---|---|
| gpt-4.1 | normal | 26 hard problems | Good for mid-difficulty |
| gpt-5 | normal | Hard problems | temp=1.0 only (API limitation) |
| gpt-5.5 | normal | Hardest problems (HumanEval/129, /132) | Solved problems no other model could |

### vLLM generation on spark (easy problems needing failures)

Used `generate_solutions_parallel.py` with 20-30 concurrent workers for ~20x speedup over serial requests on the NVIDIA GB10 GPU.

| Model | Strategy | Targets | Fail rate | Notes |
|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | normal | 48 easy problems | ~40% | Bulk of natural failures |
| deepseek-ai/deepseek-coder-1.3b-instruct | normal | 5 remaining easy | 32% | |
| microsoft/Phi-3-mini-4k-instruct | normal | 5 remaining easy | 20% | |
| mistralai/Mistral-7B-Instruct-v0.3 | normal | 5 remaining easy | 30% | |
| allenai/OLMo-2-0425-1B-Instruct | normal | 5 remaining easy | 46% | |

### API generation (easy problems, additional diversity)

| Model | Strategy | Targets | Notes |
|---|---|---|---|
| gpt-3.5-turbo | normal | 48 easy problems + 5 remaining | Low fail rate (~3.5%) on easy problems |

## Excluded Problems

Two problems are excluded from v1 for being at the extremes of difficulty:

- **HumanEval/53** (`add(x, y)` — return x + y): Too trivial. Only 3 failures out of ~226 attempts across all models. No model fails this reliably because the solution is a single expression.
- **HumanEval/145** (`order_by_points` — sort by digit sum with tricky negative handling): Too hard. Only 2 passes out of ~426 attempts across all models including GPT-5.5. The canonical solution uses a non-obvious rule where only the first digit of a negative number is negated in the digit sum (e.g., -12 → -1+2 = 1).

## Prompt Strategies

All prompt strategies are defined in `scripts/dataset_builder/solution_generator/prompt_strategies.json`. Each strategy maps to a `system_prompt` and `user_prompt` template. The `strategy` field in `solutions.jsonl` records which prompt was used.

| Strategy | Intent | System prompt summary |
|---|---|---|
| `normal` | Standard expert completion | "You are an expert Python programmer" |
| `intentional_bug` | Deliberately introduce subtle bugs (excluded from v1) | "You are a programmer who makes subtle mistakes" |
| `beginner` | Beginner-style code, non-Pythonic | "You are a beginning Python programmer" |
| `unusual` | Creative, non-obvious approaches | "You are a maverick, artistic and creative genius" |
| `refactorable` | Quick-and-dirty, needs cleanup | "You write code quickly but not always elegantly" |
| `different-style` | Deliberately different from usual | "You are bored of leetcode exercises" |
| `bad-style` | C/assembly/Haskell idioms in Python, PEP-8 violations | "Experienced in assembly, C, haskell; terrible Python style" |

## v1 Build Rules

1. Exclude all `strategy=intentional_bug` solutions
2. Exclude HumanEval/53 and HumanEval/145
3. Require >=10 pass AND >=10 fail per problem
4. **Qualified: 162/164 problems**

All solutions remain in `data/humaneval/solutions.jsonl` (append-only, never mutated). Filtering happens at build time only.

### Models contributing to v1 (18 models)

API: gpt-3.5-turbo, gpt-4, gpt-4o, gpt-4o-mini, gpt-4.1, gpt-5, gpt-5.5
Open (vLLM on spark): Llama-3.1-8B-Instruct, deepseek-coder-1.3b-instruct, Phi-3-mini-4k-instruct, Mistral-7B-Instruct-v0.3, OLMo-2-0425-1B-Instruct
External: CodeLlama-34b-Instruct, Claude Sonnet-4, DeepSeek-V3, Llama-3.3-70B, Mistral-3.2-24B, Qwen3-Coder

## Key Lessons (v1)

1. **Model diversity beats brute force.** Small/weak models (1-8B) naturally produce ~30-50% failure rates on easy problems where GPT-4o fails <1%. A mix of models gives both natural passes and natural failures.
2. **Parallelizing vLLM requests is critical.** Serial requests to vLLM on the GB10: 16s/req. With 30 concurrent workers: 0.81s/req (20x speedup).
3. **Some problems are at the extremes.** `add(x, y)` is too trivial for any model to fail; `order_by_points` is too hard for any model to solve. Accept 162/164.
4. **Intentional bug prompting is powerful but risky.** It produces 37-44% failure rates vs 3-12% from normal sampling, but the failure distribution may be systematically different. Excluded from v1 to keep the dataset "natural."

## v1 Dataset Build

### Pipeline

1. **Merge & deduplicate** — combines local + spark solutions, re-cleans with updated `clean_solution()`, removes empty solutions:
   ```bash
   .tools-venv/bin/python scripts/dataset_builder/reclean_and_merge.py
   ```
   Output: `data/humaneval/solutions_merged.jsonl` (66,946 solutions after dedup + empty removal)

2. **Stratified sampling** — splits problems, samples 30 per problem balanced 50/50 pass/fail with model/strategy diversity:
   ```bash
   .tools-venv/bin/python scripts/dataset_builder/build_humaneval_v1.py \
       --input data/humaneval/solutions_merged.jsonl \
       --output-dir data/humaneval/v1 \
       --samples-per-problem 30 \
       --seed 42
   ```
   Output: `data/humaneval/v1/train.csv` (2,400 rows) + 82 `humaneval_N.csv` test files (2,460 rows)

### Sampling details

- **Seed**: 42 (deterministic — same input produces same output)
- **Split**: 80 train problems, 82 test problems (random shuffle with seed)
- **Per problem**: 15 pass + 15 fail target, round-robin across (model, strategy) groups
- **Strategies**: 7 (normal, beginner, unusual, refactorable, different-style, bad-style, external)
- **Models**: 19
- **Filters**: excludes `intentional_bug`, HumanEval/53, HumanEval/145, empty solutions

### CSV columns

`question, answer, correct, strategy, model, temperature, task_id, error`

- `question`: function signature + docstring (the prompt)
- `answer`: cleaned function body (indented, ready to append to signature)
- `correct`: Yes/No (passed unit tests)
- `strategy`: prompt strategy used for generation
- `model`: model that generated the solution
- `temperature`: sampling temperature (blank for `external` strategy)
- `task_id`: HumanEval problem ID (e.g., HumanEval/42)
- `error`: error message for failing solutions (blank for passing)

### Strategy note: `external`

Solutions from external datasets (breath24/FailureBench, jamesmurdza/humaneval-results) where we don't know the exact prompt or temperature used. Marked `strategy=external` with blank temperature.

## Files (v1)

- `data/humaneval/solutions_merged.jsonl` — deduplicated merged pool (66,946 solutions)
- `data/humaneval/solutions.jsonl` — local raw generations (all rounds)
- `data/humaneval/solutions_spark_raw.jsonl` — spark raw generations
- `data/humaneval/problems.jsonl` — 164 HumanEval problems with unit tests
- `data/humaneval/v1/train.csv` — v1 training set (2,400 rows, 80 problems)
- `data/humaneval/v1/humaneval_*.csv` — v1 test sets (82 files, 2,460 rows total)
- `scripts/dataset_builder/reclean_and_merge.py` — merge + reclean + dedup + empty filter
- `scripts/dataset_builder/build_humaneval_v1.py` — stratified sampling + CSV output
- `scripts/dataset_builder/generate_solutions_parallel.py` — parallel vLLM generator
- `scripts/dataset_builder/generate_solutions.py` — serial generator (API + vLLM)
- `scripts/dataset_builder/import_external_solutions.py` — external repo importer
- `scripts/dataset_builder/run_multi_model_cycle.sh` — model cycling script for spark
- `scripts/dataset_builder/solution_generator/prompt_strategies.json` — all prompt templates
- `scripts/dataset_builder/solution_generator/humaneval_config.py` — clean_solution + validation
