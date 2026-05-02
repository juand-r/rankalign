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
