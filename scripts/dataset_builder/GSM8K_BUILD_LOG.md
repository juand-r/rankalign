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
