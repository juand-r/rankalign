# GSM8K Dataset Build Log

How the GSM8K rankalign dataset was built (2026-05-01).

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

## Dataset Build

```bash
python scripts/dataset_builder/build_gsm8k_dataset.py --balance --max-per-side 30
```

- 739 qualified test problems (out of 1,319)
- `--balance`: downsample majority class per problem to 50/50
- `--max-per-side 30`: cap at 30 correct + 30 incorrect per problem (to keep dataset manageable given the 1,024 solutions available per problem)
- Output: `data/gsm8k/with_solutions/{full_response,truncated_response}/`

### Final dataset (v0 — test set only)

| Split | Problems | Total rows | Pos | Neg |
|---|---|---|---|---|
| Train | 0 | 0 | 0 | 0 |
| Test | 739 | 41,180 | 20,590 | 20,590 |

Each version (full_response, truncated_response) has 739 per-problem CSV files with ~56 rows each (balanced 30+30, though some problems have fewer than 30 on one side).

**Note:** This is v0 with test-set problems only. Train-set solutions would require either generating via OpenAI API (~150K calls for 7,473 problems × 20 samples) or finding a dataset with labeled Mistral/LLM generations on GSM8K train.

## Files

- `data/gsm8k/gsm8k_train_problems.jsonl` — 7,473 train problems (questions + gold answers, no solutions yet)
- `data/gsm8k/gsm8k_test_problems.jsonl` — 1,319 test problems (questions + gold answers)
- `data/gsm8k/rlhflow_mistral_solutions.jsonl` — 1,350,656 solutions from RLHFlow (test set)
- `data/gsm8k/with_solutions/full_response/` — 739 per-problem test CSVs (complete responses)
- `data/gsm8k/with_solutions/truncated_response/` �� 739 per-problem test CSVs (answer truncated)
- `scripts/dataset_builder/build_gsm8k_dataset.py` — dataset builder
- `scripts/dataset_builder/solution_generator/gsm8k_config.py` — generation config (for future train-set generation)

## What's Missing for a Full Dataset

1. **Train solutions.** The RLHFlow dataset only covers test. To get train solutions, either:
   - Generate via `generate_solutions.py gsm8k` using OpenAI API (~150K calls)
   - Find another dataset with labeled Mistral generations on GSM8K train
   - Use dart-math-pool for positives + generate negatives via intentional bug prompting

2. **Train/test OOD split.** Currently all 739 problems are from the GSM8K test set. A proper rankalign setup would use some problems for training and hold out others for evaluation (OOD by problem, like HumanEval's 80/40 split).
