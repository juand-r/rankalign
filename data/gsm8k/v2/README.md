# GSM8K v2

Built from the 4-model × 3-strategy generation pool (May 2026).

## Filters (applied in series)

- **length** — 60 <= len(answer) <= 3000
- **qualified_problems** — ≥4 pass ∧ ≥4 fail per cell in ≥10/12 cells per problem
- **oversample** — N=4 pass + N=4 fail per qualified cell per problem
- **llm_judge_truncation** — gpt-4.1-mini T=0 + 3 few-shot; drop rows judged truncated
- **gemma_logp** — google/gemma-2-9b-it log P(completion|prompt); drop rows with logP/tok < -2
- **trim** — per cell, keep up to N=3 pass + N=3 fail (random seed=142)

## Row counts

| Stage | Rows |
|---|---|
| sampled_oversample | 49,624 |
| judge_truncated_drop | 3,186 |
| no_score_match_drop | 0 |
| logp_below_threshold_drop | 886 |
| after_all_filters | 45,552 |
| after_trim_to_3 | 36,266 |
| test_rows | 6,480 |
| train_rows | 29,786 |

## Layout

- `test/`  — 100 per-problem CSVs (held-out)
- `train/` — 469 per-problem CSVs (used by sized variants)
- `manifest.json` — full provenance
- `train_problem_ids.json` — list of train problem ids

## Sized train variants (registered in src/tasks/gsm8k.py)

- `gsm8k-v2` — 28 train problems (~2k rows)
- `gsm8k-v2-double` — 56 train problems (~4k rows)
- `gsm8k-v2-all` — all 469 train problems (~30k rows)

- `gsm8k-v2-gsm8k_test_<N>` — per-test-problem eval task (100 of these)
