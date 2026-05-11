# GSM8K v1.1

Filtered + relabeled version of v1. **v1 is unchanged.**

## Filters applied (in series)

- **relabel_dual_metric** — extract strict (`#### N` or `The answer is N.`) and flexible (lm-eval-harness alternation); correct = flexible_correct
- **length** — keep rows with 60 <= len(answer) <= 3000
- **llm_judge_truncation** — gpt-4.1-mini T=0 + 3 few-shot; drop rows judged truncated
- **gemma_logp** — google/gemma-2-9b-it scoring of log P(completion|prompt); drop rows with log_prob_per_token < -2.0

## Row counts

| Stage | Rows |
|---|---|
| v1_in | 5,328 |
| after_relabel | 5,328 |
| after_length | 4,823 |
| after_judge | 4,522 |
| after_logp_join_matched | 4,522 |
| after_logp_join_unmatched | 0 |
| after_logp_filter | 4,424 |

**Problems retained:** 111 (of 111 v1 problems)

## Schema (extends v1)

v1 columns kept: question, answer, correct, strategy, model, temperature, task_id, error

v1.1 additions:
- `old_correct` — original v1 label (string Yes/No)
- `correct` — UPDATED to flexible_correct
- `strict_extracted`, `flexible_extracted` — extracted answer strings
- `strict_correct`, `flexible_correct` — boolean
- `len_chars` — `len(answer)`
- `judge_verdict`, `judge_truncated` — gpt-4.1-mini result
- `log_prob`, `num_completion_tokens`, `log_prob_per_token` — gemma-2-9b-it scoring
- `gold_answer` — from gsm8k_test_problems.jsonl

**Build date:** 2026-05-11. **Builder:** notes/gsm8k-v1.1/build_v1_1_step*.py
