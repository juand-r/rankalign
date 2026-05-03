# HumanEval v1 — Solution Pool Analysis

Generated: 2026-05-03

## Pool Size

- **67,499 total solutions** (after merge, dedup, re-clean)
- 65,059 non-intentional-bug, non-excluded
- 162 problems (excluding HumanEval/53 and HumanEval/145)
- 19 models, 6 strategies

## Models (19)

| Model | Solutions | Pass Rate |
|-------|-----------|-----------|
| Claude Sonnet-4 | 162 | 89.5% |
| Qwen3-Coder | 162 | 87.0% |
| gpt-5.5 | 631 | 90.0% |
| gpt-4 | 653 | 78.3% |
| gpt-4o-mini | 7,549 | 73.0% |
| DeepSeek-V3 | 162 | 67.3% |
| Mistral-3.2-24B | 162 | 67.3% |
| gpt-3.5-turbo | 1,175 | 67.6% |
| gpt-4o | 3,240 | 67.0% |
| gpt-5 | 2,403 | 64.3% |
| gpt-4.1 | 1,158 | 60.3% |
| GPT-4o / Llama-3.3-70B | 162 each | 51.9% |
| deepseek-coder-1.3b | 10,115 | 46.3% |
| Phi-3-mini-4k | 9,185 | 35.2% |
| CodeLlama-34b | 502 | 34.3% |
| Llama-3.1-8B | 9,945 | 31.9% |
| Mistral-7B | 8,860 | 19.8% |
| OLMo-2-1B | 8,671 | 7.1% |

## Strategies (6)

| Strategy | Solutions | Fail Rate |
|----------|-----------|-----------|
| normal | 19,713 | 44.2% |
| beginner | 9,553 | 60.1% |
| refactorable | 8,605 | 66.1% |
| bad-style | 9,175 | 67.5% |
| different-style | 8,694 | 68.5% |
| unusual | 9,319 | 71.6% |

## Quality Checks

- **All 162 problems qualify** (≥10 pass AND ≥10 fail)
- **Garbage outputs**: 91/66,959 (0.14%) — mostly deepseek-coder-1.3b on creative strategies
- **Backtick contamination**: 510 (0.8%) — all failing, no pass contamination
- **Length overlap**: Good overlap between pass (P10=77, P90=462) and fail (P10=50, P90=724)
- **Length confounds**: Only 3/162 problems with >3x pass/fail length ratio
- **Model diversity**: 161/162 problems have ≥3 models in both pass and fail pools
- **Difficulty range**: 2.7% to 72% pass rate across problems

## clean_solution Fix

Fixed indentation handling in `clean_solution()`:
- Old: Added 4 spaces only to unindented lines → broke code where ALL lines were at base indent 0
- New: Detects pattern (second non-empty line at indent 0 = shift everything; otherwise shift only unindented)
- Result: gpt-4.1 pass rate recovered from 21.3% → 60.3%, gpt-4o-mini preserved at 73%

## Dataset (humaneval-v1)

- **Train**: 2,400 rows across 80 problems (1,200 pass / 1,200 fail)
- **Test**: 2,460 rows across 82 problems (1,227 pass / 1,233 fail)
- 30 solutions per problem, balanced pass/fail
- Stratified sampling: round-robin across (model, strategy) groups for maximum diversity
- All 19 models and 6 strategies represented
- Zero train/test problem overlap
