# humaneval-v2.1 build plan

## Goal

Build a cleaner version of humaneval-v2 that filters out garbage outputs (gibberish, model refusals, runaway generations) while preserving legitimate short and long correct/wrong solutions. Backfill from the full source pool so per-task class counts stay healthy.

## Filter (selected from filter probes 2026-05-14)

Conjunction of:

1. **`chars ∈ [10, 900]`** — answer length in characters (after `answer.strip()`)
2. **`mean log P(y | x) ≥ −5`** — mean per-token conditional log probability
3. **`total log P(y | x) ≥ −500`** — sum of per-token conditional log probability

Notes:
- `chars[10, 900]` matches the spirit of GSM8K v2's `[60, 3000]` but recalibrated for HumanEval's shorter natural answer lengths (legitimate one-liners like `return max(l)` are ~13 chars).
- `mean l ≥ −5` is a softer version of GSM8K v2's `≥ −2`. Calibrated on gemma-4-31B-it distribution: drops the catastrophic-garbage tail (~9% of wrong, ~0% of correct) without biasing the dataset toward "subtle-bug-only" wrongs.
- `total l ≥ −500` catches extreme outliers (very long + very bad responses).

On the existing v2 sample of 2,367 rows, this filter keeps:
- 2,219 rows total (93.7%)
- 1,202 / 1,205 correct (99.8%)
- 1,017 / 1,162 wrong (87.5%)
- 4 tasks fall below 10 wrong (rest are healthy)
- AUROC: raw=0.7739, lenorm=0.8399, gap=+0.066 (vs unfiltered gap +0.087)

## Filter-model choice

The log P filter requires scoring the candidate pool with **some** model to compute `mean l` and `total l`.

The thresholds above (`−5`, `−500`) are calibrated on **gemma-4-31B-it** log P (under format C + chat template).

**Use gemma-4-31B-it for both filter and eval** — same as GSM8K v2 (which used gemma-2-9b-it for both). The "circularity" is mild: the filter is for data quality (remove garbage like `'pengow'`, `'Sure'`, runaway generations), not for cherry-picking favorable test examples. The same model is used for scoring at filter time and eval time, but no individual test row is rejected on the basis of being unfavorable to lenorm vs raw — it's just rejected for being garbage by either metric.

This means the scoring step is essentially the same as v2 scoring, just applied to the (much larger) candidate pool instead of the pre-sampled v2 test set.

## Pipeline

### Step 1: Recover full candidate pool

The build log says `solutions_merged.jsonl` had 66,946 solutions after merge + dedup. Locally we only have `solutions.jsonl` (16,070 solutions); the rest came from spark (`solutions_spark_raw.jsonl`) and was merged via `reclean_and_merge.py`.

**Action**: re-run `reclean_and_merge.py` after pulling the spark file. Or, if the spark file isn't recoverable, work with what's local plus any cached intermediate files.

Inputs:
- `data/humaneval/solutions.jsonl` (local, 16k)
- `data/humaneval/solutions_spark_raw.jsonl` (recover from spark per memory entry: `~/humaneval_gen/solutions_spark.jsonl`)
- `data/humaneval/problems.jsonl` (164 HumanEval problems)

Output: `data/humaneval/solutions_merged.jsonl` (~67k solutions, post-clean, deduped)

### Step 2: Apply chars filter pre-emptively

Filter `solutions_merged.jsonl` to rows where `10 ≤ len(solution.strip()) ≤ 900`. This is cheap (no scoring required) and reduces the pool before the expensive scoring step.

Output: candidate pool, `solutions_chars_filtered.jsonl`

### Step 3: Score candidate pool with gemma-4-31B-it (format C + chat template)

Use a scoring script analogous to `notes/log_P_diff_plots/humaneval-v2/scripts/score_v2_humaneval.py`, but:
- Model: `google/gemma-4-31B-it` (same as eval — see "Filter-model choice" above)
- Prompt: format C (instruction + question + `Solution:\n{def_signature}`)
- Apply gemma's chat template (same as v2 scoring)
- For each candidate, save `logp_cond` array only (we don't need uncond or neg variants for filtering — only need them for the final eval scoring pass in step 7)
- Skip rows where the scoring fails (eg. empty completion after tokenization)

Output: `pool_scored.jsonl` — one record per candidate with per-token log P under gemma-4-31B-it.

Cost estimate: the candidate pool is ~67k after char filtering. One forward pass per row × 67k rows × ~0.3s/row on A100 ≈ 5-6 hours. ~$7 in pod compute.

(If we want to save time/money, we can skip uncond + neg variants here since this scoring is just for the filter; we'll do the full 5-prompt scoring on the smaller v2.1 sample in step 7.)

### Step 4: Apply log P filter

Drop rows where `mean l < −5` OR `total l < −500`.

Output: `pool_fully_filtered.jsonl`

### Step 5: Stratified sample → v2.1 CSVs

Modify `scripts/dataset_builder/build_humaneval_v1.py` to take a `--pool-path` argument and read from `pool_fully_filtered.jsonl` instead of `solutions_merged.jsonl`. Other build rules unchanged:
- Exclude `intentional_bug` strategy (already excluded in v1)
- Exclude HumanEval/53 and HumanEval/145
- Require ≥10 pass AND ≥10 fail per problem
- Stratified sample 30 per problem (15+15), round-robin across (model, strategy) groups
- Same seed=42

Apply `answer.strip()` to each `answer` field (matches v2's convention).

Output: `data/humaneval/v2.1/train.csv` + 82 `humaneval_*.csv` test files

### Step 6: Final scoring of v2.1 with gemma-4-31B-it (all 5 prompts)

Same scoring script as v2 (`score_v2_humaneval.py`), pointed at v2.1 data. 5 forward passes per row (cond, uncond, neg_v1, neg_v2, neg_v3), per-token log P + entropy saved.

Note: we already have logp_cond from step 3 for the rows that survive. We could in principle reuse those values to save the cond forward pass — but it's cleaner to re-run all 5 prompts from scratch on the (much smaller) v2.1 sample.

Output: `humaneval_v2_1_pertok_scores.jsonl` in `notes/log_P_diff_plots/humaneval-v2.1/`

Cost estimate: ~2,400 rows × 5 prompts × ~0.3s = ~1 hour on A100. ~$1.

### Step 7: Repeat the v2 analysis on v2.1

Same scripts as in `notes/log_P_diff_plots/humaneval-v2/`:
- AUROC macro table for all variants
- Pair classification (D/C/A/B)
- Per-task tables
- Position profile + within-class N×l correlation
- Compare v2 → v2.1 → v1 numbers

Output: `notes/log_P_diff_plots/humaneval-v2.1/V2_1_ANALYSIS_REPORT.md`

## Alternative (faster, less clean): retroactive filter on existing v2

If we want to avoid the gemma-2-9b-it scoring pass, we could apply the filter retroactively to the existing v2 scoring data:

- Skip steps 1-5
- Just drop rows from `humaneval_v2_pertok_scores.jsonl` where the filter fails
- No new compute

Pros: zero new compute, immediate result.

Cons:
- Circular: we're using gemma-4-31B-it (eval model) for filtering, which biases downstream AUROC. **Should NOT be used for the "honest" AUROC story.**
- No backfill: 4 tasks fall below 10 wrong, can't recover them.
- The v2 sample was already 30-per-problem random selection from a pre-build, so filtering it further makes per-task counts uneven.

**Use this only as a sanity-check, not as the v2.1 release.**

## Estimated effort

- Steps 1-2 (pool recovery + char filter): 1-2 hours of fiddling, mostly waiting for spark file transfer.
- Step 3 (gemma-4-31B-it scoring of full pool): ~5-6 hours of pod time. ~$7.
- Step 4 (apply filter): 5 min.
- Step 5 (build script edit + rerun): 1 hour.
- Step 6 (final 5-prompt scoring of v2.1 with gemma-4-31B-it): ~1 hour. ~$1.
- Step 7 (analysis): 1-2 hours.

**Total**: ~1 day of work, ~$8-10 in pod compute. (Most of the cost is the full-pool scoring in step 3 since we're rescoring ~67k candidates with the 31B model.)

**Optimization**: if step 3 cost is unwanted, we could reduce it by:
- (a) Subsampling the pool before scoring (e.g., score 5× target rather than the full pool). Risk: not enough headroom for backfill on some tasks.
- (b) Use vLLM for parallel scoring instead of single-row forward passes. Could cut step 3 from 5h to ~1h.

## Risks / open questions

1. **Will the spark file be recoverable?** If not, we're limited to the ~16k local solutions, which doesn't have enough wrong-class data for many tasks. The build log notes 11,680 of 16,070 are usable after dedup + intentional_bug filter; only ~51 tasks have ≥15 wrong in the local pool.
2. **Should the negated prompts also be applied as filters?** Currently no — filter is only on the positive prompt's log P. If we cared about tc-neg quality, we could add `log P(y | neg-x)` thresholds, but the user hasn't asked for this and it would complicate things.
3. **Should we use vLLM for the step-3 pool scoring?** Would speed it up ~5× but adds setup complexity. Worth it if step 3 turns out to be the bottleneck.

## Backwards compatibility

- v1 stays as-is (do not modify).
- v2 stays as-is (do not modify).
- v2.1 is a new directory: `data/humaneval/v2.1/`. New rankalign task registration adds `humaneval-v2.1` and `humaneval-v2.1-{slug}` entries (analogous to the v1 → v2 pattern in `src/tasks/humaneval.py`).
