# Score Filename Convention

This document describes the naming convention for `scores_*.csv` files in `outputs/`,
covering how they are produced, what each component means, and how to parse them.

## Filename Format

```
scores_{eval_prefix}{model_short}_{task}_{split}{v2_suffix}{metric_suffix}{tc_suffix}{lenorm_suffix}{eos_suffix}_{timestamp}.csv
```

### Components

| Component | Values | Meaning |
|-----------|--------|---------|
| `eval_prefix` | `neg-`, `self-`, `basetyp-`, `basetypneg-`, or empty | Typicality correction method used at **eval time** |
| `model_short` | e.g. `v6-google_gemma-2-9b-it-delta0.15-epoch2_...` | Model identifier (see Model Short Name below) |
| `task` | e.g. `ambigqa-american`, `hypernym-dogs`, `ifeval-prompt_42` | Eval task name |
| `split` | `test` or `train` | Data split |
| `v2_suffix` | `_v2` or empty | Present for hypernym tasks (grammar-corrected v2 data) |
| `metric_suffix` | `_log-odds` or `_log-probs` | Scoring metric (all current runs use `_log-odds`) |
| `tc_suffix` | `_tc`, `_evaltc`, or empty | See Eval TC Suffix below |
| `lenorm_suffix` | `_evallenorm` or empty | Present if `--length-normalize` was used. Does NOT affect file content. |
| `eos_suffix` | `_eos` or empty | Present if `--include-eos` was used. Includes log P(EOS) in gen_score. |
| `timestamp` | `YYYYMMDD` or `YYYYMMDD_HHMMSS` | When the eval was run |

## Eval Prefix + TC Suffix: The Three TC Types

There are three kinds of typicality correction used at eval time:

| TC type | What computes the correction | Script | Eval prefix | TC suffix |
|---------|------------------------------|--------|-------------|-----------|
| **neg** | Negated prompt: log P(y \| neg_Q) | `eval_by_claude.py --neg-typicality` | `neg-` | `_tc` |
| **self** | Model itself: unconditional log P_model(y) | `eval_by_claude.py --self-typicality` | `self-` | `_tc` (current) or `_evaltc` (older version) |
| **basetyp** | Base (pre-finetuning) model: unconditional log P_base(y) | `eval_by_claude.py --base-typicality` | `basetyp-` | `_tc` |
| **basetypneg** | Base model + negated prompt: log P_base(y \| neg_Q) | `eval_by_claude.py --base-typicality --neg-typicality` | `basetypneg-` | `_tc` |
| **gpt2** | GPT-2: log P_GPT2(y) | `eval.py --typicality-correction` | (empty) | `_evaltc` |

### Important notes

- `eval_by_claude.py` uses `_tc` suffix for all three TC types. It distinguishes them via the prefix.
- `eval.py` (older script) uses `_evaltc` suffix and has no prefix. It only supports GPT-2 TC.
- An older version of `eval_by_claude.py` used `_evaltc` instead of `_tc` for self-TC.
  Files with `self-` prefix + `_evaltc` suffix are from this older version.
- `--length-normalize` (`_evallenorm`) does NOT change the CSV content. The `gen_score_lenorm`
  column is always computed regardless of this flag. The flag only changes the filename.
  A file with `_evallenorm` and one without are functionally identical.

## Model Short Name

The `model_short` portion encodes the model identity. It is derived from the model path:

- **Base model** (e.g. `google/gemma-2-9b-it`):
  - `model_short = "v6-" + path.replace("/", "_")`
  - Example: `v6-google_gemma-2-9b-it`

- **Finetuned model** (e.g. `../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--...`):
  - `model_short = basename(path).replace("--", "_")`
  - Example: `v6-google_gemma-2-9b-it-delta0.15-epoch2_plausibleqa-all_d2g_random_alpha1.0_full-completion_force-same-x_labelonly0.1_merged`

### Model Path Components (from `ranking_loss_ref.py`)

The model save directory is built as:

```
v6-{model}--delta{delta}--epoch{epoch}--{task}{all_str}--{direction}--{split_type}{alpha_str}{tc_str}{lenorm_str}{full_completion_str}{pref_str}{nll_v_str}{nll_g_str}{fsx_str}{vlo_str}{semi_str}
```

| Component | Example | Meaning |
|-----------|---------|---------|
| `model` | `google--gemma-2-9b-it` | HuggingFace model name with `/` -> `--` |
| `delta{N}` | `delta0.15` | Delta hyperparameter |
| `epoch{N}` | `epoch2` | Which epoch checkpoint (0-indexed, final = num_epochs - 1) |
| `task` | `plausibleqa-all`, `ambigqa-all`, `hypernym-concat-bananas-to-dogs-double-all`, `ifeval-concat-all` | Training task |
| `direction` | `d2g` (V2G) or `g2d` (G2V) | Training direction |
| `split_type` | `random` | Train/test split method |
| `alpha{N}` | `alpha1.0` | Alpha hyperparameter |
| `tc_str` | `tc-neg`, `tc-self`, `tc-online`, or empty | TC at **training** time |
| `lenorm_str` | `lenorm` or empty | Length normalization at training time |
| `full_completion_str` | `full-completion` | Full completion log-probs (always present in current runs) |
| `pref_str` | `pref0.0` or empty | Preference loss weight (empty = 1.0 default) |
| `nll_v_str` | `nllv1.0` or empty | NLL validator weight (empty = 0) |
| `nll_g_str` | `nllg1.0` or empty | NLL generator weight (empty = 0) |
| `fsx_str` | `force-same-x` or empty | Force same prompt for pair training |
| `vlo_str` | `vallogodds` or empty | Validator log-odds |
| `semi_str` | `semi0.1`, `labelonly0.1`, or empty | Semi-supervised or labeled-only mode |

For 9b-it models trained with LoRA, the merged model has `_merged` appended.

## Training Variants

The 7 training loss variants used in semi-supervised experiments:

| Short name | pref | nll_v | nll_g | vallogodds | semi mode | Model path flags |
|------------|------|-------|-------|------------|-----------|------------------|
| pref-only lo | 1.0 | 0 | 0 | no | labelonly | `force-same-x--labelonly0.1` |
| pref-only vlo lo | 1.0 | 0 | 0 | yes | labelonly | `force-same-x--vallogodds--labelonly0.1` |
| pref-only semi | 1.0 | 0 | 0 | no | semi | `force-same-x--semi0.1` |
| comb lo | 1.0 | 1.0 | 1.0 | yes | labelonly | `nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1` |
| comb semi | 1.0 | 1.0 | 1.0 | yes | semi | `nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1` |
| sft semi | 0.0 | 1.0 | 1.0 | no | semi | `pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1` |
| sft lo | 0.0 | 1.0 | 1.0 | no | labelonly | `pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1` |

The **V2G baseline** (from the original RankAlign paper) uses none of these flags:
just `full-completion` with default loss weights (pref=1, nll_v=0, nll_g=0), no force-same-x,
no semi/labelonly.

## Training TC Types

| Train TC | Model path contains | Meaning |
|----------|-------------------|---------|
| plain | (no tc string) | No typicality correction during training |
| tc-neg | `tc-neg` | Negated-prompt TC during training |
| tc-self | `tc-self` | Self-model TC during training |
| tc-online | `tc-online` | GPT-2 TC during training |
| tc-neg-lenorm | `tc-neg--lenorm` | Neg TC + length norm during training |
| tc-self-lenorm | `tc-self--lenorm` | Self TC + length norm during training |

## Base Models

| Model | HuggingFace name | LoRA | `_merged` suffix |
|-------|-----------------|------|------------------|
| gemma-2-9b-it | `google/gemma-2-9b-it` | Yes | Yes |
| gemma-2-2b | `google/gemma-2-2b` | No (full finetune) | No |
| gemma-2-2b-it | `google/gemma-2-2b-it` | No (full finetune) | No |

## Eval Domains and Task Counts

| Domain | Task prefix | # tasks | Notes |
|--------|------------|---------|-------|
| PlausibleQA | `plausibleqa-nq_*`, `plausibleqa-trivia_*`, `plausibleqa-webq_*` | 100 | |
| AmbigQA | `ambigqa-*` | 17 | |
| Hypernym | `hypernym-*` | 18 | Uses `_v2` suffix |
| IFEval | `ifeval-prompt_*` | 99 | Uses `--disc-shots zero` |

## Files in `outputs-unused/`

These files were moved out of `outputs/` because they are legacy or redundant:

| Category | Reason |
|----------|--------|
| epoch0 / epoch1 models | Intermediate checkpoints, final epoch2 is authoritative |
| Per-category hypernym (e.g. `hypernym-bananas-all`) | Old per-category training, replaced by `hypernym-concat-bananas-to-dogs-double-all` |
| Duplicate lenorm files | Same content as non-lenorm version (from a separate run) |
| No-TC evals | Evals run without any typicality correction flag |
| gemma-2-2b-it no-TC lenorm | 2b-it evals with only `--length-normalize`, no TC |

## Scripts Reference

| Script | Purpose | Filename convention |
|--------|---------|-------------------|
| `scripts/ranking_loss_ref.py` | Training | Determines model save path |
| `scripts/eval.py` | Evaluation (older) | `scores_{model_short}_{task}_{split}..._evaltc_...` |
| `scripts/eval_by_claude.py` | Evaluation (current) | `scores_{prefix}{model_short}_{task}_{split}..._tc_...` |
| `scripts/run_eval_semi.sh` | Eval wrapper | Adds skip logic, calls `eval_by_claude.py` |
| `scripts/run_train_semi.sh` | Train wrapper | Handles semi-supervised args, calls `ranking_loss_ref.py` |
| `scripts/run_train_v2g.sh` | V2G baseline train | Minimal wrapper for paper baseline training |
| `scripts/inventory_scores.py` | File inventory | Checks completeness, generates LaTeX PDF |
