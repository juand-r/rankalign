# Score Filename Convention

This document describes the naming convention for `scores_*.csv` files in `outputs/`,
covering how they are produced, what each component means, and how to parse them.

> **Skip ahead:** for v7/fix1-generation runs (everything trained with
> `scripts/ranking_loss_ref_fix.py` since 2026-05-24, including all overnight
> s1–s12 cells, gemma-4-31B-it, and s13/cft cells), see
> [§ v7 / fix1 Generation](#v7--fix1-generation-added-2026-05-24) below
> first. The general filename schema is unchanged; what changed is the
> model_short construction (160-char cap + abbreviation fallback) and the
> set of recognised model-dir flags (`--ppd`, `--cft`, `--fix1`).

## Filename Format

```
scores_{eval_prefix}{model_short}_{task}_{split}{v2_suffix}{metric_suffix}{tc_suffix}{lenorm_suffix}{eos_suffix}_{timestamp}.csv
```

This format is identical for v6 and v7 runs. It's emitted by
`build_csv_filename(...)` in [`src/tasks/common.py`](../src/tasks/common.py)
(every v7 save path now goes through this single function — see
2026-05-24 entry in [`docs/UPDATES.md`](UPDATES.md)).

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

For `basetyp` and `basetypneg`, use `--base-model <name>` to specify which base model
computes the typicality scores (e.g. `--base-model google/gemma-2-2b-it`). Each finetuned
model should use its own pre-finetuning base model.

### Important notes

- `eval_by_claude.py` uses `_tc` suffix for all three TC types. It distinguishes them via the prefix.
- `eval.py` (older script) uses `_evaltc` suffix and has no prefix. It only supports GPT-2 TC.
- An older version of `eval_by_claude.py` used `_evaltc` instead of `_tc` for self-TC.
  Files with `self-` prefix + `_evaltc` suffix are from this older version.
- In `scripts/dashboard_viz_refactor.py` (current policy), only `self-` prefixed files are ingested.
  Files with empty eval prefix are ignored. By default, no-prefix `_evaltc` files (from `eval.py`) are
  ignored, while prefixed `_evaltc` files (older `eval_by_claude.py`) are still allowed.
- `--length-normalize` (`_evallenorm`) does NOT change the CSV content. The `gen_score_lenorm`
  column is always computed regardless of this flag. The flag only changes the filename.
  A file with `_evallenorm` and one without are functionally identical.

## v7 / fix1 Generation (added 2026-05-24)

All models trained with [`scripts/ranking_loss_ref_fix.py`](../scripts/ranking_loss_ref_fix.py)
(the active training script) save under `/datastor2/jdr/rankalign/models2/`
with a directory name prefixed `v7-` and ending in `--fix1` (or
`--fix1_merged` for LoRA models that get merged at save time). The
canonical v7 directory shape is:

```
v7-{model//--}-delta{delta}-epoch{epoch}--{task}-all--{direction}--{split}--alpha{alpha}{tc}{full-completion}{pref?}{nllv?}{nllg?}{fsx?}{ppd?}{cft?}{vlo?}{semi-or-labelonly}--fix1[_merged]
```

with each token glued by `--` and the trailing `_merged` only for
LoRA-trained adapters that were merged into a full model at save time
(gemma-2-9b-it; **not** for gemma-4-31B-it, which uses `--gemma4-lora`
and skips the merge).

### v7 flag tokens you'll see in `model_short`

| Token | Source flag | Meaning |
|---|---|---|
| `delta{N.NN}` | `--delta` or `--delta-bins` (computed) | Pair margin. With `--delta-bins 10` it's `(p95-p5)/10`, two-decimal-truncated. |
| `epoch{N}` | (auto, picked by eval) | Walltime-killed runs may save through 0/1 only — eval picks the latest with `ls -dt`. |
| `tc-self` / `tc-neg` | `--self-typcorr` / `--neg-typcorr` | TC at **training** time (eval-side TC is a separate axis). |
| `full-completion` | (always present in current runs) | Multi-token completion log-probs. |
| `pref{N.N}` | `--preference_loss_weight` | OMITTED when value is the default 1.0. So absent ↔ pref=1.0. |
| `nllv{N.N}` | `--nll_validator_weight` | OMITTED when 0.0. Absent ↔ nll_v=0. |
| `nllg{N.N}` | `--nll_generator_weight` | OMITTED when 0.0. Absent ↔ nll_g=0. |
| `force-same-x` | `--force-same-x` | Within-prompt pair sampling. |
| `ppd` | `--per-prompt-delta` | Per-prompt p5/p95 delta computation. Only meaningful with `force-same-x`. **v7-only token.** |
| `cft` | `--consistency-ft` | SFT-only filter: keep only labeled items where binarized validator and generator scores agree. **v7-only token.** Only valid in s13. |
| `vallogodds` | `--validator-log-odds` | Use log-odds in validator scoring. |
| `semi{N.N}` / `labelonly{N.N}` | `--semi 0.1` or `--labelonly 0.1` | Semi-supervised vs labeled-only. Mutually exclusive. |
| `fix1` | (always present for v7) | Distinguishes the v7/fix1 generation from legacy v6. |

The complete dir-name template lives at
`scripts/ranking_loss_ref_fix.py` line 2182 (search for the f-string
that joins all the `*_str` fragments).

### Three on-disk forms of `model_short`

Because of an early eval bug (subsequently fixed) and because long flag
stacks blow past Linux's 255-char `NAME_MAX` filename limit, the
`model_short` you'll see embedded in `scores_*.csv` filenames takes one of
three shapes for v7-trained models. **All three are recognised by the
parser at `src/checkpoint_name_parser.py`**, so the v7 table-builders
treat them uniformly.

| Form | What it looks like | When emitted |
|---|---|---|
| **A** legacy abs-path-embedded | `v6-_datastor2_jdr_rankalign_models2_v7-google--gemma-2-9b-it-delta1.42-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--tc-self--full-completion--semi0.1--fix1[_merged]` | Pre-fix `eval_by_claude.py` (commits before `6890d295` 2026-05-24 17:47) wrote `model_short = 'v6-' + abs_path.replace('/', '_')`, embedding the whole `/datastor2/...` path. **No new files of this form will ever be written.** Existing CSVs are still on disk. |
| **B** un-abbreviated (current default) | `v7-google--gemma-2-9b-it-delta1.42-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1[_merged]` | The basename of the model dir, used as-is whenever `len(basename) ≤ 160`. This is the default for short cells (e.g. RankAlign / s2 / s6). |
| **C** abbreviated | `v7-gemma-2-9b-it-d1.42-e2-membership-sans-rosch-v0-all-tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1` | When `len(basename) > 160`, `build_model_short` falls back to `to_hf_repo_name(parsed, prefix='', max_len=None)` which produces an abbreviated form. Used for high-flag-count cells (s4/s7 with fsx+ppd+vlo+tc). All v7 flag info is preserved — just shorter. |

The abbreviation map (form B → form C) is implemented in
[`src/checkpoint_name_parser.py:to_hf_repo_name()`](../src/checkpoint_name_parser.py):

```
delta{x}     → d{x}        nllv{x}        → nv{x}
epoch{n}     → e{n}         nllg{x}        → ng{x}
tc-self      → tcs          vallogodds     → vlo
tc-neg       → tcn          force-same-x   → fsx
tc-online    → tco          ppd            → ppd  (kept verbatim)
lenorm       → ln           cft            → cft  (kept verbatim)
pref{x}      → p{x}         semi{x}        → sm{x}
                            labelonly{x}   → lo{x}
                            fix1           → fix1 (kept verbatim)
```

### How `build_model_short()` decides which form to emit

[`src/tasks/common.py:build_model_short()`](../src/tasks/common.py),
called by every CSV-save path in `eval_by_claude.py` since
2026-05-24 (commit `6890d295`):

1. If `modelname` is an **absolute path**: `raw = basename(modelname)`.
   *(Fixes the legacy abs-path-embedded form-A bug.)*
2. Else if `modelname` is HF-style (`org/name`): `raw = "v6-" + name.replace("/", "_")`.
3. If `len(raw) ≤ 160`: return `raw` verbatim → **form B**.
4. Else: try `to_hf_repo_name(parse_checkpoint_name(raw), prefix='', max_len=None)`.
   If that fits in 160 chars: return it → **form C**. (Currently the
   max observed abbreviated length is ~100 chars, so this always fits.)
5. Last-resort: `raw[:151] + '_' + md5(raw)[:8]` → an md5-suffixed
   truncation. Should never happen with current settings; kept as a
   safety net.

The total CSV filename overhead beyond `model_short` is ~90 chars
(eval prefix + task + suffixes + timestamp), so a 160-char `model_short`
cap keeps everything under the 255-byte `NAME_MAX` limit.

### Parsing a `model_short` programmatically

Use [`src/checkpoint_name_parser.py:parse_checkpoint_name()`](../src/checkpoint_name_parser.py).
It handles all three forms (A/B/C) transparently — strips the abs-path
wrapper if present, dispatches to the abbreviated-form parser when
`-d{N}-e{N}-` is detected, else parses the un-abbreviated form. Returns
a structured dict with keys `version`, `model_short`, `delta`, `epoch`,
`task_segment`, `tc`, `pref`, `nll_v`, `nll_g`, `force_same_x`, `ppd`,
`cft`, `vallogodds`, `semi`, `labelonly`, `fix1`, etc.

For matching against an expected setting (used by the v7 table builders),
use `matches_v7_setting(model_short, model=..., task_segment=..., **expected_flags)`
in the same module — it parses, compares fields, and returns bool.

### v7 table builders that consume these CSVs

- [`scripts/_build_rosch_table_v7.py`](../scripts/_build_rosch_table_v7.py)
- [`scripts/_build_persona_v1_table_v7.py`](../scripts/_build_persona_v1_table_v7.py)

Both refactored on 2026-05-24 to use structural matching via
`matches_v7_setting()`, so they handle forms A/B/C uniformly without
needing a regex pass. To add a new setting (e.g. s13/cft), append a
METHODS row with the relevant flag kwargs.

---

## Model Short Name

The `model_short` portion encodes the model identity. It is derived from the model path:

- **Base model** (e.g. `google/gemma-2-9b-it`):
  - `model_short = "v6-" + path.replace("/", "_")`
  - Example: `v6-google_gemma-2-9b-it`

- **Finetuned model** (e.g. `../models/v6-google--gemma-2-9b-it-delta0.15-epoch2--...`):
  - `model_short = basename(path).replace("--", "_")`
  - Example: `v6-google_gemma-2-9b-it-delta0.15-epoch2_plausibleqa-all_d2g_random_alpha1.0_full-completion_force-same-x_labelonly0.1_merged`

> *Note*: the rules above describe the **legacy v6** branch of
> `build_model_short`, kept for HF-style and relative-path inputs. For
> v7 absolute-path inputs (the default for all overnight runs), see the
> [v7 / fix1 Generation](#v7--fix1-generation-added-2026-05-24)
> section above — that's the path everything currently flows through.

### Model Path Components (from `ranking_loss_ref.py`)

The model save directory is built as:

```
v6-{model}--delta{delta}--epoch{epoch}--{task}{all_str}--{direction}--{split_type}{alpha_str}{tc_str}{lenorm_str}{full_completion_str}{eos_str}{pref_str}{nll_v_str}{nll_g_str}{fsx_str}{vlo_str}{semi_str}
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
| `eos_str` | `eos` or empty | EOS included in completion scoring during training (`--include-eos`) |
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

## Dashboard Heatmap Row Key (current)

For `scripts/dashboard_viz_refactor.py`, each heatmap row key is:

```
{category}-{mode}-{regime}[-tco][-tcself][-tcneg][-norm][-v][-eos]
```

Where:

- `category`: `Base`, `S`, or `U`
- `mode`: `Pref`, `SFT`, or `Comb`
- `regime`: `all` (no `semi*`/`labelonly*` token), `semi`, or `labelonly`

This ensures rows differ when training regime differs (e.g. `semi0.1` vs `labelonly0.1`).
Aggregated heatmaps average over datasets (tasks) within the **same row key** only, so
they do not mix training regimes or training-TC variants.

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

## Output Directories

| Directory | Contents |
|-----------|----------|
| `outputs/` | All non-EOS eval score files (standard location) |
| `outputs-eos-models/` | Eval score files for EOS-trained models (from `models-eos/`) |

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

| Script / module | Purpose | Filename role |
|---|---|---|
| [`scripts/ranking_loss_ref.py`](../scripts/ranking_loss_ref.py) | Training (legacy v6) | Constructs `v6-` save dir name (line ~2658). |
| [`scripts/ranking_loss_ref_fix.py`](../scripts/ranking_loss_ref_fix.py) | **Training (current v7/fix1)** | Constructs `v7-` save dir name (line ~2182). Adds `--ppd`, `--cft`, `--fix1` flags. |
| [`scripts/eval.py`](../scripts/eval.py) | Evaluation (legacy) | Emits `scores_{model_short}_{task}_{split}..._evaltc_...`. GPT-2 TC only. |
| [`scripts/eval_by_claude.py`](../scripts/eval_by_claude.py) | **Evaluation (current)** | All 8 CSV-save paths now go through `build_csv_filename()` (since 2026-05-24 commit `6890d295`). Emits `scores_{prefix}{model_short}_{task}_{split}..._tc_...`. |
| [`src/tasks/common.py`](../src/tasks/common.py) | Filename builder | `build_csv_filename()` and `build_model_short()` — single source of truth for all `scores_*.csv` filenames. Enforces 160-char model_short cap with abbreviation fallback. |
| [`src/checkpoint_name_parser.py`](../src/checkpoint_name_parser.py) | Filename parser | `parse_checkpoint_name()` → structured dict, handles forms A/B/C. `to_hf_repo_name()` → abbreviated form C. `matches_v7_setting()` → structural matcher used by v7 table builders. |
| [`scripts/run_eval_semi.sh`](../scripts/run_eval_semi.sh) | Eval wrapper | Adds skip logic, calls `eval_by_claude.py`. |
| [`scripts/run_train_semi.sh`](../scripts/run_train_semi.sh) | Train wrapper | Reads `VENV` env var (gemma-4 needs `/datastor2/jdr/venvs/gemma4`). Plumbs `--gemma4-lora` and `--consistency-ft`. |
| [`scripts/_overnight_launch.sh`](../scripts/_overnight_launch.sh) | v7 dispatcher | One-call train+eval launcher: `bash _overnight_launch.sh DATASET MODEL SETTING`. Builds the COMMON_FLAGS list per s1..s13. |
| [`scripts/_build_rosch_table_v7.py`](../scripts/_build_rosch_table_v7.py) | v7 table builder (rosch) | Uses `matches_v7_setting()` structurally; handles forms A/B/C. |
| [`scripts/_build_persona_v1_table_v7.py`](../scripts/_build_persona_v1_table_v7.py) | v7 table builder (persona) | Same. |
| [`scripts/inventory_scores.py`](../scripts/inventory_scores.py) | File inventory | Checks completeness, generates LaTeX PDF. |

## Quick reference: how to read a real v7 filename

Take this CSV name and decompose it:

```
scores_basetyp-v7-google--gemma-2-2b-it-delta0.85-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--tc-self--full-completion--force-same-x--ppd--semi0.1--fix1_rosch-vegetable_test_log-odds_tc_20260524.csv
```

| Slice | Value | Meaning |
|---|---|---|
| `scores_` | — | Always present. |
| `basetyp-` | eval_prefix | TC at **eval** time = base-typcorr (use base model's unconditional log-prob). |
| `v7-google--gemma-2-2b-it-...--fix1` | model_short (form B) | The v7-trained model. |
|   `delta0.85` | | Delta computed from p95/p5 of validator scores. |
|   `epoch2` | | Saved checkpoint, end of training. |
|   `membership-sans-rosch-v0-all` | | Trained on this task. |
|   `tc-self` | | Used self-typcorr at training. |
|   `force-same-x--ppd` | | fsx ON + per-prompt-delta. |
|   `semi0.1` | | Semi-supervised w/ 10% labeled. |
|   `fix1` | | v7/fix1 generation. |
| `_rosch-vegetable` | task | Eval task. |
| `_test` | split | Test data. |
| `_log-odds` | metric_suffix | Validator log-odds (always for v7). |
| `_tc` | tc_suffix | Eval-side TC active. |
| `_20260524` | timestamp | Date of eval. |
| `.csv` | — | — |

Setting identification: `pref` absent → 1.0 (default), `nllv`/`nllg` absent → 0
(default), `tc-self` + fsx + ppd + (vlo absent) + semi0.1 → matches **s5**
(RA + PMI + fsx, NLL absent, vlo absent — the s5 launcher case in
`_overnight_launch.sh`).
