# Setting s13: SFT + `--consistency-ft`

Added 2026-05-24. New training setting that pairs the SFT-lo flag set (s1) with
the new `--consistency-ft` flag in `scripts/ranking_loss_ref_fix.py`.

## What `--consistency-ft` does

Before training begins:

1. Computes per-item validator scores (already done by the existing pre-pass
   over the discriminator-style "Yes/No" prompt: `logprobs_last_layer[i]`).
2. Runs an extra forward-only pass to compute per-item generator scores
   `gen_scores[i] = log P(completion_i | gen_prompt_i)` (raw, no length-norm,
   no typicality correction).
3. Computes thresholds `t_v = mean(val)`, `t_g = mean(gen)`.
   - Under `--semi-supervised`, both means are computed over labeled items
     only (so the threshold reflects the supervised distribution).
4. Binarizes each item: `bv = 1 if val > t_v else 0`, same for `bg`.
5. **Drops** any *labeled* item where `bv != bg`. Unlabeled items pass
   through unchanged.
6. Pair construction + training proceed on the filtered set.

Stats logged to `<models-dir>/training_run_logs/<...>.json` under the
`"consistency_ft"` key (`t_v`, `t_g`, `n_labeled_kept/dropped`,
`n_unlabeled_kept`, and the 4-cell `(bv, bg)` count breakdown).

## Argparse enforcement (verified 2026-05-24)

Argparse rejects any combination that isn't SFT-with-no-fsx:

- `--preference_loss_weight` must be `0`.
- `--nll_validator_weight` and `--nll_generator_weight` must both be `> 0`.
- `--force-same-x` must be OFF.

`--labeled-only`, `--semi-supervised`, or neither are all OK.

## Backwards compatibility

- Default off. When unset:
  - No extra forward pass, no items dropped.
  - `consistency_ft_stats` stays `None`; JSON log records `"consistency_ft": null`.
  - `--cft` is omitted from the save-dir name.
- Existing s1–s12 runs are bit-for-bit unchanged.

## s13 in the dispatcher

`scripts/_overnight_launch.sh` was extended (2026-05-24) with a new SETTING
case `s13` and a new DATASET case `humaneval`. Mirrors SFT-lo (s1) flags
otherwise:

| Flag                | s1 (SFT-lo)         | s13 (SFT + cft)     |
|---------------------|---------------------|---------------------|
| `LOSS`              | sft                 | sft                 |
| `SEMI_MODE`         | labelonly           | labelonly           |
| `RATIO`             | 0.1                 | 0.1                 |
| fsx                 | OFF                 | OFF                 |
| `--validator-log-odds` | OFF             | OFF                 |
| typicality (train)  | none                | none                |
| `--delta-bins`      | 10                  | 10                  |
| `--num_epochs`      | 3 (default)         | 3 (default)         |
| `--consistency-ft`  | —                   | **ON**              |
| TC eval list        | self                | self, neg           |

Save-dir suffix for s13:
`-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1`

## Scope (per user, 2026-05-24)

| Train task                | Eval tasks                                     | Models                                                   | GPUs |
|---------------------------|------------------------------------------------|----------------------------------------------------------|------|
| `membership-sans-rosch-v0`| 10 rosch tasks                                 | gemma-2-2b, gemma-2-2b-it, gemma-2-9b-it                 | 1    |
| `persona-v1`              | 6 persona-v1 test tasks                        | gemma-2-2b, gemma-2-2b-it, gemma-2-9b-it                 | 1    |
| `ifeval-concat`           | 21 `ifeval-prompt_*` tasks                     | gemma-2-2b, gemma-2-2b-it; gemma-2-9b-it (2 GPUs)        | 1/2  |
| `humaneval-v2.1correct-upper` | 82 `humaneval-v2.1correct-upper-humaneval_*` | google/gemma-4-31B-it (3 GPUs, gemma4-lora, grad ckpt) | 3    |

Total: **10 train cells** (3 datasets × 3 gemma-2 models + 1 gemma-4 humaneval).

## Gemma-4-31B-it special handling

`_overnight_launch.sh` auto-detects `MODEL` containing `gemma-4-31B-it` and:

- Sets `VENV_OVERRIDE=/datastor2/jdr/venvs/gemma4` (transformers 5.x — the
  default `venv_lexcons` has 4.46.x and won't load Gemma-4).
- Adds `--gemma4-lora` (regex target_modules; skips `merge_and_unload` at
  save → no `_merged` sibling dir, eval points at the adapter dir directly).
- Adds `--gradient-checkpointing` (VRAM).
- Forces `--disc-shots zero` (per `run_settings_v21correct_upper.sh`).
- Allocates 3 GPUs, 192 GB RAM, 24 h walltime.

The override flows through:
- `scripts/run_train_semi.sh`: reads `VENV` env var (added 2026-05-24).
- `scripts/run_eval_semi.sh`: reads `VENV` env var (added 2026-05-24).
- The eval `--wrap` cmd injects `export VENV=...` so the dependency-chained
  eval job picks the right venv.

## Launch commands (DO NOT RUN until cluster is free)

Pre-flight:

```bash
cd /datastor1/jdr/gv-gap/rankalign
squeue -u $USER -o "%i %T %D %C %m %b %M %l %j" | head    # confirm queue is light
df -h /datastor2                                          # confirm space
```

The 10 cells (each does its own train + eval-chain submit):

```bash
# 3 × membership-sans-rosch-v0 → eval on rosch
bash scripts/_overnight_launch.sh membership gemma-2-2b      s13
bash scripts/_overnight_launch.sh membership gemma-2-2b-it   s13
bash scripts/_overnight_launch.sh membership gemma-2-9b-it   s13

# 3 × persona-v1 → eval on 6 persona-v1 test tasks
bash scripts/_overnight_launch.sh persona    gemma-2-2b      s13
bash scripts/_overnight_launch.sh persona    gemma-2-2b-it   s13
bash scripts/_overnight_launch.sh persona    gemma-2-9b-it   s13

# 3 × ifeval-concat → eval on 21 ifeval-prompt_* tasks
bash scripts/_overnight_launch.sh ifeval     gemma-2-2b      s13
bash scripts/_overnight_launch.sh ifeval     gemma-2-2b-it   s13
bash scripts/_overnight_launch.sh ifeval     gemma-2-9b-it   s13

# 1 × humaneval-v2.1correct-upper → eval on 82 humaneval test tasks
bash scripts/_overnight_launch.sh humaneval  gemma-4-31B-it  s13
```

Dryrun any single cell first:

```bash
DRYRUN=1 bash scripts/_overnight_launch.sh humaneval gemma-4-31B-it s13
```

Each call submits one train job and one eval job per TC variant
(`self` + `neg` = 2 evals per cell), with the eval(s) gated by
`--dependency=afterany:<train_jobid>`. So 10 cells → 30 slurm jobs total.

## Sanity-check the trained dir naming

After (or partway through) training, expected save-dir for s13 looks like:

```
/datastor2/jdr/rankalign/models2/v7-google--gemma-2-2b-it-delta0.34-epoch2--persona-v1-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1
```

Note `--cft` between `--nllg1.0` and `--labelonly0.1`. The eval script
uses a glob with `delta*` (since `--delta-bins 10` auto-computes) and
`epoch[012]` (so an early-killed run is still evaluable).

## Verifying the JSON log captured the filter

After training, look at:

```bash
ls -lt /datastor2/jdr/rankalign/models2/training_run_logs/ | head
jq '.consistency_ft' /datastor2/jdr/rankalign/models2/training_run_logs/<latest>.json
```

Expected fields: `t_v`, `t_g`, `threshold_basis`, `n_total_pre_filter`,
`n_labeled_total/kept/dropped`, `n_unlabeled_kept`, `label_cells_bv_bg`.

## File summary of 2026-05-24 edits

- `scripts/ranking_loss_ref_fix.py`: `--consistency-ft` flag + argparse
  enforcement, pre-pass + filter block, `--cft` save-dir suffix, JSON-log
  field, wandb config.
- `scripts/run_train_semi.sh`: `--consistency-ft` and
  `--gradient-checkpointing` pass-through; `VENV` env var override.
- `scripts/run_eval_semi.sh`: `VENV` env var override.
- `scripts/_overnight_launch.sh`: SETTING case `s13`; DATASET case
  `humaneval`; gemma-4-31B-it auto-detection (venv override + `--gemma4-lora`
  + `--gradient-checkpointing` + `--disc-shots zero` + 3 GPUs);
  `CFT_STR` slot in the GLOB_PATH; correct `MERGED_SUFFIX` handling for
  `--gemma4-lora` (no `_merged` dir).
