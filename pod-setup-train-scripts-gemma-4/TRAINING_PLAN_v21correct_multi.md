# Training Plan: humaneval-v2.1correct-multi × gemma-4-31B-it

**Dataset:** `humaneval-v2.1correct-multi` (82 tasks)  
**Model:** `google/gemma-4-31B-it`  
**Epochs:** 3  
**Date planned:** 2026-05-19

## Pod Assignments

Each pod trains two settings sequentially (first setting completes before second starts).
All pods use `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with venv at `/workspace/.venv`.

| Pod | RunPod ID | GPUs | SSH | Setting A (first) | Setting B (second) |
|-----|-----------|------|-----|-------------------|--------------------|
| h100-train-4 | d4gum0mttfqkre | 5×H100 NVL | 205.196.17.114:11663 | 1 (SFT-lo) | 2 (RankAlign) |
| h100-train-5 | e3fkcad376ki1q | 4×H100 SXM | 103.207.149.104:14325 | 3 (New+fsx) | 4 (New+fsx+tc) |
| h100-train-6 | gj0nyp3g8gjk05 | 4×H100 SXM | 64.247.201.55:12004 | 5 (RankAlign+fsx+tc) | 6 (RankAlign+tc) |
| h100-train-7 | sspiiuxdyjbgmm | 4×H100 SXM | 103.207.149.92:18255 | 7 (New+fsx+negtc) | 8 (RankAlign+fsx+negtc) |
| h100-train-8 | owfszqz3sqenn5 | 4×H100 SXM | 103.207.149.86:10892 | 9 (RankAlign+negtc) | 10 (RankAlign+fsx) |

## The 10 Settings

COMMON flags (shared by all settings, defined in `run_arm_3epoch.sh`):
```
--model google/gemma-4-31B-it
--num_epochs 3
--task humaneval-v2.1correct-multi
--train_g_or_d g
--split_type random
--nll_validator_weight 0        ← overridden by comb/sft settings
--nll_generator_weight 0        ← overridden by comb/sft settings
--preference_loss_weight 1      ← overridden by sft setting
--all --delta 0.15
--semi-supervised 0.1           ← replaced by --labeled-only 0.1 for setting #1
--disc-shots zero
--lora --gradient_checkpointing
--models-dir /workspace/models_g4it
--total_samples 5110
```

| # | Name | Additional flags (on top of COMMON) | TC at eval |
|---|------|--------------------------------------|------------|
| 1 | SFT-lo | `--labeled-only 0.1` (replaces --semi-supervised), `--preference_loss_weight 0 --nll_validator_weight 1 --nll_generator_weight 1` | `--self-typicality` and `--neg-typicality`, both with `--base-typicality` |
| 2 | RankAlign | *(COMMON as-is, no changes)* | `--self-typicality` and `--neg-typicality`, both with `--base-typicality` |
| 3 | New+fsx | `--force-same-x --validator-log-odds --nll_validator_weight 1 --nll_generator_weight 1` | `--self-typicality` and `--neg-typicality`, both with `--base-typicality` |
| 4 | New+fsx+tc | `--force-same-x --validator-log-odds --nll_validator_weight 1 --nll_generator_weight 1 --self-typicality` | `--self-typicality --base-typicality` |
| 5 | RankAlign+fsx+tc | `--force-same-x --self-typicality` | `--self-typicality --base-typicality` |
| 6 | RankAlign+tc | `--self-typicality` | `--self-typicality --base-typicality` |
| 7 | New+fsx+negtc | `--force-same-x --validator-log-odds --nll_validator_weight 1 --nll_generator_weight 1 --neg-typicality` | `--neg-typicality --base-typicality` |
| 8 | RankAlign+fsx+negtc | `--force-same-x --neg-typicality` | `--neg-typicality --base-typicality` |
| 9 | RankAlign+negtc | `--neg-typicality` | `--neg-typicality --base-typicality` |
| 10 | RankAlign+fsx | `--force-same-x` | `--self-typicality` and `--neg-typicality`, both with `--base-typicality` |

**Notes:**
- Settings #5 and #8 intentionally omit `--validator-log-odds` during training (bug fix vs. old gemma-2 runs where vlo was incorrectly included).
- All eval runs use `--base-typicality --base-model google/gemma-4-31B-it` (offline TC reference = frozen base).
- Settings #1, #2, #3, #10 (non-TC trained) are evaluated with both `--self-typicality` and `--neg-typicality` to get all eval columns.

## Diagnostic structure

- **Does TC alone help?** → #6 / #9 vs #2 vs #1
- **Does neg-TC alone help?** → #9 vs #6 vs #2
- **Does fsx alone help?** → #10 vs #2
- **Does comb+fsx help?** → #3 vs #2
- **Does TC help on top of comb+fsx?** → #4 vs #3 (self), #7 vs #3 (neg)
- **Does fsx add anything given TC?** → #6 vs #5 (self), #9 vs #8 (neg)
