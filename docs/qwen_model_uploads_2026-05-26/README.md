# Qwen3.5-9B RankAlign models — provenance & artifacts (2026-05-26)

This folder documents three **Qwen3.5-9B** RankAlign-trained models that were produced on
RunPod pods, evaluated, and then uploaded to HuggingFace (org **`latkes`**). They had been
trained with `--no-upload-hf`, so they originally lived only on pod-local disk; this folder
is the durable record of what they are and how to reproduce them.

> If you are reading this later and wondering "what are these qwen models and where did they
> come from" — this is the answer. The merged weights are on HF (links below); this folder
> holds the training logs, LoRA adapter configs, and notes.

## The three models

| # | Model (HF repo on `latkes`) | Task | Setting | Δ (delta) | Epochs | Source dir name (on pod) |
|---|---|---|---|---|---|---|
| 1 | `latkes/rankalign-v7-qwen3.5-9b-ifeval-s1-ep2` | ifeval-concat | **s1** (SFT label-only) | 0.84 (delta-bins) | 3 (ep2 = final) | `v7-Qwen--Qwen3.5-9B-delta0.84-epoch2--ifeval-concat-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1_merged` |
| 2 | `latkes/rankalign-v7-qwen3.5-9b-ifeval-s13-ep0` | ifeval-concat | **s13** (SFT + consistency-FT) | 0.15 (fixed) | 1 (ep0 = final) | `v7-Qwen--Qwen3.5-9B-delta0.15-epoch0--ifeval-concat-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1_merged` |
| 3 | `latkes/rankalign-v7-qwen3.5-9b-membership-s13-ep0` | membership-sans-rosch-v0 | **s13** (SFT + consistency-FT) | 0.15 (fixed) | 1 (ep0 = final) | `v7-Qwen--Qwen3.5-9B-delta0.15-epoch0--membership-sans-rosch-v0-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--cft--labelonly0.1--fix1_merged` |

**Epoch labels are 0-indexed checkpoint indices.** `ep2` = the 3rd (final) epoch of a 3-epoch run;
`ep0` = the 1st (and only) epoch of a 1-epoch run. The two s13 models were trained for **1 epoch**
(`EPOCHS=1`), so `epoch0` is their final/used checkpoint.

A 4th model from the same eval batch — **qwen membership-s1** (rosch-s1 eval, delta 1.54) — was
already on HF at `latkes/rankalign-v7-qwen3.5-9b-membership-s1-ep2` and is **not** re-uploaded here.

## What "s1" and "s13" mean

These are RankAlign training-setting codes (the same scheme as the gemma-2-9b-it runs):

- **s1 = SFT label-only.** Supervised fine-tuning on labeled examples only; no preference/ranking
  loss. Flags: `--nll_validator_weight 1 --nll_generator_weight 1 --preference_loss_weight 0
  --labeled-only 0.1 --validator-log-odds`.
- **s13 = SFT + consistency fine-tuning (CFT).** Label-only SFT plus a consistency-FT objective.
  Flags: `--preference_loss_weight 0 --nll_validator_weight 1 --nll_generator_weight 1
  --consistency-ft --labeled-only 0.1` (no force-same-x, no typicality term, no validator-log-odds).
  This is reflected in the `cft` token in the dir name.

Only **s2** in this project is "RankAlign" proper; s1/s13 are baselines/variants. (Naming kept for
consistency with the on-disk dirs and sibling HF repos.)

## LoRA / training configuration

All three are LoRA fine-tunes of base **`Qwen/Qwen3.5-9B`** (merged into full weights for the HF
upload). From `adapter_config.json` (see `adapter_meta/`):

- `r = 16`, `lora_alpha = 32`, `lora_dropout = 0.1`, `bias = none`
- target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- `task_type = CAUSAL_LM`, PEFT 0.14.0
- delta-bins: 10 (the per-bin delta scheme); s1 used a delta-bins value of 0.84, the s13 runs used
  fixed delta 0.15.

The raw LoRA adapters (`adapter_model.safetensors`, ~116 MB each) are **not** stored in git — they
travel inside the HF model repos alongside the merged weights. Only the `adapter_config.json` and
the (empty template) adapter `README.md` are kept here for quick reference.

## How to reproduce

Training and eval were driven by committed cell scripts in `private_projects/rankalign/scripts/`:

- **s1 (ifeval):** `scripts/run_qwen35_cell.sh ifeval s1`
- **s13 (ifeval):** `EPOCHS=1 scripts/run_qwen35_v7b_cell.sh ifeval s13`
- **s13 (membership):** `EPOCHS=1 scripts/run_qwen35_v7b_cell.sh membership s13`
  - (the two s13 cells were run together via `scripts/run_qwen35_v7b_s13_both.sh`, `EPOCHS=1`)

The cell scripts encode the exact per-setting loss flags, the LoRA config, `--delta` / `--delta-bins`,
and the merge step. The python training command itself is **not** echoed verbatim in the logs (the
scripts don't `set -x`), so the scripts are the source of truth for the exact invocation. Base model:
`Qwen/Qwen3.5-9B`. Models dir on pod: `/workspace/models_q35/`.

Eval was run with **NO_BASE=1** (drops `--base-typicality` for ~2x speed; produces `self-`/`neg-`
scored CSVs only, no PMI-base columns).

## Evaluation results (for reference)

Scored eval CSVs are committed in the repo:
- ifeval (held-out test prompts 1–21, n=20 — prompt 14 was a data gap):
  `outputs_gemma4_from_pod-v7b/qw35_ifeval/scores_{self,neg}-eval_model_s{1,13}_ifeval-prompt_*.csv`
- rosch (10 membership categories, n=10):
  `outputs_gemma4_from_pod-v7b/qw35_persona_member/scores_{self,neg}-eval_model_s13_rosch-*.csv`

Summary metrics (×100, mean ± SE; computed via `scripts/summarize_scores_file.py`):

**ifeval-s1** (self eval, n=20): gen_roc Raw 52.0±4.5 / TC(self) 73.1±3.4 · val_roc 57.5±3.5 · val_acc 44.7±3.2

**ifeval-s13** (n=20): gen_roc Raw 52.5±4.6 / TC(self) 76.3±3.0 / TC(neg) 50.3±3.7 · val_roc 54.1±3.5 · val_acc 48.3±3.4

**rosch-s13** (n=10): gen_roc Raw 77.7±2.0 / TC(self) 80.1±1.9 / TC(neg) 86.1±2.8 · val_roc 95.0±2.0 · val_acc 84.3±3.3

Note: for ifeval, self-typicality correction lifts gen_roc sharply while neg-typicality does not;
for rosch (membership), neg-typicality helps *more* than self. See git history around 2026-05-26.

## Files in this folder

```
README.md                         <- this file
logs/
  cell_ifeval_s1.log.gz           <- training+eval log, ifeval-s1 (pod 64.247.201.47:11402)
  cell_ifeval_s13.log.gz          <- training+eval log, ifeval-s13 (pod xo5ntpx2v4qzce)
  cell_membership_s13.log.gz      <- training+eval log, membership-s13 (pod xo5ntpx2v4qzce)
  qwen_s13_both.log.gz            <- wrapper log covering both s13 cells
adapter_meta/
  ifeval-s1/      adapter_config.json + adapter README (PEFT template)
  ifeval-s13/     adapter_config.json + adapter README (PEFT template)
  membership-s13/ adapter_config.json + adapter README (PEFT template)
```

## Provenance summary

- **Trained:** 2026-05-25 → 2026-05-26 on RunPod (TAUR + personal accounts).
- **Pods:** ifeval-s1 on `64.247.201.47:11402`; both s13 models on `xo5ntpx2v4qzce` (TAUR).
- **Why uploaded late:** training used `--no-upload-hf`; the merged weights existed only on pod-local
  disk until this upload. The s13 models were trained the night of the paper deadline.
- **Upload target:** `latkes` org (TAUR-dev private storage was full at the time).
