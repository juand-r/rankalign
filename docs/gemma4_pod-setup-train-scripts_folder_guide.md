# What's in `pod-setup-train-scripts-gemma-4/`

**TL;DR — the folder name is misleading.** It is *not* only pod scripts. It holds the
gemma‑4‑31B‑it humaneval training/eval tooling, and over time it accumulated **two
different kinds** of runner: original **RunPod pod** scripts *and* **mll slurm** sbatch
launchers. They look similar but run in completely different places. The actual training
algorithm is **not** in this folder.

This note exists so editing one of these files isn't alarming: an `mll_*.sbatch` is a
cluster launcher, safe to edit for mll runs — it is **not** the script that ran the
original pods.

## The three categories

### 1. mll slurm launchers (`#SBATCH`, run on the mll cluster)
These are sbatch wrappers submitted with `sbatch ... .sbatch`. Editing them affects
**mll** runs only.
- `mll_g4_train_eval.sbatch` — train + inline eval ONE gemma‑4 setting on humaneval (the
  s2/s4 **wandb reruns on mll** used this; now extended to s1/s3/s4/s7/s13). Calls the
  genctx trainer copy.
- `mll_train_arm.sbatch`, `mll_eval_base.sbatch`, `mll_smoke_fix1.sbatch`,
  `mll_g4_parallel_eval.sh`, `recover_gemma_cm_s4_basetyp.sbatch` — other mll train/eval/
  smoke/recovery wrappers.

### 2. RunPod pod scripts (`.sh`, run *inside* a pod over SSH)
These bootstrap a pod and run the original gemma‑4 humaneval TC runs on RunPod. They are
the historical pod recipes — leave them as a record unless deliberately re-running pods.
- `bootstrap_fresh_pod.sh`, `bootstrap_v21correct_upper.sh`, `bootstrap_v21correct_multi.sh`
- `run_settings_v21correct_upper.sh`, `run_settings_v21correct_multi.sh`, `run_arm_3epoch.sh`
- `real_pipeline_g4it.sh`, `sequential_pipeline.sh`, `recovery_pipeline.sh`,
  `recovery_pipeline_v2.sh`, `train_seq_v21cu.sh`, `eval_base_multi.sh`,
  `download_scores_s3s4s7.sh`, `watch_recovery.sh`

### 3. Helpers / docs
- `merge_no_tc.py` — merge a no‑TC LoRA adapter into a full model.
- `README.md` — original folder readme (pod‑centric).
- `TRAINING_PLAN_v21correct_multi.md` — the multi training plan.

## Where the actual trainer lives (NOT in this folder)
- `scripts/ranking_loss_ref_fix.py` — the canonical RankAlign/FLORA trainer. **Never
  edited for the humaneval BPE‑seam fix.**
- `scripts/ranking_loss_ref_fix_genctx.py` — an isolated **copy** of the above with a
  pair‑level drop filter for the ~6 humaneval items whose prompt→completion tokenization
  is inconsistent. The mll gemma/qwen humaneval launchers call this copy; the original is
  untouched. See `docs/qwen_humaneval_genctx_fix.md`.

## Rule of thumb
- `mll_*.sbatch` → a cluster launcher; safe to edit for mll work.
- everything else `.sh` here → a pod recipe / historical record; don't edit without intent.
- the training algorithm is in `scripts/`, never here.
