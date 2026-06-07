# TODO — retrain on mll to recover wandb training curves

**Created 2026-06-06.** Goal: get **wandb training curves** for the paper models that were
**pod-trained with `--no-wandb`** (so no curves exist) or whose only wandb runs are early
fixed-δ0.15 experiments (≠ the delta-bins paper models). We retrain those cells on **mll with
wandb ONLINE**. See `inventory-jun1-2026/WANDB_RUN_MAP.md` for the existing-coverage evidence.

## Already covered — do NOT retrain
- **gemma-2-9b-it × rosch/membership** — delta-bins finished wandb runs already exist for the
  May-24 mll batch (s2–s7, s11–s13). ✅ (Only exception: s1 is fixed-δ-only; optional.)

## Retrain matrix — 3 cells × 6 settings (s1, s2, s3, s4, s7, s13)

| # | Cell | Settings | Launcher |
|---|------|----------|----------|
| A | **ifeval × gemma-2-9b-it** | s1, s2, s3, s4, s7, s13 | `scripts/run_rerun_9bit_ifeval_wandb.sh` (being written) |
| B | **ifeval × qwen3.5-9b** | s1, s2, s3, s4, s7, s13 | `scripts/run_qwen35_cell_mll.sbatch` (+ add s3, s13) |
| C | **rosch × qwen3.5-9b** | s1, s2, s3, s4, s7, s13 | `scripts/run_qwen35_cell_mll.sbatch` (+ add s3, s13) |

### Per-cell setting checklist
**A · ifeval gemma-2-9b-it** — [ ] s1 [ ] s2 [ ] s3 [ ] s4 [ ] s7 [ ] s13
**B · ifeval qwen3.5-9b**   — [ ] s1 [ ] s2 [ ] s3 [ ] s4 [ ] s7 [ ] s13
**C · rosch qwen3.5-9b**    — [ ] s1 [ ] s2 [ ] s3 [ ] s4 [ ] s7 [ ] s13

## Conventions for the rerun (keep new runs distinguishable from old)
- **Models dir:** `MODELS_DIR=/datastor2/jdr/rankalign/models2-rerun-wandb` (separate from
  `models2/` so reruns don't collide with / overwrite the originals).
- **wandb:** ONLINE (mll is logged in → project `rankalign`). Tag/name each run so it's easy to
  filter new-vs-old on the cloud: pass **`--wandb_run_name`** with a `rerun-wandb-…` prefix, e.g.
  `--wandb_run_name "rerun-wandb-<model>-<task>-<setting>"` (the trainer accepts `--wandb_run_name`,
  `ranking_loss_ref_fix.py` L2409). NB the cell launchers don't pass it yet — add it.
- **Trainer:** `ranking_loss_ref_fix.py` (NEVER the obsolete `ranking_loss_ref_qwen.py`).
  qwen → `--lora` (not `--gemma4-lora`); venv `/datastor2/jdr/venvs/gemma4` + moe.py patch.
- **2 GPUs** (`gres=gpu:2`); ifeval adds `--max-seq-len 1024 --gradient_checkpointing`.

## Setting flags (for s3 + s13, which the qwen launcher doesn't have yet)
- **s3 New+fsx:** comb (`nllv1 nllg1 pref1`) + `--force-same-x --per-prompt-delta --shape-budget-mode global --validator-log-odds`, `--semi-supervised 0.1`, no TC. delta-bins 10.
- **s13 SFT+cft:** `--preference_loss_weight 0 --nll_validator_weight 1 --nll_generator_weight 1 --consistency-ft --labeled-only 0.1`, no fsx, no TC, no vlo. (NB s13 pod runs used **fixed δ0.15, 1 epoch**; decide whether the rerun matches that or uses delta-bins/3-epoch — flag for the paper.)
- s1/s2/s4/s7 already in `run_qwen35_cell_mll.sbatch`.

## Qwen mll venv — READY (2026-06-06)
The qwen mll pipeline is proven. The Qwen3.5 gated-delta-rule attention is slow on the gemma4
venv (torch fallback, ~11.5 s/it on 2x A40); a dedicated **`/datastor2/jdr/venvs/qwen35`** venv
(gemma4 clone + `fla-core==0.5.0` + `triton==3.3.0` + `bitsandbytes==0.49.2`) gets the fla fast
kernel → **~4 s/it (~2.9x), ~17 h per 3-epoch run** (canary job 43931). Build/repro:
**`setup-mll-qwen35-fla.sh`**. `run_qwen35_cell_mll.sbatch` already defaults to this venv.
**Submit qwen reruns with `--time=24:00:00`** (17 h fits; 2 h does NOT).

## Order / status
- [ ] **A (ifeval gemma-2-9b-it)** — user driving via `run_rerun_9bit_ifeval_wandb.sh` (jobs 43886-43901, running).
- [x] **B + C (qwen ifeval + rosch) LAUNCHED 2026-06-06 23:18 CT** — all 10 cells (s1/s2/s3/s4/s7 × ifeval/membership,
      **s13 SKIPPED per user**), `TRAIN_ONLY=1`, `--time=24:00:00`. Jobs `qwrr-ifeval-s{1,2,3,4,7}` =
      43933-43937, `qwrr-membership-s{1,2,3,4,7}` = 43938-43941 + 43943 (s7 resubmit after a transient
      concurrent-download race). All RUNNING at ~4 s/it. wandb: `rerun-wandb-qwen3.5-9b-*` on juand-r/rankalign.
      Models → `models2-rerun-wandb/`. **Eval skipped (TRAIN_ONLY)** — run `EVAL_ONLY=1` later if scores wanted.
- [x] `s3` + `s13` cases added to `run_qwen35_cell_mll.sbatch`.

### Supervision (live)
- **Backbone:** laptop crontab `:07,:37` → `~/.claude/training_monitor/tick_qwen35_rerun.sh` → mll supervisor
  `scripts/monitor_qwen35_rerun.sh` (resubmits FAILED/TIMEOUT/NODE_FAIL, cap 3; writes `.monitor/ALL_DONE.flag`).
- **In-session:** CronCreate heartbeats `a11e2da7` (:12,:42), `9d1f926f` (:27,:57).
- **TEARDOWN when done:** `crontab -l | grep -v tick_qwen35_rerun | crontab -` + CronDelete both heartbeats.

## Caveat
Reruns are **fresh trainings** (seed 42 + same code/data → near-identical to the uploaded
checkpoints, but not bit-identical). Fine for recovering curves; flag if the paper needs the exact
original weights' curves.
