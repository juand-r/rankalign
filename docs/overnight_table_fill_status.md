# Overnight eval-fill status (2026-06-08 → 09)

Goal: produce the **missing eval variants** (with-base AND without-base/own) for all
paper settings, to fill `docs/original_v7_results.tex`. **mll only, eval-only, no retrain.**

## Correction to earlier note
Checkpoints are NOT lost — they exist on mll:
- gemma ifeval: `models2/` (2) + `models2-rerun-wandb/` (6)
- qwen ifeval: `models2-rerun-wandb/` (5)
- qwen membership(rosch): `models2-rerun-wandb/` (5)
- gemma rosch: `models2/` (originals; already fully evaled in `v7_rosch_all_metrics`)

The gap is **eval variants**, not checkpoints: we don't have all of {PMI self, PMI base,
Neg self, Neg base} on all settings.

## Pipeline (working — same one that produced the qwen ifeval ID scores)
- **qwen**: `sbatch --export=ALL,EVAL_ONLY=1[,NO_BASE=1],IFEVAL_SPLIT=ood scripts/run_qwen35_cell_mll.sbatch <ifeval|membership> <sN>`
  - `NO_BASE=1` → own (`self-`/`neg-`); omit → base (`basetyp-`/`basetypneg-`).
  - EVAL_MODES auto by setting: s4=self only, s7=neg only, s1/s2/s3/s13=both.
  - scores → `outputs-rerun-wandb/`.
- **gemma ifeval own**: `scripts/run_rerun_9bit_ifeval_wandb.sh`.
- (NOT `_eval_only.sh` — that's gemma-only and has a documented ifeval no-op bug.)

## Eval matrix
- **qwen ifeval OOD**: own s1/s2*/s3/s4/s7 (`NO_BASE=1`); base s1/s3.  (*s2 = canary 44259)
- **qwen rosch (membership)**: own+base s1/s2/s3/s4; base s7 (neg-own already exists).
- **gemma ifeval OOD own**: s1/s2/s3/s4/s7.

## Status
- **Canary 44259** = qwen ifeval s2 own-OOD — verifying scores before batching the rest.
- s13 (jobs 44257/44258) = LOW priority (user); let finish, harvest if done.
- Supervised by CronCreate `65ff1c5f` (:11/:41 CT). Recompute must ADD `outputs-rerun-wandb`
  as a source. Commit/push from LAPTOP (mll repo behind origin).
