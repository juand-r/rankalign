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

## Provenance policy (user-approved 2026-06-08, last word before sleep)
Prefer ORIGINAL checkpoints where available; use RERUN where originals are lost; **label
provenance per cell**.
- **Original on HF `latkes`** (downloading → `models2-original-hf/`): qwen ifeval **s1, s2**;
  qwen rosch(membership) **s1, s7** (ep2). gemma ifeval: 2 originals in `models2/`.
  (TAUR-dev v7 gemma-2-9b-it repos not visible — likely re-privatized; retry w/ token.)
- **Rerun** (`models2-rerun-wandb/`): all qwen + gemma ifeval settings — fallback for the rest.

## Status
- **Canary 44259** = qwen ifeval s2 own-OOD (rerun) — verifying before batching.
- **HF original download** launched → `models2-original-hf/` (4 qwen ep2 models); marker
  `.dl_orig_done`. Eval these for original provenance (supersede rerun for s1/s2 ifeval, s1/s7 rosch).
- s13 (jobs 44257/44258) = LOW priority; harvest if done, don't chase.
- Supervised by CronCreate **`979ef49d`** (:11/:41 CT). Recompute must ADD `outputs-rerun-wandb`.
  Commit/push from LAPTOP (mll repo behind origin; fetch+rebase first).

## Task #24 (after original table done)
Eval the RERUN checkpoints for any remaining missing variants and build a **parallel rerun
LaTeX table** — original-vs-rerun comparison (how different are two independent trains per setting).

## Launched jobs (2026-06-08 ~21:1x CT) — qwen rerun matrix, → outputs-rerun-wandb
Canary verified VALID (qwen ifeval s2 own-OOD, 80-row scores, healthy tc gen_roc).
- qwen ifeval OWN (self-/neg-, OOD): 44260=s1 44261=s3 44262=s4(self) 44263=s7(neg)  [s2=canary 44259]
- qwen ifeval BASE (OOD):            44264=s1 44265=s3   [s2/s4/s7 base already on disk]
- qwen rosch OWN:                    44266=s1 44267=s2 44268=s3 44269=s4   [s7 neg-own exists]
- qwen rosch BASE:                   44270=s1 44271=s2 44272=s3 44273=s4 44274=s7
TODO next fires: gemma ifeval OOD own (run_rerun_9bit_ifeval_wandb.sh, s1-s7);
ORIGINAL HF qwen evals (models2-original-hf: ifeval-s1/s2, membership-s1/s7) — supersede
rerun for those cells with provenance=orig. Then recompute (+outputs-rerun-wandb) + rebuild table.
