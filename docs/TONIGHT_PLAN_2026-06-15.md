# Tonight's plan — 2026-06-15 (autonomous, user asleep)

Two independent runs on mll. **Goal: results by morning.**

## A) QWEN-3.5-9B HumanEval (primary task tonight)
Train+eval Qwen/Qwen3.5-9B, settings s2 (RankAlign) + s4 (New+fsx+self-tc), on
humaneval-v2.1correct-{upper,multi} = 4 jobs. Outputs -> models2-rerun-wandb /
outputs-rerun-wandb. Launcher: `scripts/mll_qwen35_he_train_eval.sbatch` (calls the
ISOLATED trainer copy `ranking_loss_ref_fix_genctx.py`; shared trainer untouched).

### State at handoff
- Blocker = generator sequence longer than the disc-derived `max_context_length`, which
  truncates the completion tail and the trainer's `_check_tail` guard aborts.
- Fix = add a fixed MARGIN to `max_context_length` (pure padding headroom; drops nothing;
  guard still backstops). `_GEN_CTX_MARGIN` in the copy.
- Tried: +10 (smoke 44982) FAILED (still truncated). Now at **+128**, smoke **44983** running.

### Step 1 — get a CLEAN smoke
When the current smoke finishes:
- **CLEAN** (trains past the truncation, produces scores_ CSV with val_score+gen_score,
  0 NaN, correct has yes&no) -> go to Step 2.
- **tail-mismatch again** (margin still too small) -> **ESCALATE the margin and re-smoke**:
  128 -> 256 -> 512 -> 1024 (double each time). Edit `_GEN_CTX_MARGIN` in the local copy,
  commit, push, upload to mll, resubmit smoke. Repeat until CLEAN. (User: "I need results —
  if 128 fails, increase to 256", etc.) Only STOP if even 1024 fails (that would not be a
  margin problem — report it).
- **other error** -> diagnose; fix the launcher/copy if clearly safe and re-smoke, else report.

### Step 2 — launch the 4 FULL jobs (once smoke is CLEAN)
`cd /datastor2/jdr/rankalign` then:
```
sbatch --job-name=q35cu-s2 --export=ALL scripts/mll_qwen35_he_train_eval.sbatch 2
sbatch --job-name=q35cu-s4 --export=ALL scripts/mll_qwen35_he_train_eval.sbatch 4
sbatch --job-name=q35cm-s2 --export=ALL,HE_TASK=humaneval-v2.1correct-multi scripts/mll_qwen35_he_train_eval.sbatch 2
sbatch --job-name=q35cm-s4 --export=ALL,HE_TASK=humaneval-v2.1correct-multi scripts/mll_qwen35_he_train_eval.sbatch 4
touch /datastor2/jdr/rankalign/.q35_full_launched
```
(marker prevents double-launch). 2x A40 each, 3 epochs, wandb online (q35-cu/cm-s2/s4).
Each job trains then inline-evals (mode x {bt1,bt0} x 82 tasks) with done-markers (resumable).

### Step 3 — monitor + harvest
- Monitor each ~hourly. If a cell dies (gone from squeue, no epoch2 merged dir), read its
  log, resubmit that one cell.
- Expected scores when all done = **984** (s2: 4 combos x82=328/dataset; s4: 2x82=164/dataset;
  upper 328+164, multi 328+164).
- When ALL 4 jobs gone + scores plateaued -> **HARVEST (task #26)**:
  `scripts/summarize_scores_file.py` over the scores_ -> metrics CSVs -> LaTeX tables with
  **Pearson corr, gen ROC, val acc, val ROC** -> compile PDF under docs/ -> commit+push -> alert.
  Then delete the qwen cron.

## B) GEMMA-4-31B-it HumanEval (already running, separate)
- 4 train jobs 44957(cu-s2)/44958(cu-s4)/44965(cm-s2)/44966(cm-s4); 12 dependent eval jobs
  44959-44964, 44967-44972. Dirs: gemma-4-models-mll-tmp{,-multi} / outputs_gemma4_mll_tmp{,-multi}.
- s2 trainings DONE, their evals streaming (259 upper + 43 multi scores at handoff). s4
  trainings on epoch2, finishing in a few hours -> their evals release on afterok.
- Monitor = cron `25cebb68` (hourly). Harvest when all 12 eval jobs clear; then delete that cron.

## Invariants / guardrails
- Shared `ranking_loss_ref_fix.py` stays BYTE-IDENTICAL. Only the isolated copy carries the margin.
- The margin is pure padding -> NO effect on results/science; the `_check_tail` guard guarantees
  no silent truncation at any margin value.
- qwen cron only touches q35* jobs; gemma cron only touches 44957-44972. Don't cross.
- NO SSH hammering: on auth/publickey failure, `raca auth mll` AT MOST ONCE, then back off.
- Commit + push every change.

## Crons / pollers active
- `8807a109` — qwen driver (will be replaced with escalation logic), every :09/:29/:49.
- `25cebb68` — gemma hourly monitor.
- background poll watching the current qwen smoke (re-invokes faster than the cron).
