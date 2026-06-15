# gemma-4-31B-it s2/s4 on mll (humaneval-v2.1correct-upper) — overnight status

**Goal (by morning):** s2 (RankAlign) + s4 (FLORA-PMI / New+fsx+tc) gemma-4-31B-it
trained on mll **with WANDB curves**, then evaluated → **6 score-sets**:
- s2: {self, neg} × {base-typ (bt1), no-base (bt0)} = 4
- s4: {self} × {bt1, bt0} = 2

Each score-set = `scores_*` CSVs for **all 82** `humaneval-v2.1correct-upper` test tasks.
Plus: summarized metrics CSVs + a summary LaTeX table compiled to PDF.

## Config (verified)
- Trainer: `ranking_loss_ref_fix.py --gemma4-lora --lora --delta-bins 10` (bug-fixed; the others lack fixes).
- venv `/datastor2/jdr/venvs/gemma4` (torch 2.5.1, transformers 5.8.0.dev0, peft 0.14).
- Slurm: **allnodes** (raca default), 9 CPU / 96G / **4×A40** (gpu:4 for eval-bt1 headroom). NOT p-gdurret.
- **Nothing in $HOME** (20G quota): HF cache `/datastor2/jdr/.cache/huggingface`; HF token `/datastor2/jdr/.hftoken`; WANDB key from `/u/jdr/.bashrc` (online, project `rankalign`).
- Models → `/datastor2/jdr/rankalign/gemma-4-models-mll-tmp/`; scores → `/datastor2/jdr/rankalign/outputs_gemma4_mll_tmp/`.
- Launcher (committed): `pod-setup-train-scripts-gemma-4/mll_g4_train_eval.sbatch <2|4>` (`SMOKE=1` for the smoke test).

## Plan / progress (updated 2026-06-13 ~05:10 UTC)
1. [done] Verified env, keys, model cache, repo, tasks, GPUs.
2. [done] **SMOKE** validated the hard parts: 31B loads across 4×A40, **LoRA 122M/31.4B (--gemma4-lora) trains @ ~4.7 s/it, no OOM, WANDB ONLINE** (juand-r/rankalign). Caught + fixed one bug: s2 adapter glob was missing `--fix1` (pod recipe used the old v6 name; s2 was never retrained with fix.py there) — committed.
3. [done] **Launched REAL s2 (job 44939) + s4 (44940)** — 3 epochs, 5110 samples, all 82 tasks, real dirs, wandb online. Both RUNNING on node-002/003.
4. [done] **EVAL path validated**: s2 smoke produced a real `scores_basetyp-…fix1_…humaneval_1_…csv` (28 rows: correct y/n + val_score + gen_score), base-typ 2-model load, **no OOM** on 4×A40. Pipeline is green end-to-end (train→save→eval→scores).
5. [pending] When training done (~ETA below): parallel per-combo eval (6 jobs) → scores_ for all 82 tasks.
6. [pending] Harvest: scores_ → metrics CSVs → summary LaTeX → PDF.

## TIMING REALITY (important)
Training measured at **~4.7 s/it**; 5110×3 ≈ 15,330 steps → **~20h training per model**, + eval.
It is ~05:10 UTC; morning is ~9h out. **Full 3-epoch train+eval will NOT all finish by morning.**
What you WILL have by morning: both models partway trained (~1.3 epochs) with **live wandb curves**
(the stated priority), the validated pipeline, and honest ETAs. Not cutting epochs without your OK.

## Notes / risks
- gemma-4-31B is large; 3 epochs × 5110 + 82-task eval × combos is **many hours** — may not all finish by morning. Will report honest ETAs + partials. Eval is per-task (model reload each) so it's the slow part; investigating batching.
- Supervision: recurring cron `71e0112e` (every :08/:38) advances/monitors. Hardened bg monitors used for decision points.

_(updated through the night)_

## 2026-06-13 05:46 UTC — training healthy, smokes done
- Smokes COMPLETE: validated all 4 eval prefixes (basetyp/basetypneg/self/neg), s2 + s4 adapter naming, no OOM.
- Real training in **epoch 1**: s2 (44939) ~387/5110 @4.65s/it (~6h/ep); s4 (44940) ~166/5110 @5.4s/it (~7h/ep).
- ETA: s2 ~18h, s4 ~21h for 3 epochs. epoch checkpoints save each ~6-7h. Eval (parallel, 6 jobs) after final epoch.
- Only 2 real jobs in queue; logs now clean (smoke clobbering ended).

## 2026-06-13 ~10:50 CDT — RE-LAUNCH after power outage + added correct-MULTI
Power outage (~4:45 AM CDT) killed the original run mid-epoch-1 (no checkpoint); nothing usable survived. Re-launched fresh, now BOTH tasks:
- **correct-UPPER:** train 44957(s2)/44958(s4); dependent eval 44959-44962(s2)/44963-44964(s4). dirs: gemma-4-models-mll-tmp / outputs_gemma4_mll_tmp. wandb g4-cu-s2/s4.
- **correct-MULTI:** train 44965(s2)/44966(s4); dependent eval 44967-44970(s2)/44971-44972(s4). dirs: *-multi. wandb g4-cm-s2/s4.
- Structure: TRAIN-ONLY jobs (NO_EVAL=1) + parallel per-combo eval jobs with --dependency=afterok (eval auto-runs after each train succeeds). 4 train RUNNING, 12 eval PENDING(Dependency).
- Monitor: in-session hourly cron 25cebb68 (no-hammer safeguards; self-harvests + self-deletes when all 12 eval done). NOT a system crontab.
- ETA ~20h train + ~5h parallel eval. 6 score-sets per task (s2:4, s4:2) x 82 tasks.
