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
4. [in progress] Validating the **eval path** via s2 smoke eval-only (job 44941) — first scores_ file or OOM check.
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
