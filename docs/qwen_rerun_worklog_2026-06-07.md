# Qwen3.5-9B wandb-rerun + corrections — work log (2026-06-07/08)

Record of everything done in the Qwen3.5-9B rerun session: the wandb-curve reruns, three
bugs found and fixed, the table-builder changes, the vlo investigation, the base re-eval, and
the generated LaTeX tables. Companion to [`disc_shots_membership_bug_2026-06-07.md`](disc_shots_membership_bug_2026-06-07.md).

All paths are on **mll** under `/datastor2/jdr/rankalign/` unless noted. Branch: `longform`.

## 0. Goal
Recover **wandb training curves** for Qwen3.5-9B on **membership/rosch (hyponymy)** and **ifeval**
(pod-trained originals used `--no-wandb`). Launcher: `scripts/run_qwen35_cell_mll.sbatch`
(mll, 2× A40, venv `/datastor2/jdr/venvs/qwen35` = gemma4 clone + flash-linear-attention).
Settings s1/s2/s3/s4/s7 (**s13 skipped** for qwen). Models → `models2-rerun-wandb/`.

## 1. Bug — disc-shots ZERO vs FEW (membership)
- **Found:** original qwen membership runs (and the first rerun batch) used `disc_shots=zero`;
  gemma-2-9b-it membership uses **few**. Verified from the bundled `training_log.log.gz` in
  `latkes/rankalign-v7-qwen3.5-9b-membership-s7-ep2` (`gen_shots: zero, disc_shots: zero`).
  The rule (confirmed empirically across all eval logs): **few for disc everywhere EXCEPT
  ifeval** (ifeval has no few-shot impl → zero is correct for all models).
- **Fix:** `run_qwen35_cell_mll.sbatch` now sets `DISC_SHOTS` per task (membership/persona=few,
  ifeval=zero) and threads it into **both** train and eval calls.
- **Action:** archived the 30 wrong (zero) membership dirs →
  `models2-rerun-wandb-WRONG-disczero-membership/`; **retrained + re-evaled** the 5 membership
  cells with `disc_shots=few` (`Overriding disc_shots to: few` validated). mll jobs 43949–43953.

## 2. Bug — scores_ filenames lost provenance
- **Found:** the launcher symlinked each eval model to a generic `eval_model_qw_<setting>`, so
  scores files were `scores_<mode>-eval_model_qw_s2_<task>_...csv` — losing model/delta/epoch/
  method and breaking `checkpoint_name_parser` / `_build_*_table_v7.py`.
- **Fix:** launcher now names the eval symlink via
  `to_hf_repo_name(parse_checkpoint_name(<model dir basename>), prefix='')` →
  `v7-Qwen3.5-9B-d<delta>-e<epoch>-<task>-<method>-fix1` (the canonical abbreviated form, <255 ch).
  Renamed already-produced files in place with `scripts/_rename_qwen_rerun_scores.py`.

## 3. Outputs isolation
- Rerun eval scores were going to the canonical `outputs/`. Moved them to
  **`outputs-rerun-wandb/`** (alongside cell-A gemma ifeval reruns) so they never overwrite the
  canonical scores. Launcher now defaults `OUTPUTS_DIR=outputs-rerun-wandb` (env-overridable).
  No overwrite occurred (rerun filenames carry today's date + distinct model field).

## 4. Table builders generalized + vlo handling
- `_build_rosch_table_v7.py` / `_build_ifeval_table_v7.py`: replaced the hardcoded
  `gemma-2-{MODEL}` with a **`MODEL_REGISTRY`** (key → model_short, base token); added
  `qwen3.5-9b`. Deleted the redundant `_build_qwen_table_v7.py`.
- **vlo finding:** the **SFT (s1)** and **RankAlign (s2)** rows are defined as **no-vlo**. All
  gemma SFT/RankAlign models (every size, both tasks) are `--full-completion--semi0.1--fix1` =
  **no vlo**; the qwen reruns (and originals — all qwen launchers set `VLO_FLAG` for s1/s2) are
  `--vallogodds--...` = **with vlo**. So the matcher correctly rejected the qwen SFT/RankAlign.
- **Does vlo matter?** In `ranking_loss_ref_fix.py`, the validator score that drives preference
  **pair selection** is `log P(Yes) − log P(No)` (vlo) vs `log P(Yes)` (no-vlo): pairs are formed
  by sorting on it and thresholding `(v_j−v_i)>delta`. In the abstract these differ — BUT
  **empirically `P(Yes)+P(No)≈1`**, so `log-odds = logit(P(Yes))` is **monotone** in `P(Yes)`,
  the sort order is preserved, and the effect is **negligible** (only marginal boundary pairs).
  **Decision (user):** do NOT retrain; instead make the builders **accept SFT/RankAlign with or
  without vlo** (`vallogodds=None` skips the check in `matches_v7_setting`) and **label the
  variant `+vlo`**. gemma cells stay plain; qwen show `+vlo`.

## 5. Base row — zero-disc → few re-eval
- The qwen Base row (`v6-Qwen_Qwen3.5-9B`, rosch) was the old **zero-disc** May-25 eval,
  inconsistent with the few-trained rows on ρ/ROC_V/Acc_V (ROC_G is disc-independent).
- **Fix:** `scripts/run_qwen_base_rosch_eval.sh` re-evals Qwen base on 10 rosch tasks × {self,neg}
  with `--disc-shots few --validator-log-odds` → `outputs-rerun-wandb/` (newest-wins dedup
  supersedes the old). mll job 43984; `disc_shots: few` validated (20/20). New Base row matches
  the paper (ROC_G 73.6 vs 73.5; ρ_self 45.6 vs 42.9).

## 6. LaTeX tables
- `scripts/_build_paper_tables_latex.py` reads the per-metric `*_table_cells.csv`
  (rosch → `metrics-from-scores/`, ifeval → `metrics-from-scores-rerun-wandb/`) and emits
  **`docs/qwen_rerun_tables.tex`**: (1) ROC_G + ρ with **self-TC and neg-TC as separate rows**
  (unlike `tab:main-results-multi`, which reports per-task best); (2) ROC_V + Acc_V (eval-TC
  independent → one row). Notes: `†` incomplete cells (every IFEval cell is 20/21 — `ifeval-
  prompt_14` data file missing), `---` not-run (qwen has no Consistency-FT/s13), `+vlo` label.
  **Epoch:** all cells **epoch 2**; the paper reports **RankAlign at epoch 1** → those rows are
  not epoch-matched. (Confirmed with user: we always use epoch 2.)

## Key jobs / paths / commits
- mll jobs: membership 43949–43953; ifeval EVAL_ONLY 43973/43974/43981/43982/43983; base 43984.
- Models: `models2-rerun-wandb/v7-Qwen--Qwen3.5-9B-*epoch{0,1,2}*[_merged]`; wrong-zero archive
  `models2-rerun-wandb-WRONG-disczero-membership/`.
- Scores: `outputs-rerun-wandb/scores_*v7-Qwen3.5-9B-*` (trained) + `*v6-Qwen_Qwen3.5-9B*` (base).
- wandb: `wandb.ai/juand-r/rankalign`, runs `rerun-wandb-qwen3.5-9b-*` (+`-discfew` for the
  corrected membership batch).
- Scripts added/changed: `run_qwen35_cell_mll.sbatch`, `monitor_qwen35_rerun.sh`,
  `_rename_qwen_rerun_scores.py`, `run_qwen_base_rosch_eval.sh`, `_build_paper_tables_latex.py`,
  `_build_{rosch,ifeval}_table_v7.py`, `src/checkpoint_name_parser.py` (deleted
  `_build_qwen_table_v7.py`).

## Still open (optional)
- HF upload of the corrected qwen membership models + scores, superseding the old zero-disc
  `latkes/rankalign-v7-qwen3.5-9b-membership-*` repos.
- The original qwen ifeval base is zero-shot — **correct for ifeval** (no action).
