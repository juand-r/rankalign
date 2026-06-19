# HumanEval training-dynamics report — section-by-section plan

Goal: reproduce **every** section of the original `analysis/report.tex` (membership/IFEval,
gemma-2-9b-it + qwen-3.5-9b) for **HumanEval**, comparing **upper vs multi** across
**gemma-4-31b + qwen-3.5-9b**. This plan maps each original section to its generating script,
the data the new version needs, what we have, and status.

## Data availability (the binding constraints)

| Data | Status |
|---|---|
| TEST per-problem metrics (gen_roc/val_roc/val_acc/pearson/spearman) | **HAVE** — `docs/he_harvest_2026-06-17/*` (all settings, both models, both ds) |
| TEST raw **per-candidate** scores (gen+val per solution) | on mll: `outputs_gemma4_mll_tmp[-multi]/`, `outputs-rerun-wandb/` — **needs pull** |
| TRAIN per-problem metrics | gemma s2/s4 **HAVE** (`docs/he_trainset_2026-06-19/`); gemma s3/s7 + all qwen = **trs2 in flight** |
| TRAIN raw per-candidate scores | gemma s2/s4 on mll (`outputs-he-trainset/`); rest = trs2 in flight |
| WandB logs (loss curves, components, score evolution) | **HAVE** — all runs online, project `rankalign`, runs `g4-{cu,cm}-s{S}` / `q35-{cu,cm}-s{S}` — **needs pull** |
| Held-out val loss (pref/NLL-G/NLL-V per ckpt) | **NOT computed** for HE — needs new `compute_val_loss.py` jobs (decision needed) |
| Per-category structure | **N/A** for HumanEval (no categories) — substitute per-problem / per-`strategy` |

Status legend: **NOW** = buildable from data on the laptop; **PULL** = needs a fetch/compute
from mll but no new GPU jobs; **trs2** = waiting on the running train-set batch; **JOBS** =
needs new GPU compute; **ADAPT/N-A** = needs a HumanEval-specific substitute.

---

## Section-by-section

### §1 Q1: When Does Each Component Help?
- **1.1 Main comparison table** (gen ROC tc). Orig script: `analyze_all_combos.py`.
  New: built by `build_he_report_tables.py` → `he_q1_test.tex`. **DONE (test).**
  Train counterpart: gemma s2/s4 now; full grid pending **trs2**.
- **1.2 Key findings** (prose). The upper-vs-multi flip, gemma-specific. **DONE (test).**
- **1.3 Visualization** `unified_genroc_comparison.png` (bar chart per setting×model×ds).
  New: adapt `analyze_all_combos.py` plotting to HE aggregated CSVs. **NOW** (test) / trs2 (train).

### §2 Q2: Training Diagnostics
- **2.1 WandB loss curves** (`wandb_loss_methods_*`, `wandb_loss_tc_*`). Orig:
  `analyze_wandb_training.py`. New: pull runs `g4-/q35-{cu,cm}-s{2,3,4,7,1,13}` from project
  `rankalign`; plot methods (SFT/RA/Ours) + TC (s4/s7 vs s3) per model×ds. **PULL.**
- **2.2 Loss component breakdown** (Total/Pref/NLL-G/NLL-V table + `wandb_score_evolution.png`).
  Orig: `analyze_wandb_detailed.py`. New: from same wandb history (final-third means). **PULL.**
  Key question: does RankAlign's NLL-G stay bounded on HE (it exploded on IFEval)?
- **2.3 Held-out validation loss** (`val_loss_curves.png`). Orig: `compute_val_loss.py` +
  `plot_val_loss.py` — recomputes pref/NLL-G/NLL-V on a test split per checkpoint.
  New: needs HE test-split loss recompute per ckpt (base/ep0/1/2) for each model×setting×ds.
  **JOBS** (new GPU compute, ~like trs2 in cost). DECISION: include or defer?
- **2.4 Concordance table** (gen/val agreement fraction). Orig: `analyze_all_combos.py`.
  New: NOT in per-problem metrics → compute from raw per-candidate scores. **PULL** (test) / trs2 (train).
- **2.5 Spearman table**. New: per-problem `spearman` already in metrics CSVs → aggregate. **NOW** (test).
- **2.6 Pearson table**. New: per-problem `pearson` already in metrics CSVs → aggregate. **NOW** (test).
- **2.7 Score-delta histograms** (`*_delta_histograms.png`). Orig: `analyze_all_combos.py`.
  New: needs raw per-candidate gen/val scores. **PULL.**
- **2.8 Training dynamics** (`*_dynamics_key_settings.png`, `unified_concordance_comparison.png`)
  — metric vs epoch. New: needs per-epoch metrics (base/ep0/1/2). Test: only ep2 evaluated, so
  the epoch axis is a **TRAIN-set** story → pending **trs2** (+ gemma s2/s4 now).
- **2.9 Generator-validator scatter** (`gen_vs_val_scatter.png`). Orig: `analyze_per_category.py`.
  New: scatter of gen vs val per candidate (Base/s3/s4), positives vs negatives. Needs raw
  per-candidate scores. **PULL.** (This is its OWN subsection — not to be lumped.)
- **2.10 Per-category analysis** (`per_category_improvement_distribution.png`). Orig:
  `analyze_per_category.py`. HumanEval has no categories → **ADAPT**: substitute per-problem
  improvement distribution, or stratify by the dataset `strategy` column (normal/beginner/...).

### §3 Focused TC Comparison
- **3.1 Self-TC vs No-TC (s4 vs s3)**, **3.2 Neg-TC vs No-TC (s7 vs s3)** + `tc_comparison_*`,
  `tc_dynamics_*`. Orig: `plot_tc_comparison.py`. New (TEST): s3→s4 (self mode), s3→s7 (neg
  mode) deltas from the metrics CSVs — **NOW**. Train/epoch dynamics — pending **trs2**.

### §4 Train vs Test Generalization
- Table + `train_vs_test_comparison.png`. Orig: `compare_train_test.py`. New: gemma s2/s4
  **DONE** (`he_traintest_gemma.tex`); s3/s7 + qwen pending **trs2**. This is the Q3 centerpiece.
- (LATER) the ΔlogP-stratified gap analysis (parked at user request) slots in here.

### §5 Conclusions
- Write last, once §2–§4 are populated.

---

## Recommended build order

1. **NOW (laptop, no waiting):** §2.5 Spearman, §2.6 Pearson, §3 TC-comparison deltas (test),
   §1.3 unified gen-ROC bar plot. → fills 4 holes immediately.
2. **PULL (wandb, no GPU):** §2.1 loss curves, §2.2 loss-component breakdown + score evolution.
   → fills the whole WandB block.
3. **PULL (raw candidate scores from mll, no GPU):** §2.4 concordance, §2.7 delta histograms,
   §2.9 scatter. → fills the diagnostics block (test; train s2/s4 too).
4. **trs2 lands:** §1.1 train grid, §2.8 dynamics-over-epochs, §3 train-side, §4 full grid.
5. **DECISION:** §2.3 held-out val loss — needs new GPU jobs; include or defer?
6. **ADAPT:** §2.10 per-category → per-problem / per-strategy substitute.
7. **LATER:** ΔlogP-stratified overfitting analysis (§4 addendum).
8. §5 conclusions last.

## Scripts to (re)use / adapt
- `analyze_all_combos.py` → main table, concordance, spearman, pearson, dynamics, delta hist, unified plots.
- `analyze_wandb_training.py` / `analyze_wandb_detailed.py` → §2.1, §2.2 (point at HE run names).
- `compute_val_loss.py` + `plot_val_loss.py` → §2.3 (needs HE test-split + new jobs).
- `compare_train_test.py` → §4.
- `plot_tc_comparison.py` → §3.
- `analyze_per_category.py` → §2.9 scatter + §2.10 (adapt; no categories).
All currently read membership/IFEval score dirs; each needs HE score dirs + the HE setting map.
