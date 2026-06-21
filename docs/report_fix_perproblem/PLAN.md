# Per-problem fix of the original report (`analysis/report.tex` → `report_fix.tex`)

**Goal (user, 2026-06-20 night):** the original report computes its headline metrics on a
GLOBAL POOL of all candidates per (model, task, setting, epoch). They must instead be computed
**per problem, then averaged across problems, with standard errors**. Write a modified report
with `fix` appended to the filename: `analysis/report_fix.tex`.

## Problem unit (confirmed by user + data)
- **IFEval** → group by `prompt` (the `ifeval-prompt_N` instruction). 79 train prompts, 40
  candidates each, every group has both classes. Label col `correct`.
- **Membership** → group by `category`. 68 train categories, 18–96 candidates each, both classes.
  Label col `label`.
- SE = `std(per-problem values, ddof=1) / sqrt(n_problems)`.

## Data — NO GPU NEEDED (pure re-aggregation)
Train-dynamics score files already exist, pooled per (model,task,setting,epoch), WITH the
grouping column: `/datastor2/jdr/rankalign/outputs-trainset-dynamics/` (116 files, mll).
Each file has per-candidate rows incl. `gen_score`, `gen_score_typcorr`, `val_score`, and the
group key. (The "197k rows" scare was `wc -l` miscounting multi-line response fields; pandas
reads 3,160 clean rows.) So the flaw is purely in the AGGREGATION code
(`analyze_all_combos.py::compute_metrics` runs one `roc_auc_score` over the whole file).

## Scope — tables to recompute as per-problem mean ± SE (epoch 2, self-TC)
1. Q1 Main: Gen ROC AUC (TC, self-scoring) — base/s1/s2/s3/s4 × 4 cells.
2. Concordance (self-TC).
3. Spearman (self-TC).
4. Pearson (self-TC).
5. Unified ROC figure (s1..s4 self-TC; s7 neg-TC) — regenerate with error bars.
6. Per-Category (membership) — already per-category; keep, add SE to the distribution prose.
7. Train vs Test — needs TEST per-problem files (rosch test per-category; ifeval OOD test
   per-prompt). Locate under outputs/ or outputs-trainset-dynamics; recompute per-problem.
Loss-curve / loss-breakdown / held-out-loss sections: UNCHANGED (not per-problem ROC metrics).

## Pipeline (all committed, run on mll qwen35 venv)
- `analysis/scripts/build_report_fix_metrics.py` — reuses `analyze_all_combos`'s
  `COMBO_CONFIG` + `find_files_for_combo` (file→setting/epoch/tc_eval matching), replaces the
  global-pool metric with a per-problem one; writes `analysis/tables/report_fix_metrics.csv`
  (tidy: combo,setting,epoch,tc_eval, {gen_roc,val_roc,concordance,spearman,pearson}_{mean,se,n})
  and emits the LaTeX tables.
- Then write `analysis/report_fix.tex` (copy of report.tex with the recomputed tables + a note
  explaining the per-problem methodology), compile, commit, push.

## Status log
- 2026-06-20: verified grouping + data availability; no GPU needed. Writing metrics script.
- 2026-06-20 (done): `report_fix.tex` complete, compiles clean (18pp, 0 undefined refs), pushed.
  - Recomputed per-problem mean$\pm$SE: Q1 gen ROC, concordance, Spearman, Pearson (4 cells),
    unified ROC bar chart (with SE error bars), train-vs-test (membership/rosch per-category).
  - Pure re-aggregation, no GPU. Pipeline: `build_report_fix_metrics.py`.
  - Honest findings: IFEval ROC rises substantially once prompts are unpooled (base 0.670->0.792
    etc.); on Gemma/IFEval s3 and s4 become statistically tied (was "s4 best everywhere");
    TC-promotes-generalization (train-vs-test) holds under per-category recomputation.
  - NOT changed (retained from original, noted in-report): per-epoch dynamics, TC-comparison,
    score-delta, gen-val scatter figures (pooled visualizations; corrected numbers are in the
    tables). Loss-curve / loss-breakdown / held-out-loss sections: unchanged by design.
  - train-vs-test omits s5/s11 (absent from the per-category rosch test artifact) and s1/s7
    (different scoring direction; excluded from the PMI-base comparison).
  - Artifacts committed: analysis/report_fix.{tex,pdf}, analysis/tables/fix_*.tex,
    report_fix_metrics.csv, analysis/plots/fix_unified_genroc.png.
