# Morning summary — original-v7 table fill (overnight 2026-06-08 → 09)

## Bottom line
The **original-v7 results table is complete**: `docs/original_v7_results.{tex,pdf}` —
both models (gemma-2-9b-it, Qwen3.5-9B) × both tasks (Hyponymy, IFEval) × all four
eval-TC variants (PMI self, PMI base, Neg self, Neg base), **IFEval column is all-OOD**
(renamed "IFEval (OOD)"), provenance-labeled. Pushed to `longform`.

## What filled the gaps (mll only, eval-only, no retrain — as instructed)
The gap was missing **eval variants** (own vs base), not checkpoints. Ran the variant
evals on mll via `run_qwen35_cell_mll.sbatch` (qwen) + `run_gemma9b_cell_mll.sbatch`
(gemma copy, gemma4 venv) in EVAL_ONLY mode (`NO_BASE=1` → own; default → base):
- **qwen ifeval OOD**: own (s1/s2/s3/s4/s7) + base — jobs 44259–44265.
- **qwen rosch**: own + base (s1–s4, s7) — jobs 44266–44274.
- **gemma ifeval OOD own**: s1/s2/s4/s7 — jobs 44276–44282.
Coverage 308 → 430 populated (model,task,method,metric,variant) entries.

## Provenance (labeled per cell; `^r` = rerun)
- **Hyponymy / gemma-2-9b-it** — ORIGINAL (from `v7_rosch_all_metrics`, local-mll). No mark.
- **IFEval / gemma-2-9b-it** — base = ORIGINAL pod eval (no mark); **own = rerun** (`^r`)
  (own-OOD was never run in the original v7; the rerun checkpoints supplied it).
- **Qwen3.5-9B (both tasks)** — **rerun** (`^r`): the original qwen checkpoints were lost
  (pod NVMe), so per your approval these use the wandb-rerun models.

## Off-template note
FLORA-PMI (s4, tc-self) shows only self/PMI variants; FLORA-Neg (s7, tc-neg) only neg —
matching the training template (the pod recompute had over-populated the off-template
own-ID values; suppressed).

## NOT done — needs your call (deliberately deferred, not blocked on compute)
1. **Original HF qwen checkpoints** (`latkes`: ifeval s1/s2, rosch s1/s7) are **downloaded**
   to `models2-original-hf/` (68 GB) but **not yet evaluated**. Reason: their dirs are named
   `ifeval-s1-ep2` (not the parseable `v7-Qwen--…` form), and the README/logs don't carry the
   canonical name/delta — so evaling them needs a constructed `EVAL_NAME` with a distinct delta.
   That's doable but risks mis-attributing orig-vs-rerun in the shared table, and provenance
   correctness is your priority — so I left it for us to do together (~30 min). Once evaled,
   these supersede the qwen s1/s2 ifeval + s1/s7 rosch cells as provenance=orig.
2. **Task #24 — parallel rerun table / original-vs-rerun comparison.** The rerun evals are
   done (they're what filled the qwen + gemma-ifeval-own cells). A clean "two independent
   trains" comparison is most meaningful once the qwen originals (item 1) are evaled.

## Key files / jobs
- Table: `docs/original_v7_results.{tex,pdf}`; builder `scripts/_build_original_v7_table.py`.
- Eval sbatches: `scripts/run_qwen35_cell_mll.sbatch`, `scripts/run_gemma9b_cell_mll.sbatch`
  (+ `MODEL_DIR_OVERRIDE` / `EVAL_MODES_OVERRIDE` for off-naming/off-template cases).
- Rerun scores: `outputs-rerun-wandb/`; cells: `metrics-from-scores{,-rerun-wandb}/`.
- Recompute (pod eval_model): `scripts/_recompute_v7_eval_metrics.py`.
- Earlier comparison vs paper: `docs/paper_vs_original_v7.{tex,pdf}` (predates this fill;
  re-run it against the now-complete table when convenient).

## UPDATE (~04:30 CT): original-HF qwen evals DONE + verified
All 8 jobs (44283–44290) COMPLETED. `outputs-original-hf/` now has the **true
paper-checkpoint** qwen values (sentinel `delta0.001`, separate dir, no rerun mixing):
- ifeval OOD: s1 + s2, all 4 variants (self-/neg-/basetyp-/basetypneg-, 20 prompts each)
- rosch: s1 (self+neg+base), s7 (neg-own + neg-base)
Spot-verified valid (e.g. ifeval s2 own tc gen_roc 0.64, 80 rows).

### Integration recipe (the ~20-min step to do together — I stopped here on purpose)
These should supersede the rerun (`^r`) values for **qwen ifeval s1/s2 + rosch s1/s7**,
marked provenance=orig. Cleanest path (avoids polluting the rerun cells):
1. Add an env-gated `EXTRA_SEARCH_DIR` + `METRICS_DIR_OVERRIDE` to
   `_build_ifeval_table_v7.py` / `_build_rosch_table_v7.py` (additive, no default change),
   point them at `outputs-original-hf/` → `metrics-from-scores-orig-hf/`.
2. Add an `ORIG_CELLS` source to `_build_original_v7_table.py` (provenance "O", a distinct
   mark e.g. `^o`) that overrides rerun for exactly those 4 cells.
3. Rebuild + verify the 4 qwen cells changed to orig values (others unchanged), commit.
I left this for you because it's a provenance-merge judgment call and the cell-builders
need a (small, safe) edit — your call on the orig-vs-rerun presentation.

## Task #24 (rerun comparison) — status
The rerun evals are done (they fill the current table's qwen + gemma-ifeval-own cells).
A true "original vs rerun" comparison is meaningful once the orig-HF cells above are
integrated — so it's the natural next step after the integration, with you.
