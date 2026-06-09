# Original v7 results — full accounting (provenance + gaps)

**2026-06-08.** Reconstruction of the **v7 delta-bins** eval numbers (the ones the paper
`main.tex` was built from), shown as the 4 eval-time typicality variants per method.
Output: [`original_v7_results.tex`](original_v7_results.tex) / `.pdf`.

**v7 delta-bins ONLY** — excludes v6, v7b (fixed delta-0.15), and the June 2026 wandb rerun.

## Where the values come from (two places — pod + local mll)

The v7 evals were produced in two places / two filename schemes:

| Source | Scheme | Covers | Builder reads |
|---|---|---|---|
| **local mll** | full model-name | gemma-2-9b-it **rosch** (Hyponymy), all 4 variants × 5 metrics | `docs/v7_rosch_all_metrics_20260525_003506.md` (complete, matches paper) |
| **pod-v7** | `eval_model_sN` | gemma **ifeval**, qwen **ifeval**, qwen **rosch** | `metrics-from-scores/v7_recompute_eval_metrics.csv` |

The pod CSV is recomputed from raw scores by `scripts/_recompute_v7_eval_metrics.py`
(reads `outputs_gemma4_from_pod-v7/{ra9b_ifeval, qw35_ifeval, qw35_persona_member}`,
excludes v7b, dedups the qw35 grab-bag). The pre-computed markdown tables
(`pod-results-*`, `v7_ra9b_results`) were **not** used — they are incomplete (literal
`soon` placeholders for ifeval).

Inventory of every v7 score location: `scripts/_inventory_v7_scores.py`.

## IFEval split caveat (important)

- **base-typ ifeval** (PMI base / Neg base) exists on the **OOD** set (20 prompts) — marked `o`.
- **own-typ ifeval** (PMI self / Neg self) exists **only on ID** (79 prompts) — marked `i`.
  **own-OOD ifeval was never run in v7** (only v6 had it). The paper IFEval column is OOD
  and clearly used own-typ, so it cannot be reproduced exactly from v7 on-disk data.

## What is filled vs. gaps

**Filled:**
- **Hyponymy / G2-9b-it** — complete (all methods × 4 variants × ROC_G/ρ/ROC_V/Acc_V). Matches paper.
- **IFEval / G2-9b-it** — own-ID (s1/s2/s3/s4-self/s7-neg) + base-OOD (s1/s2/s3/s4/s7), all metrics.
- **IFEval / Qwen** — base-OOD (s2/s4/s7) + Consistency-FT own/base (s13). Validator metrics too.
- **Hyponymy / Qwen** — Consistency-FT (s13) + FLORA-Neg (s7 neg-own).

**Genuine v7 gaps (not on disk as v7 delta-bins):**
- **Hyponymy / Qwen** — Base, SFT, RankAlign, FLORA-PMI: no usable v7 rosch eval (this is *why*
  the June rerun was run). Only sparse cells exist.
- **IFEval / G2-9b-it** — own-**OOD** (never run in v7); Base row + Consistency-FT (s13) not in the
  pod `eval_model_sN` set (base is in the full-name store; s13 ifeval is in the gen_roc cells only).
- **Hyponymy / G2-9b-it** — Consistency-FT (s13): absent from `v7_rosch_all_metrics`.
- Validator Base rows (ifeval): base-model eval is in the full-name store, not the pod recompute.

These gaps are smaller than the initial table suggested, but real. Filling the last few
(gemma ifeval Base/s13, qwen rosch) would need either the full-name local store parsed or a
targeted re-eval.
