# Qwen3.5-9B IFEval — run-to-run training variance across three independent reruns (v1/v2/v3)

**Date:** 2026-06-27 (Central Time)
**Model:** Qwen/Qwen3.5-9B · **Task:** IFEval · **Split:** OOD test (prompts 1–21, 20 prompts)
**Metric:** per-problem generator ROC = ROC of `gen_score_typcorr` vs `correct` computed
*within each prompt file*, then averaged over the 20 OOD prompts (± SE = std(ddof=1)/√n).

## What this measures

Three **independent** train→eval reruns of the same recipe, to quantify how much the gen ROC
moves from one training run to the next (data splits are fixed; the pair shuffle, DataLoader
shuffle, and GPU non-determinism are not seeded, so there is genuine run-to-run variance):

| label | directory | notes |
|---|---|---|
| v1 | `outputs-rerun-wandb`    | original rerun batch (single run/setting on OOD; 0608) |
| v2 | `outputs-rerun-wandb-v2` | second rerun (0610 base-typ pass, 0612 no-base pass) |
| v3 | `outputs-rerun-wandb-v3` | third rerun, this session (0626–0627) |

All three use the same per-setting delta tag (s2/s3/s4/s7 = 0.96, s1 = 0.84), same OOD prompt
set, same eval. `gen_score_typcorr` depends on the typicality eval mode, so the four modes are
kept separate: `self` / `neg` = no-base normalization; `basetyp` / `basetypneg` = base-model
normalization.

## Headline finding

**Your intuition is confirmed: RankAlign is the high-variance method.** SFT and New+fsx are
rock-stable across reruns (< 1 ROC point of spread); FLORA-PMI is moderate (~2–3); RankAlign
swings ~8 points in `self` mode and ~15 points in `basetyp` mode. v1's RankAlign run is an
anomalously low draw; v2 and v3 agree closely.

### `self` typicality mode (the one flagged as high-variance)

| Method | v1 | v2 | v3 | range / std |
|---|---|---|---|---|
| SFT (s1) | 66.2 ± 3.6 | 66.6 ± 4.0 | 65.9 ± 4.0 | 0.7 / 0.4 |
| **RankAlign (s2)** | **67.9 ± 3.6** | **76.0 ± 3.0** | **75.4 ± 3.9** | **8.0 / 4.5** |
| New+fsx (s3) | 80.7 ± 2.9 | 80.6 ± 3.0 | 80.9 ± 2.8 | 0.3 / 0.2 |
| FLORA-PMI (s4) | 79.6 ± 3.3 | 78.2 ± 3.3 | 80.9 ± 3.0 | 2.7 / 1.4 |

### `basetyp` mode (RankAlign variance even larger)

| Method | v1 | v2 | v3 | range / std |
|---|---|---|---|---|
| SFT | 62.5 ± 4.1 | 61.0 ± 4.3 | 61.1 ± 4.2 | 1.5 / 0.9 |
| **RankAlign** | **58.3 ± 3.9** | **72.6 ± 3.7** | **73.8 ± 3.9** | **15.5 / 8.6** |
| New+fsx | 74.8 ± 3.5 | 76.1 ± 3.3 | 75.4 ± 3.2 | 1.3 / 0.7 |
| FLORA-PMI | 74.5 ± 3.5 | 75.5 ± 3.3 | 76.5 ± 3.1 | 2.0 / 1.0 |

### `neg` mode

| Method | v1 | v2 | v3 | range / std |
|---|---|---|---|---|
| SFT | 48.8 ± 2.8 | 46.0 ± 3.1 | 47.9 ± 2.4 | 2.8 / 1.5 |
| RankAlign | 57.7 ± 4.2 | 56.5 ± 4.8 | 53.9 ± 3.1 | 3.8 / 1.9 |
| New+fsx | 62.0 ± 3.6 | 59.1 ± 3.9 | 56.6 ± 3.9 | 5.4 / 2.7 |
| FLORA-Neg (s7) | 66.0 ± 3.0 | 64.2 ± 3.4 | 66.0 ± 3.0 | 1.8 / 1.0 |

### `basetypneg` mode

| Method | v1 | v2 | v3 | range / std |
|---|---|---|---|---|
| SFT | 49.9 ± 4.3 | 48.6 ± 4.3 | 48.9 ± 3.9 | 1.3 / 0.7 |
| **RankAlign** | **53.1 ± 4.2** | **69.2 ± 3.6** | **69.3 ± 4.1** | **16.2 / 9.3** |
| New+fsx | 65.3 ± 3.5 | 70.7 ± 3.6 | 70.5 ± 3.3 | 5.4 / 3.0 |
| FLORA-Neg | 68.9 ± 3.5 | 70.5 ± 3.0 | 71.9 ± 3.0 | 3.0 / 1.5 |

(n = 20 prompts for every cell.)

## On the 83.2 RankAlign value

The three reruns (delta 0.96) cluster RankAlign `self` gen ROC at ~76 (best), never reaching
83. The **83.2** came from the *original pod checkpoint at delta 0.001* — a different procedure,
not the rerun recipe. So 83.2 reflects partly the different delta and partly a favorable draw;
it is **not** reproducible under the rerun recipe, where RankAlign genuinely varies run-to-run
(std ~4.5 in `self`, ~8.6 in `basetyp`), and v1 happened to be a low outlier.

## Provenance / reproducibility

- **Training:** all 5 settings COMPLETED (mll jobs 45635–45640; s1 the 19 h long pole).
  Checkpoints in `models2-rerun-wandb-v3/`.
- **Eval:** default base-typicality pass (`basetyp`/`basetypneg`) + `NO_BASE=1` second pass
  (`self`/`neg`, jobs 45654–45658) → all four modes present (320 CSVs), matching v2.
- **Committed scripts** (branch `longform`):
  - launcher: `scripts/_qwen_ifeval_repro_v3.sh`
  - second eval pass: `scripts/_qwen_ifeval_repro_v3_nobase.sh`
  - comparison: `analysis/scripts/compute_rerun_v123_genroc.py` (prints the full per-run
    enumeration before the summary; picks the most-complete run per cell, not latest-date).
- **Raw scores:** `/datastor2/jdr/rankalign/outputs-rerun-wandb-v3/`.
