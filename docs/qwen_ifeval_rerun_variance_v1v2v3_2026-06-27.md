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

## On the 83.2 RankAlign value — resolved: run-to-run VARIANCE, not delta or environment

This was first attributed to a different delta: the original checkpoint's name carries
`delta0.001`. **That was a red herring — `delta0.001` is a placeholder**, inserted when the
original checkpoint was recovered from latkes and renamed to a canonical parseable form
(`scripts/_orig_in_gemma4_eval.sh`: *"delta0.001 = placeholder"*). The training log embedded in
the original checkpoint (`models2-original-hf/ifeval-s2-ep2/training_log.log.gz`) shows it was
trained on a RunPod pod (2026-05-24) with `--delta 0.15 --delta-bins 10 → auto-δ 0.9627` — the
**same delta regime as every rerun**. So the original and the reruns differ only in the unseeded
pair shuffle + DataLoader shuffle + GPU nondeterminism. The remaining suspect was the *training
environment* (the original pod: torch 2.4.1 / transformers 5.8.1; the mll reruns: torch 2.5.1 /
transformers 5.8.0.dev0).

### Pod-environment replication (3 fresh RunPod runs, 2026-06-29)

The original recipe was re-run **three times on fresh TAUR RunPod H100 pods in the exact original
environment** (`runpod/pytorch:2.4.0` → torch 2.4.1, `requirements-gemma4.txt` → transformers
5.8.1, Qwen3.5 torch-fallback attention) — i.e. the literal pipeline that produced the 83.2:

| eval mode | run1 | run2 | run3 | mean ± std |
|---|---|---|---|---|
| **self** | **82.2 ± 2.4** | 70.3 ± 4.7 | 65.9 ± 3.9 | **72.8 ± 6.9** |
| neg | 38.0 ± 3.5 | 55.8 ± 3.9 | 61.1 ± 3.6 | 51.6 |
| basetyp | 78.8 ± 2.7 | 63.1 ± 4.7 | 63.1 ± 3.9 | 68.3 |
| basetypneg | 71.5 ± 3.3 | 59.2 ± 4.5 | 61.4 ± 4.1 | 64.0 |

(n = 20 OOD prompts per cell.)

**Verdict: variance, not environment.** The pod-environment `self` mean (72.8 ± 6.9) is
indistinguishable from the mll-environment mean (73.1 ± 4.5) — the environment does **not**
systematically produce 83. But run1 independently reached **82.2**, so the original 83.2 is a
*reachable high draw* of a wide distribution, not a reproducible environment effect. Pooling all
six independent runs (mll v1/v2/v3 + pod run1/2/3), RankAlign `self` gen ROC spans
**65.9 → 82.2** (~16 points). The honest, representative RankAlign number on this cell is
**~73 ± ~6**, in either environment — and the stable methods (New+fsx, FLORA-PMI at ~80 ± ~1)
beat that representative mean.

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

### Pod-environment replication (2026-06-29)

- **3 fresh TAUR RunPod H100 pods**, original recipe + environment (image `runpod/pytorch:2.4.0`,
  `requirements-gemma4.txt`, `run_qwen35_cell.sh ifeval s2`, 3 epochs, `--delta 0.15 --delta-bins 10`).
- **Models (public):** `latkes/rankalign-v7-qwen3.5-9b-ifeval-s2-variance-run{1,2,3}` (merged + training log).
- **Committed scripts** (`longform`): `scripts/run_qwen35_variance_overnight.sh`,
  `scripts/upload_qwen_variance_to_latkes.py`, `analysis/scripts/compute_pod_variance_genroc.py`.
- **Raw scores (local):** `~/.claude/training_monitor/qwen_variance/scores/run{1,2,3}/` (80 CSVs each).
- Per-run training logs embedded in each latkes repo confirm `auto-δ 0.96` (same regime as mll).
