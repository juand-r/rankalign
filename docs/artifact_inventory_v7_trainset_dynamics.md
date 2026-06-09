# Artifact Inventory: v7 Training & Train-Set Dynamics

**Last verified:** 2026-06-08

This document records where all artifacts live for the four model/task combinations
that have v7 train-set dynamics evaluations.

---

## Directory Map

| Artifact type | Location |
|---|---|
| **Trained models (with wandb)** | `/datastor2/jdr/rankalign/models2-rerun-wandb/` |
| **Trained models (pre-wandb, gemma membership)** | `/datastor2/jdr/rankalign/models2/` |
| **Training run logs (JSON)** | `<models-dir>/training_run_logs/` |
| **WandB local artifacts** | `/datastor2/jdr/rankalign/scripts/wandb/` (122 runs) |
| **WandB online dashboard** | https://wandb.ai/juand-r/rankalign |
| **Test-set scores (wandb-era)** | `/datastor2/jdr/rankalign/outputs-rerun-wandb/` |
| **Test-set scores (pre-wandb, gemma)** | `/datastor2/jdr/rankalign/outputs/` |
| **Train-set dynamics scores** | `/datastor2/jdr/rankalign/outputs-trainset-dynamics/` |
| **Aggregated metrics (wandb reruns)** | `metrics-from-scores-rerun-wandb/` (both repos) |
| **Aggregated metrics (original)** | `metrics-from-scores/` (both repos) |
| **Slurm logs** | `/datastor2/jdr/logs/<jobid>.{out,err}` |

---

## 1. IFEval + gemma-2-9b-it

**Settings trained:** s1, s2, s3, s4, s7, s13 (6 settings)

### (a) Training & Test Scores

| Item | Status | Location |
|---|---|---|
| WandB logs | **YES** (6 runs) | `scripts/wandb/run-20260606_*` |
| WandB run names | `rerun-wandb-20260606-gemma-2-9b-it-ifeval-s{1,2,3,4,7,13}` | dashboard |
| Training run logs | 6 JSON files | `models2-rerun-wandb/training_run_logs/*gemma*ifeval*` |
| Models (3 epochs each) | 18 `_merged` dirs | `models2-rerun-wandb/v7-google--gemma-2-9b-it-*ifeval*` |
| Test scores (ep2, self+neg TC) | 990 CSV files | `outputs-rerun-wandb/scores_*gemma*ifeval*` |
| Base model test scores | 198 CSV files | `outputs/scores_basetyp*google_gemma-2-9b-it_ifeval*` |

### (b) Train-Set Dynamics (5 settings, excl s13)

| Setting | Delta | Self-TC eval | Neg-TC eval |
|---|---|---|---|
| Base model | — | basetyp ✓ | basetypneg ✓ |
| s1 (SFT-lo) | d1.29 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s2 (RankAlign basic) | d1.38 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s3 (tc-self full) | d1.94 | ep0,ep1,ep2 ✓ | — |
| s4 (RankAlign full) | d1.94 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s7 (tc-neg full) | d1.94 | — | ep0,ep1,ep2 ✓ |

**Total files:** 26 ✓ (2 base + 6 + 6 + 3 + 6 + 3)

---

## 2. IFEval + Qwen3.5-9B

**Settings trained:** s1, s2, s3, s4, s7 (5 settings)

### (a) Training & Test Scores

| Item | Status | Location |
|---|---|---|
| WandB logs | **YES** (5 runs) | `scripts/wandb/run-20260606_*` |
| WandB run names | `rerun-wandb-qwen3.5-9b-ifeval-s{1,2,3,4,7}` | dashboard |
| Training run logs | 6 JSON files | `models2-rerun-wandb/training_run_logs/*Qwen*ifeval*` |
| Models (3 epochs each) | 15 `_merged` dirs | `models2-rerun-wandb/v7-Qwen--Qwen3.5-9B-*ifeval*` |
| Test scores (ep2, self+neg TC) | 160 CSV files | `outputs-rerun-wandb/scores_*Qwen*ifeval*` |
| Base model test scores | 198 CSV files | `outputs/scores_*Qwen*Qwen3.5-9B_ifeval*` |

### (b) Train-Set Dynamics (5 settings)

| Setting | Delta | Self-TC eval | Neg-TC eval |
|---|---|---|---|
| Base model | — | basetyp ✓ | basetypneg ✓ |
| s1 (SFT-lo) | d0.84 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s2 (RankAlign basic) | d0.96 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s3 (tc-self full) | d0.96 | ep0,ep1,ep2 ✓ | — |
| s4 (RankAlign full) | d0.96 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s7 (tc-neg full) | d0.96 | — | ep0,ep1,ep2 ✓ |

**Total files:** 26 ✓ (2 base + 6 + 6 + 3 + 6 + 3)

**Venv note:** Qwen3.5 models require `/datastor2/jdr/venvs/qwen35` (transformers 5.8.0.dev0).

---

## 3. Membership/rosch + gemma-2-9b-it

**Settings trained:** s1–s7, s11, s12, s13 (10 settings total; s13 excluded from dynamics)

### (a) Training & Test Scores

| Item | Status | Location |
|---|---|---|
| WandB logs | **NO** (trained pre-wandb, May 24) | N/A |
| Training run logs | 29 JSON files | `models2/training_run_logs/*gemma*membership*` |
| Models (3 epochs each) | 29 `_merged` dirs | `models2/v7-google--gemma-2-9b-it-*membership*` |
| Test scores (ep2, self+neg TC) | 730 CSV files | `outputs/scores_*gemma*rosch*` |
| Base model test scores | 20 CSV files | `outputs/scores_*google_gemma-2-9b-it_rosch*` |

### (b) Train-Set Dynamics (9 settings, excl s13)

| Setting | Delta | Self-TC eval | Neg-TC eval |
|---|---|---|---|
| Base model | — | basetyp ✓ | — (not needed) |
| s1 (SFT-lo) | d1.43 | ep0,ep1 ✓ (no ep2 model) | ep0,ep1 ✓ |
| s2 (RankAlign basic) | d1.42 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s3 (tc-self full) | d2.69 | ep0,ep1,ep2 ✓ | — |
| s4 (RankAlign full) | d2.69 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s5 (tc-self basic) | d1.42 | ep0,ep1,ep2 ✓ | — |
| s6 (tc-self fsx+ppd) | d1.42 | ep0,ep1,ep2 ✓ | — |
| s7 (tc-neg full) | d2.69 | — | ep0,ep1,ep2 ✓ |
| s11 (tc-self vlo) | d2.69 | ep0,ep1,ep2 ✓ | — |
| s12 (tc-neg vlo) | d2.69 | — | ep0,ep1,ep2 ✓ |

**Total files:** 38 (1 base + 37 epoch evals; some dates 0607 and 0608 due to reruns)

**Note:** No neg-TC base eval exists here, but this is expected — basetyp scoring of the
unfinetuned model compares it against itself, which is trivially zero and not meaningful.

---

## 4. Membership/rosch + Qwen3.5-9B

**Settings trained:** s1, s2, s3, s4, s7 (5 settings)

### (a) Training & Test Scores

| Item | Status | Location |
|---|---|---|
| WandB logs | **YES** (5 runs) | `scripts/wandb/run-20260607_*` |
| WandB run names | `rerun-wandb-qwen3.5-9b-membership-s{1,2,3,4,7}` | dashboard |
| Training run logs | 9 JSON files | `models2-rerun-wandb/training_run_logs/*Qwen*membership*` |
| Models (3 epochs each) | 15 `_merged` dirs | `models2-rerun-wandb/v7-Qwen--Qwen3.5-9B-*membership*` |
| Test scores (ep2, self+neg TC) | 100 CSV files | `outputs-rerun-wandb/scores_*Qwen*rosch*` |
| Base model test scores | 20 CSV files | `outputs-rerun-wandb/scores_*v6-Qwen_Qwen3.5-9B_rosch*` |

### (b) Train-Set Dynamics (5 settings)

| Setting | Delta | Self-TC eval | Neg-TC eval |
|---|---|---|---|
| Base model | — | basetyp ✓ | basetypneg ✓ |
| s1 (SFT-lo) | d1.53 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s2 (RankAlign basic) | d1.54 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s3 (tc-self full) | d1.54 | ep0,ep1,ep2 ✓ | — |
| s4 (RankAlign full) | d1.54 | ep0,ep1,ep2 ✓ | ep0,ep1,ep2 ✓ |
| s7 (tc-neg full) | d1.54 | — | ep0,ep1,ep2 ✓ |

**Total files:** 26 ✓ (2 base + 6 + 6 + 3 + 6 + 3)

**Venv note:** Qwen3.5 models require `/datastor2/jdr/venvs/qwen35` (transformers 5.8.0.dev0).

---

## Notes

- **Gemma membership: no WandB training logs** (trained pre-wandb, May 24).
  This is a known limitation; original training_run_logs JSON files exist.
- Base model basetyp evals that exist in other combos (gemma ifeval, qwen ifeval,
  qwen membership) are trivially zero since they compare the model against itself,
  but were run for completeness/plotting. Their absence from gemma membership is not a gap.

---

## File Naming Conventions

### Train-set dynamics score files
```
scores_{scoring}_{model-identifier}_{task}_train_log-odds_tc_{date}.csv
```
- `{scoring}`: `basetyp` (self-TC) or `basetypneg` (neg-TC)
- For base model: `v6-{provider}_{model}` (e.g. `v6-google_gemma-2-9b-it`)
- For fine-tuned (short): `v7-{model}-d{delta}-e{epoch}-{task}-all-{flags}`
- For fine-tuned (long): `v7-{provider}--{model}-delta{d}-epoch{e}--{task}-all--{full-flags}_merged`

### Test-set score files
```
scores_{scoring}_{model-identifier}_{eval-task}_test_log-odds_tc_{date}.csv
```
- Evaluated on individual sub-tasks (e.g. `ifeval-prompt_1`, `rosch-bird`)

### Metric builder scripts
- `scripts/_build_ifeval_table_v7.py` — ifeval metrics (ID/OOD split)
- `scripts/_build_rosch_table_v7.py` — rosch/membership metrics
- Both support `MODEL_REGISTRY` dict for flexible model selection
- Env vars: `IFEVAL_MODEL` / `ROSCH_MODEL`, `IFEVAL_METRIC` / `ROSCH_METRIC`
