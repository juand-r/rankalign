# Pod Results — GenROC — 2026-05-25T18:29:11Z

All cells: **GenROC × 100**, mean ± SE across the eval-task split for that section (no ± when a single task). Scope: **v7 only** (v6 excluded), models **Qwen3.5-9B** and **gemma-2-9b-it**, evaluated on RunPod and downloaded locally.

**Delta regimes** (a property of the trained model):
- **delta-bins 10** — every model whose name encodes a non-0.15 delta (delta0.96, delta1.89, delta1.94, delta2.49, …). Read directly from the model path.
- **delta 0.15 (fixed)** — the `eval_model_sN` symlink runs (the v7b batch). The delta is not in the filename; **UNCONFIRMED pending a check of the still-running v7b training pods.**

**Columns** = scoring method at eval time. Raw = log P(y|x); basetyp-/self- = PMI vs base/self; basetypneg-/neg- = Neg vs base/self. `N/A` = that eval variant was not run for the setting (s4 ran basetyp+self only; s7 ran basetypneg+neg only).

**Cells:** `--` = no data. `soon` = eval actively in flight (results expected shortly). Train column: ✓ = eval data present · soon = in flight · – = not run.

> ifeval **OOD** = prompts 1–21 (fully held out). ifeval **ID** = prompts 22–109 (50% of completions held out). These are the *data* split, independent of delta.

## gemma-2-9b-it × ifeval OOD — delta 0.15 (fixed)

> Delta value UNCONFIRMED (symlink hides it); treat as the v7b/delta-0.15 batch pending pod check.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 50.2 ± 4.2 | 57.2 ± 4.3 | -- | 49.3 ± 4.0 | -- |
| 2 RankAlign | ✓ | 56.3 ± 4.5 | 62.8 ± 4.8 | -- | 59.6 ± 4.9 | -- |
| 3 New + fsx [-TC] | ✓ | 60.9 ± 4.1 | 73.0 ± 3.7 | -- | 65.2 ± 3.7 | -- |
| 4 New + PMI + fsx | ✓ | 60.9 ± 4.0 | 81.8 ± 2.4 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 62.2 ± 4.1 | N/A | N/A | 68.1 ± 3.4 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × ifeval OOD — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 43.0 ± 3.5 | 52.6 ± 4.4 | -- | -- | -- |
| 2 RankAlign | ✓ | 48.9 ± 3.7 | 56.7 ± 4.7 | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 54.5 ± 4.1 | 71.6 ± 4.8 | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 53.4 ± 3.4 | 82.0 ± 3.1 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 54.4 ± 3.7 | N/A | N/A | 65.9 ± 3.7 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × ifeval ID — delta 0.15 (fixed)

> Delta value UNCONFIRMED (symlink hides it); treat as the v7b/delta-0.15 batch pending pod check.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | soon | soon | soon | soon | soon | soon |
| 2 RankAlign | soon | soon | soon | soon | soon | soon |
| 3 New + fsx [-TC] | soon | soon | soon | soon | soon | soon |
| 4 New + PMI + fsx | soon | soon | soon | soon | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | soon | soon | N/A | N/A | soon | soon |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × persona ID — delta 0.15 (fixed)

> Delta value UNCONFIRMED (symlink hides it); treat as the v7b/delta-0.15 batch pending pod check.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 99.0 ± 0.5 | 99.2 ± 0.3 | -- | -- | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | – | -- | -- | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | – | -- | N/A | N/A | -- | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × persona ID — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 74.5 ± 2.4 | 80.2 ± 2.8 | -- | 82.8 ± 3.2 | -- |
| 2 RankAlign | ✓ | 96.6 ± 1.6 | 98.7 ± 0.4 | -- | 98.1 ± 0.7 | -- |
| 3 New + fsx [-TC] | ✓ | 86.7 ± 1.1 | 93.5 ± 0.5 | -- | 93.8 ± 0.8 | -- |
| 4 New + PMI + fsx | ✓ | 82.5 ± 2.2 | 90.2 ± 1.7 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 83.2 ± 1.2 | N/A | N/A | 92.0 ± 0.8 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × persona OOD — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 35.2 ± 8.0 | 37.2 ± 8.4 | -- | 50.4 ± 8.9 | -- |
| 2 RankAlign | ✓ | 52.2 ± 9.0 | 78.2 ± 3.9 | -- | 84.7 ± 3.2 | -- |
| 3 New + fsx [-TC] | ✓ | 42.0 ± 8.2 | 52.3 ± 5.2 | -- | 67.4 ± 6.1 | -- |
| 4 New + PMI + fsx | ✓ | 41.0 ± 9.0 | 48.5 ± 7.7 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 42.1 ± 9.9 | N/A | N/A | 67.5 ± 9.1 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × rosch — delta 0.15 (fixed)

> Delta value UNCONFIRMED (symlink hides it); treat as the v7b/delta-0.15 batch pending pod check.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 83.6 ± 1.7 | 83.9 ± 1.0 | -- | 80.2 ± 1.3 | -- |
| 2 RankAlign | ✓ | 87.3 ± 3.5 | 83.4 ± 2.4 | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 87.0 ± 1.4 | 85.1 ± 1.4 | -- | 79.8 ± 2.0 | -- |
| 4 New + PMI + fsx | ✓ | 87.3 ± 1.4 | 88.6 ± 1.1 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 84.5 ± 2.1 | N/A | N/A | 85.1 ± 1.4 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × rosch — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 95.3 | 93.8 | -- | -- | -- |
| 2 RankAlign | ✓ | 89.7 ± 1.6 | 89.6 ± 1.2 | -- | 86.4 ± 1.5 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | – | -- | -- | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | – | -- | N/A | N/A | -- | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × ifeval OOD — delta 0.15 (fixed)

> Delta value UNCONFIRMED (symlink hides it); treat as the v7b/delta-0.15 batch pending pod check.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 66.3 ± 3.8 | 80.5 ± 2.8 | -- | 73.8 ± 2.9 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 53.9 ± 4.5 | 76.2 ± 3.3 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 55.0 ± 4.4 | N/A | N/A | 64.9 ± 3.2 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × ifeval OOD — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 61.1 ± 3.7 | 79.3 ± 3.3 | -- | -- | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 46.9 ± 3.7 | 76.8 ± 3.8 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 50.3 ± 3.7 | N/A | N/A | 66.7 ± 3.4 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × persona ID — delta 0.15 (fixed)

> Delta value UNCONFIRMED (symlink hides it); treat as the v7b/delta-0.15 batch pending pod check.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 84.7 ± 2.6 | -- | 91.8 ± 1.4 | -- | 87.9 ± 2.6 |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | – | -- | -- | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | – | -- | N/A | N/A | -- | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × persona ID — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 98.0 ± 0.9 | 99.5 ± 0.2 | -- | 99.5 ± 0.3 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | – | -- | -- | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | – | -- | N/A | N/A | -- | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × persona OOD — delta 0.15 (fixed)

> Delta value UNCONFIRMED (symlink hides it); treat as the v7b/delta-0.15 batch pending pod check.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 35.6 ± 6.6 | -- | 45.2 ± 6.2 | -- | 89.8 ± 1.4 |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | – | -- | -- | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | – | -- | N/A | N/A | -- | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × persona OOD — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 51.5 ± 8.2 | 72.1 ± 6.0 | -- | 78.7 ± 6.1 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | – | -- | -- | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | – | -- | N/A | N/A | -- | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × rosch — delta 0.15 (fixed)

> Delta value UNCONFIRMED (symlink hides it); treat as the v7b/delta-0.15 batch pending pod check.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | – | -- | -- | -- | -- | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | – | -- | -- | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 80.9 ± 2.1 | N/A | N/A | -- | 87.5 ± 2.9 |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × rosch — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 88.2 ± 1.8 | 83.2 ± 2.1 | -- | 82.2 ± 2.0 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 81.3 ± 1.9 | 87.3 ± 2.1 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 80.6 ± 2.2 | N/A | N/A | 88.2 ± 2.1 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |
