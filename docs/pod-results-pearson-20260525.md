# Pod Results — Pearson(gen, val) — 2026-05-25T18:34:17Z

All cells: **Pearson(gen, val) × 100**, mean ± SE across the eval-task split for that section (no ± when a single task). Scope: **v7 only** (v6 excluded), models **Qwen3.5-9B** and **gemma-2-9b-it**, evaluated on RunPod and downloaded locally.

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
| 1 SFT labelonly 10% | ✓ | 4.1 ± 8.1 | 4.6 ± 8.6 | -- | 1.6 ± 7.3 | -- |
| 2 RankAlign | ✓ | 19.4 ± 7.6 | 25.9 ± 7.7 | -- | 24.3 ± 7.9 | -- |
| 3 New + fsx [-TC] | ✓ | 36.6 ± 6.6 | 49.4 ± 5.9 | -- | 44.9 ± 5.8 | -- |
| 4 New + PMI + fsx | ✓ | 40.1 ± 7.0 | 61.2 ± 6.3 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 41.8 ± 6.9 | N/A | N/A | 52.6 ± 5.6 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × ifeval OOD — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | -5.7 ± 7.6 | 0.5 ± 8.7 | -- | -- | -- |
| 2 RankAlign | ✓ | 26.5 ± 10.3 | 31.3 ± 9.3 | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 27.0 ± 8.1 | 43.2 ± 9.1 | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 29.2 ± 6.3 | 58.2 ± 9.2 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 30.3 ± 6.5 | N/A | N/A | 55.1 ± 5.7 | -- |
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
| 4 New + PMI + fsx | ✓ | 55.0 ± 3.5 | 75.2 ± 2.2 | 75.0 ± 2.3 | N/A | N/A |
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
| 2 RankAlign | ✓ | 85.0 ± 2.1 | 85.3 ± 0.9 | -- | -- | -- |
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
| 1 SFT labelonly 10% | ✓ | 39.4 ± 4.0 | 50.6 ± 4.4 | -- | 54.9 ± 5.0 | -- |
| 2 RankAlign | ✓ | 75.4 ± 3.5 | 75.5 ± 2.8 | -- | 75.4 ± 2.3 | -- |
| 3 New + fsx [-TC] | ✓ | 62.0 ± 0.9 | 74.3 ± 0.7 | -- | 74.9 ± 1.2 | -- |
| 4 New + PMI + fsx | ✓ | 54.5 ± 2.9 | 68.8 ± 2.6 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 55.5 ± 0.8 | N/A | N/A | 72.5 ± 1.5 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × persona OOD — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | -18.4 ± 6.4 | -17.7 ± 7.8 | -- | 4.0 ± 9.9 | -- |
| 2 RankAlign | ✓ | 6.3 ± 10.1 | 47.0 ± 4.2 | -- | 58.2 ± 5.5 | -- |
| 3 New + fsx [-TC] | ✓ | -9.7 ± 5.3 | 4.8 ± 2.4 | -- | 29.0 ± 7.1 | -- |
| 4 New + PMI + fsx | ✓ | -10.0 ± 8.2 | 0.7 ± 8.0 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | -9.0 ± 8.8 | N/A | N/A | 29.7 ± 11.0 | -- |
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
| 1 SFT labelonly 10% | ✓ | 59.6 ± 2.8 | 58.2 ± 1.9 | -- | 47.4 ± 2.1 | -- |
| 2 RankAlign | ✓ | 66.1 ± 6.0 | 57.2 ± 5.8 | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 65.1 ± 3.5 | 60.8 ± 2.5 | -- | 47.0 ± 2.7 | -- |
| 4 New + PMI + fsx | ✓ | 67.3 ± 2.7 | 67.3 ± 2.0 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 63.8 ± 3.1 | N/A | N/A | 57.3 ± 2.4 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × rosch — delta-bins 10

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 67.1 | 67.2 | -- | -- | -- |
| 2 RankAlign | ✓ | 70.5 ± 3.3 | 70.0 ± 2.6 | -- | 60.0 ± 3.0 | -- |
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
| 2 RankAlign | ✓ | 24.3 ± 7.8 | 32.7 ± 7.2 | -- | 31.0 ± 6.9 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 12.0 ± 6.5 | 22.6 ± 5.1 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 16.8 ± 6.8 | N/A | N/A | 15.9 ± 5.7 | -- |
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
| 2 RankAlign | ✓ | 40.3 ± 8.1 | 58.3 ± 5.0 | -- | -- | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | -2.9 ± 7.7 | 34.0 ± 9.0 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 3.6 ± 8.8 | N/A | N/A | 43.2 ± 5.9 | -- |
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
| 2 RankAlign | ✓ | 54.7 ± 2.8 | -- | 59.4 ± 3.8 | -- | 60.9 ± 4.5 |
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
| 2 RankAlign | ✓ | 80.1 ± 0.9 | 83.5 ± 1.0 | -- | 84.2 ± 1.0 | -- |
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
| 2 RankAlign | ✓ | -6.1 ± 9.4 | -- | -0.9 ± 8.5 | -- | 66.6 ± 2.3 |
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
| 2 RankAlign | ✓ | 9.2 ± 7.1 | 38.9 ± 4.0 | -- | 49.0 ± 4.9 | -- |
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
| 7 New + NegTC + fsx | ✓ | 55.2 ± 3.2 | N/A | N/A | -- | 63.5 ± 5.3 |
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
| 2 RankAlign | ✓ | 70.4 ± 3.0 | 61.2 ± 2.8 | -- | 59.5 ± 3.0 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 56.0 ± 3.1 | 69.4 ± 2.6 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 54.8 ± 3.2 | N/A | N/A | 69.4 ± 2.9 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |
