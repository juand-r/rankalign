# Pod Results — Spearman(gen, val) — 2026-05-25T19:21:04Z

All cells: **Spearman(gen, val) × 100**, mean ± SE across the eval-task split for that section (no ± when a single task). Scope: **v7 only** (v6 excluded), models **Qwen3.5-9B** and **gemma-2-9b-it**, evaluated on RunPod and downloaded locally.

**Groups** (per model × eval-set, shown in this order):
1. **delta-bins 10 · eval_model_sN (canonical)** — the primary eval (downloaded model symlinked as `eval_model_sN`; 20 OOD prompts + ID). Training delta recovered from the source HF repo commit messages: gemma-2-9b-it ifeval s1=1.89, s2=1.93, s3/s4/s7=1.94 (all delta-bins 10). Qwen experiments are delta-bins 10 throughout (per design).
2. **delta-bins 10 · named deltaX.XX (earlier on-pod eval)** — an earlier/partial eval whose filename encodes the delta. Shown for comparison; differs from the canonical eval mainly because it covers fewer prompts.
3. **delta 0.15 (fixed) · v7b** — the fixed delta=0.15 batch (v7b training pods confirmed `--delta 0.15`). Eval mostly not started -> `soon`.

**Columns** = scoring method at eval time. Raw = log P(y|x); basetyp-/self- = PMI vs base/self; basetypneg-/neg- = Neg vs base/self. `N/A` = variant not run for that setting (s4 = basetyp+self only; s7 = basetypneg+neg only).

**Cells:** `--` = no data · `soon` = eval in flight. Train: ✓ data present · soon in flight · – not run.

> ifeval **OOD** = prompts 1–21 (fully held out); **ID** = prompts 22–109 (50% completions held out). This is the data split, independent of delta.

## gemma-2-9b-it × ifeval OOD — delta-bins 10 · eval_model_sN (canonical)

> Per-setting training delta (HF-commit provenance): s1=1.89, s2=1.93, s3/s4/s7=1.94 — all delta-bins 10.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 8.2 ± 7.6 | 7.9 ± 8.3 | -- | 5.4 ± 7.0 | -- |
| 2 RankAlign | ✓ | 21.0 ± 7.9 | 29.3 ± 7.8 | -- | 25.7 ± 7.8 | -- |
| 3 New + fsx [-TC] | ✓ | 36.9 ± 5.8 | 46.5 ± 4.9 | -- | 39.9 ± 4.3 | -- |
| 4 New + PMI + fsx | ✓ | 34.5 ± 6.8 | 49.4 ± 5.6 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 40.9 ± 6.8 | N/A | N/A | 47.4 ± 4.4 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × ifeval OOD — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | -2.5 ± 7.2 | 3.7 ± 7.3 | -- | -- | -- |
| 2 RankAlign | ✓ | 26.4 ± 8.7 | 33.3 ± 7.1 | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 23.0 ± 4.7 | 38.7 ± 5.4 | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 27.7 ± 5.3 | 52.6 ± 7.9 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 27.7 ± 6.0 | N/A | N/A | 47.0 ± 4.6 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × ifeval ID — delta-bins 10 · eval_model_sN (canonical)

> Per-setting training delta (HF-commit provenance): s1=1.89, s2=1.93, s3/s4/s7=1.94 — all delta-bins 10.

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | soon | soon | soon | soon | soon | soon |
| 2 RankAlign | soon | soon | soon | soon | soon | soon |
| 3 New + fsx [-TC] | soon | soon | soon | soon | soon | soon |
| 4 New + PMI + fsx | ✓ | 43.7 ± 3.0 | 58.7 ± 2.2 | 58.2 ± 2.3 | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | soon | soon | N/A | N/A | soon | soon |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × persona ID — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 52.1 ± 2.6 | 55.8 ± 4.8 | -- | 58.6 ± 5.2 | -- |
| 2 RankAlign | ✓ | 75.8 ± 0.2 | 74.5 ± 1.4 | -- | 74.5 ± 1.2 | -- |
| 3 New + fsx [-TC] | ✓ | 67.7 ± 0.5 | 71.9 ± 1.0 | -- | 71.5 ± 0.6 | -- |
| 4 New + PMI + fsx | ✓ | 61.4 ± 1.7 | 67.7 ± 0.7 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 62.9 ± 1.9 | N/A | N/A | 70.9 ± 0.4 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × persona ID — delta 0.15 (fixed) · v7b

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 80.2 ± 0.1 | 73.9 ± 0.3 | -- | -- | -- |
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

## gemma-2-9b-it × persona OOD — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | -14.7 ± 4.1 | -15.2 ± 6.4 | -- | 6.5 ± 7.2 | -- |
| 2 RankAlign | ✓ | 8.7 ± 11.5 | 47.9 ± 3.4 | -- | 58.4 ± 3.3 | -- |
| 3 New + fsx [-TC] | ✓ | -5.8 ± 4.5 | 4.6 ± 2.1 | -- | 29.8 ± 6.4 | -- |
| 4 New + PMI + fsx | ✓ | -7.9 ± 5.3 | -0.7 ± 4.3 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | -6.5 ± 8.1 | N/A | N/A | 30.4 ± 9.5 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × rosch — delta-bins 10 · eval_model_sN (canonical)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 63.3 ± 2.5 | 60.3 ± 2.2 | -- | 49.5 ± 2.7 | -- |
| 2 RankAlign | – | -- | -- | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 68.5 ± 3.1 | 61.8 ± 2.7 | -- | 49.8 ± 3.0 | -- |
| 4 New + PMI + fsx | ✓ | 70.8 ± 2.7 | 68.6 ± 2.5 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 67.6 ± 2.6 | N/A | N/A | 61.4 ± 2.5 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × rosch — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 66.5 | 68.0 | -- | -- | -- |
| 2 RankAlign | ✓ | 72.5 ± 2.6 | 70.8 ± 2.4 | -- | 62.1 ± 3.0 | -- |
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

## gemma-2-9b-it × rosch — delta 0.15 (fixed) · v7b

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 64.1 ± 4.0 | 56.5 ± 6.0 | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 72.7 ± 4.3 | 62.9 ± 4.7 | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 70.3 | 70.9 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 65.5 ± 2.6 | N/A | N/A | 55.3 ± 6.7 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × ifeval OOD — delta-bins 10 · eval_model_sN (canonical)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 21.6 ± 7.2 | 31.6 ± 7.1 | -- | 28.5 ± 6.4 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 9.4 ± 6.3 | 23.7 ± 5.1 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 13.1 ± 6.3 | N/A | N/A | 13.7 ± 5.4 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × ifeval OOD — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 31.2 ± 8.6 | 56.5 ± 5.1 | -- | -- | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | -0.3 ± 7.0 | 34.2 ± 7.6 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 5.8 ± 7.4 | N/A | N/A | 37.6 ± 4.9 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × persona ID — delta-bins 10 · eval_model_sN (canonical)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 59.0 ± 3.8 | -- | 62.0 ± 3.4 | -- | 64.3 ± 3.1 |
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

## Qwen3.5-9B × persona ID — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 79.0 ± 1.2 | 78.8 ± 1.6 | -- | 79.7 ± 1.4 | -- |
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

## Qwen3.5-9B × persona OOD — delta-bins 10 · eval_model_sN (canonical)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | -4.3 ± 12.1 | -- | 0.1 ± 9.6 | -- | 68.5 ± 0.5 |
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

## Qwen3.5-9B × persona OOD — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 11.6 ± 9.8 | 37.0 ± 4.2 | -- | 47.7 ± 4.8 | -- |
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

## Qwen3.5-9B × rosch — delta-bins 10 · eval_model_sN (canonical)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | – | -- | -- | -- | -- | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | – | -- | -- | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 60.6 ± 3.1 | N/A | N/A | -- | 67.1 ± 5.0 |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## Qwen3.5-9B × rosch — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | – | -- | -- | -- | -- | -- |
| 2 RankAlign | ✓ | 71.2 ± 3.2 | 62.4 ± 2.8 | -- | 60.1 ± 3.4 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 60.0 ± 3.1 | 71.7 ± 3.0 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 59.6 ± 3.3 | N/A | N/A | 71.8 ± 3.4 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |
