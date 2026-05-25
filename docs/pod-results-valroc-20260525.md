# Pod Results — ValROC — 2026-05-25T19:21:04Z

All cells: **ValROC × 100**, mean ± SE across the eval-task split for that section (no ± when a single task). Scope: **v7 only** (v6 excluded), models **Qwen3.5-9B** and **gemma-2-9b-it**, evaluated on RunPod and downloaded locally.

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
| 1 SFT labelonly 10% | ✓ | 53.2 ± 4.4 | 53.2 ± 4.4 | -- | 53.2 ± 4.4 | -- |
| 2 RankAlign | ✓ | 70.5 ± 4.0 | 70.5 ± 4.0 | -- | 70.5 ± 4.0 | -- |
| 3 New + fsx [-TC] | ✓ | 75.3 ± 3.8 | 75.3 ± 3.8 | -- | 75.3 ± 3.8 | -- |
| 4 New + PMI + fsx | ✓ | 75.3 ± 3.9 | 75.3 ± 3.9 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 73.7 ± 4.0 | N/A | N/A | 73.7 ± 4.0 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × ifeval OOD — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 77.4 ± 4.5 | 77.4 ± 4.5 | -- | -- | -- |
| 2 RankAlign | ✓ | 61.5 ± 4.9 | 61.5 ± 4.9 | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 77.0 ± 5.2 | 77.0 ± 5.2 | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 76.3 ± 4.4 | 76.3 ± 4.4 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 75.0 ± 4.6 | N/A | N/A | 75.0 ± 4.6 | -- |
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
| 4 New + PMI + fsx | ✓ | 83.1 ± 1.4 | 83.1 ± 1.4 | 83.1 ± 1.4 | N/A | N/A |
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
| 1 SFT labelonly 10% | ✓ | 100.0 ± 0.0 | 100.0 ± 0.0 | -- | 100.0 ± 0.0 | -- |
| 2 RankAlign | ✓ | 99.2 ± 0.8 | 99.2 ± 0.8 | -- | 99.2 ± 0.8 | -- |
| 3 New + fsx [-TC] | ✓ | 100.0 ± 0.0 | 100.0 ± 0.0 | -- | 100.0 ± 0.0 | -- |
| 4 New + PMI + fsx | ✓ | 100.0 ± 0.0 | 100.0 ± 0.0 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 100.0 ± 0.0 | N/A | N/A | 100.0 ± 0.0 | -- |
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
| 2 RankAlign | ✓ | 100.0 ± 0.0 | 100.0 ± 0.0 | -- | -- | -- |
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
| 1 SFT labelonly 10% | ✓ | 93.7 ± 6.3 | 93.7 ± 6.3 | -- | 93.7 ± 6.3 | -- |
| 2 RankAlign | ✓ | 99.4 ± 0.6 | 99.4 ± 0.6 | -- | 99.4 ± 0.6 | -- |
| 3 New + fsx [-TC] | ✓ | 94.7 ± 5.3 | 94.7 ± 5.3 | -- | 94.7 ± 5.3 | -- |
| 4 New + PMI + fsx | ✓ | 94.0 ± 5.9 | 94.0 ± 5.9 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 96.7 ± 3.2 | N/A | N/A | 96.7 ± 3.2 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × rosch — delta-bins 10 · eval_model_sN (canonical)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 94.5 ± 1.9 | 94.5 ± 1.9 | -- | 94.5 ± 1.9 | -- |
| 2 RankAlign | – | -- | -- | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 95.0 ± 1.9 | 95.0 ± 1.9 | -- | 95.0 ± 1.9 | -- |
| 4 New + PMI + fsx | ✓ | 94.7 ± 2.0 | 94.7 ± 2.0 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 94.6 ± 1.9 | N/A | N/A | 94.6 ± 1.9 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |

## gemma-2-9b-it × rosch — delta-bins 10 · named deltaX.XX (earlier on-pod eval)

| Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self) |
|---|---|---|---|---|---|---|
| 0 Base | (base model) | -- | -- | -- | -- | -- |
| 1 SFT labelonly 10% | ✓ | 98.4 | 98.4 | -- | -- | -- |
| 2 RankAlign | ✓ | 95.2 ± 1.8 | 95.2 ± 1.8 | -- | 95.2 ± 1.8 | -- |
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
| 2 RankAlign | ✓ | 95.0 ± 3.5 | 95.0 ± 3.5 | -- | -- | -- |
| 3 New + fsx [-TC] | ✓ | 95.3 ± 2.4 | 95.3 ± 2.4 | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 99.6 | 99.6 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 94.9 ± 3.7 | N/A | N/A | 94.9 ± 3.7 | -- |
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
| 2 RankAlign | ✓ | 66.6 ± 4.4 | 66.6 ± 4.4 | -- | 66.6 ± 4.4 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 65.2 ± 3.8 | 65.2 ± 3.8 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 60.9 ± 4.0 | N/A | N/A | 60.9 ± 4.0 | -- |
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
| 2 RankAlign | ✓ | 82.4 ± 3.7 | 82.4 ± 3.7 | -- | -- | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 82.3 ± 3.6 | 82.3 ± 3.6 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 85.1 ± 3.3 | N/A | N/A | 85.1 ± 3.3 | -- |
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
| 2 RankAlign | ✓ | 95.9 ± 1.7 | -- | 95.9 ± 1.7 | -- | 95.9 ± 1.7 |
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
| 2 RankAlign | ✓ | 98.4 ± 0.5 | 98.4 ± 0.5 | -- | 98.4 ± 0.5 | -- |
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
| 2 RankAlign | ✓ | 98.7 ± 0.8 | -- | 98.7 ± 0.8 | -- | 98.7 ± 0.8 |
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
| 2 RankAlign | ✓ | 98.7 ± 0.7 | 98.7 ± 0.7 | -- | 98.7 ± 0.7 | -- |
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
| 7 New + NegTC + fsx | ✓ | 95.4 ± 1.9 | N/A | N/A | -- | 95.4 ± 1.9 |
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
| 2 RankAlign | ✓ | 95.6 ± 1.8 | 95.6 ± 1.8 | -- | 95.6 ± 1.8 | -- |
| 3 New + fsx [-TC] | – | -- | -- | -- | -- | -- |
| 4 New + PMI + fsx | ✓ | 96.0 ± 1.7 | 96.0 ± 1.7 | -- | N/A | N/A |
| 5 RA + PMI + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 6 RA + PMI [+TC] | – | -- | -- | -- | -- | -- |
| 7 New + NegTC + fsx | ✓ | 95.6 ± 1.7 | N/A | N/A | 95.6 ± 1.7 | -- |
| 8 RA + NegTC + fsx [-NLL] | – | -- | -- | -- | -- | -- |
| 9 RA + NegTC [+TC] | – | -- | -- | -- | -- | -- |
| 11 New + PMI [-fsx] | – | -- | -- | -- | -- | -- |
| 12 New + NegTC [-fsx] | – | -- | -- | -- | -- | -- |
| 13 SFT + CFT | – | -- | -- | -- | -- | -- |
