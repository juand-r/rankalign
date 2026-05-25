# Gemma-2-9b-it v7 Results — 2026-05-25

Model: gemma-2-9b-it, trained with rankalign v7 (epoch2), evaluated with base-model typicality correction.

**Columns**: gen_roc × 100 unless noted.
- Raw = log P(y|x)
- PMI base = (log P(y|x) − log P_base(y|x)) / len  [from basetyp- eval files]
- Neg base = (log P(y|x) − log P_neg(y|x)) / len  [from basetypneg- eval files]
- LenNorm = Raw / num_tokens
- PMI+Len = PMI base / num_tokens
- Neg+Len = Neg base / num_tokens
- ValROC = ROC-AUC of validation score
- N/A = this eval type was not run for this setting (s4 only runs PMI; s7 only runs Neg)

---

## Persona

Tasks trained on: persona-v1-all
- **ID** (label-flipped, in-domain): psychopathy, machiavellianism, narcissism
- **OOD** (held-out): desire-to-create-allies, interest-in-music, interest-in-science

### Persona — all 6 tasks (mean gen_roc × 100)

| Setting | Label | Raw | PMI base | Neg base | LenNorm | PMI+Len | Neg+Len | ValROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1 | SFT labelonly 10% | 54.9 | 58.7 | 66.6 | 67.9 | 58.1 | 62.5 | 96.8 |
| s2 | RankAlign | 74.4 | 88.4 | 91.4 | 85.7 | 89.6 | 90.3 | 99.3 |
| s3 | New + fsx [-TC] | 64.4 | 72.9 | 80.6 | 79.1 | 72.2 | 76.0 | 97.3 |
| s4 | New + PMI + fsx | 61.8 | 69.3 | N/A | 76.7 | 69.9 | N/A | 97.0 |
| s7 | New + NegTC + fsx | 62.7 | N/A | 79.8 | 77.4 | N/A | 74.2 | 98.4 |

### Persona — ID (3 in-domain tasks, mean gen_roc × 100)

| Setting | Label | Raw | PMI base | Neg base | LenNorm | PMI+Len | Neg+Len | ValROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1 | SFT labelonly 10% | 74.5 | 80.2 | 82.8 | 76.0 | 78.2 | 79.3 | 100.0 |
| s2 | RankAlign | 96.6 | 98.7 | 98.1 | 90.8 | 98.6 | 98.3 | 99.2 |
| s3 | New + fsx [-TC] | 86.7 | 93.5 | 93.8 | 87.5 | 92.7 | 92.5 | 100.0 |
| s4 | New + PMI + fsx | 82.5 | 90.2 | N/A | 84.4 | 89.2 | N/A | 100.0 |
| s7 | New + NegTC + fsx | 83.2 | N/A | 92.0 | 84.4 | N/A | 89.9 | 100.0 |

### Persona — OOD (3 held-out tasks, mean gen_roc × 100)

| Setting | Label | Raw | PMI base | Neg base | LenNorm | PMI+Len | Neg+Len | ValROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1 | SFT labelonly 10% | 35.2 | 37.2 | 50.4 | 59.8 | 38.0 | 45.6 | 93.7 |
| s2 | RankAlign | 52.2 | 78.2 | 84.7 | 80.6 | 80.5 | 82.3 | 99.4 |
| s3 | New + fsx [-TC] | 42.0 | 52.3 | 67.4 | 70.7 | 51.7 | 59.6 | 94.7 |
| s4 | New + PMI + fsx | 41.0 | 48.5 | N/A | 69.0 | 50.7 | N/A | 94.0 |
| s7 | New + NegTC + fsx | 42.1 | N/A | 67.5 | 70.5 | N/A | 58.5 | 96.7 |

#### Per-task breakdown (Raw gen_roc × 100)

### Persona — Raw

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| persona-v1-psychopathy | 77.7 | 98.9 | 88.1 | 86.4 | 85.2 |
| persona-v1-machiavellianism | 76.0 | 97.4 | 87.5 | 82.5 | 83.3 |
| persona-v1-narcissism | 69.9 | 93.5 | 84.5 | 78.7 | 81.1 |
| persona-v1-desire-to-create-allies | 23.0 | 35.7 | 27.5 | 25.7 | 25.8 |
| persona-v1-interest-in-music | 50.3 | 66.7 | 56.0 | 57.0 | 60.1 |
| persona-v1-interest-in-science | 32.3 | 54.2 | 42.6 | 40.3 | 40.5 |

#### Per-task breakdown (PMI base gen_roc × 100)

### Persona — PMI base

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| persona-v1-psychopathy | 82.2 | 98.9 | 93.5 | 92.2 | N/A |
| persona-v1-machiavellianism | 83.8 | 99.1 | 94.3 | 91.5 | N/A |
| persona-v1-narcissism | 74.6 | 98.0 | 92.7 | 86.8 | N/A |
| persona-v1-desire-to-create-allies | 26.2 | 70.3 | 42.7 | 34.9 | N/A |
| persona-v1-interest-in-music | 53.6 | 81.8 | 60.4 | 61.6 | N/A |
| persona-v1-interest-in-science | 31.7 | 82.4 | 53.6 | 49.0 | N/A |

#### Per-task breakdown (Neg base gen_roc × 100)

### Persona — Neg base

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| persona-v1-psychopathy | 85.8 | 99.0 | 94.5 | N/A | 92.4 |
| persona-v1-machiavellianism | 86.1 | 98.5 | 94.7 | N/A | 93.1 |
| persona-v1-narcissism | 76.5 | 96.8 | 92.3 | N/A | 90.5 |
| persona-v1-desire-to-create-allies | 34.9 | 78.5 | 55.2 | N/A | 49.8 |
| persona-v1-interest-in-music | 65.9 | 86.5 | 73.4 | N/A | 79.9 |
| persona-v1-interest-in-science | 50.5 | 89.2 | 73.6 | N/A | 72.8 |

---

## Membership / Rosch

10 rosch category-typicality tasks (all OOD w.r.t. the membership-sans-rosch training split).

### Rosch — all 10 tasks (mean gen_roc × 100)

| Setting | Label | Raw | PMI base | Neg base | LenNorm | PMI+Len | Neg+Len | ValROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1 | SFT labelonly 10% | 83.6 | 83.9 | 80.2 | 85.0 | 74.1 | 70.6 | 94.5 |
| s2 | RankAlign | — | — | — | — | — | — | — |
| s3 | New + fsx [-TC] | 87.0 | 85.1 | 79.8 | 87.0 | 74.2 | 69.8 | 95.0 |
| s4 | New + PMI + fsx | 87.3 | 88.6 | N/A | 89.5 | 75.8 | N/A | 94.7 |
| s7 | New + NegTC + fsx | 84.5 | N/A | 85.1 | 88.1 | N/A | 72.3 | 94.6 |

#### Per-task breakdown (Raw gen_roc × 100)

### Rosch — Raw

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| rosch-bird | 95.3 | — | 97.5 | 97.6 | 97.6 |
| rosch-carpenters-tool | 84.2 | — | 82.1 | 81.7 | 79.8 |
| rosch-clothing | 82.4 | — | 87.5 | 89.7 | 89.6 |
| rosch-fruit | 80.3 | — | 88.4 | 86.2 | 83.4 |
| rosch-furniture | 76.4 | — | 86.1 | 86.1 | 74.5 |
| rosch-sport | 88.9 | — | 89.1 | 89.4 | 90.3 |
| rosch-toy | 80.3 | — | 82.5 | 82.8 | 79.3 |
| rosch-vegetable | 81.6 | — | 85.8 | 84.9 | 81.5 |
| rosch-vehicle | 82.7 | — | 85.1 | 86.5 | 85.3 |
| rosch-weapon | 83.6 | — | 85.4 | 88.1 | 84.0 |

#### Per-task breakdown (PMI base gen_roc × 100)

### Rosch — PMI base

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| rosch-bird | 88.7 | — | 87.1 | 91.9 | N/A |
| rosch-carpenters-tool | 83.5 | — | 81.8 | 84.9 | N/A |
| rosch-clothing | 77.8 | — | 79.5 | 85.1 | N/A |
| rosch-fruit | 86.5 | — | 91.1 | 92.8 | N/A |
| rosch-furniture | 84.8 | — | 92.1 | 94.0 | N/A |
| rosch-sport | 85.0 | — | 84.1 | 86.4 | N/A |
| rosch-toy | 81.2 | — | 81.4 | 85.0 | N/A |
| rosch-vegetable | 85.1 | — | 86.2 | 90.6 | N/A |
| rosch-vehicle | 85.9 | — | 87.2 | 89.5 | N/A |
| rosch-weapon | 80.7 | — | 80.5 | 85.3 | N/A |

#### Per-task breakdown (Neg base gen_roc × 100)

### Rosch — Neg base

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| rosch-bird | 79.4 | — | 74.8 | N/A | 85.7 |
| rosch-carpenters-tool | 78.9 | — | 77.9 | N/A | 79.3 |
| rosch-clothing | 72.4 | — | 71.9 | N/A | 83.6 |
| rosch-fruit | 85.9 | — | 89.3 | N/A | 91.7 |
| rosch-furniture | 83.9 | — | 89.1 | N/A | 88.6 |
| rosch-sport | 76.3 | — | 71.5 | N/A | 79.1 |
| rosch-toy | 80.6 | — | 79.5 | N/A | 81.6 |
| rosch-vegetable | 77.6 | — | 77.2 | N/A | 84.7 |
| rosch-vehicle | 82.7 | — | 82.3 | N/A | 87.5 |
| rosch-weapon | 84.5 | — | 84.0 | N/A | 89.3 |

---

## IFEval

20 prompts (1-13, 15-21; prompt 14 has no data).
Training task: `ifeval-concat`. Prompts 1-21 are OOD (never appeared in training); prompts 22+ are ID but not evaluated here.

### IFEval — all 20 prompts (mean gen_roc × 100)

| Setting | Label | Raw | PMI base | Neg base | LenNorm | PMI+Len | Neg+Len | ValROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s1 | SFT labelonly 10% | 50.2 | 57.2 | 49.3 | 53.8 | 58.3 | 51.0 | 53.2 |
| s2 | RankAlign | 56.3 | 62.8 | 59.6 | 60.4 | 62.1 | 60.1 | 70.5 |
| s3 | New + fsx [-TC] | 60.9 | 73.0 | 65.2 | 67.8 | 74.7 | 68.4 | 75.3 |
| s4 | New + PMI + fsx | 60.9 | 81.8 | N/A | 68.2 | 78.3 | N/A | 75.3 |
| s7 | New + NegTC + fsx | 62.2 | N/A | 68.1 | 68.8 | N/A | 69.1 | 73.7 |

#### Per-prompt breakdown (Raw gen_roc × 100)

### IFEval — Raw

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| ifeval-prompt_1 | 51.5 | 72.1 | 65.5 | 66.3 | 64.8 |
| ifeval-prompt_2 | 49.9 | 59.9 | 52.9 | 52.7 | 55.0 |
| ifeval-prompt_3 | 59.3 | 47.7 | 67.0 | 65.0 | 67.8 |
| ifeval-prompt_4 | 31.8 | 41.9 | 40.1 | 41.2 | 39.8 |
| ifeval-prompt_5 | 43.9 | 36.9 | 59.4 | 54.8 | 49.3 |
| ifeval-prompt_6 | 55.1 | 62.1 | 61.5 | 59.7 | 58.7 |
| ifeval-prompt_7 | 43.1 | 32.2 | 55.4 | 63.9 | 64.3 |
| ifeval-prompt_8 | 37.6 | 53.1 | 56.5 | 56.8 | 62.5 |
| ifeval-prompt_9 | 39.0 | 30.9 | 41.2 | 42.6 | 44.4 |
| ifeval-prompt_10 | 38.8 | 64.3 | 65.5 | 64.0 | 69.6 |
| ifeval-prompt_11 | 24.4 | 47.5 | 25.8 | 28.0 | 26.4 |
| ifeval-prompt_12 | 20.5 | 44.3 | 41.9 | 39.8 | 39.7 |
| ifeval-prompt_13 | 58.9 | 52.5 | 60.7 | 62.6 | 66.5 |
| ifeval-prompt_15 | 61.0 | 84.6 | 80.9 | 81.3 | 84.0 |
| ifeval-prompt_16 | 70.3 | 23.2 | 77.0 | 69.7 | 74.8 |
| ifeval-prompt_17 | 38.6 | 57.6 | 52.0 | 54.0 | 52.3 |
| ifeval-prompt_18 | 37.0 | 62.0 | 52.9 | 55.6 | 59.7 |
| ifeval-prompt_19 | 62.5 | 59.7 | 64.3 | 62.4 | 65.8 |
| ifeval-prompt_20 | 96.7 | 94.2 | 98.1 | 98.6 | 98.0 |
| ifeval-prompt_21 | 83.3 | 100.0 | 99.8 | 99.9 | 99.8 |

#### Per-prompt breakdown (PMI base gen_roc × 100)

### IFEval — PMI base

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| ifeval-prompt_1 | 52.0 | 84.1 | 81.6 | 93.9 | N/A |
| ifeval-prompt_2 | 58.0 | 62.5 | 74.7 | 72.0 | N/A |
| ifeval-prompt_3 | 84.4 | 48.0 | 91.2 | 95.0 | N/A |
| ifeval-prompt_4 | 40.7 | 75.4 | 69.7 | 94.2 | N/A |
| ifeval-prompt_5 | 44.1 | 34.8 | 70.9 | 78.3 | N/A |
| ifeval-prompt_6 | 61.1 | 65.4 | 71.4 | 69.2 | N/A |
| ifeval-prompt_7 | 69.5 | 34.9 | 81.6 | 87.0 | N/A |
| ifeval-prompt_8 | 46.9 | 55.5 | 72.6 | 86.3 | N/A |
| ifeval-prompt_9 | 44.6 | 44.0 | 45.2 | 71.3 | N/A |
| ifeval-prompt_10 | 56.0 | 71.9 | 84.6 | 93.2 | N/A |
| ifeval-prompt_11 | 29.4 | 48.8 | 41.9 | 69.1 | N/A |
| ifeval-prompt_12 | 24.8 | 70.7 | 46.5 | 79.0 | N/A |
| ifeval-prompt_13 | 61.4 | 42.4 | 58.8 | 66.9 | N/A |
| ifeval-prompt_15 | 63.5 | 84.6 | 84.3 | 83.5 | N/A |
| ifeval-prompt_16 | 77.1 | 22.4 | 89.6 | 84.7 | N/A |
| ifeval-prompt_17 | 44.1 | 70.5 | 61.0 | 85.5 | N/A |
| ifeval-prompt_18 | 36.9 | 86.3 | 67.5 | 87.5 | N/A |
| ifeval-prompt_19 | 67.6 | 60.1 | 67.2 | 62.0 | N/A |
| ifeval-prompt_20 | 97.6 | 94.1 | 99.8 | 79.9 | N/A |
| ifeval-prompt_21 | 83.9 | 100.0 | 100.0 | 97.4 | N/A |

#### Per-prompt breakdown (Neg base gen_roc × 100)

### IFEval — Neg base

| Task | s1 | s2 | s3 | s4 | s7 |
| --- | --- | --- | --- | --- | --- |
| ifeval-prompt_1 | 49.8 | 81.9 | 78.5 | N/A | 87.6 |
| ifeval-prompt_2 | 48.8 | 58.4 | 59.3 | N/A | 67.3 |
| ifeval-prompt_3 | 56.3 | 46.5 | 61.9 | N/A | 47.6 |
| ifeval-prompt_4 | 31.3 | 58.4 | 54.3 | N/A | 76.2 |
| ifeval-prompt_5 | 41.3 | 33.5 | 69.2 | N/A | 46.6 |
| ifeval-prompt_6 | 56.8 | 64.8 | 68.4 | N/A | 60.4 |
| ifeval-prompt_7 | 53.2 | 33.3 | 75.8 | N/A | 78.9 |
| ifeval-prompt_8 | 36.1 | 54.0 | 65.5 | N/A | 74.3 |
| ifeval-prompt_9 | 38.5 | 41.1 | 41.1 | N/A | 52.1 |
| ifeval-prompt_10 | 42.7 | 61.5 | 65.9 | N/A | 75.3 |
| ifeval-prompt_11 | 26.1 | 48.4 | 33.4 | N/A | 47.1 |
| ifeval-prompt_12 | 23.5 | 68.4 | 46.3 | N/A | 57.6 |
| ifeval-prompt_13 | 56.2 | 34.7 | 53.9 | N/A | 58.6 |
| ifeval-prompt_15 | 55.8 | 82.6 | 79.3 | N/A | 78.7 |
| ifeval-prompt_16 | 50.1 | 17.0 | 64.9 | N/A | 50.3 |
| ifeval-prompt_17 | 39.9 | 68.1 | 59.9 | N/A | 71.0 |
| ifeval-prompt_18 | 35.8 | 86.3 | 65.6 | N/A | 85.2 |
| ifeval-prompt_19 | 61.5 | 59.3 | 61.7 | N/A | 62.0 |
| ifeval-prompt_20 | 96.4 | 94.0 | 99.4 | N/A | 92.5 |
| ifeval-prompt_21 | 85.7 | 100.0 | 99.6 | N/A | 91.9 |
