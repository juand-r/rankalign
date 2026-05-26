# Gemma-2-2b-it × Rosch — full v7 aggregation (fresh from disk)

_Aggregated 130 unique CSV files. Cells: mean ± SE across the 10 rosch tasks._

**Eval-column convention:** `Raw` = log P(y|x); `PMI {self,base}` = tc-correction using {fine-tuned, base} model logprobs as the typicality reference; `Neg {self,base}` = same with the negated-prompt alternative. For the **Base** row only, `PMI self == PMI base` and `Neg self == Neg base` since the base model IS its own self.

**Settings table:**

| # | Setting | Train fingerprint | Epoch | Note |
|---|---|---|---|---|
| 0 | Base | `v6-google_gemma-2-2b-it` | — |  |
| 1 | SFT labelonly 10% | `—` | — | no eval CSVs on disk — never trained or evaluated |
| 2 | RankAlign | `v7-google--gemma-2-2b-it-delta0.85-epoch2--membership-sans-rosch…` | e2 |  |
| 3 | New + fsx [-TC] | `v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-nv1-ng1-v…` | e2 |  |
| 4 | New + PMI + fsx | `v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcs-nv1-n…` | e2 |  |
| 7 | New + NegTC + fsx | `v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcn-nv1-n…` | e2 |  |
| 11 | New + PMI [-fsx] | `v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcs-nv1-n…` | e2 |  |
| 12 | New + NegTC [-fsx] | `v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcn-nv1-n…` | e2 |  |
| 13 | SFT + CFT | `v7-gemma-2-2b-it-d0.81-e2-membership-sans-rosch-v0-all-p0-nv1-ng…` | e2 |  |

### GenROC × 100

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 64.7 ± 2.9 | 72.3 ± 3.7 | 72.3 ± 3.7 | 79.8 ± 3.4 | 79.8 ± 3.4 |
| 1 | SFT labelonly 10% | — | -- | -- | -- | -- | -- |
| 2 | RankAlign | e2 | 79.2 ± 2.8 | -- | 78.2 ± 3.2 | -- | -- |
| 3 | New + fsx [-TC] | e2 | 81.2 ± 2.9 | -- | 81.8 ± 2.4 | -- | 79.0 ± 1.6 |
| 4 | New + PMI + fsx | e2 | 72.9 ± 3.8 | -- | 82.1 ± 2.7 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 73.8 ± 3.3 | -- | -- | -- | 81.7 ± 2.3 |
| 11 | New + PMI [-fsx] | e2 | 77.1 ± 3.7 | -- | 85.0 ± 2.6 | -- | -- |
| 12 | New + NegTC [-fsx] | e2 | 77.9 ± 3.4 | -- | -- | -- | 83.5 ± 1.7 |
| 13 | SFT + CFT | e2 | 84.9 ± 1.7 | 88.1 ± 1.3 | 85.5 ± 1.3 | 87.1 ± 2.1 | 83.7 ± 1.0 |

### ValROC × 100

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 90.3 ± 2.4 | 90.3 ± 2.4 | 90.3 ± 2.4 | 90.3 ± 2.4 | 90.3 ± 2.4 |
| 1 | SFT labelonly 10% | — | -- | -- | -- | -- | -- |
| 2 | RankAlign | e2 | 88.5 ± 2.8 | -- | 88.5 ± 2.8 | -- | -- |
| 3 | New + fsx [-TC] | e2 | 88.1 ± 2.9 | -- | 88.1 ± 2.9 | -- | 88.1 ± 2.9 |
| 4 | New + PMI + fsx | e2 | 81.7 ± 4.0 | -- | 81.7 ± 4.0 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 90.2 ± 2.8 | -- | -- | -- | 90.2 ± 2.8 |
| 11 | New + PMI [-fsx] | e2 | 89.4 ± 2.7 | -- | 89.4 ± 2.7 | -- | -- |
| 12 | New + NegTC [-fsx] | e2 | 87.1 ± 2.6 | -- | -- | -- | 87.1 ± 2.6 |
| 13 | SFT + CFT | e2 | 89.7 ± 3.0 | 89.7 ± 3.0 | 89.7 ± 3.0 | 89.7 ± 3.0 | 89.7 ± 3.0 |

### ValAcc × 100

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 79.6 ± 2.2 | 79.6 ± 2.2 | 79.6 ± 2.2 | 79.6 ± 2.2 | 79.6 ± 2.2 |
| 1 | SFT labelonly 10% | — | -- | -- | -- | -- | -- |
| 2 | RankAlign | e2 | 75.4 ± 3.1 | -- | 75.4 ± 3.1 | -- | -- |
| 3 | New + fsx [-TC] | e2 | 77.4 ± 2.2 | -- | 77.4 ± 2.2 | -- | 77.4 ± 2.2 |
| 4 | New + PMI + fsx | e2 | 72.6 ± 3.6 | -- | 72.6 ± 3.6 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 78.2 ± 2.9 | -- | -- | -- | 78.2 ± 2.9 |
| 11 | New + PMI [-fsx] | e2 | 71.0 ± 2.3 | -- | 71.0 ± 2.3 | -- | -- |
| 12 | New + NegTC [-fsx] | e2 | 77.9 ± 3.1 | -- | -- | -- | 77.9 ± 3.1 |
| 13 | SFT + CFT | e2 | 80.7 ± 3.0 | 80.7 ± 3.0 | 80.7 ± 3.0 | 80.7 ± 3.0 | 80.7 ± 3.0 |

### Pearson r

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 0.237 ± 0.047 | 0.441 ± 0.040 | 0.441 ± 0.040 | 0.486 ± 0.045 | 0.486 ± 0.045 |
| 1 | SFT labelonly 10% | — | -- | -- | -- | -- | -- |
| 2 | RankAlign | e2 | 0.549 ± 0.042 | -- | 0.543 ± 0.032 | -- | -- |
| 3 | New + fsx [-TC] | e2 | 0.581 ± 0.056 | -- | 0.587 ± 0.020 | -- | 0.463 ± 0.034 |
| 4 | New + PMI + fsx | e2 | 0.477 ± 0.056 | -- | 0.583 ± 0.036 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 0.416 ± 0.048 | -- | -- | -- | 0.580 ± 0.030 |
| 11 | New + PMI [-fsx] | e2 | 0.502 ± 0.053 | -- | 0.659 ± 0.030 | -- | -- |
| 12 | New + NegTC [-fsx] | e2 | 0.485 ± 0.051 | -- | -- | -- | 0.603 ± 0.023 |
| 13 | SFT + CFT | e2 | 0.565 ± 0.038 | 0.651 ± 0.029 | 0.572 ± 0.026 | 0.640 ± 0.027 | 0.496 ± 0.024 |

### Spearman r

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 0.318 ± 0.049 | 0.474 ± 0.045 | 0.474 ± 0.045 | 0.501 ± 0.044 | 0.501 ± 0.044 |
| 1 | SFT labelonly 10% | — | -- | -- | -- | -- | -- |
| 2 | RankAlign | e2 | 0.550 ± 0.044 | -- | 0.527 ± 0.032 | -- | -- |
| 3 | New + fsx [-TC] | e2 | 0.616 ± 0.055 | -- | 0.573 ± 0.019 | -- | 0.473 ± 0.033 |
| 4 | New + PMI + fsx | e2 | 0.536 ± 0.050 | -- | 0.582 ± 0.034 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 0.444 ± 0.037 | -- | -- | -- | 0.599 ± 0.025 |
| 11 | New + PMI [-fsx] | e2 | 0.570 ± 0.049 | -- | 0.688 ± 0.020 | -- | -- |
| 12 | New + NegTC [-fsx] | e2 | 0.542 ± 0.042 | -- | -- | -- | 0.596 ± 0.029 |
| 13 | SFT + CFT | e2 | 0.602 ± 0.038 | 0.642 ± 0.024 | 0.570 ± 0.024 | 0.634 ± 0.026 | 0.516 ± 0.027 |
