# Gemma-2-2b (NO -it) × Rosch — full v7 aggregation

_Aggregated 160 unique CSV files. Cells: mean ± SE across the 10 rosch tasks (bird, carpenters-tool, clothing, fruit, furniture, sport, toy, vegetable, vehicle, weapon)._

**Eval-column convention:** `Raw` = log P(y|x); `PMI {self,base}` = tc-correction using {fine-tuned, base} model logprobs as the typicality reference; `Neg {self,base}` = same but using the negated-prompt alternative. For the **Base** row only, `PMI self == PMI base` and `Neg self == Neg base` since the base model IS its own self.

**Settings table:**

| # | Setting | Train fingerprint | Epoch | Note |
|---|---|---|---|---|
| 0 | Base | `v6-google_gemma-2-2b` | — | |
| 1 | SFT labelonly 10% | `v7-google--gemma-2-2b-delta0.09-epoch0--membership-sans-rosch-v0…` | e0 | |
| 2 | RankAlign | `v7-google--gemma-2-2b-delta0.10-epoch1--membership-sans-rosch-v0…` | e1 | |
| 3 | New + fsx [-TC] | `v7-gemma-2-2b-d0.19-e1-membership-sans-rosch-v0-all-nv1-ng1-vlo-…` | e1 | |
| 4 | New + PMI + fsx | `v7-gemma-2-2b-d0.19-e1-membership-sans-rosch-v0-all-tcs-nv1-ng1-…` | e1 | |
| 5 | RA + PMI + fsx [-NLL] | `v7-google--gemma-2-2b-delta0.10-epoch2--membership-sans-rosch-v0…` | e2 | |
| 7 | New + NegTC + fsx | `v7-gemma-2-2b-d0.19-e2-membership-sans-rosch-v0-all-tcn-nv1-ng1-…` | e2 | |
| 11 | New + PMI [-fsx] | `v7-google--gemma-2-2b-delta0.10-epoch2--membership-sans-rosch-v0…` | e2 | |
| 13 | SFT + CFT | `v7-google--gemma-2-2b-delta0.09-epoch2--membership-sans-rosch-v0…` | e2 | |

### GenROC × 100

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 74.6 ± 1.6 | 77.3 ± 1.8 | 77.3 ± 1.8 | 77.7 ± 4.0 | 77.7 ± 4.0 |
| 1 | SFT labelonly 10% | e0 | 82.3 ± 1.7 | -- | 82.9 ± 1.7 | -- | 80.5 ± 2.0 |
| 2 | RankAlign | e1 | 76.8 ± 3.1 | -- | 79.5 ± 2.9 | -- | 63.5 ± 4.9 |
| 3 | New + fsx [-TC] | e1 | 74.8 ± 3.0 | -- | 77.3 ± 2.6 | -- | 61.8 ± 3.7 |
| 4 | New + PMI + fsx | e1 | 76.8 ± 2.4 | -- | 81.6 ± 2.4 | -- | -- |
| 5 | RA + PMI + fsx [-NLL] | e2 | 76.4 ± 2.7 | -- | 81.8 ± 2.8 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 79.2 ± 3.0 | -- | -- | -- | 75.6 ± 3.5 |
| 11 | New + PMI [-fsx] | e2 | 73.5 ± 2.9 | -- | 79.3 ± 3.0 | -- | -- |
| 13 | SFT + CFT | e2 | 80.9 ± 2.1 | 82.8 ± 2.0 | 82.5 ± 2.0 | 69.5 ± 4.6 | 81.0 ± 2.3 |

### ValROC × 100

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 89.7 ± 2.6 | 89.7 ± 2.6 | 89.7 ± 2.6 | 89.7 ± 2.6 | 89.7 ± 2.6 |
| 1 | SFT labelonly 10% | e0 | 88.2 ± 3.5 | -- | 88.2 ± 3.5 | -- | 88.2 ± 3.5 |
| 2 | RankAlign | e1 | 86.3 ± 3.7 | -- | 86.3 ± 3.7 | -- | 86.3 ± 3.7 |
| 3 | New + fsx [-TC] | e1 | 86.4 ± 3.8 | -- | 86.4 ± 3.8 | -- | 86.4 ± 3.8 |
| 4 | New + PMI + fsx | e1 | 89.9 ± 2.5 | -- | 89.9 ± 2.5 | -- | -- |
| 5 | RA + PMI + fsx [-NLL] | e2 | 89.1 ± 2.7 | -- | 89.1 ± 2.7 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 89.4 ± 3.2 | -- | -- | -- | 89.4 ± 3.2 |
| 11 | New + PMI [-fsx] | e2 | 84.4 ± 3.4 | -- | 84.4 ± 3.4 | -- | -- |
| 13 | SFT + CFT | e2 | 88.3 ± 3.1 | 88.3 ± 3.1 | 88.3 ± 3.1 | 88.3 ± 3.1 | 88.3 ± 3.1 |

### ValAcc × 100

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 78.8 ± 3.4 | 78.8 ± 3.4 | 78.8 ± 3.4 | 78.8 ± 3.4 | 78.8 ± 3.4 |
| 1 | SFT labelonly 10% | e0 | 76.2 ± 3.2 | -- | 76.2 ± 3.2 | -- | 76.2 ± 3.2 |
| 2 | RankAlign | e1 | 71.7 ± 4.1 | -- | 71.7 ± 4.1 | -- | 71.7 ± 4.1 |
| 3 | New + fsx [-TC] | e1 | 69.2 ± 2.4 | -- | 69.2 ± 2.4 | -- | 69.2 ± 2.4 |
| 4 | New + PMI + fsx | e1 | 80.0 ± 2.8 | -- | 80.0 ± 2.8 | -- | -- |
| 5 | RA + PMI + fsx [-NLL] | e2 | 78.8 ± 3.7 | -- | 78.8 ± 3.7 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 80.7 ± 3.2 | -- | -- | -- | 80.7 ± 3.2 |
| 11 | New + PMI [-fsx] | e2 | 57.1 ± 2.5 | -- | 57.1 ± 2.5 | -- | -- |
| 13 | SFT + CFT | e2 | 78.8 ± 2.7 | 78.8 ± 2.7 | 78.8 ± 2.7 | 78.8 ± 2.7 | 78.8 ± 2.7 |

### Pearson r

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 0.464 ± 0.039 | 0.545 ± 0.032 | 0.545 ± 0.032 | 0.407 ± 0.048 | 0.407 ± 0.048 |
| 1 | SFT labelonly 10% | e0 | 0.482 ± 0.041 | -- | 0.539 ± 0.031 | -- | 0.464 ± 0.048 |
| 2 | RankAlign | e1 | 0.500 ± 0.056 | -- | 0.580 ± 0.024 | -- | 0.267 ± 0.067 |
| 3 | New + fsx [-TC] | e1 | 0.469 ± 0.049 | -- | 0.600 ± 0.022 | -- | 0.230 ± 0.061 |
| 4 | New + PMI + fsx | e1 | 0.497 ± 0.039 | -- | 0.649 ± 0.028 | -- | -- |
| 5 | RA + PMI + fsx [-NLL] | e2 | 0.533 ± 0.050 | -- | 0.659 ± 0.025 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 0.563 ± 0.050 | -- | -- | -- | 0.513 ± 0.050 |
| 11 | New + PMI [-fsx] | e2 | 0.301 ± 0.069 | -- | 0.483 ± 0.032 | -- | -- |
| 13 | SFT + CFT | e2 | 0.535 ± 0.036 | 0.608 ± 0.026 | 0.604 ± 0.025 | 0.360 ± 0.058 | 0.504 ± 0.043 |

### Spearman r

| # | Setting | Epoch | Raw | PMI self | PMI base | Neg self | Neg base |
|---|---|---|---|---|---|---|---|
| 0 | Base | — | 0.478 ± 0.040 | 0.551 ± 0.032 | 0.551 ± 0.032 | 0.430 ± 0.054 | 0.430 ± 0.054 |
| 1 | SFT labelonly 10% | e0 | 0.581 ± 0.055 | -- | 0.577 ± 0.037 | -- | 0.489 ± 0.053 |
| 2 | RankAlign | e1 | 0.511 ± 0.064 | -- | 0.584 ± 0.023 | -- | 0.268 ± 0.064 |
| 3 | New + fsx [-TC] | e1 | 0.499 ± 0.064 | -- | 0.599 ± 0.029 | -- | 0.248 ± 0.062 |
| 4 | New + PMI + fsx | e1 | 0.545 ± 0.043 | -- | 0.652 ± 0.035 | -- | -- |
| 5 | RA + PMI + fsx [-NLL] | e2 | 0.558 ± 0.055 | -- | 0.672 ± 0.026 | -- | -- |
| 7 | New + NegTC + fsx | e2 | 0.614 ± 0.056 | -- | -- | -- | 0.551 ± 0.051 |
| 11 | New + PMI [-fsx] | e2 | 0.327 ± 0.067 | -- | 0.504 ± 0.035 | -- | -- |
| 13 | SFT + CFT | e2 | 0.599 ± 0.039 | 0.639 ± 0.022 | 0.636 ± 0.021 | 0.370 ± 0.065 | 0.525 ± 0.042 |
