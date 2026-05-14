# membership-sans-rosch-v0 (gemma-2-2b, epoch2) → rosch — per-task gen-ROC × 100

Rows: 10 rosch tasks ordered by item-overlap with the membership training pool (high overlap on top, rosch-sport at the bottom).

Columns: 4 self-eval variants (Base, RankAlign, SFT, Offline self-TC) followed by 4 neg-eval variants (Base, RankAlign, SFT, Offline neg-TC). The Base-self / Base-neg columns and SFT-self / SFT-neg columns are different metrics on the same checkpoint.

**Bold = highest value in the row across all 8 columns.** Note this comparison mixes self and neg eval refs, which are different metrics — interpret "row max" as a quick visual read, not a rigorous comparison.

Long-form metrics: [quickiter_metrics_long_membership_to_rosch.csv](quickiter_metrics_long_membership_to_rosch.csv)

| task (overlap) | Base (self) | RankAlign (self) | SFT (self) | Offline self-TC (self) | Base (neg) | RankAlign (neg) | SFT (neg) | Offline neg-TC (neg) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rosch-bird (89%) | 70.35 | 79.57 | **89.50** | 87.56 | 59.58 | 71.15 | 84.92 | 68.85 |
| rosch-carpenters-tool (61%) | 71.09 | 63.98 | 73.80 | 73.17 | 74.36 | **77.96** | 64.96 | 64.43 |
| rosch-fruit (60%) | 76.73 | 85.46 | 81.15 | 91.67 | 84.98 | **95.46** | 82.54 | 81.43 |
| rosch-vehicle (56%) | 85.69 | 92.30 | 86.62 | **93.07** | 77.04 | 77.93 | 47.89 | 87.98 |
| rosch-furniture (45%) | 80.19 | 87.63 | 89.34 | 93.25 | 93.00 | **97.66** | 94.03 | 83.44 |
| rosch-vegetable (44%) | 82.77 | 80.31 | 80.07 | 86.40 | **92.10** | 89.99 | 77.39 | 74.74 |
| rosch-toy (42%) | 79.17 | 77.16 | 79.48 | 78.67 | 75.54 | **81.02** | 77.16 | 57.14 |
| rosch-clothing (36%) | 69.06 | 88.15 | 88.71 | **90.83** | 54.83 | 74.66 | 66.71 | 89.57 |
| rosch-weapon (36%) | 77.72 | 75.88 | 78.83 | 78.54 | 86.99 | **87.39** | 79.37 | 73.84 |
| rosch-sport (9%) | 80.41 | 85.92 | 82.07 | **90.57** | 78.69 | 76.39 | 73.68 | 81.21 |
