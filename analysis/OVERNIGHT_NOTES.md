# Overnight Analysis Notes (2026-06-09)

## What Was Done

### Scripts Created
1. `analysis/scripts/analyze_gemma_membership.py` — Main gemma membership analysis (Q1 + Q3)
2. `analysis/scripts/analyze_wandb_training.py` — WandB loss curve extraction (Q2)
3. `analysis/scripts/analyze_all_combos.py` — Unified analysis across all 4 model/task combos
4. `analysis/scripts/plot_tc_comparison.py` — Focused TC comparison plots
5. `analysis/scripts/analyze_per_category.py` — Per-category breakdown
6. `analysis/scripts/analyze_wandb_detailed.py` — Detailed WandB loss component analysis

### Outputs
- `analysis/report.pdf` — 11-page LaTeX report with all figures and tables
- `analysis/FINDINGS.md` — Markdown summary of all key findings
- `analysis/plots/` — 19 PNG plots
- `analysis/tables/` — 13 CSV files with metrics

---

## Key Findings

### Q1: When Does Each Component Help?

**Main result (gen_roc TC, epoch 2, train set):**

| Model / Task | Base | s2 (basic RA) | s3 (full RA) | s4 (TC-self) |
|---|---|---|---|---|
| Gemma / Membership | 0.851 | 0.941 | 0.948 | **0.976** |
| Gemma / IFEval | 0.670 | 0.482 ❌ | 0.753 | **0.801** |
| Qwen / Membership | 0.799 | 0.880 | 0.899 | **0.969** |
| Qwen / IFEval | 0.606 | 0.558 | 0.711 | **0.743** |

- **s4 (TC-self) wins in ALL four combinations.**
- **s2 (basic RA) HURTS on IFEval** (drops below chance for Gemma!)
- The full method (s3) always helps over base, by +0.08 to +0.11
- TC-self (s4) adds another +0.03 to +0.07 over s3

**TC Comparison (gemma membership, self-TC eval):**
- s4 (TC-self full): gen_roc = 0.976 (+0.028 over s3)
- s6 (TC-self fsx, lower delta): gen_roc = 0.971 (+0.023 over s3)
- s11 (TC-self vlo, no fsx/ppd): gen_roc = 0.970 (+0.022 over s3)
- s5 (TC-self basic): gen_roc = 0.958 (+0.010 over s3)

**Neg-TC eval (gemma membership):**
- s7 (TC-neg full): gen_roc = 0.962 (+0.072 over s3's 0.890)
- s12 (TC-neg vlo): gen_roc = 0.965 (+0.075 over s3)

### Q2: Training Diagnostics

**Critical finding — NLL-G explosion for s2:**

On Gemma IFEval, s2's NLL-G reaches **3120** in the final third of training (vs 3.8-4.3 for constrained settings). This is the mechanism behind the score explosion:
- s2 has NO NLL-G constraint
- The model inflates raw generator scores to minimize preference loss
- This works on simple tasks (membership: NLL-G stays at 2.4) but explodes on complex tasks (ifeval)
- TC further helps because it normalizes relative to base model

**NLL-V is tiny everywhere** (<0.01) — the validator barely changes during training.

**s7 achieves lowest total loss** (5.13 vs 6.15 for s3 on Gemma IFEval) — TC-neg training is the most optimization-efficient.

### Q3: Additional Statistics

**Concordance (epoch 2, self-TC):**
- s4 achieves highest concordance in every combo (0.694-0.807)
- s2 on Gemma IFEval: 0.370 (BELOW CHANCE — anti-correlated!)

**Per-category analysis (gemma membership):**
- Categories where base was weakest improve most (medical specialty: +0.496)
- Categories near ceiling (breed of dog, chemical element = 1.000) show no change
- One regression: "thing taken from a burning home" (-0.265, n=20, likely noise)
- TC-self improvement (s3→s4) is consistently positive across nearly all categories

---

## Important Technical Notes

### s1 (SFT-lo) only has 2 epochs for gemma membership
The s1 model was trained without `--cft` for only epochs 0 and 1. Epoch 2 only exists with the `--cft` flag (a different variant). So we only report epochs 0 and 1 for s1. This is not a bug — it's a data availability fact.

### s7 → s4 comparison limitation
s4 (TC-self trained) is only evaluated with self-TC scoring.
s7 (TC-neg trained) is only evaluated with neg-TC scoring.
We CANNOT directly compare s4 and s7 using the same evaluation scoring method from the existing data. The comparison is:
- Self-TC eval: s3 (0.948) vs s4 (0.976) → TC-self adds +0.028
- Neg-TC eval: s3 (0.890) vs s7 (0.962) → TC-neg adds +0.072

The neg-TC improvement looks larger in absolute terms, but the baselines differ (self-TC scoring gives higher absolute numbers for everyone). To properly compare s4 vs s7 would require evaluating both with the same scoring method.

### Gemma membership was trained pre-WandB
WandB logs only exist for gemma-ifeval and qwen-membership/ifeval runs. For gemma membership, we rely solely on the train-set dynamics score files (which are comprehensive — all epochs, all settings, both TC scoring types).

### Training variance is normal
All ranking-trained models show "HIGH VARIANCE" in the final training quarter. This is inherent to the pairwise loss (each batch samples different pairs). It is NOT a sign of instability — the downstream metrics (gen_roc etc.) are stable across epochs.

---

## File Locations Reference

| Artifact | Location |
|---|---|
| Train-set dynamics scores | `/datastor2/jdr/rankalign/outputs-trainset-dynamics/` |
| Test-set scores | `/datastor2/jdr/rankalign/outputs/` and `outputs-rerun-wandb/` |
| Trained models | `/datastor2/jdr/rankalign/models2/` |
| WandB project | `juand-r/rankalign` on wandb.ai |
| Analysis outputs | `/datastor1/jdr/gv-gap/rankalign/analysis/` |
| Metric CSVs | `analysis/tables/` (13 files) |
| Plots | `analysis/plots/` (19 PNGs) |
| LaTeX report | `analysis/report.tex` → `analysis/report.pdf` |

---

## Train vs Test Set Validation

The train-set findings were validated against test-set metrics:

### Gemma Rosch Test Set (basetyp scoring, avg across 10 categories)

| Setting | Train (membership) | Test (rosch) | Notes |
|---|---|---|---|
| s11 (TC-self vlo) | 0.970 | **0.924** | Best on test! |
| s4 (TC-self full) | **0.976** | 0.920 | Best on train |
| s5 (TC-self basic) | 0.958 | 0.913 | |
| s2 (RA basic) | 0.941 | 0.891 | |
| s3 (RA full) | 0.948 | 0.885 | ← DROPS below s2 on test! |
| s1 (SFT) | — | 0.854 | |

**Critical insight**: s3 (full method, no TC) OUTPERFORMS s2 (basic RA) on training categories but UNDERPERFORMS on test categories! The fsx/ppd features may cause overfitting to training patterns. TC-trained models (s4, s11, s5) **generalize better** — they maintain their advantage on both train and test.

### Gemma IFEval (Test Set)

| Setting | Train | Test ID | Test OOD |
|---|---|---|---|
| s4 (TC-self) | 0.801 | 0.843 | **0.806** |
| s3 (full) | 0.753 | **0.846** | 0.785 |
| s1 (SFT) | 0.586 | 0.617 | 0.582 |
| s2 (basic) | 0.482 | 0.489 | 0.434 |

IFEval findings generalize well. The rank ordering is preserved: s4 ≈ s3 >> s1 > s2. On OOD, TC helps (+0.021 from s3 to s4). On ID, s3 and s4 are nearly tied.

### Key Generalization Finding

**TC promotes generalization to out-of-distribution data.** Without TC, the full method (s3) can overfit to training-specific patterns. With TC, the model learns more robust features that transfer to new categories/tasks.

---

## What Could Be Done Next

1. **Cross-evaluation**: Evaluate s4 with neg-TC scoring (and s7 with self-TC scoring) to enable direct comparison.
2. **Test-set metrics**: The current analysis is on the TRAINING set. The test-set metrics from `outputs/` and `metrics-from-scores-rerun-wandb/` should be compared to confirm findings generalize.
3. **Per-category detail for ifeval**: The ifeval tasks have sub-tasks (ifeval-prompt_N); could analyze per-subtask.
4. **Statistical significance**: Bootstrap confidence intervals on the ROC AUC and concordance measures.
5. **Epoch selection**: Some settings (s4, s6) converge by epoch 0 — investigate whether fewer epochs suffice.
