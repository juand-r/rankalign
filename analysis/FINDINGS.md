# Analysis Findings: When Does RankAlign / TC Work?

**Generated:** 2026-06-09  
**Data source:** Train-set dynamics scores (`outputs-trainset-dynamics/`)  
**Evaluation:** Self-TC scoring, epoch 2 (final model), on training split.

---

## Executive Summary

1. **RankAlign consistently improves over the base model** on both tasks, but the *full method* (s3) is essential — basic RankAlign (s2) can actually *hurt* on ifeval.
2. **Self-typicality correction (TC-self, s4) provides the largest and most consistent improvement** over the already-strong full method (s3), across all four model/task combos.
3. **The NLL-V/G + fsx + ppd + vlo components are critical** for ifeval — without them (s2), generator scores explode and correlation drops below chance.
4. **Score spread (gen_delta_mean) is a useful diagnostic**: explosion indicates poor training dynamics.

---

## Q1: When Does Each Component Help?

### Comparison: Base → s2 (basic RA) → s3 (full method) → s4 (TC-self)

| Model / Task | Base | s2 (basic) | s3 (full) | s4 (TC-self) |
|---|---|---|---|---|
| **Gemma / Membership** | 0.851 | 0.941 (+0.090) | 0.948 (+0.097) | **0.976** (+0.125) |
| **Gemma / IFEval** | 0.670 | 0.482 (−0.188!) | 0.753 (+0.083) | **0.801** (+0.131) |
| **Qwen / Membership** | 0.799 | 0.880 (+0.081) | 0.899 (+0.100) | **0.969** (+0.170) |
| **Qwen / IFEval** | 0.606 | 0.558 (−0.048) | 0.711 (+0.105) | **0.743** (+0.137) |

**Key observations:**
- s4 (TC-self) is the **best setting in every single combination**.
- s2 (basic RA) *hurts* on ifeval (both models). This is because without NLL constraints, generator scores diverge.
- The improvement from s3 → s4 (adding TC-self training) is +0.03 to +0.07 on gen_roc — a consistent and meaningful gain.

### Neg-TC Comparison (s7 vs s3)

For neg-TC scoring (evaluating with negative typicality correction):

| Model / Task | s3 | s7 (TC-neg) | s12 (TC-neg vlo) |
|---|---|---|---|
| **Gemma / Membership** | 0.890 | 0.962 (+0.072) | **0.965** (+0.075) |
| **Gemma / IFEval** | — | — | — |

TC-neg training similarly helps when evaluated with neg-TC scoring.

---

## Q2: Training Diagnostics

### WandB Loss Component Breakdown (Final Third of Training)

**Gemma IFEval (critical finding!):**

| Setting | Total Loss | Pref Loss | NLL-G | NLL-V |
|---|---|---|---|---|
| s1 (SFT) | 5.99 | 0.00 | 5.99 | 0.0002 |
| **s2 (basic)** | 11.56 | 11.56 | **3119.8** | 0.00 |
| s4 (TC-self) | 6.40 | 2.64 | 3.76 | 0.001 |
| s3 (full) | 6.15 | 1.81 | 4.33 | 0.009 |
| s7 (TC-neg) | 5.13 | 1.37 | 3.77 | 0.001 |

**ROOT CAUSE of s2 failure**: Without NLL-G constraint, the generator's NLL explodes to **3120** (vs 3.8-4.3 for constrained settings). The model produces arbitrarily large generator scores to minimize the unconstrained preference loss.

**Qwen Membership:**

| Setting | Total Loss | Pref Loss | NLL-G | NLL-V |
|---|---|---|---|---|
| s1 (SFT) | 2.75 | 0.00 | 2.75 | 0.0001 |
| s2 (basic) | 0.04 | 0.04 | 2.42 | 0.00 |
| s4 (TC-self) | 0.82 | 0.04 | 0.77 | 0.0001 |
| s3 (full) | 0.80 | 0.03 | 0.77 | 0.0000 |
| s7 (TC-neg) | 0.77 | 0.05 | 0.72 | 0.0001 |

**Key insight**: On the simpler membership task, even s2 keeps NLL-G reasonable (2.42) without explicit constraint. The constraint is only critical for complex tasks (ifeval) where the optimization landscape allows score explosion.

### Why s2 Fails on IFEval but Works on Membership

The mechanism is clear from WandB data:
1. **s2 has NO NLL-G constraint** — only the preference loss.
2. On ifeval (complex, many tokens per example), the model finds it "easier" to maximize preference by inflating raw scores rather than learning genuine features.
3. On membership (simple, few tokens per example), the optimization landscape naturally constrains scores — there's less room to exploit.
4. TC further helps because it normalizes scores relative to a base model, preventing drift.

### Score Explosion Diagnostic

| Setting / Task | gen_delta_mean | gen_roc (tc) | Status |
|---|---|---|---|
| Gemma ifeval s2 | **13,421** | 0.482 | ❌ Exploded |
| Gemma ifeval s3 | 691 | 0.753 | ✓ Normal |
| Gemma ifeval s4 | 216 | 0.801 | ✓ Best |
| Gemma membership s2 | 9.1 | 0.941 | ✓ Normal |
| Gemma membership s3 | 10.9 | 0.948 | ✓ Normal |

### Additional WandB Findings

- **NLL-V is always tiny** (<0.01 for all settings). The validator barely changes during training — it's frozen/near-frozen. This is expected given the log-odds formulation.
- **s7 (TC-neg) achieves the lowest total loss** (5.13 on Gemma IFEval) — TC-neg training is the most efficient at optimization.
- **Preference loss decreases for all constrained settings** — the model IS learning to rank, but the NLL regularization prevents degenerate solutions.

---

## Q3: Additional Statistics

### Concordance (fraction of pairs where gen and val agree on ordering)

| Model / Task | Base | s1 | s2 | s3 | s4 | s7 |
|---|---|---|---|---|---|---|
| **Gemma / Membership** | 0.705 | — | 0.794 | 0.787 | 0.803 | — |
| **Gemma / IFEval** | 0.548 | 0.561 | 0.370 | 0.677 | 0.694 | — |
| **Qwen / Membership** | 0.671 | 0.734 | 0.759 | 0.752 | 0.807 | — |
| **Qwen / IFEval** | 0.545 | 0.553 | 0.571 | 0.641 | 0.646 | — |

- s4 (TC-self) achieves the highest concordance in every combo.
- s2 on gemma ifeval has concordance 0.370 (below chance!) — confirming generator scores are anti-correlated with validator.

### Spearman Correlation (gen vs val)

| Model / Task | Base | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| **Gemma / Membership** | 0.585 | — | 0.798 | 0.795 | **0.826** |
| **Gemma / IFEval** | 0.141 | 0.178 | −0.374 | 0.504 | **0.567** |
| **Qwen / Membership** | 0.496 | 0.665 | 0.723 | 0.715 | **0.830** |
| **Qwen / IFEval** | 0.119 | 0.151 | 0.205 | 0.404 | **0.431** |

### Training Dynamics (epoch progression)

For gemma membership (self-TC scoring):
- **s4 (TC-self)**: starts strong at ep0 (0.975) and stays flat — converges fast.
- **s3 (RA full)**: gradually improves ep0 (0.937) → ep2 (0.948).
- **s2 (RA basic)**: also gradual improvement ep0 (0.899) → ep2 (0.941).
- **s11 (TC-self vlo)**: highest spearman (0.865) despite slightly lower gen_roc.

---

## Per-Category Analysis (Gemma Membership)

### Which categories benefit most from TC?

Top 5 improvements (Base → s4):
| Category | Base gen_roc | s4 gen_roc | Improvement |
|---|---|---|---|
| medical specialty | 0.504 | 1.000 | **+0.496** |
| female first name | 0.570 | 0.994 | +0.423 |
| thing that makes noise | 0.572 | 0.962 | +0.390 |
| male first name | 0.700 | 1.000 | +0.300 |
| state | 0.718 | 1.000 | +0.282 |

**Pattern**: Categories where the base model was weakest (gen_roc ~0.5-0.7) benefit the most from training. Categories already near ceiling (1.0) show minimal improvement.

### One regression
- "thing taken from a burning home": 0.670 → 0.405 (-0.265, n=20). This is a small, ambiguous category where the model may have overfit to training signal.

### TC-specific improvement (s3 → s4)
The distribution of per-category improvements from adding TC-self training is consistently positive. Out of 68 categories, the large majority improve with TC-self. See `analysis/plots/per_category_improvement_distribution.png`.

---

## Key Takeaways for the Paper

1. **TC-self is the clear winner** — it improves every metric across all combos.
2. **The NLL-V/G constraints are essential** for ifeval (prevents score explosion).
3. **Basic RankAlign (s2) is insufficient for complex tasks** (ifeval) but works for simpler ones (membership).
4. **Gen_delta_mean as a diagnostic**: values >1000 indicate training instability.
5. **Concordance is a useful metric** beyond ROC AUC — it directly measures gen-val alignment.

---

## Files Generated

### Plots
- `analysis/plots/gemma_membership_dynamics_key_settings.png` — gen_roc and spearman across epochs
- `analysis/plots/gemma_membership_genroc_bar_ep2.png` — bar chart of all settings at ep2
- `analysis/plots/gemma_membership_delta_histograms.png` — pairwise score delta distributions
- `analysis/plots/gemma_membership_concordance_ep2.png` — concordance bar chart
- `analysis/plots/unified_genroc_comparison.png` — all combos, key settings
- `analysis/plots/unified_improvement_over_base.png` — delta over base
- `analysis/plots/unified_concordance_comparison.png` — concordance across combos
- `analysis/plots/wandb_loss_curves_gemma-9b-it_ifeval.png` — training loss curves
- `analysis/plots/wandb_loss_curves_qwen3.5-9b_membership.png`
- `analysis/plots/wandb_loss_curves_qwen3.5-9b_ifeval.png`
- `analysis/plots/wandb_score_stats_*.png` — training score statistics

### Tables (CSV)
- `analysis/tables/gemma_membership_all_metrics.csv` — full metrics for all gemma membership files
- `analysis/tables/all_combos_metrics.csv` — unified metrics across all combos
- `analysis/tables/wandb_training_summary_*.csv` — final training loss values
