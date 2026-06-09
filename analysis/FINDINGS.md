# Analysis Findings: When Does RankAlign / TC Work?

**Generated:** 2026-06-09  
**Data source:** Train-set dynamics scores (`outputs-trainset-dynamics/`)  
**Evaluation:** Self-TC scoring, epoch 2 (final model), on training split.

---

## Executive Summary

1. **RankAlign consistently improves over the base model** on both tasks, but the *full method* (s4) is essential — basic RankAlign (s2) can actually *hurt* on ifeval.
2. **Self-typicality correction (TC-self, s3) provides the largest and most consistent improvement** over the already-strong full method (s4), across all four model/task combos.
3. **The NLL-V/G + fsx + ppd + vlo components are critical** for ifeval — without them (s2), generator scores explode and correlation drops below chance.
4. **Score spread (gen_delta_mean) is a useful diagnostic**: explosion indicates poor training dynamics.

---

## Q1: When Does Each Component Help?

### Comparison: Base → s2 (basic RA) → s4 (full method) → s3 (TC-self)

| Model / Task | Base | s2 (basic) | s4 (full) | s3 (TC-self) |
|---|---|---|---|---|
| **Gemma / Membership** | 0.851 | 0.941 (+0.090) | 0.948 (+0.097) | **0.976** (+0.125) |
| **Gemma / IFEval** | 0.670 | 0.482 (−0.188!) | 0.753 (+0.083) | **0.801** (+0.131) |
| **Qwen / Membership** | 0.799 | 0.880 (+0.081) | 0.899 (+0.100) | **0.969** (+0.170) |
| **Qwen / IFEval** | 0.606 | 0.558 (−0.048) | 0.711 (+0.105) | **0.743** (+0.137) |

**Key observations:**
- s3 (TC-self) is the **best setting in every single combination**.
- s2 (basic RA) *hurts* on ifeval (both models). This is because without NLL constraints, generator scores diverge.
- The improvement from s4 → s3 (adding TC-self training) is +0.03 to +0.07 on gen_roc — a consistent and meaningful gain.

### Neg-TC Comparison (s7 vs s4)

For neg-TC scoring (evaluating with negative typicality correction):

| Model / Task | s4 | s7 (TC-neg) | s12 (TC-neg vlo) |
|---|---|---|---|
| **Gemma / Membership** | 0.890 | 0.962 (+0.072) | **0.965** (+0.075) |
| **Gemma / IFEval** | — | — | — |

TC-neg training similarly helps when evaluated with neg-TC scoring.

---

## Q2: Training Diagnostics

### WandB Loss Curves

**Gemma IFEval:**
- All settings show HIGH VARIANCE in the final training quarter (inherent to the ranking loss).
- s3 shows NLL-V increasing — the validator NLL gets slightly worse as training focuses on ranking.
- No clear divergence or catastrophic issues for the "good" settings.

**Qwen IFEval:**
- s2 and s4 show LOSS INCREASED flag — total loss drifts upward.
- This correlates with s2's poor test performance (generator explosion).

**Qwen Membership:**
- s1 is clean (OK). All others show final-quarter variance (normal for ranking).

### Diagnostic: Score Explosion in s2/IFEval

| Setting / Task | gen_delta_mean | gen_roc (tc) | Status |
|---|---|---|---|
| Gemma ifeval s2 | **13,421** | 0.482 | ❌ Exploded |
| Gemma ifeval s4 | 691 | 0.753 | ✓ Normal |
| Gemma ifeval s3 | 216 | 0.801 | ✓ Best |
| Gemma membership s2 | 9.1 | 0.941 | ✓ Normal |
| Gemma membership s4 | 10.9 | 0.948 | ✓ Normal |

The gen_delta_mean (mean absolute pairwise difference in generator scores) is an excellent diagnostic.
Values >1000 suggest unstable training. For s2 on ifeval, it's 13,421 — confirming complete score explosion.

---

## Q3: Additional Statistics

### Concordance (fraction of pairs where gen and val agree on ordering)

| Model / Task | Base | s1 | s2 | s4 | s3 | s7 |
|---|---|---|---|---|---|---|
| **Gemma / Membership** | 0.705 | — | 0.794 | 0.787 | 0.803 | — |
| **Gemma / IFEval** | 0.548 | 0.561 | 0.370 | 0.677 | 0.694 | — |
| **Qwen / Membership** | 0.671 | 0.734 | 0.759 | 0.752 | 0.807 | — |
| **Qwen / IFEval** | 0.545 | 0.553 | 0.571 | 0.641 | 0.646 | — |

- s3 (TC-self) achieves the highest concordance in every combo.
- s2 on gemma ifeval has concordance 0.370 (below chance!) — confirming generator scores are anti-correlated with validator.

### Spearman Correlation (gen vs val)

| Model / Task | Base | s1 | s2 | s4 | s3 |
|---|---|---|---|---|---|
| **Gemma / Membership** | 0.585 | — | 0.798 | 0.795 | **0.826** |
| **Gemma / IFEval** | 0.141 | 0.178 | −0.374 | 0.504 | **0.567** |
| **Qwen / Membership** | 0.496 | 0.665 | 0.723 | 0.715 | **0.830** |
| **Qwen / IFEval** | 0.119 | 0.151 | 0.205 | 0.404 | **0.431** |

### Training Dynamics (epoch progression)

For gemma membership (self-TC scoring):
- **s3 (TC-self)**: starts strong at ep0 (0.975) and stays flat — converges fast.
- **s4 (RA full)**: gradually improves ep0 (0.937) → ep2 (0.948).
- **s2 (RA basic)**: also gradual improvement ep0 (0.899) → ep2 (0.941).
- **s11 (TC-self vlo)**: highest spearman (0.865) despite slightly lower gen_roc.

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
