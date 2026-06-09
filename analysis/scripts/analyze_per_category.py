#!/usr/bin/env python3
"""
Per-category analysis: which categories benefit most from TC?
Also produces scatter plots of gen vs val scores per category.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
SCORES_DIR = Path("/datastor2/jdr/rankalign/outputs-trainset-dynamics")
PLOTS_DIR = REPO / "analysis" / "plots"
TABLES_DIR = REPO / "analysis" / "tables"

# Key files for comparison
FILES = {
    "Base (self-TC)": SCORES_DIR / "scores_basetyp-v6-google_gemma-2-9b-it_membership-sans-rosch-v0_train_log-odds_tc_20260607.csv",
    "s3 (self-TC)": SCORES_DIR / "scores_basetyp-v7-gemma-2-9b-it-d2.69-e2-membership-sans-rosch-v0-all-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1_membership-sans-rosch-v0_train_log-odds_tc_20260607.csv",
    "s4 (self-TC)": SCORES_DIR / "scores_basetyp-v7-gemma-2-9b-it-d2.69-e2-membership-sans-rosch-v0-all-tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1_membership-sans-rosch-v0_train_log-odds_tc_20260608.csv",
    "s7 (neg-TC)": SCORES_DIR / "scores_basetypneg-v7-gemma-2-9b-it-d2.69-e2-membership-sans-rosch-v0-all-tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1_membership-sans-rosch-v0_train_log-odds_tc_20260607.csv",
    "s3 (neg-TC)": SCORES_DIR / "scores_basetypneg-v7-gemma-2-9b-it-d2.69-e2-membership-sans-rosch-v0-all-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1_membership-sans-rosch-v0_train_log-odds_tc_20260607.csv",
}


def compute_per_category_metrics(filepath, gen_col="gen_score_typcorr"):
    """Compute ROC AUC and spearman per category."""
    df = pd.read_csv(filepath)
    y = df["label"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).values.astype(float)
    gen = df[gen_col].values.astype(float) if gen_col in df.columns else None
    val = df["val_score"].values.astype(float)
    categories = df["category"].values

    results = []
    for cat in sorted(set(categories)):
        mask = categories == cat
        if mask.sum() < 4:
            continue
        yc = y[mask]
        if len(set(yc[~np.isnan(yc)])) < 2:
            continue

        vc = val[mask]
        row = {"category": cat, "n": int(mask.sum())}

        try:
            row["val_roc"] = roc_auc_score(yc, vc)
        except:
            row["val_roc"] = np.nan

        if gen is not None:
            gc = gen[mask]
            valid = ~(np.isnan(gc) | np.isnan(yc))
            if valid.sum() >= 4:
                try:
                    row["gen_roc"] = roc_auc_score(yc[valid], gc[valid])
                except:
                    row["gen_roc"] = np.nan
                try:
                    row["spearman"], _ = spearmanr(gc[valid], vc[valid])
                except:
                    row["spearman"] = np.nan
                row["gen_mean"] = np.mean(gc[valid])
                row["gen_std"] = np.std(gc[valid])
            else:
                row["gen_roc"] = np.nan
                row["spearman"] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("  PER-CATEGORY ANALYSIS: Gemma membership")
    print("=" * 70)

    all_cat_metrics = {}
    for label, filepath in FILES.items():
        if not filepath.exists():
            print(f"  SKIP {label}: file not found")
            continue
        metrics = compute_per_category_metrics(filepath)
        metrics["model"] = label
        all_cat_metrics[label] = metrics
        print(f"  {label}: {len(metrics)} categories computed")

    # Combine
    combined = pd.concat(all_cat_metrics.values(), ignore_index=True)
    combined.to_csv(TABLES_DIR / "gemma_membership_per_category_metrics.csv", index=False)

    # Analysis: which categories improve most from Base → s4?
    if "Base (self-TC)" in all_cat_metrics and "s4 (self-TC)" in all_cat_metrics:
        base = all_cat_metrics["Base (self-TC)"].set_index("category")
        s4 = all_cat_metrics["s4 (self-TC)"].set_index("category")

        common = base.index.intersection(s4.index)
        improvements = pd.DataFrame({
            "base_gen_roc": base.loc[common, "gen_roc"],
            "s4_gen_roc": s4.loc[common, "gen_roc"],
            "improvement": s4.loc[common, "gen_roc"] - base.loc[common, "gen_roc"],
            "n": base.loc[common, "n"],
        }).dropna()

        improvements = improvements.sort_values("improvement", ascending=False)

        print(f"\n  Top 10 categories that improved MOST (Base → s4):")
        for cat, row in improvements.head(10).iterrows():
            print(f"    {cat:<35}: {row['base_gen_roc']:.3f} → {row['s4_gen_roc']:.3f} "
                  f"(+{row['improvement']:.3f}, n={int(row['n'])})")

        print(f"\n  Bottom 10 categories (least improvement or regression):")
        for cat, row in improvements.tail(10).iterrows():
            print(f"    {cat:<35}: {row['base_gen_roc']:.3f} → {row['s4_gen_roc']:.3f} "
                  f"({row['improvement']:+.3f}, n={int(row['n'])})")

        # Save full table
        improvements.to_csv(TABLES_DIR / "gemma_membership_category_improvements.csv")

        # Plot: improvement distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.hist(improvements["improvement"].values, bins=30, color='steelblue', alpha=0.8, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel("Δ Gen ROC AUC (s4 - Base)")
        ax.set_ylabel("Number of categories")
        ax.set_title("Distribution of per-category improvements\n(Base → s4 TC-self, gemma membership)")
        ax.text(0.02, 0.95, f"Mean Δ = {improvements['improvement'].mean():.3f}\n"
                f"Median Δ = {improvements['improvement'].median():.3f}\n"
                f"Categories improved: {(improvements['improvement'] > 0).sum()}/{len(improvements)}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # s4 vs s3 improvement
        if "s3 (self-TC)" in all_cat_metrics:
            s3 = all_cat_metrics["s3 (self-TC)"].set_index("category")
            common2 = s3.index.intersection(s4.index)
            tc_improvement = s4.loc[common2, "gen_roc"] - s3.loc[common2, "gen_roc"]
            tc_improvement = tc_improvement.dropna()

            ax = axes[1]
            ax.hist(tc_improvement.values, bins=30, color='coral', alpha=0.8, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel("Δ Gen ROC AUC (s4 - s3)")
            ax.set_ylabel("Number of categories")
            ax.set_title("Distribution of TC-self improvement\n(s3 → s4, gemma membership)")
            ax.text(0.02, 0.95, f"Mean Δ = {tc_improvement.mean():.3f}\n"
                    f"Median Δ = {tc_improvement.median():.3f}\n"
                    f"Categories improved: {(tc_improvement > 0).sum()}/{len(tc_improvement)}",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "per_category_improvement_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: analysis/plots/per_category_improvement_distribution.png")

    # Scatter plot: gen score vs val score for key settings
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (label, filepath) in enumerate([
        ("Base (self-TC)", FILES["Base (self-TC)"]),
        ("s3 (self-TC)", FILES["s3 (self-TC)"]),
        ("s4 (self-TC)", FILES["s4 (self-TC)"]),
    ]):
        if idx >= 3:
            break
        df = pd.read_csv(filepath)
        y = df["label"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        gen = df["gen_score_typcorr"].values if "gen_score_typcorr" in df.columns else None
        val = df["val_score"].values

        if gen is None:
            continue

        ax = axes[idx]
        pos_mask = y == 1
        neg_mask = y == 0

        ax.scatter(gen[pos_mask], val[pos_mask], alpha=0.3, s=10, c='green', label='Positive')
        ax.scatter(gen[neg_mask], val[neg_mask], alpha=0.3, s=10, c='red', label='Negative')
        ax.set_xlabel("Generator score (TC)")
        ax.set_ylabel("Validator score")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)

        # Add correlation annotation
        mask = ~(np.isnan(gen) | np.isnan(val))
        if mask.sum() > 0:
            rho, _ = spearmanr(gen[mask], val[mask])
            ax.text(0.02, 0.02, f"ρ = {rho:.3f}", transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle("Generator vs Validator scores (gemma membership, epoch 2)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gen_vs_val_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/gen_vs_val_scatter.png")

    # Neg-TC scatter comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (label, filepath) in enumerate([
        ("s3 (neg-TC)", FILES["s3 (neg-TC)"]),
        ("s7 (neg-TC)", FILES["s7 (neg-TC)"]),
    ]):
        df = pd.read_csv(filepath)
        y = df["label"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        gen = df["gen_score_typcorr"].values if "gen_score_typcorr" in df.columns else None
        val = df["val_score"].values

        if gen is None:
            continue

        ax = axes[idx]
        pos_mask = y == 1
        neg_mask = y == 0

        ax.scatter(gen[pos_mask], val[pos_mask], alpha=0.3, s=10, c='green', label='Positive')
        ax.scatter(gen[neg_mask], val[neg_mask], alpha=0.3, s=10, c='red', label='Negative')
        ax.set_xlabel("Generator score (neg-TC)")
        ax.set_ylabel("Validator score")
        ax.set_title(f"{label}")
        ax.legend(fontsize=8)

        mask = ~(np.isnan(gen) | np.isnan(val))
        if mask.sum() > 0:
            rho, _ = spearmanr(gen[mask], val[mask])
            ax.text(0.02, 0.02, f"ρ = {rho:.3f}", transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle("Neg-TC scoring: s3 vs s7 (gemma membership, epoch 2)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gen_vs_val_scatter_neg_tc.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/gen_vs_val_scatter_neg_tc.png")

    print("\n\nDone!")


if __name__ == "__main__":
    main()
