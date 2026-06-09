#!/usr/bin/env python3
"""
Comprehensive analysis of gemma-2-9b-it membership train-set dynamics.

Answers:
  Q1: When does RankAlign / our method / TC improve?
  Q3: Concordance, delta histograms, and additional stats.

Outputs to analysis/plots/ and analysis/tables/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parent.parent.parent
SCORES_DIR = Path("/datastor2/jdr/rankalign/outputs-trainset-dynamics")
PLOTS_DIR = REPO / "analysis" / "plots"
TABLES_DIR = REPO / "analysis" / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Setting definitions for gemma-2-9b-it membership
# Maps setting name -> (delta, short-name suffix pattern, tc_type, description)
SETTINGS = {
    "s1 (SFT-lo)":       {"delta": "1.43", "pattern": "p0-nv1-ng1-lo0.1-fix1", "tc_train": "none",
                           "desc": "SFT with labeled-only"},
    "s2 (RA basic)":     {"delta": "1.42", "long_pattern": "full-completion--semi0.1--fix1",
                           "tc_train": "none", "desc": "Basic RankAlign"},
    "s3 (RA full)":      {"delta": "2.69", "pattern": "nv1-ng1-vlo-fsx-ppd-sm0.1-fix1",
                           "tc_train": "none", "desc": "RankAlign + NLL-V/G + fsx + ppd + vlo"},
    "s4 (TC-self)":      {"delta": "2.69", "pattern": "tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1",
                           "tc_train": "self", "desc": "s3 + self-typicality at training"},
    "s7 (TC-neg)":       {"delta": "2.69", "pattern": "tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1",
                           "tc_train": "neg", "desc": "s3 + neg-typicality at training"},
    "s5 (TC-self basic)": {"delta": "1.42", "long_pattern": "tc-self--full-completion--semi0.1--fix1",
                            "tc_train": "self", "desc": "Basic RA + self-TC"},
    "s6 (TC-self fsx)":  {"delta": "1.42", "pattern": "tcs-fsx-ppd-sm0.1-fix1",
                           "tc_train": "self", "desc": "RA + fsx + ppd + self-TC"},
    "s11 (TC-self vlo)": {"delta": "2.69", "pattern": "tcs-nv1-ng1-vlo-sm0.1-fix1",
                           "tc_train": "self", "desc": "s3-minus-fsx-ppd + self-TC"},
    "s12 (TC-neg vlo)":  {"delta": "2.69", "pattern": "tcn-nv1-ng1-vlo-sm0.1-fix1",
                           "tc_train": "neg", "desc": "s3-minus-fsx-ppd + neg-TC"},
}


def find_score_files():
    """Find and categorize all gemma membership train-set dynamics files."""
    all_files = sorted(SCORES_DIR.glob("scores_*membership*"))
    gemma_files = [f for f in all_files if "Qwen" not in f.name]

    inventory = []
    for f in gemma_files:
        name = f.name
        # Determine TC scoring type
        if name.startswith("scores_basetypneg"):
            tc_eval = "neg"
        elif name.startswith("scores_basetyp"):
            tc_eval = "self"
        else:
            continue

        # Base model
        if "-v6-" in name:
            inventory.append({"file": f, "setting": "Base", "epoch": -1, "tc_eval": tc_eval})
            continue

        # Extract epoch
        epoch = None
        for ep in [0, 1, 2]:
            if f"-e{ep}-" in name or f"-epoch{ep}--" in name:
                epoch = ep
                break
        if epoch is None:
            continue

        # Match to setting using ordered rules (most specific first)
        matched_setting = _match_setting(name)
        if matched_setting:
            inventory.append({"file": f, "setting": matched_setting, "epoch": epoch,
                              "tc_eval": tc_eval})

    df = pd.DataFrame(inventory)
    # Deduplicate: prefer most recent file (by date in filename) for same setting/epoch/tc_eval
    if not df.empty:
        df["date"] = df["file"].apply(lambda f: f.name.split("_")[-1].replace(".csv", ""))
        df = df.sort_values("date", ascending=False).drop_duplicates(
            subset=["setting", "epoch", "tc_eval"], keep="first"
        ).drop(columns=["date"]).reset_index(drop=True)
    return df


def _match_setting(name):
    """Match a filename to a setting. Order matters: most specific patterns first."""
    # Short-name format patterns (d{delta}-e{epoch}-...-{flags})
    # S4: tc-self full (has tcs- prefix + fsx-ppd)
    if "tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d2.69-" in name or "-delta2.69-" in name):
        return "s4 (TC-self)"
    # S7: tc-neg full (has tcn- prefix + fsx-ppd)
    if "tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d2.69-" in name or "-delta2.69-" in name):
        return "s7 (TC-neg)"
    # S11: tc-self vlo (tcs- but no fsx-ppd)
    if "tcs-nv1-ng1-vlo-sm0.1-fix1" in name and ("-d2.69-" in name or "-delta2.69-" in name):
        return "s11 (TC-self vlo)"
    # S12: tc-neg vlo (tcn- but no fsx-ppd)
    if "tcn-nv1-ng1-vlo-sm0.1-fix1" in name and ("-d2.69-" in name or "-delta2.69-" in name):
        return "s12 (TC-neg vlo)"
    # S3: no TC, full features (nv1-ng1-vlo-fsx-ppd but no tcs/tcn prefix)
    if "nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and "tcs-" not in name and "tcn-" not in name:
        if "-d2.69-" in name or "-delta2.69-" in name:
            return "s3 (RA full)"
    # S6: tc-self with fsx-ppd but at lower delta
    if "tcs-fsx-ppd-sm0.1-fix1" in name and ("-d1.42-" in name or "-delta1.42-" in name):
        return "s6 (TC-self fsx)"
    # S1: SFT-lo (p0-nv1-ng1-lo0.1-fix1)
    if "p0-nv1-ng1-lo0.1-fix1" in name and ("-d1.43-" in name or "-delta1.43-" in name):
        return "s1 (SFT-lo)"
    # Long-format names
    # S2: basic RA (full-completion--semi0.1--fix1, no TC)
    if "full-completion--semi0.1--fix1" in name and "tc-" not in name:
        if "-delta1.42-" in name:
            return "s2 (RA basic)"
    # S5: tc-self basic
    if "tc-self--full-completion--semi0.1--fix1" in name and "-delta1.42-" in name:
        return "s5 (TC-self basic)"
    # Also check for long-format s3
    if "full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "tc-" not in name and "-delta2.69-" in name:
            return "s3 (RA full)"
    # Long-format s4
    if "tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "-delta2.69-" in name:
            return "s4 (TC-self)"
    # Long-format s7
    if "tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "-delta2.69-" in name:
            return "s7 (TC-neg)"
    return None


def load_and_compute_metrics(filepath):
    """Load a score file and compute all metrics."""
    df = pd.read_csv(filepath)

    # Get label column
    label_col = None
    for col in ["label", "correct", "gpt4_ground_truth"]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        return None

    # Convert labels to binary
    s = df[label_col].astype(str).str.strip().str.lower()
    y = s.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}).values.astype(float)

    val = df["val_score"].values.astype(float)

    results = {"n": len(df), "n_pos": int(np.nansum(y == 1)), "n_neg": int(np.nansum(y == 0))}

    # Metrics for each gen variant
    for variant, col in [("raw", "gen_score"), ("tc", "gen_score_typcorr"),
                         ("lenorm", "gen_score_lenorm"), ("tc_lenorm", "gen_score_typcorr_lenorm")]:
        if col not in df.columns:
            continue
        gen = df[col].values.astype(float)
        mask = ~(np.isnan(gen) | np.isnan(val) | np.isnan(y))
        if mask.sum() < 4 or len(set(y[mask])) < 2:
            continue

        g, v, yy = gen[mask], val[mask], y[mask]

        try:
            results[f"{variant}_gen_roc"] = roc_auc_score(yy, g)
        except:
            results[f"{variant}_gen_roc"] = np.nan
        try:
            results[f"{variant}_val_roc"] = roc_auc_score(yy, v)
        except:
            results[f"{variant}_val_roc"] = np.nan
        results[f"{variant}_val_acc"] = accuracy_score(yy, (v > 0).astype(int))
        try:
            results[f"{variant}_pearson"], _ = pearsonr(g, v)
        except:
            results[f"{variant}_pearson"] = np.nan
        try:
            results[f"{variant}_spearman"], _ = spearmanr(g, v)
        except:
            results[f"{variant}_spearman"] = np.nan

        # Additional stats: concordance
        # Fraction of pairs where gen and val agree on ordering
        n_items = len(g)
        if n_items > 1:
            concordant = 0
            discordant = 0
            tied = 0
            # Use vectorized approach for efficiency (sample if too many pairs)
            max_pairs = 50000
            if n_items * (n_items - 1) // 2 > max_pairs:
                np.random.seed(42)
                idx_i = np.random.randint(0, n_items, max_pairs)
                idx_j = np.random.randint(0, n_items, max_pairs)
                # Avoid i==j
                valid = idx_i != idx_j
                idx_i, idx_j = idx_i[valid], idx_j[valid]
            else:
                from itertools import combinations
                pairs = list(combinations(range(n_items), 2))
                idx_i = np.array([p[0] for p in pairs])
                idx_j = np.array([p[1] for p in pairs])

            gen_diff = g[idx_i] - g[idx_j]
            val_diff = v[idx_i] - v[idx_j]
            concordant = np.sum((gen_diff > 0) & (val_diff > 0)) + np.sum((gen_diff < 0) & (val_diff < 0))
            discordant = np.sum((gen_diff > 0) & (val_diff < 0)) + np.sum((gen_diff < 0) & (val_diff > 0))
            total_pairs = concordant + discordant
            if total_pairs > 0:
                results[f"{variant}_concordance"] = concordant / total_pairs
            else:
                results[f"{variant}_concordance"] = np.nan

        # Generator score spread stats
        results[f"{variant}_gen_mean"] = np.mean(g)
        results[f"{variant}_gen_std"] = np.std(g)
        results[f"{variant}_gen_iqr"] = np.percentile(g, 75) - np.percentile(g, 25)
        results[f"{variant}_val_mean"] = np.mean(v)
        results[f"{variant}_val_std"] = np.std(v)

    return results


def compute_delta_histograms(filepath, variant_col="gen_score_typcorr"):
    """Compute histogram of pairwise deltas for generator and validator scores."""
    df = pd.read_csv(filepath)
    if variant_col not in df.columns:
        return None, None

    gen = df[variant_col].dropna().values
    val = df["val_score"].dropna().values

    # Sample pairs for histogram
    np.random.seed(42)
    n = len(gen)
    n_pairs = min(100000, n * (n - 1) // 2)
    idx_i = np.random.randint(0, n, n_pairs)
    idx_j = np.random.randint(0, n, n_pairs)
    valid = idx_i != idx_j
    idx_i, idx_j = idx_i[valid], idx_j[valid]

    gen_deltas = np.abs(gen[idx_i] - gen[idx_j])
    val_deltas = np.abs(val[idx_i] - val[idx_j])

    return gen_deltas, val_deltas


def main():
    print("=" * 70)
    print("  ANALYSIS: gemma-2-9b-it membership train-set dynamics")
    print("=" * 70)

    # 1. Find and categorize all score files
    inv = find_score_files()
    print(f"\nFound {len(inv)} score files")
    print(f"Settings covered: {sorted(inv['setting'].unique())}")
    print(f"TC eval types: {sorted(inv['tc_eval'].unique())}")
    print()

    # 2. Compute metrics for all files
    all_metrics = []
    for _, row in inv.iterrows():
        metrics = load_and_compute_metrics(row["file"])
        if metrics is None:
            continue
        metrics["setting"] = row["setting"]
        metrics["epoch"] = row["epoch"]
        metrics["tc_eval"] = row["tc_eval"]
        metrics["filename"] = row["file"].name
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(TABLES_DIR / "gemma_membership_all_metrics.csv", index=False)
    print(f"Computed metrics for {len(metrics_df)} files")
    print(f"Saved to analysis/tables/gemma_membership_all_metrics.csv")

    # 3. Build comparison table: epoch 2 (final model) across settings
    final_models = metrics_df[metrics_df["epoch"] == 2].copy()
    base_models = metrics_df[metrics_df["setting"] == "Base"].copy()

    # For the main comparison, use tc_eval='self' with gen_score_typcorr (the standard metric)
    # This is the "tc" variant in our metrics
    print("\n" + "=" * 70)
    print("  Q1: WHEN DOES RANKALIGN / OUR METHOD / TC WORK?")
    print("  (Evaluated with self-TC scoring on train set, epoch 2)")
    print("=" * 70)

    # Filter to self-TC eval for the main comparison
    self_eval = final_models[final_models["tc_eval"] == "self"].copy()
    base_self = base_models[base_models["tc_eval"] == "self"]

    if not base_self.empty:
        base_row = base_self.iloc[0]
        print(f"\nBase model (self-TC eval):")
        print(f"  gen_roc(tc)={base_row.get('tc_gen_roc', 'N/A'):.4f}  "
              f"spearman(tc)={base_row.get('tc_spearman', 'N/A'):.4f}  "
              f"val_roc={base_row.get('tc_val_roc', 'N/A'):.4f}  "
              f"val_acc={base_row.get('tc_val_acc', 'N/A'):.4f}")

    comparison_rows = []
    for _, row in self_eval.iterrows():
        comparison_rows.append({
            "Setting": row["setting"],
            "gen_roc (tc)": row.get("tc_gen_roc", np.nan),
            "gen_roc (raw)": row.get("raw_gen_roc", np.nan),
            "spearman (tc)": row.get("tc_spearman", np.nan),
            "pearson (tc)": row.get("tc_pearson", np.nan),
            "val_roc": row.get("tc_val_roc", np.nan),
            "val_acc": row.get("tc_val_acc", np.nan),
            "concordance (tc)": row.get("tc_concordance", np.nan),
        })

    comp_df = pd.DataFrame(comparison_rows).sort_values("gen_roc (tc)", ascending=False)
    comp_df.to_csv(TABLES_DIR / "gemma_membership_comparison_self_tc_ep2.csv", index=False)

    print("\n  Setting comparison (epoch 2, self-TC scoring):")
    print(comp_df.to_string(index=False, float_format="%.4f"))

    # Also do neg-TC eval comparison
    neg_eval = final_models[final_models["tc_eval"] == "neg"].copy()
    if not neg_eval.empty:
        print("\n\n  Setting comparison (epoch 2, neg-TC scoring):")
        neg_rows = []
        for _, row in neg_eval.iterrows():
            neg_rows.append({
                "Setting": row["setting"],
                "gen_roc (tc)": row.get("tc_gen_roc", np.nan),
                "spearman (tc)": row.get("tc_spearman", np.nan),
                "val_roc": row.get("tc_val_roc", np.nan),
                "concordance (tc)": row.get("tc_concordance", np.nan),
            })
        neg_df = pd.DataFrame(neg_rows).sort_values("gen_roc (tc)", ascending=False)
        neg_df.to_csv(TABLES_DIR / "gemma_membership_comparison_neg_tc_ep2.csv", index=False)
        print(neg_df.to_string(index=False, float_format="%.4f"))

    # 4. Training dynamics: metrics across epochs
    print("\n\n" + "=" * 70)
    print("  TRAINING DYNAMICS: Metrics by epoch")
    print("=" * 70)

    dynamics_data = metrics_df[metrics_df["tc_eval"] == "self"].copy()
    for setting in sorted(dynamics_data["setting"].unique()):
        if setting == "Base":
            continue
        subset = dynamics_data[dynamics_data["setting"] == setting].sort_values("epoch")
        if subset.empty:
            continue
        print(f"\n  {setting}:")
        print(f"    {'Epoch':<6} {'gen_roc(tc)':<12} {'spearman(tc)':<13} {'val_roc':<9} {'concordance':<12}")
        for _, row in subset.iterrows():
            print(f"    ep{int(row['epoch']):<4} "
                  f"{row.get('tc_gen_roc', np.nan):<12.4f} "
                  f"{row.get('tc_spearman', np.nan):<13.4f} "
                  f"{row.get('tc_val_roc', np.nan):<9.4f} "
                  f"{row.get('tc_concordance', np.nan):<12.4f}")

    # 5. Plots
    print("\n\n" + "=" * 70)
    print("  GENERATING PLOTS")
    print("=" * 70)

    # Plot 1: gen_roc across epochs for key settings
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    key_settings = ["s1 (SFT-lo)", "s2 (RA basic)", "s3 (RA full)",
                    "s4 (TC-self)", "s7 (TC-neg)"]
    colors = {"s1 (SFT-lo)": "#1f77b4", "s2 (RA basic)": "#ff7f0e",
              "s3 (RA full)": "#2ca02c", "s4 (TC-self)": "#d62728",
              "s7 (TC-neg)": "#9467bd"}

    for setting in key_settings:
        subset = dynamics_data[(dynamics_data["setting"] == setting)].sort_values("epoch")
        if subset.empty:
            continue
        epochs = subset["epoch"].values
        gen_roc = subset["tc_gen_roc"].values
        spearman = subset["tc_spearman"].values

        axes[0].plot(epochs, gen_roc, 'o-', label=setting, color=colors.get(setting, None))
        axes[1].plot(epochs, spearman, 'o-', label=setting, color=colors.get(setting, None))

    # Add base model line
    if not base_self.empty:
        base_gen_roc = base_self.iloc[0].get("tc_gen_roc", np.nan)
        base_spearman = base_self.iloc[0].get("tc_spearman", np.nan)
        axes[0].axhline(base_gen_roc, color='gray', linestyle='--', alpha=0.7, label='Base')
        axes[1].axhline(base_spearman, color='gray', linestyle='--', alpha=0.7, label='Base')

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Generator ROC AUC (TC)")
    axes[0].set_title("Gen ROC (self-TC) across epochs\ngemma-2-9b-it, membership train set")
    axes[0].legend(fontsize=8)
    axes[0].set_xticks([0, 1, 2])

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Spearman ρ (gen vs val, TC)")
    axes[1].set_title("Spearman correlation across epochs\ngemma-2-9b-it, membership train set")
    axes[1].legend(fontsize=8)
    axes[1].set_xticks([0, 1, 2])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gemma_membership_dynamics_key_settings.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/gemma_membership_dynamics_key_settings.png")

    # Plot 2: All settings bar chart at epoch 2
    fig, ax = plt.subplots(figsize=(12, 6))
    self_ep2 = dynamics_data[(dynamics_data["epoch"] == 2)].copy()
    if not self_ep2.empty:
        self_ep2 = self_ep2.sort_values("tc_gen_roc", ascending=True)
        y_pos = range(len(self_ep2))
        bars = ax.barh(y_pos, self_ep2["tc_gen_roc"].values, color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self_ep2["setting"].values)
        ax.set_xlabel("Generator ROC AUC (TC, self-scoring)")
        ax.set_title("gemma-2-9b-it membership: Gen ROC at epoch 2 (train set, self-TC)")
        if not base_self.empty:
            ax.axvline(base_self.iloc[0].get("tc_gen_roc", 0.5), color='red',
                       linestyle='--', alpha=0.7, label='Base model')
            ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "gemma_membership_genroc_bar_ep2.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: analysis/plots/gemma_membership_genroc_bar_ep2.png")

    # Plot 3: Delta histograms for key settings (epoch 2)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plot_settings = ["Base", "s2 (RA basic)", "s3 (RA full)", "s4 (TC-self)", "s7 (TC-neg)", "s1 (SFT-lo)"]

    for idx, setting in enumerate(plot_settings):
        ax = axes[idx // 3, idx % 3]
        if setting == "Base":
            files = inv[(inv["setting"] == "Base") & (inv["tc_eval"] == "self")]
        else:
            files = inv[(inv["setting"] == setting) & (inv["tc_eval"] == "self")]
            # Get epoch 2
            files = files[files["file"].apply(lambda f: "-e2-" in f.name or "-epoch2--" in f.name)]

        if files.empty:
            ax.set_title(f"{setting}\n(no data)")
            continue

        filepath = files.iloc[0]["file"]
        gen_deltas, val_deltas = compute_delta_histograms(filepath)
        if gen_deltas is None:
            ax.set_title(f"{setting}\n(no tc column)")
            continue

        ax.hist(gen_deltas, bins=50, alpha=0.6, label=f"Gen |Δ| (μ={np.mean(gen_deltas):.1f})",
                color='blue', density=True)
        ax.hist(val_deltas, bins=50, alpha=0.6, label=f"Val |Δ| (μ={np.mean(val_deltas):.1f})",
                color='orange', density=True)
        ax.set_title(f"{setting}")
        ax.legend(fontsize=7)
        ax.set_xlabel("|score_i - score_j|")

    plt.suptitle("Pairwise score delta distributions (gemma-2-9b-it, membership train set)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gemma_membership_delta_histograms.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/gemma_membership_delta_histograms.png")

    # Plot 4: Concordance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    conc_data = dynamics_data[dynamics_data["epoch"] == 2][["setting", "tc_concordance"]].dropna()
    if not conc_data.empty:
        conc_data = conc_data.sort_values("tc_concordance", ascending=True)
        y_pos = range(len(conc_data))
        ax.barh(y_pos, conc_data["tc_concordance"].values, color='coral', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(conc_data["setting"].values)
        ax.set_xlabel("Concordance (fraction of pairs where gen & val agree on ordering)")
        ax.set_title("Generator-Validator Concordance at epoch 2\ngemma-2-9b-it, membership train set")
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "gemma_membership_concordance_ep2.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: analysis/plots/gemma_membership_concordance_ep2.png")

    print("\n\nDone! All outputs in analysis/plots/ and analysis/tables/")


if __name__ == "__main__":
    main()
