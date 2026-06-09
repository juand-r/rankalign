#!/usr/bin/env python3
"""
Focused comparison plot: TC-self (s4) vs no-TC (s3) vs TC-neg (s7).
For both self-TC eval and neg-TC eval scoring.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = REPO / "analysis" / "plots"
TABLES_DIR = REPO / "analysis" / "tables"

# Load the pre-computed metrics
metrics = pd.read_csv(TABLES_DIR / "all_combos_metrics.csv")
gemma_mem = pd.read_csv(TABLES_DIR / "gemma_membership_all_metrics.csv")


def main():
    # Focus: gemma membership, which has the most settings including s7 and s12
    # We want to compare s3 vs s4 (self-TC eval) and s3 vs s7 (neg-TC eval)

    # Self-TC evaluation
    self_eval = gemma_mem[(gemma_mem["tc_eval"] == "self") & (gemma_mem["epoch"] == 2)]
    neg_eval = gemma_mem[(gemma_mem["tc_eval"] == "neg") & (gemma_mem["epoch"] == 2)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: self-TC eval comparing s3, s4, s11
    ax = axes[0]
    settings_self = ["s3 (RA full)", "s4 (TC-self)", "s11 (TC-self vlo)", "s5 (TC-self basic)", "s6 (TC-self fsx)"]
    data_self = self_eval[self_eval["setting"].isin(settings_self)].copy()
    data_self = data_self.sort_values("tc_gen_roc", ascending=True)

    if not data_self.empty:
        y = range(len(data_self))
        colors = []
        for s in data_self["setting"]:
            if "TC-self" in s:
                colors.append("#d62728")
            else:
                colors.append("#2ca02c")

        ax.barh(y, data_self["tc_gen_roc"].values, color=colors, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(data_self["setting"].values)
        ax.set_xlabel("Gen ROC AUC (self-TC scoring)")
        ax.set_title("Self-TC eval: s3 (no TC) vs TC-self variants\nGemma membership, epoch 2")
        # Add s3 reference line
        s3_val = data_self[data_self["setting"] == "s3 (RA full)"]["tc_gen_roc"].values
        if len(s3_val) > 0:
            ax.axvline(s3_val[0], color='green', linestyle='--', alpha=0.6, label='s3 (no TC)')
        ax.legend()

    # Right: neg-TC eval comparing s3, s7, s12
    ax = axes[1]
    settings_neg = ["s3 (RA full)", "s7 (TC-neg)", "s12 (TC-neg vlo)"]
    data_neg = neg_eval[neg_eval["setting"].isin(settings_neg)].copy()
    data_neg = data_neg.sort_values("tc_gen_roc", ascending=True)

    if not data_neg.empty:
        y = range(len(data_neg))
        colors = []
        for s in data_neg["setting"]:
            if "TC-neg" in s:
                colors.append("#9467bd")
            else:
                colors.append("#2ca02c")

        ax.barh(y, data_neg["tc_gen_roc"].values, color=colors, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(data_neg["setting"].values)
        ax.set_xlabel("Gen ROC AUC (neg-TC scoring)")
        ax.set_title("Neg-TC eval: s3 (no TC) vs TC-neg variants\nGemma membership, epoch 2")
        s3_val = data_neg[data_neg["setting"] == "s3 (RA full)"]["tc_gen_roc"].values
        if len(s3_val) > 0:
            ax.axvline(s3_val[0], color='green', linestyle='--', alpha=0.6, label='s3 (no TC)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tc_comparison_self_vs_neg.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: analysis/plots/tc_comparison_self_vs_neg.png")

    # Also: epoch-by-epoch dynamics for the TC comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Dynamics: self-TC eval for s4, s3, s11
    ax = axes[0]
    self_dyn = gemma_mem[gemma_mem["tc_eval"] == "self"]
    for setting, color, ls in [("s3 (RA full)", "#2ca02c", "-"),
                                ("s4 (TC-self)", "#d62728", "-"),
                                ("s11 (TC-self vlo)", "#ff7f0e", "--")]:
        sub = self_dyn[self_dyn["setting"] == setting].sort_values("epoch")
        if not sub.empty:
            ax.plot(sub["epoch"], sub["tc_gen_roc"], 'o' + ls, label=setting, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gen ROC AUC (self-TC)")
    ax.set_title("Self-TC training dynamics\nGemma membership")
    ax.legend()
    ax.set_xticks([0, 1, 2])

    # Dynamics: neg-TC eval for s3, s7, s12
    ax = axes[1]
    neg_dyn = gemma_mem[gemma_mem["tc_eval"] == "neg"]
    for setting, color, ls in [("s3 (RA full)", "#2ca02c", "-"),
                                ("s7 (TC-neg)", "#9467bd", "-"),
                                ("s12 (TC-neg vlo)", "#ff7f0e", "--")]:
        sub = neg_dyn[neg_dyn["setting"] == setting].sort_values("epoch")
        if not sub.empty:
            ax.plot(sub["epoch"], sub["tc_gen_roc"], 'o' + ls, label=setting, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gen ROC AUC (neg-TC)")
    ax.set_title("Neg-TC training dynamics\nGemma membership")
    ax.legend()
    ax.set_xticks([0, 1, 2])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "tc_dynamics_self_and_neg.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: analysis/plots/tc_dynamics_self_and_neg.png")

    # Print the focused TC comparison numbers
    print("\n=== TC Comparison Summary (Gemma Membership) ===")
    print("\nSelf-TC eval (s3 vs s4 comparison):")
    for _, row in data_self.sort_values("tc_gen_roc", ascending=False).iterrows():
        delta_str = ""
        if "s3" not in row["setting"]:
            s3_row = data_self[data_self["setting"] == "s3 (RA full)"]
            if not s3_row.empty:
                delta = row["tc_gen_roc"] - s3_row.iloc[0]["tc_gen_roc"]
                delta_str = f" (Δ vs s3: {delta:+.4f})"
        print(f"  {row['setting']:<22}: gen_roc={row['tc_gen_roc']:.4f}{delta_str}")

    print("\nNeg-TC eval (s3 vs s7 comparison):")
    for _, row in data_neg.sort_values("tc_gen_roc", ascending=False).iterrows():
        delta_str = ""
        if "s3" not in row["setting"]:
            s3_row = data_neg[data_neg["setting"] == "s3 (RA full)"]
            if not s3_row.empty:
                delta = row["tc_gen_roc"] - s3_row.iloc[0]["tc_gen_roc"]
                delta_str = f" (Δ vs s3: {delta:+.4f})"
        print(f"  {row['setting']:<22}: gen_roc={row['tc_gen_roc']:.4f}{delta_str}")


if __name__ == "__main__":
    main()
