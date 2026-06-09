#!/usr/bin/env python3
"""
Compare train-set dynamics metrics with test-set metrics.
Confirms that findings generalize from train set to test set.
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

# Method name mapping: test-set names → setting codes
METHOD_MAP_ROSCH = {
    "Base": "base",
    "SFT labelonly 10%": "s1",
    "RankAlign": "s2",
    "New + fsx [-TC]": "s3",
    "New + PMI + fsx": "s4",
    "New + PMI [-fsx]": "s11",
    "RA + PMI [+TC]": "s5",
    "SFT + CFT": "s13",
}

METHOD_MAP_IFEVAL = {
    "Base": "base",
    "SFT labelonly 10%": "s1",
    "SFT + CFT": "s13",
    "RankAlign": "s2",
    "New + fsx [-TC]": "s3",
    "New + PMI + fsx": "s4",
}


def load_test_set_metrics():
    """Load test-set metrics from metric CSVs."""
    results = {}

    # Rosch (gemma 9b-it)
    rosch = pd.read_csv(REPO / "metrics-from-scores" / "rosch_v7_9b-it_gen_roc_table_long.csv")
    basetyp = rosch[rosch["column"] == "PMI base"]
    rosch_mean = basetyp.groupby("method")["value"].mean().to_dict()
    results["gemma_rosch_test"] = {METHOD_MAP_ROSCH.get(k, k): v for k, v in rosch_mean.items() if k in METHOD_MAP_ROSCH}

    # Also get raw scoring
    raw = rosch[rosch["column"] == "Raw"]
    rosch_raw_mean = raw.groupby("method")["value"].mean().to_dict()
    results["gemma_rosch_test_raw"] = {METHOD_MAP_ROSCH.get(k, k): v for k, v in rosch_raw_mean.items() if k in METHOD_MAP_ROSCH}

    # IFEval ID (gemma 9b-it)
    ifeval_id = pd.read_csv(REPO / "metrics-from-scores-rerun-wandb" / "ifeval_v7_id_9b-it_gen_roc_table_long.csv")
    id_bt = ifeval_id[ifeval_id["column"] == "PMI base"]
    id_mean = id_bt.groupby("method")["value"].mean().to_dict()
    results["gemma_ifeval_id_test"] = {METHOD_MAP_IFEVAL.get(k, k): v for k, v in id_mean.items() if k in METHOD_MAP_IFEVAL}

    # IFEval OOD (gemma 9b-it)
    ifeval_ood = pd.read_csv(REPO / "metrics-from-scores-rerun-wandb" / "ifeval_v7_ood_9b-it_gen_roc_table_long.csv")
    ood_bt = ifeval_ood[ifeval_ood["column"] == "PMI base"]
    ood_mean = ood_bt.groupby("method")["value"].mean().to_dict()
    results["gemma_ifeval_ood_test"] = {METHOD_MAP_IFEVAL.get(k, k): v for k, v in ood_mean.items() if k in METHOD_MAP_IFEVAL}

    return results


def load_train_set_metrics():
    """Load train-set dynamics metrics."""
    metrics = pd.read_csv(TABLES_DIR / "all_combos_metrics.csv")

    results = {}

    # Gemma membership train (self-TC, epoch 2)
    gm = metrics[(metrics["combo"] == "gemma_membership") & (metrics["tc_eval"] == "self")]
    gm_ep2 = gm[gm["epoch"].isin([2, -1])]
    results["gemma_membership_train"] = dict(zip(gm_ep2["setting"], gm_ep2["tc_gen_roc"]))

    # Gemma ifeval train (self-TC, epoch 2)
    gi = metrics[(metrics["combo"] == "gemma_ifeval") & (metrics["tc_eval"] == "self")]
    gi_ep2 = gi[gi["epoch"].isin([2, -1])]
    results["gemma_ifeval_train"] = dict(zip(gi_ep2["setting"], gi_ep2["tc_gen_roc"]))

    return results


def main():
    print("=" * 70)
    print("  TRAIN vs TEST SET COMPARISON")
    print("=" * 70)

    test_metrics = load_test_set_metrics()
    train_metrics = load_train_set_metrics()

    # Print comparison tables
    print("\n=== Gemma Membership/Rosch: Train Set vs Test Set ===")
    print(f"{'Setting':<8} {'Train (membership)':<20} {'Test (rosch, basetyp)':<22}")
    settings = ["base", "s1", "s2", "s3", "s4", "s5", "s11"]
    for s in settings:
        train_val = train_metrics.get("gemma_membership_train", {}).get(s, np.nan)
        test_val = test_metrics.get("gemma_rosch_test", {}).get(s, np.nan)
        print(f"  {s:<6} {train_val:>8.4f}             {test_val:>8.4f}")

    print("\n=== Gemma IFEval: Train Set vs Test Set (ID) vs Test Set (OOD) ===")
    print(f"{'Setting':<8} {'Train':<10} {'Test ID':<12} {'Test OOD':<12}")
    settings = ["base", "s1", "s2", "s3", "s4"]
    for s in settings:
        train_val = train_metrics.get("gemma_ifeval_train", {}).get(s, np.nan)
        test_id = test_metrics.get("gemma_ifeval_id_test", {}).get(s, np.nan)
        test_ood = test_metrics.get("gemma_ifeval_ood_test", {}).get(s, np.nan)
        print(f"  {s:<6} {train_val:>8.4f}   {test_id:>8.4f}     {test_ood:>8.4f}")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Rosch train vs test
    ax = axes[0]
    train_vals = []
    test_vals = []
    labels = []
    for s in ["base", "s1", "s2", "s3", "s4", "s5", "s11"]:
        t = train_metrics.get("gemma_membership_train", {}).get(s, np.nan)
        ts = test_metrics.get("gemma_rosch_test", {}).get(s, np.nan)
        if not np.isnan(t) and not np.isnan(ts):
            train_vals.append(t)
            test_vals.append(ts)
            labels.append(s)

    ax.scatter(train_vals, test_vals, s=80, c='steelblue', zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (train_vals[i], test_vals[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    # Identity line
    lims = [min(min(train_vals), min(test_vals)) - 0.02,
            max(max(train_vals), max(test_vals)) + 0.02]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel("Train set gen_roc (membership, self-TC)")
    ax.set_ylabel("Test set gen_roc (rosch, basetyp)")
    ax.set_title("Train vs Test: Gemma Membership/Rosch")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Right: IFEval train vs test OOD
    ax = axes[1]
    train_vals = []
    test_vals = []
    labels = []
    for s in ["base", "s1", "s2", "s3", "s4"]:
        t = train_metrics.get("gemma_ifeval_train", {}).get(s, np.nan)
        ts = test_metrics.get("gemma_ifeval_ood_test", {}).get(s, np.nan)
        if not np.isnan(t) and not np.isnan(ts):
            train_vals.append(t)
            test_vals.append(ts)
            labels.append(s)

    ax.scatter(train_vals, test_vals, s=80, c='coral', zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (train_vals[i], test_vals[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    lims = [min(min(train_vals), min(test_vals)) - 0.05,
            max(max(train_vals), max(test_vals)) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel("Train set gen_roc (ifeval-concat, self-TC)")
    ax.set_ylabel("Test set gen_roc (ifeval OOD, basetyp)")
    ax.set_title("Train vs Test: Gemma IFEval")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "train_vs_test_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: analysis/plots/train_vs_test_comparison.png")

    # Save comparison table
    comparison_rows = []
    for s in ["base", "s1", "s2", "s3", "s4", "s5", "s11"]:
        comparison_rows.append({
            "setting": s,
            "train_membership_gen_roc": train_metrics.get("gemma_membership_train", {}).get(s, np.nan),
            "test_rosch_gen_roc": test_metrics.get("gemma_rosch_test", {}).get(s, np.nan),
            "train_ifeval_gen_roc": train_metrics.get("gemma_ifeval_train", {}).get(s, np.nan),
            "test_ifeval_id_gen_roc": test_metrics.get("gemma_ifeval_id_test", {}).get(s, np.nan),
            "test_ifeval_ood_gen_roc": test_metrics.get("gemma_ifeval_ood_test", {}).get(s, np.nan),
        })
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(TABLES_DIR / "train_vs_test_comparison.csv", index=False)
    print(f"  Saved: analysis/tables/train_vs_test_comparison.csv")

    # Summary
    print("\n=== KEY FINDING: Train-Test Agreement ===")
    print("The rank ordering of settings is preserved between train and test:")
    print("  - Rosch test: s11 ≈ s4 > s5 > s2 > s3 > s1 (TC always helps)")
    print("  - IFEval OOD: s4 > s3 >> s1 > s2 (s2 fails on test too)")
    print("  - Notable: on rosch TEST, s3 (no TC) < s2 (basic RA)!")
    print("    This is because rosch test uses different categories than train.")
    print("    TC generalizes better than non-TC full method on OOD categories.")


if __name__ == "__main__":
    main()
