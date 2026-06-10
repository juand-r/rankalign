#!/usr/bin/env python3
"""
Q2: Training diagnostics from WandB logs.

Pulls loss curves (NLL-G, NLL-V, preference loss, total loss) and
training statistics from WandB API for gemma-ifeval and qwen-membership runs.

Outputs to analysis/plots/ and analysis/tables/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = REPO / "analysis" / "plots"
TABLES_DIR = REPO / "analysis" / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SETTING_LABELS = {
    "base": "Base",
    "s1": "SFT",
    "s2": "RankAlign",
    "s3": "Ours (w/o TC)",
    "s4": "Ours",
    "s7": "Ours (neg TC)",
}

# WandB run IDs for finished runs
# NOTE: gemma-membership v7 was trained before WandB logging was added.
# Older (pre-v7) runs exist but lack fix1, force-same-x, ppd, vallogodds
# so they are not directly comparable.

GEMMA_IFEVAL_RUNS = {
    "s1": "5ikunquj",
    "s2": "rd7pz236",
    "s3": "55x00ezb",
    "s4": "h3ncq9u2",
    "s7": "xj4t4ab8",
}

QWEN_MEMBERSHIP_RUNS = {
    "s1": "du2ceze3",
    "s2": "6sc1vpho",
    "s3": "r3cmhowf",
    "s4": "ictz0z9q",
    "s7": "w0htohh3",
}

QWEN_IFEVAL_RUNS = {
    "s1": None,  # will find
    "s2": None,
    "s4": None,
    "s3": None,
    "s7": None,
}


def pull_run_history(run_id, project="juand-r/rankalign", samples=500):
    """Pull training history from a wandb run."""
    import wandb
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    history = run.history(samples=samples)
    return history, run.name, run.config


def analyze_runs(runs_dict, label_prefix):
    """Analyze a set of runs and produce loss curve plots."""
    all_data = {}
    configs = {}

    for setting, run_id in runs_dict.items():
        if run_id is None:
            continue
        try:
            history, name, config = pull_run_history(run_id)
            all_data[setting] = history
            configs[setting] = config
            print(f"  Pulled {setting} ({name}): {len(history)} steps")
        except Exception as e:
            print(f"  ERROR pulling {setting} ({run_id}): {e}")

    if not all_data:
        print("  No data pulled!")
        return None

    # Plot loss curves in two figures:
    # Panel A: SFT vs RankAlign vs Ours (s1, s2, s4)
    # Panel B: Ours variants (s4, s3, s7)
    metrics_to_plot = [
        ("train/loss", "Total Loss"),
        ("train/preference_loss", "Preference Loss"),
        ("train/nll_generator_loss", "NLL Generator Loss"),
        ("train/nll_validator_loss", "NLL Validator Loss"),
    ]

    group_a = {"s1": "SFT", "s2": "RankAlign", "s4": "Ours"}
    group_b = {"s4": "Ours", "s3": "Ours (w/o TC)", "s7": "Ours (neg TC)"}

    for group, group_label, suffix in [
        (group_a, "SFT vs RankAlign vs Ours", "methods"),
        (group_b, "Ours variants (TC comparison)", "tc"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()

        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes_flat[idx]
            for setting, legend_label in group.items():
                if setting not in all_data:
                    continue
                history = all_data[setting]
                if metric in history.columns:
                    sub = history[["_step", metric]].dropna()
                    if len(sub) > 0:
                        smoothed = sub[metric].rolling(window=20, min_periods=1).mean()
                        ax.plot(sub["_step"].values, smoothed.values, label=legend_label, alpha=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel(title)
            ax.set_title(f"{title}\n({label_prefix})")
            ax.legend(fontsize=9, loc="best")

        plt.suptitle(f"{group_label} — {label_prefix}", fontsize=12, y=1.01)
        plt.tight_layout()
        fname = f"wandb_loss_{suffix}_{label_prefix.replace(' ', '_').lower()}.png"
        plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: analysis/plots/{fname}")

    # Zoomed versions with capped y-limits
    YLIM_CAPS = {
        "train/loss": 150,
        "train/preference_loss": 20,
        "train/nll_generator_loss": 150,
        "train/nll_validator_loss": 2.5,
    }
    for group, group_label, suffix in [
        (group_a, "SFT vs RankAlign vs Ours", "methods"),
        (group_b, "Ours variants (TC comparison)", "tc"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()

        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes_flat[idx]
            for setting, legend_label in group.items():
                if setting not in all_data:
                    continue
                history = all_data[setting]
                if metric in history.columns:
                    sub = history[["_step", metric]].dropna()
                    if len(sub) > 0:
                        smoothed = sub[metric].rolling(window=20, min_periods=1).mean()
                        ax.plot(sub["_step"].values, smoothed.values, label=legend_label, alpha=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel(title)
            ax.set_title(f"{title}\n({label_prefix})")
            ax.set_ylim(0, YLIM_CAPS[metric])
            ax.legend(fontsize=9, loc="best")

        plt.suptitle(f"{group_label} — {label_prefix} [zoomed]", fontsize=12, y=1.01)
        plt.tight_layout()
        fname = f"wandb_loss_{suffix}_{label_prefix.replace(' ', '_').lower()}_zoomed.png"
        plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: analysis/plots/{fname}")

    # Plot score statistics (if available)
    score_metrics = [
        ("train/score_i", "Score i (val)"),
        ("train/score_j", "Score j (val)"),
        ("train/score_gen_i", "Gen Score i"),
        ("train/score_gen_j", "Gen Score j"),
        ("train/diff", "Diff (score_i - score_j)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(score_metrics):
        if idx >= len(axes):
            break
        ax = axes[idx]
        for setting, history in all_data.items():
            if metric in history.columns:
                sub = history[["_step", metric]].dropna()
                if len(sub) > 0:
                    smoothed = sub[metric].rolling(window=20, min_periods=1).mean()
                    ax.plot(sub["_step"].values, smoothed.values, label=SETTING_LABELS.get(setting, setting), alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(f"{title}")
        ax.legend(fontsize=8)

    # Hide unused subplot
    if len(score_metrics) < len(axes):
        for i in range(len(score_metrics), len(axes)):
            axes[i].set_visible(False)

    plt.suptitle(f"Training Score Statistics ({label_prefix})", fontsize=12)
    plt.tight_layout()
    fname = f"wandb_score_stats_{label_prefix.replace(' ', '_').lower()}.png"
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/{fname}")

    # Summary table: final-step metrics
    summary_rows = []
    for setting, history in all_data.items():
        row = {"setting": setting}
        for metric, _ in metrics_to_plot:
            if metric in history.columns:
                vals = history[metric].dropna()
                if len(vals) > 0:
                    # Use last 10% as "final" metrics
                    final_vals = vals.iloc[-max(1, len(vals)//10):]
                    row[metric.replace("train/", "") + "_final_mean"] = final_vals.mean()
                    row[metric.replace("train/", "") + "_final_std"] = final_vals.std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_name = f"wandb_training_summary_{label_prefix.replace(' ', '_').lower()}.csv"
    summary_df.to_csv(TABLES_DIR / csv_name, index=False)
    print(f"  Saved: analysis/tables/{csv_name}")

    # Diagnostic checks
    print(f"\n  === Diagnostics for {label_prefix} ===")
    for setting, history in all_data.items():
        issues = []
        if "train/loss" in history.columns:
            loss = history["train/loss"].dropna()
            if len(loss) > 10:
                # Check for loss explosion
                if loss.iloc[-1] > loss.iloc[0] * 2:
                    issues.append("LOSS INCREASED (possible divergence)")
                # Check for NaN
                if loss.isna().any():
                    issues.append("NaN in loss")
                # Check for very high variance in last quarter
                last_quarter = loss.iloc[-len(loss)//4:]
                if last_quarter.std() > last_quarter.mean() * 0.5:
                    issues.append("HIGH VARIANCE in final quarter")

        if "train/nll_generator_loss" in history.columns:
            nll_g = history["train/nll_generator_loss"].dropna()
            if len(nll_g) > 10 and nll_g.iloc[-1] > nll_g.iloc[0] * 1.5:
                issues.append("NLL-G increased significantly")

        if "train/nll_validator_loss" in history.columns:
            nll_v = history["train/nll_validator_loss"].dropna()
            if len(nll_v) > 10 and nll_v.iloc[-1] > nll_v.iloc[0] * 1.5:
                issues.append("NLL-V increased significantly")

        status = "OK" if not issues else " | ".join(issues)
        print(f"    {setting}: {status}")

    return all_data


def main():
    import wandb  # verify import works
    print("=" * 70)
    print("  Q2: TRAINING DIAGNOSTICS FROM WANDB")
    print("=" * 70)

    # 1. Gemma IFEval runs
    print("\n--- Gemma-2-9b-it IFEval training runs ---")
    gemma_data = analyze_runs(GEMMA_IFEVAL_RUNS, "gemma-9b-it ifeval")

    # 2. Qwen Membership runs
    print("\n--- Qwen3.5-9B Membership training runs ---")
    qwen_mem_data = analyze_runs(QWEN_MEMBERSHIP_RUNS, "qwen3.5-9b membership")

    # 3. Find and analyze Qwen IFEval runs
    print("\n--- Finding Qwen3.5-9B IFEval runs ---")
    api = wandb.Api()
    runs = api.runs('juand-r/rankalign', order='-created_at', per_page=50)
    qwen_ifeval = {}
    for r in runs:
        if 'qwen' in r.name and 'ifeval' in r.name and r.state == 'finished':
            for ws in ["s1", "s2", "s3", "s4", "s7"]:
                if r.name.endswith(f"-{ws}") and ws not in qwen_ifeval:
                    qwen_ifeval[ws] = r.id
    print(f"  Found Qwen IFEval runs: {qwen_ifeval}")
    if qwen_ifeval:
        qwen_ifeval_data = analyze_runs(qwen_ifeval, "qwen3.5-9b ifeval")

    print("\n\nDone!")


if __name__ == "__main__":
    main()
