#!/usr/bin/env python3
"""
Detailed WandB training analysis:
- Loss component breakdown per setting
- Score dynamics (how generator and validator scores evolve)
- Epoch-by-epoch loss analysis
- Comparison: why s2 fails on ifeval but works on membership
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

import wandb

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
    "s1": "gxm3rh44",
    "s2": "u701qq66",
    "s3": "2ayy1ty3",
    "s4": "bfrcn9l1",
    "s7": "wa42i0z3",
}


def pull_full_history(run_id, project="juand-r/rankalign"):
    """Pull a 2000-row sample of training history from wandb.

    Note: despite the legacy function name, this uses the sampled API
    (run.history(samples=N)), not the full per-step stream. For exact
    per-step values, use run.scan_history() instead. The 2000-sample view
    is sufficient for trend visualization and aggregate statistics over
    sub-windows of training; values reported from the final third should
    be read as estimates over ~660 sampled rows.
    """
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    history = run.history(samples=2000, pandas=True)
    return history, run.name, run.config


def analyze_loss_breakdown():
    """Compare loss component magnitudes between settings."""
    print("\n--- Loss Component Breakdown ---")

    all_summaries = []

    for group_name, runs in [("Gemma IFEval", GEMMA_IFEVAL_RUNS),
                              ("Qwen Membership", QWEN_MEMBERSHIP_RUNS)]:
        print(f"\n  {group_name}:")
        for setting, run_id in runs.items():
            try:
                history, name, config = pull_full_history(run_id)
            except Exception as e:
                print(f"    {setting}: ERROR {e}")
                continue

            # Get final-epoch stats (last 33% of steps)
            n = len(history)
            final_third = history.iloc[n * 2 // 3:]

            row = {
                "group": group_name,
                "setting": setting,
                "total_loss_mean": final_third["train/loss"].mean() if "train/loss" in final_third else np.nan,
                "pref_loss_mean": final_third["train/preference_loss"].mean() if "train/preference_loss" in final_third else np.nan,
                "nll_g_mean": final_third["train/nll_generator_loss"].mean() if "train/nll_generator_loss" in final_third else np.nan,
                "nll_v_mean": final_third["train/nll_validator_loss"].mean() if "train/nll_validator_loss" in final_third else np.nan,
                "total_steps": n,
            }

            # Score dynamics
            if "train/score_gen_i" in history.columns:
                row["score_gen_i_final"] = final_third["train/score_gen_i"].mean()
                row["score_gen_j_final"] = final_third["train/score_gen_j"].mean()
                row["gen_spread_final"] = (final_third["train/score_gen_i"] - final_third["train/score_gen_j"]).mean()

            if "train/score_i" in history.columns:
                row["score_val_i_final"] = final_third["train/score_i"].mean()
                row["score_val_j_final"] = final_third["train/score_j"].mean()
                row["val_spread_final"] = (final_third["train/score_i"] - final_third["train/score_j"]).mean()

            if "train/diff" in history.columns:
                row["diff_final"] = final_third["train/diff"].mean()

            all_summaries.append(row)
            print(f"    {setting}: total={row['total_loss_mean']:.4f}  "
                  f"pref={row['pref_loss_mean']:.4f}  "
                  f"nll_g={row['nll_g_mean']:.4f}  "
                  f"nll_v={row['nll_v_mean']:.4f}")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(TABLES_DIR / "wandb_loss_breakdown_detailed.csv", index=False)
    print(f"\n  Saved: analysis/tables/wandb_loss_breakdown_detailed.csv")
    return summary_df


def _plot_spread_panel(ax, runs, settings, kind, combo_label):
    """kind in {'gen', 'val'} -> picks the score column pair and label."""
    if kind == "gen":
        col_i, col_j = "train/score_gen_i", "train/score_gen_j"
        ylabel = "Gen score spread (i - j)"
        title = f"Generator score spread\n({combo_label})"
        legend_kind = "gen"
    else:
        col_i, col_j = "train/score_i", "train/score_j"
        ylabel = "Val score spread (i - j)"
        title = f"Validator score spread\n({combo_label})"
        legend_kind = "val"

    for setting in settings:
        if setting not in runs:
            continue
        history, _, _ = pull_full_history(runs[setting])
        if col_i in history.columns and col_j in history.columns:
            sub = history[["_step", col_i, col_j]].dropna()
            if len(sub) == 0:
                continue
            spread = (sub[col_i] - sub[col_j]).rolling(20, min_periods=1).mean()
            ax.plot(sub["_step"].values, spread.values,
                    label=f"{setting} {legend_kind}(i)-{legend_kind}(j)", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)


def plot_score_evolution():
    """Plot how generator/validator scores evolve during training."""
    print("\n--- Score Evolution Plots ---")

    combos = [
        ("Gemma IFEval",     GEMMA_IFEVAL_RUNS),
        ("Qwen IFEval",      QWEN_IFEVAL_RUNS),
        ("Qwen Membership",  QWEN_MEMBERSHIP_RUNS),
    ]
    fig, axes = plt.subplots(len(combos), 2, figsize=(16, 6 * len(combos)))

    for row, (combo_label, runs) in enumerate(combos):
        _plot_spread_panel(axes[row, 0], runs, ["s2", "s3", "s4"], "gen", combo_label)
        _plot_spread_panel(axes[row, 1], runs, ["s2", "s3", "s4"], "val", combo_label)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wandb_score_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/wandb_score_evolution.png")

    # Focused plot: absolute gen score magnitudes (to show explosion)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for setting, run_id in GEMMA_IFEVAL_RUNS.items():
        if setting == "s1":
            continue
        history, _, _ = pull_full_history(run_id)
        if "train/score_gen_i" in history.columns:
            steps = history["_step"].values
            gen_i = history["train/score_gen_i"].rolling(20).mean()
            ax.plot(steps, gen_i, label=f"{setting}", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Gen score (item i)")
    ax.set_title("Generator score magnitude\n(Gemma IFEval)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for setting, run_id in QWEN_MEMBERSHIP_RUNS.items():
        if setting == "s1":
            continue
        history, _, _ = pull_full_history(run_id)
        if "train/score_gen_i" in history.columns:
            steps = history["_step"].values
            gen_i = history["train/score_gen_i"].rolling(20).mean()
            ax.plot(steps, gen_i, label=f"{setting}", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Gen score (item i)")
    ax.set_title("Generator score magnitude\n(Qwen Membership)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wandb_gen_score_magnitude.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/wandb_gen_score_magnitude.png")


def plot_preference_loss_analysis():
    """Analyze how well the preference loss is being optimized."""
    print("\n--- Preference Loss Analysis ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gemma IFEval preference loss
    ax = axes[0]
    for setting, run_id in GEMMA_IFEVAL_RUNS.items():
        history, _, _ = pull_full_history(run_id)
        if "train/preference_loss" in history.columns:
            steps = history["_step"].values
            pref = history["train/preference_loss"].rolling(30).mean()
            ax.plot(steps, pref, label=f"{setting}", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Preference Loss")
    ax.set_title("Preference Loss Evolution\n(Gemma IFEval)")
    ax.legend(fontsize=8)

    # Qwen Membership preference loss
    ax = axes[1]
    for setting, run_id in QWEN_MEMBERSHIP_RUNS.items():
        history, _, _ = pull_full_history(run_id)
        if "train/preference_loss" in history.columns:
            steps = history["_step"].values
            pref = history["train/preference_loss"].rolling(30).mean()
            ax.plot(steps, pref, label=f"{setting}", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Preference Loss")
    ax.set_title("Preference Loss Evolution\n(Qwen Membership)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "wandb_preference_loss_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/wandb_preference_loss_evolution.png")


def main():
    print("=" * 70)
    print("  DETAILED WANDB TRAINING ANALYSIS")
    print("=" * 70)

    summary_df = analyze_loss_breakdown()
    plot_score_evolution()
    plot_preference_loss_analysis()

    print("\n\nDone!")


if __name__ == "__main__":
    main()
