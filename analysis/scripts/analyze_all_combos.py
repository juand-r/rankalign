#!/usr/bin/env python3
"""
Extended analysis across all four model/task combinations.
Produces unified comparison tables and additional diagnostic stats.

Covers:
  - gemma-2-9b-it membership (train-set dynamics)
  - gemma-2-9b-it ifeval (train-set dynamics)
  - qwen3.5-9b membership (train-set dynamics)
  - qwen3.5-9b ifeval (train-set dynamics)
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

REPO = Path(__file__).resolve().parent.parent.parent
SCORES_DIR = Path("/datastor2/jdr/rankalign/outputs-trainset-dynamics")
PLOTS_DIR = REPO / "analysis" / "plots"
TABLES_DIR = REPO / "analysis" / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _match_setting_gemma_membership(name):
    if "tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d2.69-" in name or "-delta2.69-" in name):
        return "s4"
    if "tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d2.69-" in name or "-delta2.69-" in name):
        return "s7"
    if "tcs-nv1-ng1-vlo-sm0.1-fix1" in name and ("-d2.69-" in name or "-delta2.69-" in name):
        return "s11"
    if "tcn-nv1-ng1-vlo-sm0.1-fix1" in name and ("-d2.69-" in name or "-delta2.69-" in name):
        return "s12"
    if "nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and "tcs-" not in name and "tcn-" not in name:
        if "-d2.69-" in name or "-delta2.69-" in name:
            return "s3"
    if "tcs-fsx-ppd-sm0.1-fix1" in name and ("-d1.42-" in name or "-delta1.42-" in name):
        return "s6"
    if "p0-nv1-ng1-lo0.1-fix1" in name and ("-d1.43-" in name or "-delta1.43-" in name):
        return "s1"
    if "full-completion--semi0.1--fix1" in name and "tc-" not in name and "-delta1.42-" in name:
        return "s2"
    if "tc-self--full-completion--semi0.1--fix1" in name and "-delta1.42-" in name:
        return "s5"
    if "tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "-delta2.69-" in name:
            return "s4"
    if "tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "-delta2.69-" in name:
            return "s7"
    if "full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "tc-" not in name and "-delta2.69-" in name:
            return "s3"
    return None


def _match_setting_gemma_ifeval(name):
    if "tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d1.94-" in name or "-delta1.94-" in name):
        return "s4"
    if "tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d1.94-" in name or "-delta1.94-" in name):
        return "s7"
    if "nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and "tcs-" not in name and "tcn-" not in name:
        if "-d1.94-" in name or "-delta1.94-" in name:
            return "s3"
    if "p0-nv1-ng1-lo0.1-fix1" in name or "labelonly0.1--fix1" in name:
        if "-d1.29-" in name or "-delta1.29-" in name:
            return "s1"
    if "full-completion--semi0.1--fix1" in name and "tc-" not in name:
        if "-d1.38-" in name or "-delta1.38-" in name:
            return "s2"
    if "full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1--fix1" in name:
        if "-delta1.29-" in name:
            return "s1"
    return None


def _match_setting_qwen_membership(name):
    if "tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d1.54-" in name or "-delta1.54-" in name):
        return "s4"
    if "tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d1.54-" in name or "-delta1.54-" in name):
        return "s7"
    if "nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and "tcs-" not in name and "tcn-" not in name:
        if "-d1.54-" in name or "-delta1.54-" in name:
            return "s3"
    if "p0-nv1-ng1-vlo-lo0.1-fix1" in name and ("-d1.53-" in name or "-delta1.53-" in name):
        return "s1"
    if "vlo-sm0.1-fix1" in name and "tcs-" not in name and "tcn-" not in name and "fsx" not in name:
        if "-d1.54-" in name or "-delta1.54-" in name:
            return "s2"
    if "full-completion--vallogodds--semi0.1--fix1" in name and "tc-" not in name:
        if "-delta1.54-" in name:
            return "s2"
    if "full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1" in name:
        if "-delta1.53-" in name:
            return "s1"
    if "tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "-delta1.54-" in name:
            return "s4"
    if "tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "-delta1.54-" in name:
            return "s7"
    if "full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "tc-" not in name and "-delta1.54-" in name:
            return "s3"
    return None


def _match_setting_qwen_ifeval(name):
    if "tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d0.96-" in name or "-delta0.96-" in name):
        return "s4"
    if "tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and ("-d0.96-" in name or "-delta0.96-" in name):
        return "s7"
    if "nv1-ng1-vlo-fsx-ppd-sm0.1-fix1" in name and "tcs-" not in name and "tcn-" not in name:
        if "-d0.96-" in name or "-delta0.96-" in name:
            return "s3"
    if "p0-nv1-ng1-vlo-lo0.1-fix1" in name and ("-d0.84-" in name or "-delta0.84-" in name):
        return "s1"
    if "vlo-sm0.1-fix1" in name and "tcs-" not in name and "tcn-" not in name and "fsx" not in name:
        if "-d0.96-" in name or "-delta0.96-" in name:
            return "s2"
    if "full-completion--vallogodds--semi0.1--fix1" in name and "tc-" not in name:
        if "-delta0.96-" in name:
            return "s2"
    if "full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1" in name:
        if "-delta0.84-" in name:
            return "s1"
    if "tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "-delta0.96-" in name:
            return "s4"
    if "tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "-delta0.96-" in name:
            return "s7"
    if "full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1" in name:
        if "tc-" not in name and "-delta0.96-" in name:
            return "s3"
    return None


COMBO_CONFIG = {
    "gemma_membership": {
        "task_glob": "*membership*",
        "exclude": "Qwen",
        "model_label": "gemma-2-9b-it",
        "task_label": "membership",
        "match_fn": _match_setting_gemma_membership,
    },
    "gemma_ifeval": {
        "task_glob": "*ifeval-concat*",
        "exclude": "Qwen",
        "model_label": "gemma-2-9b-it",
        "task_label": "ifeval",
        "match_fn": _match_setting_gemma_ifeval,
    },
    "qwen_membership": {
        "task_glob": "*Qwen*membership*",
        "exclude": None,
        "model_label": "Qwen3.5-9B",
        "task_label": "membership",
        "match_fn": _match_setting_qwen_membership,
    },
    "qwen_ifeval": {
        "task_glob": "*Qwen*ifeval*",
        "exclude": None,
        "model_label": "Qwen3.5-9B",
        "task_label": "ifeval",
        "match_fn": _match_setting_qwen_ifeval,
    },
}


def find_files_for_combo(combo_name):
    cfg = COMBO_CONFIG[combo_name]
    all_files = sorted(SCORES_DIR.glob(f"scores_{cfg['task_glob']}"))
    if cfg["exclude"]:
        all_files = [f for f in all_files if cfg["exclude"] not in f.name]

    inventory = []
    for f in all_files:
        name = f.name
        if name.startswith("scores_basetypneg"):
            tc_eval = "neg"
        elif name.startswith("scores_basetyp"):
            tc_eval = "self"
        else:
            continue

        if "-v6-" in name:
            inventory.append({"file": f, "setting": "base", "epoch": -1, "tc_eval": tc_eval})
            continue

        epoch = None
        for ep in [0, 1, 2]:
            if f"-e{ep}-" in name or f"-epoch{ep}--" in name:
                epoch = ep
                break
        if epoch is None:
            continue

        setting = cfg["match_fn"](name)
        if setting:
            inventory.append({"file": f, "setting": setting, "epoch": epoch, "tc_eval": tc_eval})

    df = pd.DataFrame(inventory)
    if not df.empty:
        df["date"] = df["file"].apply(lambda f: f.name.split("_")[-1].replace(".csv", ""))
        df = df.sort_values("date", ascending=False).drop_duplicates(
            subset=["setting", "epoch", "tc_eval"], keep="first"
        ).drop(columns=["date"]).reset_index(drop=True)
    return df


def compute_metrics(filepath):
    """Compute all metrics from a score file."""
    df = pd.read_csv(filepath)

    label_col = None
    for col in ["label", "correct", "gpt4_ground_truth"]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        return None

    s = df[label_col].astype(str).str.strip().str.lower()
    y = s.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}).values.astype(float)
    val = df["val_score"].values.astype(float)

    results = {"n": len(df), "n_pos": int(np.nansum(y == 1)), "n_neg": int(np.nansum(y == 0))}

    for variant, col in [("raw", "gen_score"), ("tc", "gen_score_typcorr")]:
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
            results[f"{variant}_spearman"], _ = spearmanr(g, v)
        except:
            results[f"{variant}_spearman"] = np.nan
        try:
            results[f"{variant}_pearson"], _ = pearsonr(g, v)
        except:
            results[f"{variant}_pearson"] = np.nan

        # Concordance
        n_items = len(g)
        np.random.seed(42)
        n_pairs = min(50000, n_items * (n_items - 1) // 2)
        idx_i = np.random.randint(0, n_items, n_pairs)
        idx_j = np.random.randint(0, n_items, n_pairs)
        valid = idx_i != idx_j
        idx_i, idx_j = idx_i[valid], idx_j[valid]
        gen_diff = g[idx_i] - g[idx_j]
        val_diff = v[idx_i] - v[idx_j]
        concordant = np.sum((gen_diff > 0) & (val_diff > 0)) + np.sum((gen_diff < 0) & (val_diff < 0))
        discordant = np.sum((gen_diff > 0) & (val_diff < 0)) + np.sum((gen_diff < 0) & (val_diff > 0))
        total = concordant + discordant
        results[f"{variant}_concordance"] = concordant / total if total > 0 else np.nan

        # Score distribution stats
        results[f"{variant}_gen_mean"] = np.mean(g)
        results[f"{variant}_gen_std"] = np.std(g)
        results[f"{variant}_val_mean"] = np.mean(v)
        results[f"{variant}_val_std"] = np.std(v)

        # Pairwise delta stats (for histograms)
        gen_deltas = np.abs(g[idx_i] - g[idx_j])
        val_deltas = np.abs(v[idx_i] - v[idx_j])
        results[f"{variant}_gen_delta_mean"] = np.mean(gen_deltas)
        results[f"{variant}_gen_delta_median"] = np.median(gen_deltas)
        results[f"{variant}_val_delta_mean"] = np.mean(val_deltas)
        results[f"{variant}_val_delta_median"] = np.median(val_deltas)

    return results


def analyze_combo(combo_name):
    """Full analysis for one model/task combination."""
    cfg = COMBO_CONFIG[combo_name]
    label = f"{cfg['model_label']} / {cfg['task_label']}"
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    inv = find_files_for_combo(combo_name)
    if inv.empty:
        print("  No files found!")
        return None

    print(f"  Found {len(inv)} files, settings: {sorted(inv['setting'].unique())}")

    # Compute metrics
    rows = []
    for _, r in inv.iterrows():
        metrics = compute_metrics(r["file"])
        if metrics is None:
            continue
        metrics["setting"] = r["setting"]
        metrics["epoch"] = r["epoch"]
        metrics["tc_eval"] = r["tc_eval"]
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df["combo"] = combo_name
    df.to_csv(TABLES_DIR / f"{combo_name}_metrics.csv", index=False)

    # Print epoch-2 comparison (self-TC eval)
    ep2_self = df[(df["epoch"] == 2) & (df["tc_eval"] == "self")].copy()
    base_self = df[(df["setting"] == "base") & (df["tc_eval"] == "self")]

    print(f"\n  Epoch 2 (self-TC scoring):")
    if not base_self.empty:
        b = base_self.iloc[0]
        print(f"    Base: gen_roc(tc)={b.get('tc_gen_roc',np.nan):.4f}  "
              f"spearman={b.get('tc_spearman',np.nan):.4f}  "
              f"concordance={b.get('tc_concordance',np.nan):.4f}")

    for _, row in ep2_self.sort_values("tc_gen_roc", ascending=False).iterrows():
        print(f"    {row['setting']:<5}: gen_roc(tc)={row.get('tc_gen_roc',np.nan):.4f}  "
              f"spearman={row.get('tc_spearman',np.nan):.4f}  "
              f"concordance={row.get('tc_concordance',np.nan):.4f}  "
              f"gen_delta_mean={row.get('tc_gen_delta_mean',np.nan):.2f}")

    return df


def make_unified_plots(all_results):
    """Create unified comparison plots across all combos."""
    combined = pd.concat(all_results.values(), ignore_index=True)

    # Focus on epoch 2, self-TC, key settings (s1-s3, s7)
    key_settings = ["base", "s1", "s2", "s3", "s4", "s7"]
    ep2 = combined[(combined["epoch"].isin([2, -1])) & (combined["tc_eval"] == "self")]
    ep2 = ep2[ep2["setting"].isin(key_settings)]

    if ep2.empty:
        print("  No data for unified plots!")
        return

    # Plot: gen_roc(tc) grouped by dataset×model pair, bars = settings
    fig, ax = plt.subplots(figsize=(12, 7))
    combos = ["gemma_membership", "gemma_ifeval", "qwen_membership", "qwen_ifeval"]
    combo_labels = ["Gemma\nMembership", "Gemma\nIFEval", "Qwen\nMembership", "Qwen\nIFEval"]
    n_settings = len(key_settings)
    x = np.arange(len(combos))
    width = 0.8 / n_settings

    for i, s in enumerate(key_settings):
        vals = []
        for combo in combos:
            subset = ep2[ep2["combo"] == combo]
            row = subset[subset["setting"] == s]
            if not row.empty:
                vals.append(row.iloc[0].get("tc_gen_roc", np.nan))
            else:
                vals.append(np.nan)
        offset = (i - (n_settings - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=s, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels)
    ax.set_ylabel("Generator ROC AUC (TC)")
    ax.set_title("Generator ROC AUC across settings and model/task combos\n(epoch 2, self-TC scoring, train set)")
    ax.legend(title="Setting")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "unified_genroc_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/unified_genroc_comparison.png")

    # Plot: improvement over base, grouped by dataset×model pair, bars = settings
    fig, ax = plt.subplots(figsize=(12, 7))
    settings_no_base = [s for s in key_settings if s != "base"]
    n_snb = len(settings_no_base)
    width2 = 0.8 / n_snb

    # Precompute base values per combo
    base_vals = {}
    for combo in combos:
        subset = ep2[ep2["combo"] == combo]
        base_row = subset[subset["setting"] == "base"]
        if not base_row.empty:
            base_vals[combo] = base_row.iloc[0].get("tc_gen_roc", np.nan)
        else:
            base_vals[combo] = np.nan

    for i, s in enumerate(settings_no_base):
        improvements = []
        for combo in combos:
            subset = ep2[ep2["combo"] == combo]
            row = subset[subset["setting"] == s]
            bv = base_vals.get(combo, np.nan)
            if not row.empty and not np.isnan(bv):
                val = row.iloc[0].get("tc_gen_roc", np.nan)
                improvements.append(val - bv if not np.isnan(val) else np.nan)
            else:
                improvements.append(np.nan)
        offset = (i - (n_snb - 1) / 2) * width2
        ax.bar(x + offset, improvements, width2, label=s, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels)
    ax.set_ylabel("Δ Gen ROC AUC over Base")
    ax.set_title("Improvement over base model (gen_roc TC)\n(epoch 2, self-TC scoring, train set)")
    ax.legend(title="Setting")
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "unified_improvement_over_base.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/unified_improvement_over_base.png")

    # Plot: Concordance comparison, grouped by dataset×model pair, bars = settings
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, s in enumerate(key_settings):
        vals = []
        for combo in combos:
            subset = ep2[ep2["combo"] == combo]
            row = subset[subset["setting"] == s]
            if not row.empty:
                vals.append(row.iloc[0].get("tc_concordance", np.nan))
            else:
                vals.append(np.nan)
        offset = (i - (n_settings - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=s, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels)
    ax.set_ylabel("Concordance (gen-val pair agreement)")
    ax.set_title("Generator-Validator Concordance\n(epoch 2, self-TC scoring, train set)")
    ax.legend(title="Setting")
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "unified_concordance_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: analysis/plots/unified_concordance_comparison.png")


def main():
    print("=" * 70)
    print("  COMPREHENSIVE ANALYSIS: ALL MODEL/TASK COMBINATIONS")
    print("=" * 70)

    all_results = {}
    for combo_name in COMBO_CONFIG:
        result = analyze_combo(combo_name)
        if result is not None:
            all_results[combo_name] = result

    # Unified plots
    print(f"\n{'='*70}")
    print("  UNIFIED COMPARISON PLOTS")
    print(f"{'='*70}")
    make_unified_plots(all_results)

    # Save combined CSV
    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined.to_csv(TABLES_DIR / "all_combos_metrics.csv", index=False)
        print(f"\n  Saved: analysis/tables/all_combos_metrics.csv")

    print("\n\nDone!")


if __name__ == "__main__":
    main()
