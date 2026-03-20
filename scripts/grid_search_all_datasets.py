#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified grid search over (alpha, c) in gen - alpha*typ + c*len
for all three datasets (IFEval, PlausibleQA, Hypernym) and both
typicality sources (GPT-2, Self).

Produces side-by-side heatmaps:
  Row 1: [IFEval Corr] [PlausibleQA Corr] [Hypernym Corr]
  Row 2: [IFEval ROC]  [PlausibleQA ROC]  [Hypernym ROC]

One figure per typicality source.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import glob
import re
import json

HYPERNYM_TASKS = ['bananas', 'bazookas', 'cabinets', 'cars', 'chairs', 'crows', 'diapers', 'dogs']

# =========================================================================
# Data loaders
# =========================================================================

def load_plausibleqa(outputs_dir, self_typicality=False):
    prefix = 'self-' if self_typicality else ''
    pattern = str(Path(outputs_dir) / f'scores_{prefix}v6-google_gemma-2-2b_plausibleqa-*_test_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))
    seen = {}
    for fp in files:
        if 'plausibleqa_v0_old' in fp:
            continue
        m = re.search(r'plausibleqa-(\w+_\d+)_test', fp)
        if m:
            seen[m.group(1)] = fp
    all_rows = []
    for prompt_id, fp in sorted(seen.items()):
        df = pd.read_csv(fp)
        df['prompt_id'] = prompt_id
        df['typicality'] = df['gen_score'] - df['gen_score_typcorr']
        if df['gpt4_ground_truth'].dtype == object:
            df['label'] = df['gpt4_ground_truth'].str.strip().str.lower().map(
                {'yes': 1, 'no': 0}).fillna(0).astype(int)
        else:
            df['label'] = df['gpt4_ground_truth'].astype(int)
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    print(f"PlausibleQA ({('self' if self_typicality else 'GPT-2')}): {len(df)} rows, {df['prompt_id'].nunique()} prompts")
    return df


def load_hypernym(outputs_dir, self_typicality=False):
    prefix = 'self-' if self_typicality else ''
    pattern = str(Path(outputs_dir) / f'scores_{prefix}v6-google_gemma-2-2b_hypernym-*_test_v2_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))
    seen = {}
    for fp in files:
        m = re.search(r'hypernym-([a-zA-Z]+)_test', fp)
        if m:
            noun = m.group(1)
            if noun in HYPERNYM_TASKS:
                seen[noun] = fp
    all_rows = []
    for noun, fp in sorted(seen.items()):
        df = pd.read_csv(fp)
        df['prompt_id'] = noun
        df['typicality'] = df['gen_score'] - df['gen_score_typcorr']
        if df['gpt4_ground_truth'].dtype == object:
            df['label'] = df['gpt4_ground_truth'].str.strip().str.lower().map(
                {'yes': 1, 'no': 0}).fillna(0).astype(int)
        else:
            df['label'] = df['gpt4_ground_truth'].astype(int)
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    print(f"Hypernym ({('self' if self_typicality else 'GPT-2')}): {len(df)} rows, {df['prompt_id'].nunique()} tasks")
    return df


def load_ifeval(outputs_dir, self_typicality=False):
    prefix = 'self-' if self_typicality else ''
    pattern = str(Path(outputs_dir) / f'scores_{prefix}gemma-2-2b_ifeval-prompt_*_test_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))
    seen = {}
    for fp in files:
        m = re.search(r'ifeval-prompt_(\d+)_test', fp)
        if m:
            prompt_num = int(m.group(1))
            seen[f'prompt_{prompt_num}'] = fp
    all_rows = []
    for prompt_id, fp in sorted(seen.items(), key=lambda x: int(x[0].split('_')[1])):
        df = pd.read_csv(fp)
        df['prompt_id'] = prompt_id
        df['typicality'] = df['gen_score'] - df['gen_score_typcorr']
        if df['correct'].dtype == object:
            df['label'] = df['correct'].str.strip().str.lower().map(
                {'yes': 1, 'no': 0}).fillna(0).astype(int)
        else:
            df['label'] = df['correct'].astype(int)
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    print(f"IFEval ({('self' if self_typicality else 'GPT-2')}): {len(df)} rows, {df['prompt_id'].nunique()} prompts")
    return df


# =========================================================================
# Grid search core
# =========================================================================

def compute_metrics(corrected, val, labels):
    """Compute Pearson correlation and GenROC."""
    labels = np.asarray(labels)
    corrected = np.asarray(corrected)
    val = np.asarray(val)
    mask = ~(np.isnan(corrected) | np.isnan(val) | np.isinf(corrected))
    if mask.sum() < 3:
        return np.nan, np.nan
    corr = pearsonr(corrected[mask], val[mask])[0]
    try:
        gen_roc = roc_auc_score(labels[mask], corrected[mask])
    except ValueError:
        gen_roc = np.nan
    return corr, gen_roc


def grid_search(df, alphas, cs):
    """
    LOO grid search: for each prompt, evaluate on all other prompts.
    Returns results array of shape (len(alphas), len(cs), 2) where
    last dim = [mean_corr, mean_genroc].
    """
    prompts = sorted(df['prompt_id'].unique())
    # Try numeric sorting if applicable
    try:
        prompts = sorted(prompts, key=lambda x: int(x.split('_')[1]))
    except (IndexError, ValueError):
        prompts = sorted(prompts)

    n_prompts = len(prompts)
    results = np.zeros((len(alphas), len(cs), 2))

    for pid in prompts:
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        lab = sub['label'].values

        for ai, alpha in enumerate(alphas):
            for ci, c in enumerate(cs):
                corrected = gen - alpha * typ + c * leng
                corr, roc = compute_metrics(corrected, val, lab)
                if not np.isnan(corr):
                    results[ai, ci, 0] += corr / n_prompts
                if not np.isnan(roc):
                    results[ai, ci, 1] += roc / n_prompts

    return results


def find_best(results, alphas, cs):
    """Find best (alpha, c) for each metric."""
    best_corr_idx = np.unravel_index(np.nanargmax(results[:, :, 0]), results[:, :, 0].shape)
    best_roc_idx = np.unravel_index(np.nanargmax(results[:, :, 1]), results[:, :, 1].shape)
    return {
        'best_alpha_corr': alphas[best_corr_idx[0]],
        'best_c_corr': cs[best_corr_idx[1]],
        'best_corr': results[best_corr_idx[0], best_corr_idx[1], 0],
        'best_roc_at_corr': results[best_corr_idx[0], best_corr_idx[1], 1],
        'best_alpha_roc': alphas[best_roc_idx[0]],
        'best_c_roc': cs[best_roc_idx[1]],
        'best_roc': results[best_roc_idx[0], best_roc_idx[1], 1],
        'best_corr_at_roc': results[best_roc_idx[0], best_roc_idx[1], 0],
    }


# =========================================================================
# Plotting
# =========================================================================

def plot_comparison_heatmaps(all_results, alphas, cs, typ_source, outputs_dir):
    """
    Create 2×3 figure:
      Row 0: Correlation heatmaps for [IFEval, PlausibleQA, Hypernym]
      Row 1: GenROC heatmaps for [IFEval, PlausibleQA, Hypernym]
    """
    dataset_names = ['IFEval', 'PlausibleQA', 'Hypernym']
    metric_names = ['Correlation (%)', 'GenROC (%)']
    extent = [cs[0] - 0.25, cs[-1] + 0.25, alphas[-1] + 0.125, alphas[0] - 0.125]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, dname in enumerate(dataset_names):
        results = all_results[dname]
        if results is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                     fontsize=14, transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{dname}\n{metric_names[row]}')
            continue

        best = find_best(results, alphas, cs)

        for row, (metric_idx, metric_name) in enumerate([(0, 'Correlation (%)'), (1, 'GenROC (%)')]):
            ax = axes[row, col]
            data = results[:, :, metric_idx] * 100

            # Use consistent color ranges per metric across datasets
            im = ax.imshow(data, cmap='RdYlGn', aspect='auto', extent=extent)

            ax.set_xlabel('c (length coeff.)', fontsize=10)
            if col == 0:
                ax.set_ylabel('α (typicality coeff.)', fontsize=10)

            ax.set_title(f'{dname}\n{metric_name}', fontsize=12)

            # Mark best point
            if metric_idx == 0:
                ba, bc = best['best_alpha_corr'], best['best_c_corr']
                bv = best['best_corr'] * 100
            else:
                ba, bc = best['best_alpha_roc'], best['best_c_roc']
                bv = best['best_roc'] * 100
            ax.plot(bc, ba, 'r*', markersize=12, label=f'Best: {bv:.1f}%')

            # Mark PMI (alpha=1, c=0)
            ax.plot(0, 1.0, 'ko', markersize=6, label='PMI')

            ax.legend(fontsize=8, loc='upper right')
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    suffix = 'self' if typ_source == 'self' else 'gpt2'
    fig.suptitle(f'Grid Search: gen − α·typ + c·len  |  Typicality: {typ_source.upper()}\n'
                 f'(within-prompt evaluation, averaged across prompts)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = str(Path(outputs_dir) / f'grid_search_comparison_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_comparison_heatmaps_fine(all_results_fine, all_alphas_fine, all_cs_fine,
                                  typ_source, metric_name, outputs_dir):
    """
    Create 1×3 fine-grid figure for a single metric.
    """
    dataset_names = ['IFEval', 'PlausibleQA', 'Hypernym']
    metric_idx = 0 if 'orr' in metric_name else 1
    metric_label = 'Correlation (%)' if metric_idx == 0 else 'GenROC (%)'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, dname in enumerate(dataset_names):
        ax = axes[col]
        if dname not in all_results_fine or all_results_fine[dname] is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.set_title(f'{dname}')
            continue

        results = all_results_fine[dname]
        alphas = all_alphas_fine[dname]
        cs = all_cs_fine[dname]
        extent = [cs[0] - 0.05, cs[-1] + 0.05, alphas[-1] + 0.05, alphas[0] - 0.05]

        data = results[:, :, metric_idx] * 100
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', extent=extent)
        ax.set_xlabel('c (length coeff.)', fontsize=10)
        if col == 0:
            ax.set_ylabel('α (typicality coeff.)', fontsize=10)
        ax.set_title(f'{dname}', fontsize=12)

        best_idx = np.unravel_index(np.nanargmax(data), data.shape)
        ax.plot(cs[best_idx[1]], alphas[best_idx[0]], 'r*', markersize=12,
                label=f'Best: {data[best_idx]:.1f}%')
        ax.plot(0, 1.0, 'ko', markersize=6, label='PMI')
        ax.legend(fontsize=8, loc='upper right')
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    suffix = 'self' if typ_source == 'self' else 'gpt2'
    metric_suffix = 'corr' if metric_idx == 0 else 'roc'
    fig.suptitle(f'Fine Grid: {metric_label}  |  Typicality: {typ_source.upper()}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = str(Path(outputs_dir) / f'grid_search_fine_{metric_suffix}_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Grid search across all datasets')
    parser.add_argument('--outputs-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'outputs'))
    args = parser.parse_args()
    outputs_dir = args.outputs_dir

    # Coarse grid
    alphas = np.arange(0.0, 3.1, 0.25)
    cs = np.arange(-2.0, 4.1, 0.5)

    loaders = {
        'IFEval': load_ifeval,
        'PlausibleQA': load_plausibleqa,
        'Hypernym': load_hypernym,
    }

    summary_rows = []

    for typ_source_flag, typ_label in [(False, 'gpt2'), (True, 'self')]:
        print(f"\n{'#' * 80}")
        print(f"# TYPICALITY SOURCE: {typ_label.upper()}")
        print(f"{'#' * 80}")

        all_results = {}
        all_dfs = {}

        for dname, loader in loaders.items():
            try:
                df = loader(outputs_dir, self_typicality=typ_source_flag)
                if len(df) == 0:
                    print(f"  {dname}: No data found, skipping")
                    all_results[dname] = None
                    continue
                all_dfs[dname] = df
            except Exception as e:
                print(f"  {dname}: Failed to load: {e}")
                all_results[dname] = None
                continue

            print(f"  Running coarse grid search for {dname}...")
            results = grid_search(df, alphas, cs)
            all_results[dname] = results

            best = find_best(results, alphas, cs)
            print(f"  Best Corr: α={best['best_alpha_corr']:.2f}, c={best['best_c_corr']:.2f} → "
                  f"Corr={best['best_corr']*100:.1f}%, GenROC={best['best_roc_at_corr']*100:.1f}%")
            print(f"  Best ROC:  α={best['best_alpha_roc']:.2f}, c={best['best_c_roc']:.2f} → "
                  f"Corr={best['best_corr_at_roc']*100:.1f}%, GenROC={best['best_roc']*100:.1f}%")

            # PMI reference
            ai_1 = np.argmin(np.abs(alphas - 1.0))
            ci_0 = np.argmin(np.abs(cs - 0.0))
            pmi_corr = results[ai_1, ci_0, 0]
            pmi_roc = results[ai_1, ci_0, 1]
            print(f"  PMI (α=1,c=0): Corr={pmi_corr*100:.1f}%, GenROC={pmi_roc*100:.1f}%")

            # Raw (alpha=0, c=0)
            ai_0 = np.argmin(np.abs(alphas - 0.0))
            raw_corr = results[ai_0, ci_0, 0]
            raw_roc = results[ai_0, ci_0, 1]
            print(f"  Raw (α=0,c=0): Corr={raw_corr*100:.1f}%, GenROC={raw_roc*100:.1f}%")

            summary_rows.append({
                'dataset': dname,
                'typicality': typ_label,
                'n_prompts': df['prompt_id'].nunique(),
                'n_rows': len(df),
                'best_alpha_corr': best['best_alpha_corr'],
                'best_c_corr': best['best_c_corr'],
                'best_corr': best['best_corr'],
                'best_roc_at_corr': best['best_roc_at_corr'],
                'best_alpha_roc': best['best_alpha_roc'],
                'best_c_roc': best['best_c_roc'],
                'best_roc': best['best_roc'],
                'best_corr_at_roc': best['best_corr_at_roc'],
                'pmi_corr': pmi_corr,
                'pmi_roc': pmi_roc,
                'raw_corr': raw_corr,
                'raw_roc': raw_roc,
            })

        # Plot coarse comparison
        plot_comparison_heatmaps(all_results, alphas, cs, typ_label, outputs_dir)

        # Fine grid near optima (per dataset, per metric)
        for metric_name, metric_key in [('Corr', 'best_alpha_corr'), ('ROC', 'best_alpha_roc')]:
            all_results_fine = {}
            all_alphas_fine = {}
            all_cs_fine = {}

            for dname in loaders:
                if all_results.get(dname) is None or dname not in all_dfs:
                    all_results_fine[dname] = None
                    continue

                best = find_best(all_results[dname], alphas, cs)
                if metric_name == 'Corr':
                    center_a, center_c = best['best_alpha_corr'], best['best_c_corr']
                else:
                    center_a, center_c = best['best_alpha_roc'], best['best_c_roc']

                alphas_f = np.arange(max(0, center_a - 0.5), center_a + 0.55, 0.05)
                cs_f = np.arange(center_c - 1.0, center_c + 1.05, 0.1)

                results_f = grid_search(all_dfs[dname], alphas_f, cs_f)
                all_results_fine[dname] = results_f
                all_alphas_fine[dname] = alphas_f
                all_cs_fine[dname] = cs_f

                best_f = find_best(results_f, alphas_f, cs_f)
                if metric_name == 'Corr':
                    print(f"  Fine {dname} Best Corr: α={best_f['best_alpha_corr']:.2f}, "
                          f"c={best_f['best_c_corr']:.2f} → {best_f['best_corr']*100:.1f}%")
                else:
                    print(f"  Fine {dname} Best ROC:  α={best_f['best_alpha_roc']:.2f}, "
                          f"c={best_f['best_c_roc']:.2f} → {best_f['best_roc']*100:.1f}%")

            plot_comparison_heatmaps_fine(all_results_fine, all_alphas_fine, all_cs_fine,
                                          typ_label, metric_name, outputs_dir)

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = str(Path(outputs_dir) / 'grid_search_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # Print summary table
    print(f"\n{'=' * 100}")
    print("SUMMARY TABLE")
    print(f"{'=' * 100}")
    print(f"{'Dataset':<14} {'Typ':<6} {'N_pr':>5} {'N_row':>6} | "
          f"{'α_corr':>6} {'c_corr':>6} {'Corr':>6} {'ROC@C':>6} | "
          f"{'α_roc':>6} {'c_roc':>6} {'ROC':>6} {'C@ROC':>6} | "
          f"{'PMI_C':>6} {'PMI_R':>6} {'Raw_C':>6} {'Raw_R':>6}")
    print("-" * 120)
    for _, r in summary_df.iterrows():
        print(f"{r['dataset']:<14} {r['typicality']:<6} {r['n_prompts']:>5} {r['n_rows']:>6} | "
              f"{r['best_alpha_corr']:>6.2f} {r['best_c_corr']:>6.2f} "
              f"{r['best_corr']*100:>5.1f}% {r['best_roc_at_corr']*100:>5.1f}% | "
              f"{r['best_alpha_roc']:>6.2f} {r['best_c_roc']:>6.2f} "
              f"{r['best_roc']*100:>5.1f}% {r['best_corr_at_roc']*100:>5.1f}% | "
              f"{r['pmi_corr']*100:>5.1f}% {r['pmi_roc']*100:>5.1f}% "
              f"{r['raw_corr']*100:>5.1f}% {r['raw_roc']*100:>5.1f}%")


if __name__ == '__main__':
    main()
