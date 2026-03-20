#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive grid search over ALL correction formulas analyzed in the reports,
across all three datasets (IFEval, PlausibleQA, Hypernym) and both
typicality sources (GPT-2, Self).

Formulas with 2D hyperparameter sweeps:
  F1: gen - alpha*typ + c*len              (alpha, c)
  F2: (gen - alpha*typ) / len^a            (alpha, a)
  F3: gen - b1*typ - b2*typ^2              (b1, b2)
  F4: (gen - alpha*typ) / len^a + c*len    (alpha, c) with per-token norm (a fixed at 1)
      Actually: gen/len^a - b*typ/len^a     (a, b)

For each formula & typicality source, produces:
  Row 1: [IFEval Corr] [PlausibleQA Corr] [Hypernym Corr]
  Row 2: [IFEval ROC]  [PlausibleQA ROC]  [Hypernym ROC]
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

HYPERNYM_TASKS = ['bananas', 'bazookas', 'cabinets', 'cars', 'chairs', 'crows', 'diapers', 'dogs']
DATASET_NAMES = ['IFEval', 'PlausibleQA', 'Hypernym']

# =========================================================================
# Data loaders (unified column names: typicality, label)
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
    if not all_rows:
        return pd.DataFrame()
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
    if not all_rows:
        return pd.DataFrame()
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
    if not all_rows:
        return pd.DataFrame()
    df = pd.concat(all_rows, ignore_index=True)
    print(f"IFEval ({('self' if self_typicality else 'GPT-2')}): {len(df)} rows, {df['prompt_id'].nunique()} prompts")
    return df


# =========================================================================
# Metrics
# =========================================================================

def compute_metrics(corrected, val, labels):
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


# =========================================================================
# Formula definitions
# =========================================================================

def formula_alpha_typ_c_len(gen, typ, leng, params):
    """F1: gen - alpha*typ + c*len"""
    alpha, c = params
    return gen - alpha * typ + c * leng


def formula_alpha_typ_div_len_a(gen, typ, leng, params):
    """F2: (gen - alpha*typ) / len^a"""
    alpha, a = params
    safe_len = np.maximum(leng, 1.0)
    return (gen - alpha * typ) / safe_len**a


def formula_b1_typ_b2_typ2(gen, typ, leng, params):
    """F3: gen - b1*typ - b2*typ^2"""
    b1, b2 = params
    return gen - b1 * typ - b2 * typ**2


def formula_gen_len_a_minus_b_typ_len_a(gen, typ, leng, params):
    """F4: gen/len^a - b*typ/len^a  (per-token-like normalization)"""
    a, b = params
    safe_len = np.maximum(leng, 1.0)
    return gen / safe_len**a - b * typ / safe_len**a


def formula_alpha_typ_c_len_b2_typ2(gen, typ, leng, params):
    """F5: gen - alpha*typ + c*len - b2*typ^2  (linear + quadratic typ + length)"""
    alpha, c, b2 = params
    return gen - alpha * typ + c * leng - b2 * typ**2


FORMULAS = {
    'F1: gen − α·typ + c·len': {
        'func': formula_alpha_typ_c_len,
        'param_names': ('α', 'c'),
        'grids': {
            'coarse': (np.arange(0.0, 3.1, 0.25), np.arange(-2.0, 4.1, 0.5)),
        },
        'ref_points': [
            ((0.0, 0.0), 'Raw', 'bs'),
            ((1.0, 0.0), 'PMI', 'ko'),
        ],
    },
    'F2: (gen − α·typ) / len^a': {
        'func': formula_alpha_typ_div_len_a,
        'param_names': ('α', 'a'),
        'grids': {
            'coarse': (np.arange(0.0, 3.1, 0.25), np.arange(0.0, 2.05, 0.15)),
        },
        'ref_points': [
            ((0.0, 0.0), 'Raw', 'bs'),
            ((1.0, 0.0), 'PMI', 'ko'),
            ((1.0, 0.5), 'PMI/√len', 'md'),
            ((1.0, 1.0), 'PMI/len', 'c^'),
        ],
    },
    'F3: gen − b₁·typ − b₂·typ²': {
        'func': formula_b1_typ_b2_typ2,
        'param_names': ('b₁', 'b₂'),
        'grids': {
            'coarse': (np.arange(-1.0, 3.1, 0.25), np.arange(-0.01, 0.011, 0.001)),
        },
        'ref_points': [
            ((0.0, 0.0), 'Raw', 'bs'),
            ((1.0, 0.0), 'PMI', 'ko'),
        ],
    },
    'F4: gen/len^a − b·typ/len^a': {
        'func': formula_gen_len_a_minus_b_typ_len_a,
        'param_names': ('a', 'b'),
        'grids': {
            'coarse': (np.arange(0.0, 2.05, 0.15), np.arange(0.0, 3.1, 0.25)),
        },
        'ref_points': [
            ((0.0, 0.0), 'Raw', 'bs'),
            ((0.0, 1.0), 'PMI', 'ko'),
            ((0.5, 1.0), 'PMI/√len', 'md'),
            ((1.0, 1.0), 'PMI/len', 'c^'),
        ],
    },
}


# =========================================================================
# Grid search
# =========================================================================

def grid_search_formula(df, formula_func, param1_vals, param2_vals):
    """
    Within-prompt grid search for a 2D formula.
    Returns results array (len(p1), len(p2), 2) where last dim = [corr, roc].
    """
    prompts = sorted(df['prompt_id'].unique())
    try:
        prompts = sorted(prompts, key=lambda x: int(x.split('_')[1]))
    except (IndexError, ValueError):
        prompts = sorted(prompts)

    n_prompts = len(prompts)
    results = np.zeros((len(param1_vals), len(param2_vals), 2))

    for pid in prompts:
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        lab = sub['label'].values

        for i, p1 in enumerate(param1_vals):
            for j, p2 in enumerate(param2_vals):
                corrected = formula_func(gen, typ, leng, (p1, p2))
                corr, roc = compute_metrics(corrected, val, lab)
                if not np.isnan(corr):
                    results[i, j, 0] += corr / n_prompts
                if not np.isnan(roc):
                    results[i, j, 1] += roc / n_prompts

    return results


# =========================================================================
# Plotting
# =========================================================================

def plot_formula_comparison(all_results, p1_vals, p2_vals, formula_info,
                            typ_source, outputs_dir, formula_key):
    """
    Create 2x3 figure for one formula:
      Row 0: Correlation for [IFEval, PlausibleQA, Hypernym]
      Row 1: GenROC for [IFEval, PlausibleQA, Hypernym]
    """
    pname1, pname2 = formula_info['param_names']
    metric_names = ['Correlation (%)', 'GenROC (%)']
    extent = [p2_vals[0], p2_vals[-1], p1_vals[-1], p1_vals[0]]

    # Add small padding for extent
    dp2 = (p2_vals[-1] - p2_vals[0]) / (len(p2_vals) - 1) / 2 if len(p2_vals) > 1 else 0.5
    dp1 = (p1_vals[-1] - p1_vals[0]) / (len(p1_vals) - 1) / 2 if len(p1_vals) > 1 else 0.5
    extent = [p2_vals[0] - dp2, p2_vals[-1] + dp2,
              p1_vals[-1] + dp1, p1_vals[0] - dp1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, dname in enumerate(DATASET_NAMES):
        results = all_results.get(dname)
        if results is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                     fontsize=14, transform=axes[row, col].transAxes)
                axes[row, col].set_title(f'{dname}\n{metric_names[row]}')
            continue

        for row, (metric_idx, metric_name) in enumerate([(0, 'Correlation (%)'),
                                                           (1, 'GenROC (%)')]):
            ax = axes[row, col]
            data = results[:, :, metric_idx] * 100

            # Fixed color ranges for cross-plot comparability
            if metric_idx == 0:  # Correlation
                vmin, vmax = -40, 100
            else:  # GenROC
                vmin, vmax = 0, 100
            im = ax.imshow(data, cmap='RdYlGn', aspect='auto', extent=extent,
                           vmin=vmin, vmax=vmax)
            ax.set_xlabel(f'{pname2}', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{pname1}', fontsize=10)
            ax.set_title(f'{dname}\n{metric_name}', fontsize=12)

            # Mark best
            best_idx = np.unravel_index(np.nanargmax(data), data.shape)
            best_p1 = p1_vals[best_idx[0]]
            best_p2 = p2_vals[best_idx[1]]
            best_val = data[best_idx]
            ax.plot(best_p2, best_p1, 'r*', markersize=12,
                    label=f'Best: {best_val:.1f}%')

            # Mark reference points
            for ref_pt, ref_lbl, ref_fmt in formula_info.get('ref_points', []):
                ref_p1, ref_p2 = ref_pt
                # Only plot if within grid range
                if (p1_vals[0] <= ref_p1 <= p1_vals[-1] and
                    p2_vals[0] <= ref_p2 <= p2_vals[-1]):
                    ref_i = np.argmin(np.abs(p1_vals - ref_p1))
                    ref_j = np.argmin(np.abs(p2_vals - ref_p2))
                    ref_val = data[ref_i, ref_j]
                    ax.plot(ref_p2, ref_p1, ref_fmt, markersize=7,
                            label=f'{ref_lbl}: {ref_val:.1f}%')

            ax.legend(fontsize=8, loc='upper right')
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

    suffix = 'self' if typ_source == 'self' else 'gpt2'
    safe_key = formula_key.split(':')[0].strip()
    fig.suptitle(f'{formula_key}  |  Typicality: {typ_source.upper()}\n'
                 f'(within-prompt evaluation, averaged across prompts)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = str(Path(outputs_dir) / f'grid_search_{safe_key}_{suffix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


# =========================================================================
# Main
# =========================================================================

def cache_path(outputs_dir, formula_key, typ_label, dname):
    """Path for cached grid search results."""
    safe_key = formula_key.split(':')[0].strip()
    return str(Path(outputs_dir) / f'grid_cache_{safe_key}_{typ_label}_{dname}.npz')


def save_cache(path, results, p1_vals, p2_vals, n_prompts):
    """Save grid search results to disk."""
    np.savez(path, results=results, p1_vals=p1_vals, p2_vals=p2_vals,
             n_prompts=np.array([n_prompts]))
    print(f"    Cached: {path}")


def load_cache(path, p1_vals, p2_vals):
    """Load cached grid search results. Returns results array or None if cache invalid."""
    try:
        data = np.load(path)
        cached_p1 = data['p1_vals']
        cached_p2 = data['p2_vals']
        # Verify grid matches
        if (len(cached_p1) == len(p1_vals) and len(cached_p2) == len(p2_vals) and
            np.allclose(cached_p1, p1_vals) and np.allclose(cached_p2, p2_vals)):
            n_prompts = int(data['n_prompts'][0])
            print(f"    Loaded from cache ({n_prompts} prompts)")
            return data['results'], n_prompts
    except Exception:
        pass
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Grid search all formulas across all datasets')
    parser.add_argument('--outputs-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'outputs'))
    parser.add_argument('--replot', action='store_true',
                        help='Skip computation, just re-plot from cached results')
    args = parser.parse_args()
    outputs_dir = args.outputs_dir

    loaders = {
        'IFEval': load_ifeval,
        'PlausibleQA': load_plausibleqa,
        'Hypernym': load_hypernym,
    }

    all_summary = []

    for typ_flag, typ_label in [(False, 'gpt2'), (True, 'self')]:
        print(f"\n{'#' * 80}")
        print(f"# TYPICALITY SOURCE: {typ_label.upper()}")
        print(f"{'#' * 80}")

        # Load all datasets (skip if --replot)
        datasets = {}
        if not args.replot:
            for dname, loader in loaders.items():
                try:
                    df = loader(outputs_dir, self_typicality=typ_flag)
                    if len(df) > 0:
                        datasets[dname] = df
                    else:
                        print(f"  {dname}: No data")
                except Exception as e:
                    print(f"  {dname}: Failed: {e}")

        # Run each formula
        for formula_key, formula_info in FORMULAS.items():
            print(f"\n--- {formula_key} ---")
            func = formula_info['func']
            p1_vals, p2_vals = formula_info['grids']['coarse']

            all_results = {}
            for dname in DATASET_NAMES:
                cp = cache_path(outputs_dir, formula_key, typ_label, dname)

                if args.replot:
                    # Load from cache only
                    cached, n_prompts = load_cache(cp, p1_vals, p2_vals)
                    if cached is not None:
                        all_results[dname] = cached
                        results = cached
                    else:
                        print(f"  {dname}: No cache found, skipping")
                        all_results[dname] = None
                        continue
                else:
                    if dname not in datasets:
                        all_results[dname] = None
                        continue

                    df = datasets[dname]
                    n_prompts = df['prompt_id'].nunique()
                    results = grid_search_formula(df, func, p1_vals, p2_vals)
                    all_results[dname] = results

                    # Save cache
                    save_cache(cp, results, p1_vals, p2_vals, n_prompts)

                # Find bests
                best_corr_idx = np.unravel_index(np.nanargmax(results[:, :, 0]),
                                                  results[:, :, 0].shape)
                best_roc_idx = np.unravel_index(np.nanargmax(results[:, :, 1]),
                                                 results[:, :, 1].shape)

                pn1, pn2 = formula_info['param_names']
                print(f"  {dname}: Best Corr {pn1}={p1_vals[best_corr_idx[0]]:.3f}, "
                      f"{pn2}={p2_vals[best_corr_idx[1]]:.3f} → "
                      f"Corr={results[best_corr_idx[0], best_corr_idx[1], 0]*100:.1f}%, "
                      f"ROC={results[best_corr_idx[0], best_corr_idx[1], 1]*100:.1f}%")
                print(f"  {dname}: Best ROC  {pn1}={p1_vals[best_roc_idx[0]]:.3f}, "
                      f"{pn2}={p2_vals[best_roc_idx[1]]:.3f} → "
                      f"Corr={results[best_roc_idx[0], best_roc_idx[1], 0]*100:.1f}%, "
                      f"ROC={results[best_roc_idx[0], best_roc_idx[1], 1]*100:.1f}%")

                all_summary.append({
                    'formula': formula_key,
                    'typicality': typ_label,
                    'dataset': dname,
                    'n_prompts': n_prompts,
                    'best_p1_corr': p1_vals[best_corr_idx[0]],
                    'best_p2_corr': p2_vals[best_corr_idx[1]],
                    'best_corr': results[best_corr_idx[0], best_corr_idx[1], 0],
                    'best_roc_at_corr': results[best_corr_idx[0], best_corr_idx[1], 1],
                    'best_p1_roc': p1_vals[best_roc_idx[0]],
                    'best_p2_roc': p2_vals[best_roc_idx[1]],
                    'best_roc': results[best_roc_idx[0], best_roc_idx[1], 1],
                    'best_corr_at_roc': results[best_roc_idx[0], best_roc_idx[1], 0],
                })

            plot_formula_comparison(all_results, p1_vals, p2_vals, formula_info,
                                     typ_label, outputs_dir, formula_key)

    # Save summary
    summary_df = pd.DataFrame(all_summary)
    summary_path = str(Path(outputs_dir) / 'grid_search_all_formulas_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    # Print summary
    print(f"\n{'=' * 120}")
    print("SUMMARY TABLE")
    print(f"{'=' * 120}")
    for formula_key in FORMULAS:
        pn1, pn2 = FORMULAS[formula_key]['param_names']
        print(f"\n{formula_key}")
        print(f"{'Dataset':<14} {'Typ':<6} | "
              f"{pn1+'_c':>8} {pn2+'_c':>8} {'Corr':>6} {'ROC@C':>6} | "
              f"{pn1+'_r':>8} {pn2+'_r':>8} {'ROC':>6} {'C@ROC':>6}")
        print("-" * 90)
        for row in all_summary:
            if row['formula'] != formula_key:
                continue
            print(f"{row['dataset']:<14} {row['typicality']:<6} | "
                  f"{row['best_p1_corr']:>8.3f} {row['best_p2_corr']:>8.3f} "
                  f"{row['best_corr']*100:>5.1f}% {row['best_roc_at_corr']*100:>5.1f}% | "
                  f"{row['best_p1_roc']:>8.3f} {row['best_p2_roc']:>8.3f} "
                  f"{row['best_roc']*100:>5.1f}% {row['best_corr_at_roc']*100:>5.1f}%")


if __name__ == '__main__':
    main()
