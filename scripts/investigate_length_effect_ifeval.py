#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate the length effect in typicality correction for IFEVAL tasks.

Adapted from investigate_length_effect_hypernym.py for ifeval data structure:
  - ~40 ifeval prompts (prompt_1 through prompt_40)
  - CSV pattern: scores_{prefix}gemma-2-2b_ifeval-prompt_N_test_log-odds_evaltc_*.csv
  - Note: ifeval uses 'gemma-2-2b' (not 'v6-google_gemma-2-2b')

Note: IFEval completions are very long (500-1000 tokens), so length effects
may behave differently than in hypernym (1-5 tokens) or plausibleqa (1-30 tokens).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import argparse
import sys
import glob
import re

sys.path.insert(0, str(Path(__file__).parent))


def load_all_ifeval(outputs_dir, self_typicality=False):
    """Load all ifeval CSVs into a single DataFrame with prompt_id column."""
    prefix = 'self-' if self_typicality else ''
    pattern = str(Path(outputs_dir) / f'scores_{prefix}gemma-2-2b_ifeval-prompt_*_test_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))

    seen = {}
    for fp in files:
        m = re.search(r'ifeval-prompt_(\d+)_test', fp)
        if m:
            prompt_num = int(m.group(1))
            prompt_id = f'prompt_{prompt_num}'
            seen[prompt_id] = fp

    all_rows = []
    for prompt_id, fp in sorted(seen.items(), key=lambda x: int(x[0].split('_')[1])):
        df = pd.read_csv(fp)
        df['prompt_id'] = prompt_id
        df['typicality_gpt2'] = df['gen_score'] - df['gen_score_typcorr']
        # Normalize labels — 'correct' column with Yes/No
        if df['correct'].dtype == object:
            df['correct'] = df['correct'].str.strip().str.lower().map(
                {'yes': 1, 'no': 0}).fillna(0).astype(int)
        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)
    print(f"Loaded {len(df)} rows across {df['prompt_id'].nunique()} ifeval prompts")
    return df


def compute_metrics(corrected, val, labels):
    """Compute correlation and GenROC."""
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
# Analysis 1: Correlation structure
# =========================================================================

def analyze_correlations(df, outputs_dir, suffix=''):
    """Pairwise correlations between gen, typ, len, val --- pooled and within-prompt."""

    n_total = len(df)
    n_prompts = df['prompt_id'].nunique()
    print(f"\n{'=' * 80}")
    print("ANALYSIS 1: CORRELATION STRUCTURE")
    print(f"{'=' * 80}")

    gen = df['gen_score'].values
    typ = df['typicality_gpt2'].values
    leng = df['num_tokens'].values.astype(float)
    val = df['val_score'].values

    vars_dict = {'gen': gen, 'typ': typ, 'len': leng, 'val': val}
    names = list(vars_dict.keys())
    n = len(names)

    print(f"\n--- Pooled Pearson correlations (N={n_total}) ---")
    print(f"{'':>6}", end='')
    for name in names:
        print(f"{name:>8}", end='')
    print()
    pooled_corr = np.zeros((n, n))
    for i in range(n):
        print(f"{names[i]:>6}", end='')
        for j in range(n):
            r = pearsonr(vars_dict[names[i]], vars_dict[names[j]])[0]
            pooled_corr[i, j] = r
            print(f"{r:>8.3f}", end='')
        print()

    print(f"\n--- Within-prompt Pearson correlations (averaged across {n_prompts} prompts) ---")
    prompt_corrs = {f'{a}-{b}': [] for a in names for b in names}
    var_cols = {'gen': 'gen_score', 'typ': 'typicality_gpt2', 'len': 'num_tokens', 'val': 'val_score'}
    for pid in df['prompt_id'].unique():
        sub = df[df['prompt_id'] == pid]
        for a in names:
            for b in names:
                va = sub[var_cols[a]].values.astype(float)
                vb = sub[var_cols[b]].values.astype(float)
                if len(va) >= 3 and np.std(va) > 1e-10 and np.std(vb) > 1e-10:
                    r = pearsonr(va, vb)[0]
                    prompt_corrs[f'{a}-{b}'].append(r)

    print(f"{'':>6}", end='')
    for name in names:
        print(f"{name:>8}", end='')
    print("  (n_prompts)")
    within_corr = np.zeros((n, n))
    for i in range(n):
        print(f"{names[i]:>6}", end='')
        for j in range(n):
            vals = prompt_corrs[f'{names[i]}-{names[j]}']
            mean_r = np.nanmean(vals) if len(vals) > 0 else float('nan')
            within_corr[i, j] = mean_r
            ntasks = len(vals)
            print(f"{mean_r:>8.3f}", end='')
        print(f"  ({ntasks})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in [(axes[0], pooled_corr, f'Pooled (N={n_total})'),
                            (axes[1], within_corr, f'Within-prompt (mean of {n_prompts})')]:
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, fontsize=12)
        ax.set_yticklabels(names, fontsize=12)
        ax.set_title(title, fontsize=14)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                       fontsize=11, color='white' if abs(mat[i,j]) > 0.5 else 'black')
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle('Correlation Structure: gen, typ, len, val (IFEval)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(str(Path(outputs_dir) / f'length_correlation_matrix_ifeval{suffix}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: length_correlation_matrix_ifeval{suffix}.png")

    return pooled_corr, within_corr, prompt_corrs


# =========================================================================
# Analysis 2: Redundancy test
# =========================================================================

def analyze_redundancy(df, outputs_dir, suffix=''):
    """Test: does gen - 1.5*typ correlate with gen - 1.0*typ + c*len?"""

    print(f"\n{'=' * 80}")
    print("ANALYSIS 2: REDUNDANCY TEST")
    print(f"{'=' * 80}")

    gen = df['gen_score'].values
    typ = df['typicality_gpt2'].values
    leng = df['num_tokens'].values.astype(float)

    pmi_15 = gen - 1.5 * typ
    pmi_10 = gen - 1.0 * typ
    diff = pmi_15 - pmi_10
    c_opt = np.cov(diff, leng)[0, 1] / np.var(leng) if np.var(leng) > 0 else 0
    print(f"Optimal c (pooled): {c_opt:.4f}")

    approx = pmi_10 + c_opt * leng
    r_approx = pearsonr(pmi_15, approx)[0]
    print(f"  r(gen - 1.5*typ, gen - 1.0*typ + c*len) = {r_approx:.6f}")

    r_typ_len = pearsonr(typ, leng)[0] if np.std(leng) > 1e-10 else 0
    print(f"  r(typ, len) = {r_typ_len:.4f}, R^2 = {r_typ_len**2:.4f}")

    within_r_typ_len = []
    for pid in df['prompt_id'].unique():
        sub = df[df['prompt_id'] == pid]
        t = sub['typicality_gpt2'].values
        l = sub['num_tokens'].values.astype(float)
        if len(t) >= 3 and np.std(l) > 1e-10 and np.std(t) > 1e-10:
            within_r_typ_len.append(pearsonr(t, l)[0])

    r2_within = np.nanmean([r**2 for r in within_r_typ_len]) if within_r_typ_len else float('nan')
    print(f"  Within-prompt r(typ, len): mean={np.mean(within_r_typ_len):.4f}")
    print(f"  Within-prompt R^2: mean={r2_within:.4f}")

    if r2_within < 0.5:
        print(f"  CONCLUSION: Length is NOT merely a proxy for typ (R^2={r2_within:.3f})")
    else:
        print(f"  CONCLUSION: Length is substantially redundant with typ (R^2={r2_within:.3f})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(leng, typ, alpha=0.3, s=20, c='steelblue')
    if np.std(leng) > 0:
        z = np.polyfit(leng, typ, 1)
        x_line = np.linspace(leng.min(), leng.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2, label=f'r={r_typ_len:.3f}')
    ax.set_xlabel('Completion length (tokens)')
    ax.set_ylabel('Typicality (log P(y))')
    ax.set_title('Pooled: typ vs len (IFEval)')
    ax.legend()

    ax = axes[1]
    if within_r_typ_len:
        ax.hist(within_r_typ_len, bins=min(15, len(within_r_typ_len)),
                edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(within_r_typ_len), color='red', linestyle='--', linewidth=2,
                   label=f'mean={np.mean(within_r_typ_len):.3f}')
    ax.set_xlabel('Within-prompt r(typ, len)')
    ax.set_ylabel('Count (prompts)')
    ax.set_title('Distribution of within-prompt r(typ, len)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(Path(outputs_dir) / f'length_redundancy_analysis_ifeval{suffix}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: length_redundancy_analysis_ifeval{suffix}.png")

    return {
        'r_typ_len_pooled': r_typ_len,
        'r_typ_len_within_mean': np.mean(within_r_typ_len) if within_r_typ_len else 0,
        'r2_within_mean': r2_within,
        'c_opt_pooled': c_opt,
    }


# =========================================================================
# Analysis 3: Grid search over (alpha, c)
# =========================================================================

def grid_search_alpha_c(df, outputs_dir, suffix=''):
    """Grid search over alpha and c in gen - alpha*typ + c*len."""

    print(f"\n{'=' * 80}")
    print("ANALYSIS 3: GRID SEARCH over (alpha, c) in gen - alpha*typ + c*len")
    print(f"{'=' * 80}")

    alphas = np.arange(0.0, 3.1, 0.25)
    cs = np.arange(-2.0, 4.1, 0.5)
    prompts = sorted(df['prompt_id'].unique(), key=lambda x: int(x.split('_')[1]))

    results = np.zeros((len(alphas), len(cs), 2))
    for pi, pid in enumerate(prompts):
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        lab = sub['correct'].values

        for ai, alpha in enumerate(alphas):
            for ci, c in enumerate(cs):
                corrected = gen - alpha * typ + c * leng
                corr, roc = compute_metrics(corrected, val, lab)
                results[ai, ci, 0] += corr / len(prompts)
                results[ai, ci, 1] += roc / len(prompts)

    best_corr_idx = np.unravel_index(np.nanargmax(results[:, :, 0]), results[:, :, 0].shape)
    best_roc_idx = np.unravel_index(np.nanargmax(results[:, :, 1]), results[:, :, 1].shape)

    best_alpha_corr = alphas[best_corr_idx[0]]
    best_c_corr = cs[best_corr_idx[1]]
    best_corr = results[best_corr_idx[0], best_corr_idx[1], 0]
    best_roc_at_corr = results[best_corr_idx[0], best_corr_idx[1], 1]

    best_alpha_roc = alphas[best_roc_idx[0]]
    best_c_roc = cs[best_roc_idx[1]]
    best_roc = results[best_roc_idx[0], best_roc_idx[1], 1]
    best_corr_at_roc = results[best_roc_idx[0], best_roc_idx[1], 0]

    print(f"\nBest for Correlation:")
    print(f"  alpha={best_alpha_corr:.2f}, c={best_c_corr:.2f}")
    print(f"  Corr={best_corr*100:.1f}%, GenROC={best_roc_at_corr*100:.1f}%")

    print(f"\nBest for GenROC:")
    print(f"  alpha={best_alpha_roc:.2f}, c={best_c_roc:.2f}")
    print(f"  Corr={best_corr_at_roc*100:.1f}%, GenROC={best_roc*100:.1f}%")

    # Reference: PMI (alpha=1, c=0)
    ai_1 = np.argmin(np.abs(alphas - 1.0))
    ci_0 = np.argmin(np.abs(cs - 0.0))
    pmi_corr = results[ai_1, ci_0, 0]
    pmi_roc = results[ai_1, ci_0, 1]
    print(f"\nReference PMI (alpha=1.0, c=0):")
    print(f"  Corr={pmi_corr*100:.1f}%, GenROC={pmi_roc*100:.1f}%")

    # Heatmap table
    header = 'alpha\\c'
    print(f"\n--- Corr heatmap ---")
    print(f"{header:>8}", end='')
    for c in cs:
        print(f"{c:>7.1f}", end='')
    print()
    for ai, alpha in enumerate(alphas):
        print(f"{alpha:>8.2f}", end='')
        for ci, c in enumerate(cs):
            print(f"{results[ai, ci, 0]*100:>6.1f}%", end='')
        print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric_idx, title in [(axes[0], 0, 'Correlation (%)'),
                                   (axes[1], 1, 'GenROC (%)')]:
        data = results[:, :, metric_idx] * 100
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto',
                       extent=[cs[0]-0.25, cs[-1]+0.25, alphas[-1]+0.125, alphas[0]-0.125])
        ax.set_xlabel('c (length coefficient)')
        ax.set_ylabel('alpha (typicality coefficient)')
        ax.set_title(f'{title}\ngen - alpha*typ + c*len (IFEval)')
        if metric_idx == 0:
            ax.plot(best_c_corr, best_alpha_corr, 'r*', markersize=15, label='Best')
        else:
            ax.plot(best_c_roc, best_alpha_roc, 'r*', markersize=15, label='Best')
        ax.plot(0, 1.0, 'ko', markersize=8, label='PMI')
        ax.legend(fontsize=10, loc='upper right')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(Path(outputs_dir) / f'length_grid_search_heatmap_ifeval{suffix}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: length_grid_search_heatmap_ifeval{suffix}.png")

    # Fine grid near optimum
    print("\n--- Fine grid near optimum ---")
    alphas_fine = np.arange(max(0, best_alpha_corr - 0.5), best_alpha_corr + 0.55, 0.1)
    cs_fine = np.arange(best_c_corr - 1.0, best_c_corr + 1.05, 0.1)

    results_fine = np.zeros((len(alphas_fine), len(cs_fine), 2))
    for pi, pid in enumerate(prompts):
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        lab = sub['correct'].values
        for ai, alpha in enumerate(alphas_fine):
            for ci, c in enumerate(cs_fine):
                corrected = gen - alpha * typ + c * leng
                corr, roc = compute_metrics(corrected, val, lab)
                results_fine[ai, ci, 0] += corr / len(prompts)
                results_fine[ai, ci, 1] += roc / len(prompts)

    best_fine_idx = np.unravel_index(np.nanargmax(results_fine[:, :, 0]), results_fine[:, :, 0].shape)
    best_alpha_fine = alphas_fine[best_fine_idx[0]]
    best_c_fine = cs_fine[best_fine_idx[1]]
    best_corr_fine = results_fine[best_fine_idx[0], best_fine_idx[1], 0]
    best_roc_fine = results_fine[best_fine_idx[0], best_fine_idx[1], 1]

    print(f"Best (fine grid) for Correlation:")
    print(f"  alpha={best_alpha_fine:.2f}, c={best_c_fine:.2f}")
    print(f"  Corr={best_corr_fine*100:.1f}%, GenROC={best_roc_fine*100:.1f}%")

    # Fine grid for ROC
    alphas_fine_r = np.arange(max(0, best_alpha_roc - 0.5), best_alpha_roc + 0.55, 0.1)
    cs_fine_r = np.arange(best_c_roc - 1.0, best_c_roc + 1.05, 0.1)
    results_fine_r = np.zeros((len(alphas_fine_r), len(cs_fine_r), 2))
    for pi, pid in enumerate(prompts):
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        lab = sub['correct'].values
        for ai, alpha in enumerate(alphas_fine_r):
            for ci, c in enumerate(cs_fine_r):
                corrected = gen - alpha * typ + c * leng
                corr, roc = compute_metrics(corrected, val, lab)
                results_fine_r[ai, ci, 0] += corr / len(prompts)
                results_fine_r[ai, ci, 1] += roc / len(prompts)

    best_fine_r_idx = np.unravel_index(np.nanargmax(results_fine_r[:, :, 1]), results_fine_r[:, :, 1].shape)
    best_alpha_fine_r = alphas_fine_r[best_fine_r_idx[0]]
    best_c_fine_r = cs_fine_r[best_fine_r_idx[1]]
    best_roc_fine_r = results_fine_r[best_fine_r_idx[0], best_fine_r_idx[1], 1]
    best_corr_fine_r = results_fine_r[best_fine_r_idx[0], best_fine_r_idx[1], 0]

    print(f"\nBest (fine grid) for GenROC:")
    print(f"  alpha={best_alpha_fine_r:.2f}, c={best_c_fine_r:.2f}")
    print(f"  Corr={best_corr_fine_r*100:.1f}%, GenROC={best_roc_fine_r*100:.1f}%")

    return {
        'best_alpha_corr': best_alpha_fine,
        'best_c_corr': best_c_fine,
        'best_corr': best_corr_fine,
        'best_roc_at_corr': best_roc_fine,
        'best_alpha_roc': best_alpha_fine_r,
        'best_c_roc': best_c_fine_r,
        'best_roc': best_roc_fine_r,
        'best_corr_at_roc': best_corr_fine_r,
    }


# =========================================================================
# Analysis 4: Per-prompt length coefficient
# =========================================================================

def analyze_per_prompt_length(df, outputs_dir, suffix=''):
    """Fit val ~ gen + typ + len per prompt, examine length coefficient."""

    print(f"\n{'=' * 80}")
    print("ANALYSIS 4: PER-PROMPT LENGTH COEFFICIENT")
    print(f"{'=' * 80}")

    from numpy.linalg import lstsq

    results = []
    for pid in sorted(df['prompt_id'].unique(), key=lambda x: int(x.split('_')[1])):
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        n = len(sub)

        gen_z = (gen - gen.mean()) / gen.std() if gen.std() > 0 else gen * 0
        typ_z = (typ - typ.mean()) / typ.std() if typ.std() > 0 else typ * 0
        len_z = (leng - leng.mean()) / leng.std() if leng.std() > 0 else leng * 0

        X = np.column_stack([np.ones(n), gen_z, typ_z, len_z])
        coeffs, _, _, _ = lstsq(X, val, rcond=None)

        X_nolen = np.column_stack([np.ones(n), gen_z, typ_z])
        coeffs_nolen, _, _, _ = lstsq(X_nolen, val, rcond=None)

        pred_with_len = X @ coeffs
        pred_no_len = X_nolen @ coeffs_nolen
        r_with = pearsonr(pred_with_len, val)[0]
        r_without = pearsonr(pred_no_len, val)[0]

        results.append({
            'prompt_id': pid, 'n': n,
            'b_gen': coeffs[1], 'b_typ': coeffs[2], 'b_len': coeffs[3],
            'b_typ_nolen': coeffs_nolen[2],
            'r_with_len': r_with, 'r_without_len': r_without,
            'r_improvement': r_with - r_without,
        })

    df_results = pd.DataFrame(results)

    print(f"\nPer-prompt length coefficient (b_len) distribution:")
    print(f"  Mean:   {df_results['b_len'].mean():>8.4f}")
    print(f"  Median: {df_results['b_len'].median():>8.4f}")
    print(f"  Std:    {df_results['b_len'].std():>8.4f}")
    print(f"  Positive: {(df_results['b_len'] > 0).sum()}/{len(df_results)}")

    print(f"\nCorrelation improvement from adding length:")
    print(f"  Mean:   {df_results['r_improvement'].mean():>8.4f}")
    print(f"  Improved: {(df_results['r_improvement'] > 0).sum()}/{len(df_results)}")

    print(f"\n{'Prompt':>12} {'n':>4} {'b_gen':>8} {'b_typ':>8} {'b_len':>8} "
          f"{'r_with':>8} {'r_without':>8} {'delta_r':>8}")
    print("-" * 75)
    for _, row in df_results.sort_values('b_len').iterrows():
        print(f"{row['prompt_id']:>12} {row['n']:>4} {row['b_gen']:>8.3f} "
              f"{row['b_typ']:>8.3f} {row['b_len']:>8.3f} "
              f"{row['r_with_len']:>8.3f} {row['r_without_len']:>8.3f} "
              f"{row['r_improvement']:>8.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(df_results['b_len'], bins=min(15, len(df_results)),
            edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(df_results['b_len'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'mean={df_results["b_len"].mean():.3f}')
    ax.set_xlabel('Length coefficient (b_len)')
    ax.set_ylabel('Count (prompts)')
    ax.set_title('Per-prompt b_len (IFEval)')
    ax.legend()

    ax = axes[1]
    ax.scatter(df_results['b_len'], df_results['r_improvement'],
               s=60, alpha=0.8, c='steelblue', edgecolors='black', linewidth=0.5)
    for _, row in df_results.iterrows():
        ax.annotate(row['prompt_id'], (row['b_len'], row['r_improvement']),
                    fontsize=7, ha='center', va='bottom')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Length coefficient (b_len)')
    ax.set_ylabel('Correlation improvement')
    ax.set_title('Does length help per-prompt?')

    ax = axes[2]
    sorted_df = df_results.sort_values('b_len')
    colors = ['steelblue' if v > 0 else 'salmon' for v in sorted_df['b_len']]
    ax.barh(range(len(sorted_df)), sorted_df['b_len'], color=colors,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['prompt_id'], fontsize=7)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Length coefficient (b_len)')
    ax.set_title('b_len by prompt (sorted)')

    plt.tight_layout()
    plt.savefig(str(Path(outputs_dir) / f'length_per_prompt_analysis_ifeval{suffix}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: length_per_prompt_analysis_ifeval{suffix}.png")

    return df_results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Investigate length effect for ifeval')
    parser.add_argument('--outputs-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'outputs'))
    parser.add_argument('--self-typicality', action='store_true',
                        help='Use self- prefixed CSVs')
    args = parser.parse_args()

    suffix = '_self' if args.self_typicality else ''
    df = load_all_ifeval(args.outputs_dir, self_typicality=args.self_typicality)

    pooled_corr, within_corr, prompt_corrs = analyze_correlations(df, args.outputs_dir, suffix)
    redundancy_results = analyze_redundancy(df, args.outputs_dir, suffix)
    grid_results = grid_search_alpha_c(df, args.outputs_dir, suffix)
    per_prompt_results = analyze_per_prompt_length(df, args.outputs_dir, suffix)


if __name__ == '__main__':
    main()
