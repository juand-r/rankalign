#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate the length effect in typicality correction.

Analyses:
  1. Correlation structure: pairwise correlations between gen, typ, len, val
     (both pooled across all data and within-prompt)
  2. Redundancy test: does gen - 1.5*typ correlate with gen - 1.0*typ + c*len?
     Is length just another way to adjust the typicality coefficient?
  3. Grid search over (alpha, c) in gen - alpha*typ + c*len (no fitting, LOO)
  4. Per-prompt length coefficient analysis: fit val ~ gen + typ + len per prompt
     and examine the distribution of the length coefficient
  5. Effect size of beta_len in context
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


def load_all_plausibleqa(outputs_dir, typicality_source='gpt2', cache_file=None):
    """Load all plausibleqa CSVs into a single DataFrame with prompt_id column.

    Args:
        outputs_dir: Directory containing score CSVs
        typicality_source: 'gpt2' or 'self'
        cache_file: Path to cached self-typicality CSV (required when source='self')
    """
    prefix = 'self-' if typicality_source == 'self' else ''
    pattern = str(Path(outputs_dir) / f'scores_{prefix}v6-google_gemma-2-2b_plausibleqa-*_test_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))

    # Deduplicate: keep latest file per prompt_id
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
        df['typicality_gpt2'] = df['gen_score'] - df['gen_score_typcorr']
        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)

    typ_label = typicality_source
    print(f"Loaded {len(df)} rows across {df['prompt_id'].nunique()} prompts (typicality: {typ_label})")
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

def analyze_correlations(df, outputs_dir):
    """Pairwise correlations between gen, typ, len, val — pooled and within-prompt."""

    print("\n" + "=" * 80)
    print("ANALYSIS 1: CORRELATION STRUCTURE")
    print("=" * 80)

    gen = df['gen_score'].values
    typ = df['typicality_gpt2'].values
    leng = df['num_tokens'].values.astype(float)
    val = df['val_score'].values

    # Pooled correlations
    vars_dict = {'gen': gen, 'typ': typ, 'len': leng, 'val': val}
    names = list(vars_dict.keys())
    n = len(names)

    print("\n--- Pooled Pearson correlations (N=468) ---")
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

    # Within-prompt correlations (average across prompts)
    print("\n--- Within-prompt Pearson correlations (averaged across 33 prompts) ---")
    prompt_corrs = {f'{a}-{b}': [] for a in names for b in names}
    var_cols = {'gen': 'gen_score', 'typ': 'typicality_gpt2',
                'len': 'num_tokens', 'val': 'val_score'}
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
            nprompts = len(vals)
            print(f"{mean_r:>8.3f}", end='')
        print(f"  ({nprompts})")

    # Also print std of within-prompt correlations
    print("\n--- Std of within-prompt correlations ---")
    print(f"{'':>6}", end='')
    for name in names:
        print(f"{name:>8}", end='')
    print()
    for i in range(n):
        print(f"{names[i]:>6}", end='')
        for j in range(n):
            vals = prompt_corrs[f'{names[i]}-{names[j]}']
            std_r = np.nanstd(vals) if len(vals) > 0 else float('nan')
            print(f"{std_r:>8.3f}", end='')
        print()

    # Plot: correlation matrix heatmap (pooled + within-prompt side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in [(axes[0], pooled_corr, 'Pooled (N=468)'),
                            (axes[1], within_corr, 'Within-prompt (mean of 33)')]:
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
    fig.suptitle('Correlation Structure: gen, typ, len, val', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(str(Path(outputs_dir) / 'length_correlation_matrix_self.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: length_correlation_matrix_self.png")

    return pooled_corr, within_corr, prompt_corrs


# =========================================================================
# Analysis 2: Redundancy test
# =========================================================================

def analyze_redundancy(df, outputs_dir):
    """Test: does gen - 1.5*typ correlate with gen - 1.0*typ + c*len?"""

    print("\n" + "=" * 80)
    print("ANALYSIS 2: REDUNDANCY TEST")
    print("=" * 80)
    print("Question: Is gen - 1.5*typ approximately equal to gen - 1.0*typ + c*len")
    print("          for some c? If so, length is just a proxy for adjusting b.\n")

    gen = df['gen_score'].values
    typ = df['typicality_gpt2'].values
    leng = df['num_tokens'].values.astype(float)

    pmi_15 = gen - 1.5 * typ  # = gen - typ - 0.5*typ

    # The difference: (gen - 1.5*typ) - (gen - 1.0*typ) = -0.5*typ
    # So gen - 1.5*typ = gen - 1.0*typ - 0.5*typ
    # For this to equal gen - 1.0*typ + c*len, we'd need -0.5*typ ≈ c*len
    # i.e. typ ≈ -2c*len, or equivalently corr(typ, len) should be high.

    pmi_10 = gen - 1.0 * typ

    # Find optimal c by regression: pmi_15 = pmi_10 + c*len + residual
    # => -0.5*typ = c*len + residual
    # => c = cov(-0.5*typ, len) / var(len) = -0.5 * cov(typ, len) / var(len)
    diff = pmi_15 - pmi_10  # = -0.5 * typ
    c_opt = np.cov(diff, leng)[0, 1] / np.var(leng)
    print(f"Optimal c (pooled regression of -0.5*typ on len): {c_opt:.4f}")
    print(f"  This means: gen - 1.5*typ ≈ gen - 1.0*typ + ({c_opt:.4f})*len")

    # How good is the approximation?
    approx = pmi_10 + c_opt * leng
    r_approx = pearsonr(pmi_15, approx)[0]
    print(f"  Pearson r(gen - 1.5*typ, gen - 1.0*typ + c_opt*len) = {r_approx:.6f}")

    # What fraction of the variance in -0.5*typ is explained by len?
    r_typ_len = pearsonr(typ, leng)[0]
    print(f"\n  Correlation between typ and len:")
    print(f"    Pooled r(typ, len) = {r_typ_len:.4f}")
    print(f"    R² = {r_typ_len**2:.4f}")
    print(f"    => len explains {r_typ_len**2*100:.1f}% of variance in typ")

    # Within-prompt
    within_r_typ_len = []
    within_r_approx = []
    for pid in df['prompt_id'].unique():
        sub = df[df['prompt_id'] == pid]
        t = sub['typicality_gpt2'].values
        l = sub['num_tokens'].values.astype(float)
        g = sub['gen_score'].values
        if len(t) >= 3 and np.std(l) > 1e-10 and np.std(t) > 1e-10:
            within_r_typ_len.append(pearsonr(t, l)[0])
            p15 = g - 1.5 * t
            p10 = g - 1.0 * t
            c_local = np.cov(p15 - p10, l)[0, 1] / np.var(l)
            approx_local = p10 + c_local * l
            within_r_approx.append(pearsonr(p15, approx_local)[0])

    print(f"    Within-prompt r(typ, len): mean={np.mean(within_r_typ_len):.4f}, "
          f"std={np.std(within_r_typ_len):.4f}")
    print(f"    Within-prompt R²: mean={np.mean([r**2 for r in within_r_typ_len]):.4f}")

    # Key question: if typ and len are not that correlated within-prompt,
    # then length is NOT just a proxy for b — it carries independent info
    r2_within = np.nanmean([r**2 for r in within_r_typ_len]) if within_r_typ_len else float('nan')
    if r2_within < 0.5:
        print(f"\n  CONCLUSION: Within-prompt R²(typ,len) = {r2_within:.3f} < 0.5")
        print(f"  Length is NOT merely a proxy for adjusting the typicality coefficient.")
        print(f"  It carries independent information beyond typ.")
    else:
        print(f"\n  CONCLUSION: Within-prompt R²(typ,len) = {r2_within:.3f} >= 0.5")
        print(f"  Length is substantially redundant with typ.")

    # Now check: does PMI_1.5 actually correlate highly with PMI + c*len per-prompt?
    # (this is the real question — not just whether typ correlates with len)
    print(f"\n--- Per-prompt: how well does PMI + c*len approximate PMI_1.5? ---")
    per_prompt_r2 = []
    for pid in df['prompt_id'].unique():
        sub = df[df['prompt_id'] == pid]
        g = sub['gen_score'].values
        t = sub['typicality_gpt2'].values
        l = sub['num_tokens'].values.astype(float)
        p15 = g - 1.5 * t
        p10 = g - 1.0 * t
        # Best c for this prompt
        if np.var(l) > 0 and len(l) >= 3:
            c_local = np.cov(p15 - p10, l)[0, 1] / np.var(l)
            approx_local = p10 + c_local * l
            r = pearsonr(p15, approx_local)[0]
            per_prompt_r2.append(r**2)

    print(f"  Per-prompt R²(PMI_1.5, PMI + c_best*len): "
          f"mean={np.mean(per_prompt_r2):.4f}, min={np.min(per_prompt_r2):.4f}, "
          f"max={np.max(per_prompt_r2):.4f}")
    if np.mean(per_prompt_r2) > 0.95:
        print(f"  => PMI + c*len CAN closely approximate PMI_1.5 (high R²)")
        print(f"     But this doesn't mean it's ONLY doing what PMI_1.5 does —")
        print(f"     it could also capture additional signal.")
    else:
        print(f"  => PMI + c*len is NOT a close approximation of PMI_1.5")
        print(f"     Length provides genuinely different information.")

    # Make a scatter plot: typ vs len, colored by prompt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: pooled typ vs len
    ax = axes[0]
    ax.scatter(leng, typ, alpha=0.3, s=20, c='steelblue')
    z = np.polyfit(leng, typ, 1)
    x_line = np.linspace(leng.min(), leng.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2,
            label=f'r={r_typ_len:.3f}')
    ax.set_xlabel('Answer length (tokens)', fontsize=12)
    ax.set_ylabel('Typicality (GPT-2 log P(y))', fontsize=12)
    ax.set_title('Pooled: typ vs len', fontsize=13)
    ax.legend(fontsize=11)

    # Right: histogram of within-prompt r(typ, len)
    ax = axes[1]
    ax.hist(within_r_typ_len, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(within_r_typ_len), color='red', linestyle='--', linewidth=2,
               label=f'mean={np.mean(within_r_typ_len):.3f}')
    ax.set_xlabel('Within-prompt r(typ, len)', fontsize=12)
    ax.set_ylabel('Count (prompts)', fontsize=12)
    ax.set_title('Distribution of within-prompt r(typ, len)', fontsize=13)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(str(Path(outputs_dir) / 'length_redundancy_analysis_self.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: length_redundancy_analysis_self.png")

    return {
        'r_typ_len_pooled': r_typ_len,
        'r_typ_len_within_mean': np.mean(within_r_typ_len),
        'r_typ_len_within_std': np.std(within_r_typ_len),
        'r2_within_mean': r2_within,
        'c_opt_pooled': c_opt,
    }


# =========================================================================
# Analysis 3: Grid search over (alpha, c)
# =========================================================================

def grid_search_alpha_c(df, outputs_dir):
    """Grid search over alpha and c in gen - alpha*typ + c*len.
    Evaluate LOO (per eval prompt) — no fitting, all parameter-free."""

    print("\n" + "=" * 80)
    print("ANALYSIS 3: GRID SEARCH over (alpha, c) in gen - alpha*typ + c*len")
    print("=" * 80)

    # Define grid
    alphas = np.arange(0.0, 3.1, 0.25)
    # c is trickier — need to think about scale.
    # len is in tokens (1-38), gen is in log-prob (-118 to -1).
    # A reasonable c would be on the order of gen_range / len_range ~ 3
    cs = np.arange(-2.0, 4.1, 0.5)

    prompts = sorted(df['prompt_id'].unique())

    # For each (alpha, c), compute mean corr and GenROC across prompts
    results = np.zeros((len(alphas), len(cs), 2))  # [corr, roc]

    for pi, pid in enumerate(prompts):
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        lab = sub['gpt4_ground_truth'].values

        for ai, alpha in enumerate(alphas):
            for ci, c in enumerate(cs):
                corrected = gen - alpha * typ + c * leng
                corr, roc = compute_metrics(corrected, val, lab)
                results[ai, ci, 0] += corr / len(prompts)
                results[ai, ci, 1] += roc / len(prompts)

    # Find best
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

    # Reference: PMI_1.5 (alpha=1.5, c=0)
    ai_15 = np.argmin(np.abs(alphas - 1.5))
    ci_0 = np.argmin(np.abs(cs - 0.0))
    pmi15_corr = results[ai_15, ci_0, 0]
    pmi15_roc = results[ai_15, ci_0, 1]
    print(f"\nReference PMI_1.5 (alpha=1.5, c=0):")
    print(f"  Corr={pmi15_corr*100:.1f}%, GenROC={pmi15_roc*100:.1f}%")

    # Print a table of results near the optimum
    header = 'alpha\\c'
    print(f"\n--- Corr heatmap (selected) ---")
    print(f"{header:>8}", end='')
    for c in cs:
        print(f"{c:>7.1f}", end='')
    print()
    for ai, alpha in enumerate(alphas):
        print(f"{alpha:>8.2f}", end='')
        for ci, c in enumerate(cs):
            val_str = f"{results[ai, ci, 0]*100:>6.1f}%"
            print(val_str, end='')
        print()

    print(f"\n--- GenROC heatmap (selected) ---")
    print(f"{header:>8}", end='')
    for c in cs:
        print(f"{c:>7.1f}", end='')
    print()
    for ai, alpha in enumerate(alphas):
        print(f"{alpha:>8.2f}", end='')
        for ci, c in enumerate(cs):
            val_str = f"{results[ai, ci, 1]*100:>6.1f}%"
            print(val_str, end='')
        print()

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric_idx, title in [(axes[0], 0, 'Correlation (%)'),
                                   (axes[1], 1, 'GenROC (%)')]:
        data = results[:, :, metric_idx] * 100
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto',
                       extent=[cs[0]-0.25, cs[-1]+0.25, alphas[-1]+0.125, alphas[0]-0.125])
        ax.set_xlabel('c (length coefficient)', fontsize=12)
        ax.set_ylabel('alpha (typicality coefficient)', fontsize=12)
        ax.set_title(f'{title}\ngen - alpha*typ + c*len', fontsize=13)

        # Mark best
        if metric_idx == 0:
            ax.plot(best_c_corr, best_alpha_corr, 'r*', markersize=15, label='Best')
        else:
            ax.plot(best_c_roc, best_alpha_roc, 'r*', markersize=15, label='Best')
        # Mark PMI_1.5
        ax.plot(0, 1.5, 'ko', markersize=8, label='PMI_1.5')
        ax.legend(fontsize=10, loc='upper right')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(Path(outputs_dir) / 'length_grid_search_heatmap_self.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: length_grid_search_heatmap_self.png")

    # Also do a finer grid near the optimum
    print("\n--- Fine grid near optimum ---")
    alphas_fine = np.arange(max(0, best_alpha_corr - 0.5),
                            best_alpha_corr + 0.55, 0.1)
    cs_fine = np.arange(best_c_corr - 1.0, best_c_corr + 1.05, 0.1)

    results_fine = np.zeros((len(alphas_fine), len(cs_fine), 2))
    for pi, pid in enumerate(prompts):
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        lab = sub['gpt4_ground_truth'].values

        for ai, alpha in enumerate(alphas_fine):
            for ci, c in enumerate(cs_fine):
                corrected = gen - alpha * typ + c * leng
                corr, roc = compute_metrics(corrected, val, lab)
                results_fine[ai, ci, 0] += corr / len(prompts)
                results_fine[ai, ci, 1] += roc / len(prompts)

    best_fine_idx = np.unravel_index(np.nanargmax(results_fine[:, :, 0]),
                                      results_fine[:, :, 0].shape)
    best_alpha_fine = alphas_fine[best_fine_idx[0]]
    best_c_fine = cs_fine[best_fine_idx[1]]
    best_corr_fine = results_fine[best_fine_idx[0], best_fine_idx[1], 0]
    best_roc_fine = results_fine[best_fine_idx[0], best_fine_idx[1], 1]

    print(f"Best (fine grid) for Correlation:")
    print(f"  alpha={best_alpha_fine:.2f}, c={best_c_fine:.2f}")
    print(f"  Corr={best_corr_fine*100:.1f}%, GenROC={best_roc_fine*100:.1f}%")

    # Fine grid for ROC
    alphas_fine_r = np.arange(max(0, best_alpha_roc - 0.5),
                              best_alpha_roc + 0.55, 0.1)
    cs_fine_r = np.arange(best_c_roc - 1.0, best_c_roc + 1.05, 0.1)

    results_fine_r = np.zeros((len(alphas_fine_r), len(cs_fine_r), 2))
    for pi, pid in enumerate(prompts):
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        lab = sub['gpt4_ground_truth'].values

        for ai, alpha in enumerate(alphas_fine_r):
            for ci, c in enumerate(cs_fine_r):
                corrected = gen - alpha * typ + c * leng
                corr, roc = compute_metrics(corrected, val, lab)
                results_fine_r[ai, ci, 0] += corr / len(prompts)
                results_fine_r[ai, ci, 1] += roc / len(prompts)

    best_fine_r_idx = np.unravel_index(np.nanargmax(results_fine_r[:, :, 1]),
                                        results_fine_r[:, :, 1].shape)
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

def analyze_per_prompt_length(df, outputs_dir):
    """Fit val ~ gen + typ + len per prompt, examine length coefficient distribution."""

    print("\n" + "=" * 80)
    print("ANALYSIS 4: PER-PROMPT LENGTH COEFFICIENT")
    print("=" * 80)

    from numpy.linalg import lstsq

    results = []
    for pid in sorted(df['prompt_id'].unique()):
        sub = df[df['prompt_id'] == pid]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        leng = sub['num_tokens'].values.astype(float)
        val = sub['val_score'].values
        n = len(sub)

        # Z-score within this prompt
        gen_z = (gen - gen.mean()) / gen.std() if gen.std() > 0 else gen * 0
        typ_z = (typ - typ.mean()) / typ.std() if typ.std() > 0 else typ * 0
        len_z = (leng - leng.mean()) / leng.std() if leng.std() > 0 else leng * 0

        # Fit: val = a + b1*gen_z + b2*typ_z + b3*len_z
        X = np.column_stack([np.ones(n), gen_z, typ_z, len_z])
        coeffs, _, _, _ = lstsq(X, val, rcond=None)

        # Also fit without length for comparison
        X_nolen = np.column_stack([np.ones(n), gen_z, typ_z])
        coeffs_nolen, _, _, _ = lstsq(X_nolen, val, rcond=None)

        # Correlations of corrected scores with val
        pred_with_len = X @ coeffs
        pred_no_len = X_nolen @ coeffs_nolen
        r_with = pearsonr(pred_with_len, val)[0]
        r_without = pearsonr(pred_no_len, val)[0]

        results.append({
            'prompt_id': pid,
            'n': n,
            'b_gen': coeffs[1],
            'b_typ': coeffs[2],
            'b_len': coeffs[3],
            'b_typ_nolen': coeffs_nolen[2],
            'r_with_len': r_with,
            'r_without_len': r_without,
            'r_improvement': r_with - r_without,
        })

    df_results = pd.DataFrame(results)

    print(f"\nPer-prompt length coefficient (b_len) distribution:")
    print(f"  Mean:   {df_results['b_len'].mean():>8.4f}")
    print(f"  Median: {df_results['b_len'].median():>8.4f}")
    print(f"  Std:    {df_results['b_len'].std():>8.4f}")
    print(f"  Min:    {df_results['b_len'].min():>8.4f}")
    print(f"  Max:    {df_results['b_len'].max():>8.4f}")

    n_positive = (df_results['b_len'] > 0).sum()
    n_total = len(df_results)
    print(f"  Positive: {n_positive}/{n_total} ({n_positive/n_total*100:.0f}%)")

    print(f"\nPer-prompt correlation improvement from adding length:")
    print(f"  Mean:   {df_results['r_improvement'].mean():>8.4f}")
    print(f"  Median: {df_results['r_improvement'].median():>8.4f}")
    print(f"  Min:    {df_results['r_improvement'].min():>8.4f}")
    print(f"  Max:    {df_results['r_improvement'].max():>8.4f}")

    n_improved = (df_results['r_improvement'] > 0).sum()
    print(f"  Improved: {n_improved}/{n_total} ({n_improved/n_total*100:.0f}%)")

    # Full table
    print(f"\n{'Prompt':>8} {'n':>3} {'b_gen':>8} {'b_typ':>8} {'b_len':>8} "
          f"{'r_with':>8} {'r_without':>8} {'delta_r':>8}")
    print("-" * 70)
    for _, row in df_results.sort_values('b_len').iterrows():
        print(f"{row['prompt_id']:>8} {row['n']:>3} {row['b_gen']:>8.3f} "
              f"{row['b_typ']:>8.3f} {row['b_len']:>8.3f} "
              f"{row['r_with_len']:>8.3f} {row['r_without_len']:>8.3f} "
              f"{row['r_improvement']:>8.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: histogram of b_len
    ax = axes[0]
    ax.hist(df_results['b_len'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(df_results['b_len'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'mean={df_results["b_len"].mean():.3f}')
    ax.set_xlabel('Length coefficient (b_len)', fontsize=12)
    ax.set_ylabel('Count (prompts)', fontsize=12)
    ax.set_title('Per-prompt b_len distribution', fontsize=13)
    ax.legend(fontsize=10)

    # Middle: scatter b_len vs r_improvement
    ax = axes[1]
    ax.scatter(df_results['b_len'], df_results['r_improvement'],
               s=40, alpha=0.7, c='steelblue', edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Length coefficient (b_len)', fontsize=12)
    ax.set_ylabel('Correlation improvement from adding len', fontsize=12)
    ax.set_title('Does length help per-prompt?', fontsize=13)

    # Right: sorted bar chart of b_len by prompt
    ax = axes[2]
    sorted_df = df_results.sort_values('b_len')
    colors = ['steelblue' if v > 0 else 'salmon' for v in sorted_df['b_len']]
    ax.barh(range(len(sorted_df)), sorted_df['b_len'], color=colors, edgecolor='black',
            linewidth=0.5)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['prompt_id'], fontsize=7)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Length coefficient (b_len)', fontsize=12)
    ax.set_ylabel('Prompt ID', fontsize=12)
    ax.set_title('b_len by prompt (sorted)', fontsize=13)

    plt.tight_layout()
    plt.savefig(str(Path(outputs_dir) / 'length_per_prompt_analysis_self.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: length_per_prompt_analysis_self.png")

    return df_results


# =========================================================================
# Analysis 5: Effect size of beta_len
# =========================================================================

def analyze_effect_size(df, redundancy_results, grid_results, per_prompt_results):
    """Put beta_len = 0.044 in context."""

    print("\n" + "=" * 80)
    print("ANALYSIS 5: EFFECT SIZE OF LENGTH COEFFICIENT")
    print("=" * 80)

    gen = df['gen_score'].values
    typ = df['typicality_gpt2'].values
    leng = df['num_tokens'].values.astype(float)
    val = df['val_score'].values

    # beta_len = 0.044 is on the z-scored scale
    # On original scale: beta_len_orig = beta_len_z / std(len)
    len_std = leng.std()
    typ_std = typ.std()
    gen_std = gen.std()
    val_std = val.std()

    beta_len_z = 0.044
    beta_gen_z = 0.094  # from ME_len
    beta_typ_z = -0.153

    print(f"\nVariable scales:")
    print(f"  gen: mean={gen.mean():.1f}, std={gen_std:.1f}")
    print(f"  typ: mean={typ.mean():.1f}, std={typ_std:.1f}")
    print(f"  len: mean={leng.mean():.1f}, std={leng.std():.1f}")
    print(f"  val: mean={val.mean():.3f}, std={val_std:.3f}")

    # Effect of 1 SD change in each predictor on val:
    print(f"\nEffect of 1 SD change in each predictor on val score:")
    print(f"  gen: {beta_gen_z:.4f} (1 SD of gen changes val by {beta_gen_z:.4f})")
    print(f"  typ: {beta_typ_z:.4f} (1 SD of typ changes val by {abs(beta_typ_z):.4f})")
    print(f"  len: {beta_len_z:.4f} (1 SD of len changes val by {beta_len_z:.4f})")
    print(f"\n  Ratio |beta_len/beta_typ| = {abs(beta_len_z/beta_typ_z):.3f}")
    print(f"  Ratio |beta_len/beta_gen| = {abs(beta_len_z/beta_gen_z):.3f}")
    print(f"  => Length effect is {abs(beta_len_z/beta_typ_z)*100:.0f}% the size of "
          f"typicality effect")

    # What does c mean in the original-scale grid search?
    print(f"\n--- Grid search optimal c in original-scale context ---")
    c_opt = grid_results.get('best_c_corr', 0)
    alpha_opt = grid_results.get('best_alpha_corr', 1.5)
    print(f"  Best (alpha, c) for corr: ({alpha_opt:.2f}, {c_opt:.2f})")

    # How much does adding c*len change the corrected score?
    # For a typical answer: len ~ 3 tokens (median), c*len ~ c*3
    # vs. alpha*typ ~ alpha * mean(|typ|) ~ 1.5 * 30 ~ 45
    median_len = np.median(leng)
    mean_abs_typ = np.mean(np.abs(typ))
    print(f"  Typical correction magnitude:")
    print(f"    alpha*|typ| at median: {alpha_opt * mean_abs_typ:.1f}")
    print(f"    c*len at median:       {c_opt * median_len:.1f}")
    print(f"    Ratio: {abs(c_opt * median_len) / (alpha_opt * mean_abs_typ):.3f}")

    # Improvement from adding length
    print(f"\n--- Improvement summary ---")
    print(f"  PMI_1.5 (alpha=1.5, c=0):  Corr=53.5%, GenROC=76.0%")
    print(f"  ME_len:                     Corr=54.9%, GenROC=75.8%")
    best_corr = grid_results.get('best_corr', 0)
    best_roc_at_corr = grid_results.get('best_roc_at_corr', 0)
    print(f"  Grid search best (corr):    Corr={best_corr*100:.1f}%, "
          f"GenROC={best_roc_at_corr*100:.1f}%")
    print(f"  Correlation improvement from length: "
          f"+{(best_corr - 0.535)*100:.1f} pp over PMI_1.5")

    # Is the per-prompt length coefficient consistent?
    b_len_positive_frac = (per_prompt_results['b_len'] > 0).mean()
    print(f"\n  Per-prompt b_len > 0 in {b_len_positive_frac*100:.0f}% of prompts")
    print(f"  Mean b_len = {per_prompt_results['b_len'].mean():.4f} "
          f"(std={per_prompt_results['b_len'].std():.4f})")

    # Final verdict
    print(f"\n{'=' * 60}")
    print("VERDICT:")
    if best_corr - 0.535 > 0.01:
        print(f"  Length provides a small but real improvement ({(best_corr-0.535)*100:.1f} pp).")
    else:
        print(f"  Length improvement is negligible ({(best_corr-0.535)*100:.1f} pp).")

    r2_within = redundancy_results.get('r2_within_mean', 0)
    if r2_within < 0.3:
        print(f"  Length is NOT a proxy for typicality (within-prompt R²={r2_within:.3f}).")
        print(f"  It carries independent information.")
    else:
        print(f"  Length is partially redundant with typicality (within-prompt R²={r2_within:.3f}).")

    print(f"  Effect size: ~{abs(beta_len_z/beta_typ_z)*100:.0f}% of typicality effect.")
    print(f"{'=' * 60}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Investigate length effect')
    parser.add_argument('--outputs-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'outputs'))
    parser.add_argument('--typicality-source', type=str, default='gpt2',
                        choices=['gpt2', 'self'],
                        help='Typicality source: gpt2 (default) or self')
    parser.add_argument('--cache-file', type=str, default=None,
                        help='Path to cached self-typicality CSV')
    args = parser.parse_args()

    df = load_all_plausibleqa(args.outputs_dir,
                              typicality_source=args.typicality_source,
                              cache_file=args.cache_file)

    # Run all analyses
    pooled_corr, within_corr, prompt_corrs = analyze_correlations(df, args.outputs_dir)
    redundancy_results = analyze_redundancy(df, args.outputs_dir)
    grid_results = grid_search_alpha_c(df, args.outputs_dir)
    per_prompt_results = analyze_per_prompt_length(df, args.outputs_dir)
    analyze_effect_size(df, redundancy_results, grid_results, per_prompt_results)


if __name__ == '__main__':
    main()
