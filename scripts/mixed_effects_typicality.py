#!/usr/bin/env python
"""
Mixed-effects analysis of typicality correction.

Uses statsmodels MixedLM to fit:
    val ~ gen + typ + (typ | prompt)

This gives:
  - A fixed-effect coefficient for typ (the population-level optimal correction)
  - Per-prompt random slopes for typ (how much each prompt deviates)
  - Shrinkage: prompts with less data are pulled toward the population mean

The "corrected gen score" is then:
    corrected_i = gen_i - (b_fixed + b_random_prompt) * typ_i

We compare this to PMI (b=1) and PMI_1.5 (b=1.5) using LOO evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import statsmodels.formula.api as smf
import warnings
import argparse
import sys

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def load_all_plausibleqa(outputs_dir):
    """Load all plausibleqa CSVs into a single DataFrame with prompt_id column."""
    import glob
    import re

    pattern = str(Path(outputs_dir) / 'scores_v6-google_gemma-2-2b_plausibleqa-nq_*_test_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))

    all_rows = []
    for fp in files:
        m = re.search(r'plausibleqa-nq_(\d+)_test', fp)
        if not m:
            continue
        prompt_id = m.group(1)
        df = pd.read_csv(fp)
        df['prompt_id'] = prompt_id
        df['typicality_gpt2'] = df['gen_score'] - df['gen_score_typcorr']
        all_rows.append(df)

    df = pd.concat(all_rows, ignore_index=True)
    print(f"Loaded {len(df)} rows across {df['prompt_id'].nunique()} prompts")
    return df


def compute_metrics(corrected, val, labels):
    """Compute correlation and GenROC."""
    labels = np.asarray(labels)
    corrected = np.asarray(corrected)
    val = np.asarray(val)

    mask = ~(np.isnan(corrected) | np.isnan(val))
    if mask.sum() < 3:
        return {'corr': np.nan, 'gen_roc': np.nan}

    corr = pearsonr(corrected[mask], val[mask])[0]
    try:
        gen_roc = roc_auc_score(labels[mask], corrected[mask])
    except ValueError:
        gen_roc = np.nan

    return {'corr': corr, 'gen_roc': gen_roc}


def fit_mixed_model(df):
    """Fit mixed-effects model: val ~ gen + typ + (typ | prompt).

    Returns the model result and extracted coefficients.
    """
    # Standardize predictors for numerical stability
    df = df.copy()
    gen_mean, gen_std = df['gen_score'].mean(), df['gen_score'].std()
    typ_mean, typ_std = df['typicality_gpt2'].mean(), df['typicality_gpt2'].std()

    df['gen_z'] = (df['gen_score'] - gen_mean) / gen_std
    df['typ_z'] = (df['typicality_gpt2'] - typ_mean) / typ_std

    print("\nFitting mixed-effects model:")
    print("  val_score ~ gen_z + typ_z + (typ_z | prompt_id)")
    print(f"  N = {len(df)}, n_prompts = {df['prompt_id'].nunique()}")

    # Random intercept + random slope for typ_z, grouped by prompt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "val_score ~ gen_z + typ_z",
            data=df,
            groups=df["prompt_id"],
            re_formula="~typ_z"
        )
        result = model.fit(method='lbfgs', maxiter=500)

    print("\n" + "="*70)
    print("MIXED-EFFECTS MODEL SUMMARY")
    print("="*70)
    print(result.summary())

    # Extract fixed effects
    fe = result.fe_params
    print(f"\nFixed effects (standardized):")
    print(f"  Intercept: {fe['Intercept']:.4f}")
    print(f"  gen_z:     {fe['gen_z']:.4f}")
    print(f"  typ_z:     {fe['typ_z']:.4f}")

    # Convert to original scale
    # val = a + b_gen * (gen - gen_mean)/gen_std + b_typ * (typ - typ_mean)/typ_std
    # val = a + (b_gen/gen_std)*gen + (b_typ/typ_std)*typ + const
    b_gen_orig = fe['gen_z'] / gen_std
    b_typ_orig = fe['typ_z'] / typ_std
    print(f"\nFixed effects (original scale):")
    print(f"  b_gen: {b_gen_orig:.6f}")
    print(f"  b_typ: {b_typ_orig:.6f}")
    print(f"  Implied correction: corrected = gen - ({-b_typ_orig/b_gen_orig:.4f}) * typ")
    print(f"  (ratio -b_typ/b_gen = {-b_typ_orig/b_gen_orig:.4f})")

    # Extract random effects per prompt
    re_vals = result.random_effects
    print(f"\nRandom effects (per-prompt deviations from fixed typ_z coefficient):")
    print(f"  {'Prompt':>8}  {'RE_intercept':>12}  {'RE_typ_z':>10}  {'Total_typ_z':>12}  {'Implied_b':>10}")
    print(f"  {'-'*60}")

    implied_bs = {}
    for pid in sorted(re_vals.keys()):
        re = re_vals[pid]
        re_intercept = re.iloc[0] if hasattr(re, 'iloc') else re[0]
        re_typ = re.iloc[1] if hasattr(re, 'iloc') and len(re) > 1 else 0.0
        total_typ_z = fe['typ_z'] + re_typ
        # Convert total_typ_z to original scale and compute implied b
        total_typ_orig = total_typ_z / typ_std
        implied_b = -total_typ_orig / b_gen_orig if abs(b_gen_orig) > 1e-10 else np.nan
        implied_bs[pid] = implied_b
        print(f"  {pid:>8}  {re_intercept:>12.4f}  {re_typ:>10.4f}  {total_typ_z:>12.4f}  {implied_b:>10.4f}")

    print(f"\n  Mean implied b across prompts: {np.mean(list(implied_bs.values())):.4f}")
    print(f"  Std implied b across prompts:  {np.std(list(implied_bs.values())):.4f}")
    print(f"  Range: [{min(implied_bs.values()):.4f}, {max(implied_bs.values()):.4f}]")

    return result, implied_bs, {'gen_mean': gen_mean, 'gen_std': gen_std,
                                 'typ_mean': typ_mean, 'typ_std': typ_std,
                                 'b_gen_orig': b_gen_orig, 'b_typ_orig': b_typ_orig}


def loo_evaluation(df, implied_bs, model_info):
    """LOO evaluation: for each prompt, use the mixed-model predicted b
    (fitted on ALL data including this prompt's -- but shrunk),
    and compare against PMI baselines.

    Also do a stricter LOO: refit the model excluding each prompt.
    """
    prompts = sorted(df['prompt_id'].unique())

    print(f"\n{'='*70}")
    print("LOO EVALUATION")
    print(f"{'='*70}")

    # Method 1: Use the full-data mixed model's per-prompt b values
    # (This is slightly optimistic since the model saw all data)
    results_full = []
    for pid in prompts:
        mask = df['prompt_id'] == pid
        sub = df[mask]
        gen = sub['gen_score'].values
        typ = sub['typicality_gpt2'].values
        val = sub['val_score'].values
        lab = sub['gpt4_ground_truth'].values

        b_mixed = implied_bs.get(pid, 1.0)

        for method, b in [('raw', 0.0), ('PMI', 1.0), ('PMI_1.5', 1.5),
                          ('PMI2', 2.0), ('mixed_full', b_mixed)]:
            corrected = gen - b * typ
            m = compute_metrics(corrected, val, lab)
            results_full.append({
                'prompt': pid, 'method': method, 'b': b,
                'corr': m['corr'], 'gen_roc': m['gen_roc']
            })

    df_full = pd.DataFrame(results_full)

    print("\nMethod A: Mixed model fitted on ALL data, per-prompt shrunk b")
    print("(slightly optimistic — model saw all prompts)")
    print(f"\n{'Method':<15} {'Corr':>8} {'GenROC':>8}")
    print("-" * 35)
    for method in ['raw', 'PMI', 'PMI_1.5', 'PMI2', 'mixed_full']:
        sub = df_full[df_full['method'] == method]
        print(f"{method:<15} {sub['corr'].mean()*100:>7.1f}% {sub['gen_roc'].mean()*100:>7.1f}%")

    # Method 2: True LOO — refit mixed model excluding each prompt
    print(f"\nMethod B: True LOO — refit mixed model excluding each eval prompt")
    print("(this is the fair comparison)")

    results_loo = []
    for i, eval_pid in enumerate(prompts):
        # Fit on all prompts except eval_pid
        train_df = df[df['prompt_id'] != eval_pid].copy()
        eval_df = df[df['prompt_id'] == eval_pid]

        gen_mean = train_df['gen_score'].mean()
        gen_std = train_df['gen_score'].std()
        typ_mean = train_df['typicality_gpt2'].mean()
        typ_std = train_df['typicality_gpt2'].std()

        train_df['gen_z'] = (train_df['gen_score'] - gen_mean) / gen_std
        train_df['typ_z'] = (train_df['typicality_gpt2'] - typ_mean) / typ_std

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = smf.mixedlm(
                    "val_score ~ gen_z + typ_z",
                    data=train_df,
                    groups=train_df["prompt_id"],
                    re_formula="~typ_z"
                )
                res = model.fit(method='lbfgs', maxiter=500)
                fe = res.fe_params

                # For the held-out prompt, we only have fixed effects (no random effect)
                # This is the "new prompt" prediction — uses population-level b
                b_gen_orig = fe['gen_z'] / gen_std
                b_typ_orig = fe['typ_z'] / typ_std
                b_loo = -b_typ_orig / b_gen_orig if abs(b_gen_orig) > 1e-10 else 1.0
            except Exception as e:
                b_loo = 1.0  # fall back to PMI

        eval_gen = eval_df['gen_score'].values
        eval_typ = eval_df['typicality_gpt2'].values
        eval_val = eval_df['val_score'].values
        eval_lab = eval_df['gpt4_ground_truth'].values

        for method, b in [('raw', 0.0), ('PMI', 1.0), ('PMI_1.5', 1.5),
                          ('PMI2', 2.0), ('mixed_loo', b_loo)]:
            corrected = eval_gen - b * eval_typ
            m = compute_metrics(corrected, eval_val, eval_lab)
            results_loo.append({
                'eval_prompt': eval_pid, 'method': method, 'b': b,
                'corr': m['corr'], 'gen_roc': m['gen_roc']
            })

        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(prompts)} prompts done")

    df_loo = pd.DataFrame(results_loo)

    print(f"\n{'Method':<15} {'Corr':>8} {'GenROC':>8} {'Mean_b':>8}")
    print("-" * 45)
    for method in ['raw', 'PMI', 'PMI_1.5', 'PMI2', 'mixed_loo']:
        sub = df_loo[df_loo['method'] == method]
        mean_b = sub['b'].mean()
        print(f"{method:<15} {sub['corr'].mean()*100:>7.1f}% "
              f"{sub['gen_roc'].mean()*100:>7.1f}% {mean_b:>7.3f}")

    # Print per-prompt detail for mixed_loo
    print(f"\nPer-prompt mixed_loo b values:")
    mixed_sub = df_loo[df_loo['method'] == 'mixed_loo']
    for _, row in mixed_sub.iterrows():
        print(f"  Prompt {row['eval_prompt']:>6}: b={row['b']:.4f}, "
              f"corr={row['corr']*100:.1f}%, roc={row['gen_roc']*100:.1f}%")

    return df_full, df_loo


def main():
    parser = argparse.ArgumentParser(description='Mixed-effects typicality analysis')
    parser.add_argument('--outputs-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'outputs'))
    args = parser.parse_args()

    df = load_all_plausibleqa(args.outputs_dir)

    # Fit the mixed-effects model on all data
    result, implied_bs, model_info = fit_mixed_model(df)

    # LOO evaluation
    df_full, df_loo = loo_evaluation(df, implied_bs, model_info)


if __name__ == '__main__':
    main()
