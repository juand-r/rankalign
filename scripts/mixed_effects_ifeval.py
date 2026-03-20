#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended mixed-effects analysis of typicality correction for IFEVAL tasks.

Adapted from mixed_effects_hypernym.py for the ifeval task structure:
  - ~40 ifeval prompts (prompt_1 through prompt_40)
  - Each prompt acts as a "task" in the mixed-effects model grouping
  - CSV pattern: scores_{prefix}gemma-2-2b_ifeval-prompt_N_test_log-odds_evaltc_*.csv
  - Note: ifeval uses 'gemma-2-2b' (not 'v6-google_gemma-2-2b')

Fits 8 mixed-effects model specifications and evaluates with LOO.
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
import time
import glob
import re

sys.path.insert(0, str(Path(__file__).parent))


def load_all_ifeval(outputs_dir, self_typicality=False):
    """Load all ifeval CSVs into a single DataFrame with prompt_id column.

    IFEval CSVs have columns: prompt, response, num_tokens, correct,
    val_prompt, val_score, gen_score, gen_score_typcorr, gen_score_lenorm,
    gen_score_typcorr_lenorm
    """
    prefix = 'self-' if self_typicality else ''
    pattern = str(Path(outputs_dir) / f'scores_{prefix}gemma-2-2b_ifeval-prompt_*_test_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))

    # Deduplicate: keep latest file per prompt number
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
        return {'corr': np.nan, 'gen_roc': np.nan}

    corr = pearsonr(corrected[mask], val[mask])[0]
    try:
        gen_roc = roc_auc_score(labels[mask], corrected[mask])
    except ValueError:
        gen_roc = np.nan

    return {'corr': corr, 'gen_roc': gen_roc}


# =========================================================================
# Model specification framework (same as hypernym/plausibleqa version)
# =========================================================================

class MixedEffectsModel:
    def __init__(self, name, description, analogous_to):
        self.name = name
        self.description = description
        self.analogous_to = analogous_to

    def prepare_features(self, df, stats=None):
        raise NotImplementedError

    def get_formula(self):
        raise NotImplementedError

    def predict_fixed(self, fe_params, eval_df, stats):
        raise NotImplementedError

    def fit_and_predict(self, train_df, eval_df):
        train_df = train_df.copy()
        eval_df = eval_df.copy()
        train_df, stats = self.prepare_features(train_df, stats=None)
        eval_df, _ = self.prepare_features(eval_df, stats=stats)
        formula, re_formula = self.get_formula()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, data=train_df,
                                groups=train_df["prompt_id"],
                                re_formula=re_formula)
            result = model.fit(method='lbfgs', maxiter=500)
        return self.predict_fixed(result.fe_params, eval_df, stats), result


class ME_Basic(MixedEffectsModel):
    def __init__(self):
        super().__init__("ME_basic", "val ~ gen_z + typ_z + (typ_z | prompt)", "PMI, opt_pearson_typ")
    def prepare_features(self, df, stats=None):
        df = df.copy()
        if stats is None:
            stats = {'gen_mean': df['gen_score'].mean(), 'gen_std': df['gen_score'].std(),
                     'typ_mean': df['typicality_gpt2'].mean(), 'typ_std': df['typicality_gpt2'].std()}
        df['gen_z'] = (df['gen_score'] - stats['gen_mean']) / stats['gen_std']
        df['typ_z'] = (df['typicality_gpt2'] - stats['typ_mean']) / stats['typ_std']
        return df, stats
    def get_formula(self):
        return "val_score ~ gen_z + typ_z", "~typ_z"
    def predict_fixed(self, fe, eval_df, stats):
        return fe['Intercept'] + fe['gen_z'] * eval_df['gen_z'].values + fe['typ_z'] * eval_df['typ_z'].values


class ME_Intercept(MixedEffectsModel):
    def __init__(self):
        super().__init__("ME_intercept", "val ~ gen_z + typ_z + (1 | prompt)", "PMI (simpler RE)")
    def prepare_features(self, df, stats=None):
        df = df.copy()
        if stats is None:
            stats = {'gen_mean': df['gen_score'].mean(), 'gen_std': df['gen_score'].std(),
                     'typ_mean': df['typicality_gpt2'].mean(), 'typ_std': df['typicality_gpt2'].std()}
        df['gen_z'] = (df['gen_score'] - stats['gen_mean']) / stats['gen_std']
        df['typ_z'] = (df['typicality_gpt2'] - stats['typ_mean']) / stats['typ_std']
        return df, stats
    def get_formula(self):
        return "val_score ~ gen_z + typ_z", "~1"
    def predict_fixed(self, fe, eval_df, stats):
        return fe['Intercept'] + fe['gen_z'] * eval_df['gen_z'].values + fe['typ_z'] * eval_df['typ_z'].values


class ME_Len(MixedEffectsModel):
    def __init__(self):
        super().__init__("ME_len", "val ~ gen_z + typ_z + len_z + (typ_z | prompt)", "opt_typ_len")
    def prepare_features(self, df, stats=None):
        df = df.copy()
        if stats is None:
            stats = {'gen_mean': df['gen_score'].mean(), 'gen_std': df['gen_score'].std(),
                     'typ_mean': df['typicality_gpt2'].mean(), 'typ_std': df['typicality_gpt2'].std(),
                     'len_mean': df['num_tokens'].mean(), 'len_std': df['num_tokens'].std()}
        df['gen_z'] = (df['gen_score'] - stats['gen_mean']) / stats['gen_std']
        df['typ_z'] = (df['typicality_gpt2'] - stats['typ_mean']) / stats['typ_std']
        df['len_z'] = (df['num_tokens'] - stats['len_mean']) / max(stats['len_std'], 1e-8)
        return df, stats
    def get_formula(self):
        return "val_score ~ gen_z + typ_z + len_z", "~typ_z"
    def predict_fixed(self, fe, eval_df, stats):
        return (fe['Intercept'] + fe['gen_z'] * eval_df['gen_z'].values +
                fe['typ_z'] * eval_df['typ_z'].values + fe['len_z'] * eval_df['len_z'].values)


class ME_LenRE(MixedEffectsModel):
    def __init__(self):
        super().__init__("ME_len_RE", "val ~ gen_z + typ_z + len_z + (typ_z + len_z | prompt)", "opt_typ_len (full RE)")
    def prepare_features(self, df, stats=None):
        df = df.copy()
        if stats is None:
            stats = {'gen_mean': df['gen_score'].mean(), 'gen_std': df['gen_score'].std(),
                     'typ_mean': df['typicality_gpt2'].mean(), 'typ_std': df['typicality_gpt2'].std(),
                     'len_mean': df['num_tokens'].mean(), 'len_std': df['num_tokens'].std()}
        df['gen_z'] = (df['gen_score'] - stats['gen_mean']) / stats['gen_std']
        df['typ_z'] = (df['typicality_gpt2'] - stats['typ_mean']) / stats['typ_std']
        df['len_z'] = (df['num_tokens'] - stats['len_mean']) / max(stats['len_std'], 1e-8)
        return df, stats
    def get_formula(self):
        return "val_score ~ gen_z + typ_z + len_z", "~typ_z + len_z"
    def predict_fixed(self, fe, eval_df, stats):
        return (fe['Intercept'] + fe['gen_z'] * eval_df['gen_z'].values +
                fe['typ_z'] * eval_df['typ_z'].values + fe['len_z'] * eval_df['len_z'].values)


class ME_PolyTyp(MixedEffectsModel):
    def __init__(self):
        super().__init__("ME_poly_typ", "val ~ gen_z + typ_z + typ_z_sq + (typ_z | prompt)", "opt_poly2_typ")
    def prepare_features(self, df, stats=None):
        df = df.copy()
        if stats is None:
            stats = {'gen_mean': df['gen_score'].mean(), 'gen_std': df['gen_score'].std(),
                     'typ_mean': df['typicality_gpt2'].mean(), 'typ_std': df['typicality_gpt2'].std()}
        df['gen_z'] = (df['gen_score'] - stats['gen_mean']) / stats['gen_std']
        df['typ_z'] = (df['typicality_gpt2'] - stats['typ_mean']) / stats['typ_std']
        df['typ_z_sq'] = df['typ_z'] ** 2
        return df, stats
    def get_formula(self):
        return "val_score ~ gen_z + typ_z + typ_z_sq", "~typ_z"
    def predict_fixed(self, fe, eval_df, stats):
        return (fe['Intercept'] + fe['gen_z'] * eval_df['gen_z'].values +
                fe['typ_z'] * eval_df['typ_z'].values + fe['typ_z_sq'] * eval_df['typ_z_sq'].values)


class ME_PolyFull(MixedEffectsModel):
    def __init__(self):
        super().__init__("ME_poly_full",
                         "val ~ gen_z + typ_z + len_z + typ_z_sq + len_z_sq + typ_len_z + (1 | prompt)",
                         "opt_poly2_full")
    def prepare_features(self, df, stats=None):
        df = df.copy()
        if stats is None:
            stats = {'gen_mean': df['gen_score'].mean(), 'gen_std': df['gen_score'].std(),
                     'typ_mean': df['typicality_gpt2'].mean(), 'typ_std': df['typicality_gpt2'].std(),
                     'len_mean': df['num_tokens'].mean(), 'len_std': df['num_tokens'].std()}
        df['gen_z'] = (df['gen_score'] - stats['gen_mean']) / stats['gen_std']
        df['typ_z'] = (df['typicality_gpt2'] - stats['typ_mean']) / stats['typ_std']
        df['len_z'] = (df['num_tokens'] - stats['len_mean']) / max(stats['len_std'], 1e-8)
        df['typ_z_sq'] = df['typ_z'] ** 2
        df['len_z_sq'] = df['len_z'] ** 2
        df['typ_len_z'] = df['typ_z'] * df['len_z']
        return df, stats
    def get_formula(self):
        return "val_score ~ gen_z + typ_z + len_z + typ_z_sq + len_z_sq + typ_len_z", "~1"
    def predict_fixed(self, fe, eval_df, stats):
        return (fe['Intercept'] + fe['gen_z'] * eval_df['gen_z'].values +
                fe['typ_z'] * eval_df['typ_z'].values + fe['len_z'] * eval_df['len_z'].values +
                fe['typ_z_sq'] * eval_df['typ_z_sq'].values + fe['len_z_sq'] * eval_df['len_z_sq'].values +
                fe['typ_len_z'] * eval_df['typ_len_z'].values)


class ME_PerToken(MixedEffectsModel):
    def __init__(self):
        super().__init__("ME_per_token", "val ~ gen_pt_z + typ_pt_z + (typ_pt_z | prompt)", "opt_lenorm_typ, PMI_div_len")
    def prepare_features(self, df, stats=None):
        df = df.copy()
        lens = np.maximum(df['num_tokens'].values, 1).astype(float)
        df['gen_pt'] = df['gen_score'] / lens
        df['typ_pt'] = df['typicality_gpt2'] / lens
        if stats is None:
            stats = {'gen_pt_mean': df['gen_pt'].mean(), 'gen_pt_std': max(df['gen_pt'].std(), 1e-8),
                     'typ_pt_mean': df['typ_pt'].mean(), 'typ_pt_std': max(df['typ_pt'].std(), 1e-8)}
        df['gen_pt_z'] = (df['gen_pt'] - stats['gen_pt_mean']) / stats['gen_pt_std']
        df['typ_pt_z'] = (df['typ_pt'] - stats['typ_pt_mean']) / stats['typ_pt_std']
        return df, stats
    def get_formula(self):
        return "val_score ~ gen_pt_z + typ_pt_z", "~typ_pt_z"
    def predict_fixed(self, fe, eval_df, stats):
        return (fe['Intercept'] + fe['gen_pt_z'] * eval_df['gen_pt_z'].values +
                fe['typ_pt_z'] * eval_df['typ_pt_z'].values)


class ME_SqrtNorm(MixedEffectsModel):
    def __init__(self):
        super().__init__("ME_sqrt_norm", "val ~ gen_sl_z + typ_sl_z + (typ_sl_z | prompt)", "PMI_div_len05")
    def prepare_features(self, df, stats=None):
        df = df.copy()
        sqrt_lens = np.sqrt(np.maximum(df['num_tokens'].values, 1).astype(float))
        df['gen_sl'] = df['gen_score'] / sqrt_lens
        df['typ_sl'] = df['typicality_gpt2'] / sqrt_lens
        if stats is None:
            stats = {'gen_sl_mean': df['gen_sl'].mean(), 'gen_sl_std': max(df['gen_sl'].std(), 1e-8),
                     'typ_sl_mean': df['typ_sl'].mean(), 'typ_sl_std': max(df['typ_sl'].std(), 1e-8)}
        df['gen_sl_z'] = (df['gen_sl'] - stats['gen_sl_mean']) / stats['gen_sl_std']
        df['typ_sl_z'] = (df['typ_sl'] - stats['typ_sl_mean']) / stats['typ_sl_std']
        return df, stats
    def get_formula(self):
        return "val_score ~ gen_sl_z + typ_sl_z", "~typ_sl_z"
    def predict_fixed(self, fe, eval_df, stats):
        return (fe['Intercept'] + fe['gen_sl_z'] * eval_df['gen_sl_z'].values +
                fe['typ_sl_z'] * eval_df['typ_sl_z'].values)


# =========================================================================
# Full-data model fitting
# =========================================================================

def fit_all_models_full(df, models):
    """Fit each model on all data and report fixed-effect coefficients."""
    n_tasks = df['prompt_id'].nunique()
    print(f"\n{'=' * 80}")
    print(f"FULL-DATA MODEL FITS (all {n_tasks} ifeval prompts)")
    print(f"{'=' * 80}")

    full_results = {}
    for model in models:
        print(f"\n{'---' * 23}")
        print(f"Model: {model.name}")
        print(f"  Formula: {model.description}")
        print(f"  Analogous to: {model.analogous_to}")

        try:
            df_copy = df.copy()
            df_copy, stats = model.prepare_features(df_copy, stats=None)
            formula, re_formula = model.get_formula()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mlm = smf.mixedlm(formula, data=df_copy, groups=df_copy["prompt_id"],
                                  re_formula=re_formula)
                result = mlm.fit(method='lbfgs', maxiter=500)

            fe = result.fe_params
            print(f"  Fixed effects:")
            for k, v in fe.items():
                print(f"    {k:>15}: {v:>10.4f}")

            if 'gen_z' in fe and 'typ_z' in fe:
                b_gen = fe['gen_z'] / stats.get('gen_std', 1)
                b_typ = fe['typ_z'] / stats.get('typ_std', 1)
                if abs(b_gen) > 1e-10:
                    implied_b = -b_typ / b_gen
                    print(f"  Implied b (typ correction): {implied_b:.4f}")

            print(f"  Log-likelihood: {result.llf:.2f}")
            n_fe = len(fe)
            re_cov = result.cov_re
            if hasattr(re_cov, 'shape'):
                n_re_params = re_cov.shape[0] * (re_cov.shape[0] + 1) // 2
            else:
                n_re_params = 1
            n_params = n_fe + n_re_params + 1
            bic = -2 * result.llf + n_params * np.log(len(df_copy))
            print(f"  BIC: {bic:.2f}  (n_params={n_params})")

            full_results[model.name] = {
                'fe': dict(fe), 'stats': stats, 'llf': result.llf,
                'bic': bic, 'n_params': n_params, 'converged': True}
        except Exception as e:
            print(f"  FAILED: {e}")
            full_results[model.name] = {'converged': False, 'error': str(e)}

    return full_results


# =========================================================================
# LOO evaluation
# =========================================================================

def loo_evaluation(df, models):
    """LOO evaluation: for each ifeval prompt, refit on remaining prompts, predict on held-out."""
    prompts = sorted(df['prompt_id'].unique(), key=lambda x: int(x.split('_')[1]))
    n_prompts = len(prompts)

    print(f"\n{'=' * 80}")
    print(f"TRUE LOO EVALUATION ({n_prompts} ifeval prompts)")
    print(f"{'=' * 80}")

    all_results = []
    model_failures = {m.name: 0 for m in models}
    t0 = time.time()

    for i, eval_pid in enumerate(prompts):
        train_df = df[df['prompt_id'] != eval_pid].copy()
        eval_df = df[df['prompt_id'] == eval_pid].copy()

        eval_gen = eval_df['gen_score'].values
        eval_typ = eval_df['typicality_gpt2'].values
        eval_val = eval_df['val_score'].values
        eval_lab = eval_df['correct'].values

        # PMI baselines
        for method, b in [('raw', 0.0), ('PMI', 1.0), ('PMI_1.5', 1.5),
                          ('PMI2', 2.0), ('PMI_3', 3.0)]:
            corrected = eval_gen - b * eval_typ
            m = compute_metrics(corrected, eval_val, eval_lab)
            all_results.append({
                'eval_prompt': eval_pid, 'method': method,
                'corr': m['corr'], 'gen_roc': m['gen_roc'], 'n_params': 0})

        # Length-normalized PMI baselines
        eval_len = np.maximum(eval_df['num_tokens'].values, 1).astype(float)
        pmi = eval_gen - eval_typ
        for method, divisor in [('PMI_div_len', eval_len),
                                ('PMI_div_len05', np.sqrt(eval_len))]:
            corrected = pmi / divisor
            m = compute_metrics(corrected, eval_val, eval_lab)
            all_results.append({
                'eval_prompt': eval_pid, 'method': method,
                'corr': m['corr'], 'gen_roc': m['gen_roc'], 'n_params': 0})

        # Mixed-effects models
        for model in models:
            try:
                corrected, _ = model.fit_and_predict(train_df, eval_df)
                m = compute_metrics(corrected, eval_val, eval_lab)
                all_results.append({
                    'eval_prompt': eval_pid, 'method': model.name,
                    'corr': m['corr'], 'gen_roc': m['gen_roc'],
                    'n_params': len(model.get_formula()[0].split('+')) - 1})
            except Exception as e:
                model_failures[model.name] += 1
                all_results.append({
                    'eval_prompt': eval_pid, 'method': model.name,
                    'corr': np.nan, 'gen_roc': np.nan, 'n_params': 0})

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 1
        eta = (n_prompts - i - 1) / rate if rate > 0 else 0
        print(f"  Prompt {i+1}/{n_prompts} ({eval_pid}) done "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    for mname, nfail in model_failures.items():
        if nfail > 0:
            print(f"  WARNING: {mname} failed on {nfail}/{n_prompts} prompts")

    df_results = pd.DataFrame(all_results)

    baselines = ['raw', 'PMI', 'PMI_1.5', 'PMI2', 'PMI_3', 'PMI_div_len', 'PMI_div_len05']
    me_methods = [m.name for m in models]
    all_methods = baselines + me_methods

    print(f"\n{'Method':<20} {'Corr':>8} {'GenROC':>8} {'n_valid':>8}")
    print("-" * 50)
    for method in all_methods:
        sub = df_results[df_results['method'] == method]
        valid = sub.dropna(subset=['corr'])
        corr_mean = valid['corr'].mean() * 100
        roc_mean = valid['gen_roc'].mean() * 100
        print(f"{method:<20} {corr_mean:>7.1f}% {roc_mean:>7.1f}% {len(valid):>6}/{len(sub)}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    return df_results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description='Mixed-effects typicality analysis for ifeval')
    parser.add_argument('--outputs-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'outputs'))
    parser.add_argument('--skip-full', action='store_true')
    parser.add_argument('--save-csv', type=str, default=None)
    parser.add_argument('--self-typicality', action='store_true',
                        help='Use self- prefixed CSVs')
    args = parser.parse_args()

    df = load_all_ifeval(args.outputs_dir, self_typicality=args.self_typicality)

    models = [ME_Basic(), ME_Intercept(), ME_Len(), ME_LenRE(),
              ME_PolyTyp(), ME_PolyFull(), ME_PerToken(), ME_SqrtNorm()]

    print(f"\nModels to evaluate ({len(models)}):")
    for m in models:
        print(f"  {m.name:<18} {m.description}")
        print(f"  {'':18} Analogous to: {m.analogous_to}")

    if not args.skip_full:
        fit_all_models_full(df, models)

    df_loo = loo_evaluation(df, models)

    suffix = '_self' if args.self_typicality else ''
    if args.save_csv:
        csv_path = args.save_csv
    else:
        csv_path = str(Path(args.outputs_dir) / f'mixed_effects_ifeval{suffix}.csv')
    df_loo.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # LaTeX table
    print(f"\n{'=' * 80}")
    print("LATEX TABLE")
    print(f"{'=' * 80}")

    baselines = ['raw', 'PMI', 'PMI_1.5', 'PMI2', 'PMI_3', 'PMI_div_len', 'PMI_div_len05']
    me_methods = [m.name for m in models]
    all_methods = baselines + me_methods

    summary = []
    for method in all_methods:
        sub = df_loo[df_loo['method'] == method].dropna(subset=['corr'])
        summary.append({'method': method,
                        'corr': sub['corr'].mean() * 100,
                        'gen_roc': sub['gen_roc'].mean() * 100})
    summary.sort(key=lambda x: -x['gen_roc'])

    print("\\begin{tabular}{lrr}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Corr} & \\textbf{GenROC} \\\\")
    print("\\midrule")
    for s in summary:
        name = s['method'].replace('_', '\\_')
        print(f"\\texttt{{{name}}} & {s['corr']:.1f}\\% & {s['gen_roc']:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == '__main__':
    main()
