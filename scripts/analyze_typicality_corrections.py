#!/usr/bin/env python3
"""
Analyze different typicality correction methods for generator scores.

Loads base-model evaluation CSVs, applies various corrections to gen_score,
and measures G-V correlation and Gen ROC for each method.

Usage:
    source ~/venvs/venv_lexcons/bin/activate
    python analyze_typicality_corrections.py --task plausibleqa
    python analyze_typicality_corrections.py --task plausibleqa --compute-llm-typicality

Methods:
    0  Raw:           gen_score
    1  PMI:           gen_score - typicality
    2  PMI^2:         gen_score - 2*typicality
    3  Resid(typ):    residual from gen_score ~ typicality
    4  Resid(typ+len):residual from gen_score ~ typicality + length
    5  Resid(typ+len+int): residual from gen_score ~ typicality + length + typ*len
    6  PMI/len:       (gen_score - typicality) / length
    7  Lenorm-then-resid: gen_score/len residualized on typicality/len

Each method is run with two typicality sources:
    T1: P_GPT2(y | null)      — already in CSVs
    T2: P_LLM(y | null)       — computed from gemma-2-2b (--compute-llm-typicality)
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import matplotlib.pyplot as plt

# ===========================================================================
# Metrics (matching eval.py / dashboard_viz_refactor.py)
# ===========================================================================

def compute_metrics(gen_scores, val_scores, labels):
    """Compute G-V correlation and Gen/Val ROC-AUC.

    Returns dict with: corr, corr_pos, corr_neg, gen_roc, val_roc, val_acc
    """
    gen = np.asarray(gen_scores, dtype=float)
    val = np.asarray(val_scores, dtype=float)
    lab = np.asarray(labels, dtype=int)

    valid = ~(np.isnan(gen) | np.isnan(val) | np.isinf(gen) | np.isinf(val))
    if valid.sum() < 4:
        return {k: np.nan for k in ['corr', 'corr_pos', 'corr_neg', 'gen_roc', 'val_roc', 'val_acc']}

    gen, val, lab = gen[valid], val[valid], lab[valid]
    pos = lab == 1
    neg = lab == 0

    def safe_pearsonr(x, y):
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return np.nan
        return pearsonr(x, y)[0]

    def safe_roc(labels, scores):
        if len(np.unique(labels)) < 2:
            return np.nan
        try:
            return roc_auc_score(labels, scores)
        except Exception:
            return np.nan

    corr = safe_pearsonr(gen, val)
    corr_pos = safe_pearsonr(gen[pos], val[pos]) if pos.sum() > 2 else np.nan
    corr_neg = safe_pearsonr(gen[neg], val[neg]) if neg.sum() > 2 else np.nan
    gen_roc = safe_roc(lab, gen)
    val_roc = safe_roc(lab, val)

    # Val accuracy at threshold=0 (log-odds)
    preds = (val > 0).astype(int)
    val_acc = accuracy_score(lab, preds)

    return dict(corr=corr, corr_pos=corr_pos, corr_neg=corr_neg,
                gen_roc=gen_roc, val_roc=val_roc, val_acc=val_acc)


# ===========================================================================
# Correction methods
# ===========================================================================

def _find_optimal_b(fit_gen, fit_typ, fit_val, b_range=(-5.0, 5.0)):
    """Find b that maximizes corr(gen - b*typ, val) on the fit data.

    Uses scipy.optimize.minimize_scalar (bounded).
    Returns optimal b value.
    """
    from scipy.optimize import minimize_scalar

    def neg_corr(b):
        corrected = fit_gen - b * fit_typ
        if np.std(corrected) < 1e-12:
            return 0.0  # degenerate
        return -pearsonr(corrected, fit_val)[0]

    result = minimize_scalar(neg_corr, bounds=b_range, method='bounded')
    return result.x


def _find_optimal_b_multi(fit_gen, fit_features, fit_val, n_features,
                           b_range=(-5.0, 5.0)):
    """Find b1, b2, ... that maximize corr(gen - b1*f1 - b2*f2 - ..., val).

    Uses L-BFGS-B with bounds for speed (much faster than Nelder-Mead).
    fit_features: np.array of shape (n, n_features)
    Returns optimal b array.
    """
    from scipy.optimize import minimize

    def neg_corr(bs):
        corrected = fit_gen - fit_features @ bs
        s = np.std(corrected)
        if s < 1e-12:
            return 0.0
        return -pearsonr(corrected, fit_val)[0]

    # Initialize with PMI-like values (1.0 for typ, 0 for others)
    x0 = np.zeros(n_features)
    x0[0] = 1.0
    bounds = [(b_range[0], b_range[1])] * n_features
    result = minimize(neg_corr, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-8})
    return result.x


def apply_corrections(gen_scores, typicality, lengths, val_scores=None,
                      labels=None, fit_data=None):
    """Apply all correction methods.

    Args:
        gen_scores: np.array of raw generator log-probs
        typicality: np.array of typicality scores (log P(y|null))
        lengths: np.array of completion lengths in tokens
        val_scores: np.array of validator log-odds (needed for fitted methods)
        labels: np.array of binary labels (needed for AUC-optimized methods)
        fit_data: optional dict with 'gen', 'typ', 'len', 'val', 'labels'
                  arrays for fitting. If None, fit on the same data.

    Returns:
        (results_dict, coefficients_dict)
    """
    from scipy.optimize import minimize_scalar, minimize
    from scipy.stats import spearmanr

    gen = np.asarray(gen_scores, dtype=float)
    typ = np.asarray(typicality, dtype=float)
    leng = np.asarray(lengths, dtype=float)

    # Data for fitting (train set or same data)
    if fit_data is not None:
        fit_gen = np.asarray(fit_data['gen'], dtype=float)
        fit_typ = np.asarray(fit_data['typ'], dtype=float)
        fit_len = np.asarray(fit_data['len'], dtype=float)
        fit_val = np.asarray(fit_data['val'], dtype=float)
        fit_labels = np.asarray(fit_data['labels'], dtype=float) if 'labels' in fit_data else None
    else:
        fit_gen, fit_typ, fit_len = gen, typ, leng
        fit_val = np.asarray(val_scores, dtype=float) if val_scores is not None else None
        fit_labels = np.asarray(labels, dtype=float) if labels is not None else None

    results = {}
    coeffs = {}

    # ===================================================================
    # PARAMETER-FREE METHODS (no fitting, can't overfit)
    # ===================================================================

    # 0. Raw
    results['raw'] = gen.copy()

    # 1. PMI: gen - b*typ with fixed b values
    results['PMI'] = gen - typ                          # b=1.0 (standard)
    results['PMI2'] = gen - 2 * typ                     # b=2.0
    results['PMI_0.5'] = gen - 0.5 * typ                # b=0.5
    results['PMI_1.5'] = gen - 1.5 * typ                # b=1.5
    results['PMI_3'] = gen - 3.0 * typ                  # b=3.0

    # 2. PMI / length
    results['PMI_div_len'] = (gen - typ) / np.maximum(leng, 1)

    # 3. Per-token PMI: (gen - typ) / len^a for fixed a values
    results['PMI_div_len05'] = (gen - typ) / np.maximum(leng, 1)**0.5  # sqrt(len)

    # ===================================================================
    # FITTED METHODS (need val_scores for optimization)
    # ===================================================================
    if fit_val is not None:
        # --- A. Optimize Pearson corr(gen - b*typ, val) ---
        b_typ = _find_optimal_b(fit_gen, fit_typ, fit_val)
        results['opt_pearson_typ'] = gen - b_typ * typ
        coeffs['opt_pearson_typ'] = b_typ

        # --- B. Bayesian-regularized: max corr - lambda*(b-1)^2 ---
        # Shrinks b toward PMI prior (b=1). Lambda chosen empirically.
        for lam in [0.1, 0.5, 2.0]:
            def neg_corr_reg(b, _lam=lam):
                corrected = fit_gen - b * fit_typ
                if np.std(corrected) < 1e-12:
                    return 0.0
                return -pearsonr(corrected, fit_val)[0] + _lam * (b - 1.0)**2
            res = minimize_scalar(neg_corr_reg, bounds=(-5, 5), method='bounded')
            key = f'reg_typ_lam{lam}'
            results[key] = gen - res.x * typ
            coeffs[key] = res.x

        # --- C. Optimize Spearman corr (rank-based, better for small n) ---
        def neg_spearman(b):
            corrected = fit_gen - b * fit_typ
            if np.std(corrected) < 1e-12:
                return 0.0
            return -spearmanr(corrected, fit_val)[0]
        res = minimize_scalar(neg_spearman, bounds=(-5, 5), method='bounded')
        results['opt_spearman_typ'] = gen - res.x * typ
        coeffs['opt_spearman_typ'] = res.x

        # --- D. Optimize AUC directly (needs labels) ---
        if fit_labels is not None and len(np.unique(fit_labels)) == 2:
            def neg_auc(b):
                corrected = fit_gen - b * fit_typ
                try:
                    return -roc_auc_score(fit_labels, corrected)
                except ValueError:
                    return 0.0
            res = minimize_scalar(neg_auc, bounds=(-5, 5), method='bounded')
            results['opt_auc_typ'] = gen - res.x * typ
            coeffs['opt_auc_typ'] = res.x

        # --- E. Optimal b for typ + len (Pearson) ---
        fit_features_2 = np.column_stack([fit_typ, fit_len])
        eval_features_2 = np.column_stack([typ, leng])
        bs_2 = _find_optimal_b_multi(fit_gen, fit_features_2, fit_val, 2)
        results['opt_typ_len'] = gen - eval_features_2 @ bs_2
        coeffs['opt_typ_len'] = bs_2

        # --- F. Polynomial degree 2: gen ~ typ + typ^2, maximize corr ---
        fit_features_poly = np.column_stack([fit_typ, fit_typ**2])
        eval_features_poly = np.column_stack([typ, typ**2])
        bs_poly = _find_optimal_b_multi(fit_gen, fit_features_poly, fit_val, 2)
        results['opt_poly2_typ'] = gen - eval_features_poly @ bs_poly
        coeffs['opt_poly2_typ'] = bs_poly

        # --- G. Full polynomial: gen ~ typ + len + typ^2 + len^2 + typ*len ---
        fit_features_fp = np.column_stack([
            fit_typ, fit_len, fit_typ**2, fit_len**2, fit_typ * fit_len])
        eval_features_fp = np.column_stack([
            typ, leng, typ**2, leng**2, typ * leng])
        bs_fp = _find_optimal_b_multi(fit_gen, fit_features_fp, fit_val, 5)
        results['opt_poly2_full'] = gen - eval_features_fp @ bs_fp
        coeffs['opt_poly2_full'] = bs_fp

        # --- H. Logistic regression: P(correct | gen, typ, len) ---
        if fit_labels is not None and len(np.unique(fit_labels)) == 2:
            from sklearn.linear_model import LogisticRegression
            X_fit = np.column_stack([fit_gen, fit_typ, fit_len])
            X_eval = np.column_stack([gen, typ, leng])
            try:
                lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
                lr.fit(X_fit, fit_labels)
                # Use log-odds as corrected score
                results['logistic'] = lr.decision_function(X_eval)
                coeffs['logistic'] = {'gen': lr.coef_[0][0], 'typ': lr.coef_[0][1],
                                       'len': lr.coef_[0][2], 'intercept': lr.intercept_[0]}
            except Exception:
                pass

        # --- I. Length-normalized then optimize ---
        gen_pt = gen / np.maximum(leng, 1)
        typ_pt = typ / np.maximum(leng, 1)
        fit_gen_pt = fit_gen / np.maximum(fit_len, 1)
        fit_typ_pt = fit_typ / np.maximum(fit_len, 1)
        b_lenorm = _find_optimal_b(fit_gen_pt, fit_typ_pt, fit_val)
        results['opt_lenorm_typ'] = gen_pt - b_lenorm * typ_pt
        coeffs['opt_lenorm_typ'] = b_lenorm

    return results, coeffs


def get_fitted_coefficients(gen_scores, typicality, lengths, val_scores):
    """Get optimized coefficients (maximizing corr with val) for interpretability."""
    gen = np.asarray(gen_scores, dtype=float)
    typ = np.asarray(typicality, dtype=float)
    leng = np.asarray(lengths, dtype=float)
    val = np.asarray(val_scores, dtype=float)

    coeffs = {}

    b_typ = _find_optimal_b(gen, typ, val)
    coeffs['opt_typ'] = {'b_typ': b_typ}

    features_2 = np.column_stack([typ, leng])
    bs_2 = _find_optimal_b_multi(gen, features_2, val, 2)
    coeffs['opt_typ_len'] = {'b_typ': bs_2[0], 'b_len': bs_2[1]}

    features_3 = np.column_stack([typ, leng, typ * leng])
    bs_3 = _find_optimal_b_multi(gen, features_3, val, 3)
    coeffs['opt_typ_len_int'] = {'b_typ': bs_3[0], 'b_len': bs_3[1], 'b_typ*len': bs_3[2]}

    gen_pt = gen / np.maximum(leng, 1)
    typ_pt = typ / np.maximum(leng, 1)
    b_lenorm = _find_optimal_b(gen_pt, typ_pt, val)
    coeffs['opt_lenorm_typ'] = {'b_typ_per_tok': b_lenorm}

    return coeffs


# ===========================================================================
# File discovery
# ===========================================================================

def discover_plausibleqa_files(outputs_dir, self_typicality=False):
    """Find all base-model plausibleqa score CSVs.

    Returns list of (prompt_id, filepath) tuples.
    """
    import re
    prefix = 'self-' if self_typicality else ''
    pattern = os.path.join(outputs_dir, f'scores_{prefix}v6-google_gemma-2-2b_plausibleqa-*_test_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))

    # Deduplicate: keep latest file per prompt_id (files are sorted, so last wins)
    seen = {}
    for f in files:
        match = re.search(r'plausibleqa-(\w+_\d+)_test', f)
        if match:
            prompt_id = match.group(1)
            seen[prompt_id] = f

    return sorted(seen.items())


HYPERNYM_TASKS = ['bananas', 'bazookas', 'cabinets', 'cars', 'chairs', 'crows', 'diapers', 'dogs']


def discover_ifeval_files(outputs_dir, self_typicality=False):
    """Find all base-model ifeval score CSVs.

    Filename convention: scores_{prefix}gemma-2-2b_ifeval-prompt_N_test_log-odds_evaltc_*.csv
    Note: ifeval uses 'gemma-2-2b' (not 'v6-google_gemma-2-2b').

    Returns list of (prompt_id, filepath) tuples sorted by prompt number.
    """
    import re
    prefix = 'self-' if self_typicality else ''
    pattern = os.path.join(outputs_dir,
        f'scores_{prefix}gemma-2-2b_ifeval-prompt_*_test_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))

    # Deduplicate: keep latest file per prompt number (files are sorted, so last wins)
    seen = {}
    for f in files:
        match = re.search(r'ifeval-prompt_(\d+)_test', f)
        if match:
            prompt_num = int(match.group(1))
            prompt_id = f'prompt_{prompt_num}'
            seen[prompt_id] = f

    # Sort by prompt number
    return sorted(seen.items(), key=lambda x: int(x[0].split('_')[1]))


def discover_hypernym_files(outputs_dir, split='test', self_typicality=False):
    """Find all base-model hypernym score CSVs (v6, evaltc).

    Returns list of (noun, filepath) tuples.
    Only includes the 8 standard tasks: bananas, bazookas, cabinets, cars,
    chairs, crows, diapers, dogs.
    """
    import re
    prefix = 'self-' if self_typicality else ''
    pattern = os.path.join(outputs_dir,
        f'scores_{prefix}v6-google_gemma-2-2b_hypernym-*_{split}_v2_log-odds_evaltc_*.csv')
    files = sorted(glob.glob(pattern))

    # Deduplicate: keep latest file per noun (files are sorted, so last wins)
    seen = {}
    for f in files:
        match = re.search(r'hypernym-([a-zA-Z]+)_' + split, f)
        if match:
            noun = match.group(1)
            if noun in HYPERNYM_TASKS:
                seen[noun] = f

    return sorted(seen.items())


# ===========================================================================
# Load and extract data from CSVs
# ===========================================================================

def load_plausibleqa_csv(filepath):
    """Load a plausibleqa CSV and extract gen_score, typicality_gpt2, length, val_score, label."""
    df = pd.read_csv(filepath)

    gen_score = df['gen_score'].values
    gen_score_typcorr = df['gen_score_typcorr'].values
    # When loaded from "self" CSVs, this is log P_model(answer) not GPT-2
    typicality_gpt2 = gen_score - gen_score_typcorr
    lengths = df['num_tokens'].values.astype(float)
    val_score = df['val_score'].values

    # Labels
    label_col = df['gpt4_ground_truth']
    if label_col.dtype == object:
        labels = label_col.str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int).values
    else:
        labels = label_col.astype(int).values

    # Completion text (for computing LLM typicality)
    completions = df['answer'].values

    return dict(
        gen_score=gen_score,
        typicality_gpt2=typicality_gpt2,
        lengths=lengths,
        val_score=val_score,
        labels=labels,
        completions=completions,
        question=df['question'].iloc[0] if 'question' in df.columns else '',
        df=df,
    )


def load_hypernym_csv(filepath):
    """Load a hypernym CSV and extract gen_score, typicality_gpt2, length, val_score, label.

    Hypernym CSVs have columns: noun1, noun2, num_tokens, strategy, gpt4_ground_truth,
    gen_score, gen_score_typcorr, gen_score_lenorm, gen_score_typcorr_lenorm,
    val_score, gen_prompt, val_prompt
    """
    df = pd.read_csv(filepath)

    gen_score = df['gen_score'].values
    gen_score_typcorr = df['gen_score_typcorr'].values
    typicality_gpt2 = gen_score - gen_score_typcorr
    lengths = df['num_tokens'].values.astype(float)
    val_score = df['val_score'].values

    # Labels
    label_col = df['gpt4_ground_truth']
    if label_col.dtype == object:
        labels = label_col.str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int).values
    else:
        labels = label_col.astype(int).values

    # Completion text — use noun2 (the hypernym candidate)
    completions = df['noun2'].values

    return dict(
        gen_score=gen_score,
        typicality_gpt2=typicality_gpt2,
        lengths=lengths,
        val_score=val_score,
        labels=labels,
        completions=completions,
        question=df['gen_prompt'].iloc[0] if 'gen_prompt' in df.columns else '',
        df=df,
    )


def load_ifeval_csv(filepath):
    """Load an ifeval CSV and extract gen_score, typicality, length, val_score, label.

    IFEval CSVs have columns: prompt, response, num_tokens, correct,
    val_prompt, val_score, gen_score, gen_score_typcorr, gen_score_lenorm,
    gen_score_typcorr_lenorm
    """
    df = pd.read_csv(filepath)

    gen_score = df['gen_score'].values
    gen_score_typcorr = df['gen_score_typcorr'].values
    typicality_gpt2 = gen_score - gen_score_typcorr
    lengths = df['num_tokens'].values.astype(float)
    val_score = df['val_score'].values

    # Labels — 'correct' column with Yes/No
    label_col = df['correct']
    if label_col.dtype == object:
        labels = label_col.str.strip().str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int).values
    else:
        labels = label_col.astype(int).values

    # Completion text — use 'response' column
    completions = df['response'].values

    return dict(
        gen_score=gen_score,
        typicality_gpt2=typicality_gpt2,
        lengths=lengths,
        val_score=val_score,
        labels=labels,
        completions=completions,
        question=df['prompt'].iloc[0] if 'prompt' in df.columns else '',
        df=df,
    )


# ===========================================================================
# LLM-self typicality computation
# ===========================================================================

def compute_llm_typicality(completions, model_name='google/gemma-2-2b', cache_file=None):
    """Compute P_LLM(completion | null) for each completion using the scoring model itself.

    Args:
        completions: list of completion strings
        model_name: HuggingFace model name
        cache_file: optional path to cache results

    Returns:
        np.array of log-probabilities
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import math

    # Check cache
    if cache_file and os.path.exists(cache_file):
        print(f"  Loading cached LLM typicality from {cache_file}")
        cached = pd.read_csv(cache_file)
        # Build lookup
        lookup = dict(zip(cached['completion'], cached['typicality']))
        scores = []
        for c in completions:
            if c in lookup:
                scores.append(lookup[c])
            else:
                scores.append(np.nan)  # Will need to compute missing
        if not any(np.isnan(s) for s in scores):
            return np.array(scores)
        print(f"  Cache miss for {sum(np.isnan(s) for s in scores)} completions, computing...")

    print(f"  Loading model {model_name} for typicality computation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Get unique completions to avoid recomputation
    unique_completions = list(set(completions))
    print(f"  Computing typicality for {len(unique_completions)} unique completions...")

    typ_lookup = {}
    for comp in unique_completions:
        with torch.no_grad():
            # Tokenize the completion
            input_ids = tokenizer.encode(comp, add_special_tokens=False)

            if len(input_ids) == 0:
                typ_lookup[comp] = float('-inf')
                continue

            # Add BOS token as "null" context
            bos_ids = tokenizer.encode("", add_special_tokens=True)
            full_ids = bos_ids + input_ids

            input_tensor = torch.tensor([full_ids]).to(device)
            outputs = model(input_tensor)
            logits = outputs.logits[0]  # [seq_len, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)

            # Sum log-probs for each completion token
            # logits[t] predicts token t+1
            total_log_prob = 0.0
            for i, tok_id in enumerate(input_ids):
                pos = len(bos_ids) - 1 + i  # position predicting this token
                if pos < log_probs.shape[0]:
                    total_log_prob += log_probs[pos, tok_id].item()

            typ_lookup[comp] = total_log_prob

    scores = np.array([typ_lookup[c] for c in completions])

    # Cache results
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        cache_df = pd.DataFrame({
            'completion': list(typ_lookup.keys()),
            'typicality': list(typ_lookup.values())
        })
        cache_df.to_csv(cache_file, index=False)
        print(f"  Cached {len(typ_lookup)} typicality scores to {cache_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return scores


# ===========================================================================
# In-sample diagnostic analysis
# ===========================================================================

def run_insample_diagnostic(outputs_dir, self_typicality=False):
    """For each prompt, fit AND eval on same data. Print per-prompt table.

    Fitted methods now maximize corr(corrected, val) instead of MSE.
    """
    files = discover_plausibleqa_files(outputs_dir, self_typicality=self_typicality)
    if not files:
        print("No plausibleqa files found!")
        return

    print(f"Found {len(files)} plausibleqa prompt files")
    print(f"\n{'='*90}")
    print("IN-SAMPLE DIAGNOSTIC: fit and eval on same prompt")
    print("Fitted methods maximize corr(gen - b*typ, val_score)")
    print(f"{'='*90}")

    # Discover methods from first prompt
    data0 = load_plausibleqa_csv(files[0][1])
    corrected0, _ = apply_corrections(
        data0['gen_score'], data0['typicality_gpt2'], data0['lengths'],
        val_scores=data0['val_score'], labels=data0['labels'], fit_data=None)
    METHOD_ORDER = list(corrected0.keys())

    # Header for CORR table
    print(f"\n{'Prompt':>8} {'N':>3} | ", end='')
    for m in METHOD_ORDER:
        print(f' {m[:12]:>12}', end='')
    print()
    print('-' * (16 + 13*len(METHOD_ORDER)))

    all_rows = []

    for prompt_id, filepath in files:
        data = load_plausibleqa_csv(filepath)
        gen = data['gen_score']
        typ = data['typicality_gpt2']
        leng = data['lengths']
        val = data['val_score']
        lab = data['labels']

        corrected, fitted_coeffs = apply_corrections(
            gen, typ, leng, val_scores=val, labels=lab, fit_data=None)

        row = {'prompt': prompt_id, 'n': len(gen)}
        for k, v in fitted_coeffs.items():
            if isinstance(v, (int, float)):
                row[f'coeff_{k}'] = v

        print(f"{prompt_id:>8} {len(gen):>3} | ", end='')
        for m in METHOD_ORDER:
            if m in corrected:
                metrics = compute_metrics(corrected[m], val, lab)
                corr_val = metrics['corr'] * 100
                print(f' {corr_val:>9.1f}%', end='')
                row[f'{m}_corr'] = metrics['corr']
                row[f'{m}_roc'] = metrics['gen_roc']
            else:
                print(f' {"N/A":>10}', end='')
                row[f'{m}_corr'] = np.nan
                row[f'{m}_roc'] = np.nan

        print()
        all_rows.append(row)

    # Summary
    df = pd.DataFrame(all_rows)
    print('-' * (16 + 13*len(METHOD_ORDER)))
    print(f"{'MEAN':>8} {'':>3} | ", end='')
    for m in METHOD_ORDER:
        col = f'{m}_corr'
        if col in df.columns:
            print(f' {df[col].mean()*100:>11.1f}%', end='')
        else:
            print(f' {"N/A":>12}', end='')
    print()

    # GenROC table
    print(f"\n{'Prompt':>8} {'N':>3} | ", end='')
    for m in METHOD_ORDER:
        print(f' {m[:12]:>12}', end='')
    print('   (Gen ROC)')
    print('-' * (16 + 13*len(METHOD_ORDER)))

    for _, row in df.iterrows():
        print(f"{row['prompt']:>8} {int(row['n']):>3} | ", end='')
        for m in METHOD_ORDER:
            col = f'{m}_roc'
            if col in df.columns and not np.isnan(row[col]):
                print(f' {row[col]*100:>9.1f}%', end='')
            else:
                print(f' {"N/A":>10}', end='')
        print()

    print('-' * 120)
    print(f"{'MEAN':>8} {'':>3} | ", end='')
    for m in METHOD_ORDER:
        col = f'{m}_roc'
        if col in df.columns:
            print(f' {df[col].mean()*100:>9.1f}%', end='')
        else:
            print(f' {"N/A":>10}', end='')
    print()

    # Diagnostic: first prompt detailed view
    print(f"\n{'='*70}")
    print("DIAGNOSTIC: Detailed look at first prompt")
    print(f"{'='*70}")
    pid0, fp0 = files[0]
    d = load_plausibleqa_csv(fp0)
    gen = d['gen_score']
    typ = d['typicality_gpt2']
    val = d['val_score']
    lab = d['labels']

    b_opt = _find_optimal_b(gen, typ, val)
    opt_corr = gen - b_opt * typ
    pmi = gen - typ

    print(f"Prompt {pid0}: {len(gen)} items")
    print(f"Optimal b (max corr with val): {b_opt:.4f}")
    print(f"  => opt_typ = gen - {b_opt:.4f} * typ")
    print(f"  => PMI     = gen - 1.0 * typ")
    print(f"\nCorr(opt_typ, val) = {pearsonr(opt_corr, val)[0]:.4f}")
    print(f"Corr(PMI, val)     = {pearsonr(pmi, val)[0]:.4f}")
    print(f"Corr(gen, typ)     = {pearsonr(gen, typ)[0]:.4f}")
    print(f"Corr(gen, val)     = {pearsonr(gen, val)[0]:.4f}")
    print(f"Corr(typ, val)     = {pearsonr(typ, val)[0]:.4f}")

    print(f"\nPer-item comparison (first prompt):")
    print(f"{'label':>5} {'gen':>8} {'typ':>8} {'val':>8} | {'opt_typ':>8} {'PMI':>8} {'diff':>8}")
    print('-' * 70)
    for i in range(len(gen)):
        print(f"{'YES' if lab[i] else 'no':>5} {gen[i]:>8.2f} {typ[i]:>8.2f} {val[i]:>8.2f} | "
              f"{opt_corr[i]:>8.2f} {pmi[i]:>8.2f} {pmi[i]-opt_corr[i]:>8.2f}")

    return df


# ===========================================================================
# Main analysis
# ===========================================================================

def run_plausibleqa_analysis(outputs_dir, compute_llm_typ=False, cache_dir=None,
                             save_plot=None, self_typicality=False):
    """Run leave-one-out typicality correction analysis on plausibleqa data.

    For each prompt i:
      - Fit regression on prompt i's data
      - Evaluate on all other 32 prompts using those coefficients
    Non-regression methods (raw, PMI, PMI^2, PMI/len) don't need fitting,
    so they produce the same result regardless of which prompt is "fit".
    """

    files = discover_plausibleqa_files(outputs_dir, self_typicality=self_typicality)
    if not files:
        print("No plausibleqa files found!")
        return

    print(f"Found {len(files)} plausibleqa prompt files")

    # Load all data
    all_data = {}
    all_completions = []
    for prompt_id, filepath in files:
        data = load_plausibleqa_csv(filepath)
        all_data[prompt_id] = data
        all_completions.extend(data['completions'].tolist())

    prompt_ids = list(all_data.keys())
    n_prompts = len(prompt_ids)

    # Summary stats
    all_gen = np.concatenate([d['gen_score'] for d in all_data.values()])
    all_typ = np.concatenate([d['typicality_gpt2'] for d in all_data.values()])
    all_len = np.concatenate([d['lengths'] for d in all_data.values()])
    all_labels = np.concatenate([d['labels'] for d in all_data.values()])

    print(f"\nTotal data: {len(all_gen)} examples across {n_prompts} prompts")
    print(f"  Label balance: {int(all_labels.sum())} pos / {int((1-all_labels).sum())} neg")
    print(f"  Gen score range: [{all_gen.min():.2f}, {all_gen.max():.2f}]")
    print(f"  GPT-2 typicality range: [{all_typ.min():.2f}, {all_typ.max():.2f}]")
    print(f"  Length range: [{all_len.min():.0f}, {all_len.max():.0f}]")

    # Typicality sources
    typ_sources = {'gpt2': {pid: d['typicality_gpt2'] for pid, d in all_data.items()}}

    # Compute LLM-self typicality if requested
    if compute_llm_typ:
        cache_file = os.path.join(cache_dir, 'llm_typicality_plausibleqa.csv') if cache_dir else None
        unique_completions = list(set(all_completions))
        llm_scores = compute_llm_typicality(unique_completions, cache_file=cache_file)
        llm_lookup = dict(zip(unique_completions, llm_scores))

        for pid, data in all_data.items():
            data['typicality_llm'] = np.array([llm_lookup[c] for c in data['completions']])

        typ_sources['llm'] = {pid: d['typicality_llm'] for pid, d in all_data.items()}

        pooled_typ_llm = np.concatenate([d['typicality_llm'] for d in all_data.values()])
        print(f"  LLM typicality range: [{pooled_typ_llm.min():.2f}, {pooled_typ_llm.max():.2f}]")

    # METHOD_ORDER will be set dynamically after first apply_corrections call
    METHOD_ORDER = None

    # ---------------------------------------------------------------
    # Leave-one-out: fit on prompt i, evaluate on all others
    # ---------------------------------------------------------------

    # Structure: loo_results[typ_name][method] = list of dicts, each with:
    #   fit_prompt, eval_prompt, corr, gen_roc, ...
    loo_results = {}

    for typ_name, typ_per_prompt in typ_sources.items():
        print(f"\n{'='*70}")
        print(f"LEAVE-ONE-OUT ANALYSIS — Typicality source: {typ_name}")
        print(f"{'='*70}")

        loo_results[typ_name] = defaultdict(list)

        for fit_idx, fit_pid in enumerate(prompt_ids):
            # Build fit data from this one prompt (including val + labels)
            fit_data = {
                'gen': all_data[fit_pid]['gen_score'],
                'typ': typ_per_prompt[fit_pid],
                'len': all_data[fit_pid]['lengths'],
                'val': all_data[fit_pid]['val_score'],
                'labels': all_data[fit_pid]['labels'],
            }

            # Print fitted coefficients for first fit prompt
            if fit_idx == 0:
                print(f"\nExample fitted coefficients (fit on prompt {fit_pid}):")
                corrected_ex, coeffs_ex = apply_corrections(
                    fit_data['gen'], fit_data['typ'], fit_data['len'],
                    val_scores=fit_data['val'], labels=fit_data['labels'],
                    fit_data=None)
                for method, val in coeffs_ex.items():
                    if isinstance(val, np.ndarray):
                        print(f"  {method}: {val}")
                    elif isinstance(val, dict):
                        parts = [f"{k}={v:.4f}" for k, v in val.items()]
                        print(f"  {method}: {', '.join(parts)}")
                    else:
                        print(f"  {method}: b={val:.4f}")

            # Evaluate on all OTHER prompts
            for eval_pid in prompt_ids:
                if eval_pid == fit_pid:
                    continue

                eval_data = all_data[eval_pid]
                eval_typ = typ_per_prompt[eval_pid]

                corrected, _ = apply_corrections(
                    eval_data['gen_score'], eval_typ, eval_data['lengths'],
                    val_scores=eval_data['val_score'],
                    labels=eval_data['labels'],
                    fit_data=fit_data
                )

                for method_name, corrected_gen in corrected.items():
                    metrics = compute_metrics(
                        corrected_gen, eval_data['val_score'], eval_data['labels']
                    )
                    metrics['fit_prompt'] = fit_pid
                    metrics['eval_prompt'] = eval_pid
                    loo_results[typ_name][method_name].append(metrics)

        # Set METHOD_ORDER from first iteration's results
        if METHOD_ORDER is None:
            METHOD_ORDER = list(loo_results[typ_name].keys())

        # Print summary table: mean across all (fit, eval) pairs
        print(f"\nAggregated LOO results (mean over {n_prompts}x{n_prompts-1} = "
              f"{n_prompts*(n_prompts-1)} fit-eval pairs):")
        print(f"{'Method':<22} {'Corr':>8} {'Corr+':>8} {'Corr-':>8} "
              f"{'GenROC':>8} {'ValROC':>8}")
        print("-" * 70)

        for method_name in METHOD_ORDER:
            entries = loo_results[typ_name][method_name]

            def agg(key):
                vals = [e[key] for e in entries if not np.isnan(e[key])]
                return np.mean(vals) if vals else np.nan

            def agg_se(key):
                vals = [e[key] for e in entries if not np.isnan(e[key])]
                return np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan

            print(f"{method_name:<22} {agg('corr')*100:>7.1f}% {agg('corr_pos')*100:>7.1f}% "
                  f"{agg('corr_neg')*100:>7.1f}% {agg('gen_roc')*100:>7.1f}% "
                  f"{agg('val_roc')*100:>7.1f}%")

    # ---------------------------------------------------------------
    # Visualization: violin plots
    # ---------------------------------------------------------------

    plot_path = save_plot or os.path.join(outputs_dir, 'typicality_correction_loo.png')
    _make_loo_violin_plots(loo_results, prompt_ids, METHOD_ORDER, plot_path)

    # Build results DataFrame for optional CSV save
    rows = []
    for typ_name in loo_results:
        for method_name in METHOD_ORDER:
            for e in loo_results[typ_name][method_name]:
                rows.append({
                    'typ_source': typ_name,
                    'method': method_name,
                    'fit_prompt': e['fit_prompt'],
                    'eval_prompt': e['eval_prompt'],
                    'corr': e['corr'],
                    'corr_pos': e['corr_pos'],
                    'corr_neg': e['corr_neg'],
                    'gen_roc': e['gen_roc'],
                    'val_roc': e['val_roc'],
                    'val_acc': e['val_acc'],
                })
    return pd.DataFrame(rows)


def _make_loo_violin_plots(loo_results, prompt_ids, method_order, save_path):
    """Create violin/bar plots of LOO results.

    Three figures:
      1. Grouped by method (x=method, violins show distribution across fit-eval pairs)
      2. Per eval-prompt breakdown (x=eval_prompt, grouped bars for top methods)
         — shows which prompts are easy/hard to correct
      3. Regression fit sensitivity (x=fit_prompt, violins for regression methods only)
         — shows how much the choice of fit prompt matters for methods that need fitting
    """
    # Non-fitted methods (fit prompt doesn't affect results)
    NO_FIT_METHODS = {'raw', 'PMI', 'PMI2', 'PMI_div_len'}
    # Fitted methods (fit prompt matters — these optimize corr with val)
    REGRESSION_METHODS = [m for m in method_order if m not in NO_FIT_METHODS]

    for typ_name in loo_results:
        # --- Figure 1: violin per method (Corr and Gen ROC side by side) ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax_idx, metric in enumerate(['corr', 'gen_roc']):
            ax = axes[ax_idx]
            data_for_violin = []
            labels = []
            for method in method_order:
                vals = [e[metric]*100 for e in loo_results[typ_name][method]
                        if not np.isnan(e[metric])]
                data_for_violin.append(vals)
                labels.append(method)

            parts = ax.violinplot(data_for_violin, showmeans=True, showextrema=True)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(f'{metric} (%)')
            ax.set_title(f'{metric.upper()} by Method (typ={typ_name})')
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'Leave-One-Out: Distribution across fit-eval pairs (typ={typ_name})',
                     fontsize=13, y=1.02)
        fig.tight_layout()
        path1 = save_path.replace('.png', f'_by_method_{typ_name}.png')
        fig.savefig(path1, dpi=150, bbox_inches='tight')
        print(f"\nSaved method violin plot: {path1}")
        plt.close(fig)

        # --- Figure 2: per eval-prompt breakdown for top methods ---
        # For each eval prompt, average metric across all fit prompts.
        # This shows which prompts are easy vs hard to correct.
        method_mean_roc = {}
        for method in method_order:
            vals = [e['gen_roc'] for e in loo_results[typ_name][method]
                    if not np.isnan(e['gen_roc'])]
            method_mean_roc[method] = np.mean(vals) if vals else 0
        top_methods = sorted(method_mean_roc, key=method_mean_roc.get, reverse=True)[:4]

        fig, axes = plt.subplots(2, 1, figsize=(18, 10))
        colors = plt.cm.Set2(np.linspace(0, 1, len(top_methods)))

        for ax_idx, metric in enumerate(['corr', 'gen_roc']):
            ax = axes[ax_idx]

            # Compute mean metric per eval prompt, per method
            eval_means = {}  # method -> {eval_prompt -> mean}
            for method in top_methods:
                by_eval = defaultdict(list)
                for e in loo_results[typ_name][method]:
                    if not np.isnan(e[metric]):
                        by_eval[e['eval_prompt']].append(e[metric] * 100)
                eval_means[method] = {pid: np.mean(vals) for pid, vals in by_eval.items()}

            # Sort eval prompts by the best method's score
            best_method = top_methods[0]
            sorted_evals = sorted(eval_means[best_method].keys(),
                                  key=lambda p: eval_means[best_method].get(p, 0),
                                  reverse=True)

            n_methods = len(top_methods)
            bar_width = 0.8 / n_methods
            x = np.arange(len(sorted_evals))

            for m_idx, method in enumerate(top_methods):
                offsets = x + (m_idx - n_methods/2 + 0.5) * bar_width
                heights = [eval_means[method].get(pid, 0) for pid in sorted_evals]
                ax.bar(offsets, heights, bar_width, label=method, color=colors[m_idx],
                       alpha=0.8, edgecolor='white', linewidth=0.5)

            short_labels = [str(pid)[:8] for pid in sorted_evals]
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
            ax.set_ylabel(f'{metric} (%)')
            ax.set_title(f'{metric.upper()} per eval prompt (mean across fit prompts)')
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.legend(fontsize=9, loc='lower left')
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'Per-Prompt Difficulty: Which prompts are easy/hard to correct? (typ={typ_name})',
                     fontsize=13)
        fig.tight_layout()
        path2 = save_path.replace('.png', f'_by_eval_prompt_{typ_name}.png')
        fig.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"Saved eval-prompt bar plot: {path2}")
        plt.close(fig)

        # --- Figure 3: scatter — fit-prompt sensitivity for REGRESSION methods ---
        # Each column of dots = one fit prompt, each dot = one of 32 eval prompts.
        # Non-regression methods shown as horizontal band for reference.
        reg_methods_with_data = [m for m in REGRESSION_METHODS
                                 if m in loo_results[typ_name]
                                 and len(loo_results[typ_name][m]) > 0]
        if not reg_methods_with_data:
            continue

        fig, axes = plt.subplots(len(reg_methods_with_data), 2,
                                 figsize=(20, 4*len(reg_methods_with_data)))
        if len(reg_methods_with_data) == 1:
            axes = axes.reshape(1, -1)

        # Precompute PMI reference stats for horizontal band
        pmi_stats = {}
        for metric in ['corr', 'gen_roc']:
            pmi_vals = [e[metric]*100 for e in loo_results[typ_name]['PMI']
                        if not np.isnan(e[metric])]
            pmi_stats[metric] = {'mean': np.mean(pmi_vals), 'std': np.std(pmi_vals)}

        for m_idx, method in enumerate(reg_methods_with_data):
            for ax_idx, metric in enumerate(['corr', 'gen_roc']):
                ax = axes[m_idx, ax_idx]

                # Group by fit_prompt
                by_fit = defaultdict(list)
                for e in loo_results[typ_name][method]:
                    if not np.isnan(e[metric]):
                        by_fit[e['fit_prompt']].append(e[metric] * 100)

                sorted_pids = sorted(by_fit.keys(),
                                     key=lambda p: np.mean(by_fit[p]), reverse=True)

                # Scatter: x = fit prompt index, y = metric value
                for x_idx, pid in enumerate(sorted_pids):
                    vals = by_fit[pid]
                    jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(vals))
                    ax.scatter([x_idx]*len(vals) + jitter, vals,
                               s=12, alpha=0.4, color='steelblue', edgecolors='none')
                    # Mean marker
                    ax.scatter(x_idx, np.mean(vals), s=40, color='darkblue',
                               marker='_', linewidths=2, zorder=5)

                short_labels = [str(pid)[:6] for pid in sorted_pids]
                ax.set_xticks(range(len(short_labels)))
                ax.set_xticklabels(short_labels, rotation=90, fontsize=7)

                # PMI reference band
                pm = pmi_stats[metric]
                ax.axhspan(pm['mean'] - pm['std'], pm['mean'] + pm['std'],
                           color='green', alpha=0.08)
                ax.axhline(y=pm['mean'], color='green', linestyle='--', alpha=0.6,
                           label=f'PMI mean={pm["mean"]:.1f}%')

                # Overall mean for this method
                all_vals = [v for vlist in by_fit.values() for v in vlist]
                ax.axhline(y=np.mean(all_vals), color='red', linestyle=':',
                           alpha=0.6, label=f'{method} mean={np.mean(all_vals):.1f}%')

                ax.set_ylabel(f'{metric} (%)')
                ax.set_title(f'{method}: {metric.upper()} by fit prompt')
                ax.legend(fontsize=8, loc='lower left')
                ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'Regression Methods: How much does fit-prompt choice matter? (typ={typ_name})\n'
                     f'Each column = 32 eval prompts; green band = PMI reference',
                     fontsize=12)
        fig.tight_layout()
        path3 = save_path.replace('.png', f'_regression_fit_sensitivity_{typ_name}.png')
        fig.savefig(path3, dpi=150, bbox_inches='tight')
        print(f"Saved regression fit scatter plot: {path3}")
        plt.close(fig)


def run_hypernym_analysis(outputs_dir, compute_llm_typ=False, cache_dir=None,
                          save_plot=None, self_typicality=False):
    """Run leave-one-out typicality correction analysis on hypernym data.

    Same structure as run_plausibleqa_analysis but for hypernym tasks.
    Each hypernym task (bananas, dogs, etc.) is treated as one "prompt".
    LOO: fit on task i, evaluate on all other tasks.
    """

    files = discover_hypernym_files(outputs_dir, self_typicality=self_typicality)
    if not files:
        print("No hypernym files found!")
        return None

    print(f"Found {len(files)} hypernym task files")
    for noun, fp in files:
        print(f"  {noun}: {os.path.basename(fp)}")

    # Load all data
    all_data = {}
    all_completions = []
    for noun, filepath in files:
        data = load_hypernym_csv(filepath)
        all_data[noun] = data
        all_completions.extend(data['completions'].tolist())

    prompt_ids = list(all_data.keys())
    n_prompts = len(prompt_ids)

    # Summary stats
    all_gen = np.concatenate([d['gen_score'] for d in all_data.values()])
    all_typ = np.concatenate([d['typicality_gpt2'] for d in all_data.values()])
    all_len = np.concatenate([d['lengths'] for d in all_data.values()])
    all_labels = np.concatenate([d['labels'] for d in all_data.values()])

    print(f"\nTotal data: {len(all_gen)} examples across {n_prompts} hypernym tasks")
    print(f"  Label balance: {int(all_labels.sum())} pos / {int((1-all_labels).sum())} neg")
    print(f"  Gen score range: [{all_gen.min():.2f}, {all_gen.max():.2f}]")
    print(f"  GPT-2 typicality range: [{all_typ.min():.2f}, {all_typ.max():.2f}]")
    print(f"  Length range: [{all_len.min():.0f}, {all_len.max():.0f}]")

    # Typicality sources
    typ_sources = {'gpt2': {pid: d['typicality_gpt2'] for pid, d in all_data.items()}}

    # Compute LLM-self typicality if requested
    if compute_llm_typ:
        cache_file = os.path.join(cache_dir, 'llm_typicality_hypernym.csv') if cache_dir else None
        unique_completions = list(set(all_completions))
        llm_scores = compute_llm_typicality(unique_completions, cache_file=cache_file)
        llm_lookup = dict(zip(unique_completions, llm_scores))

        for pid, data in all_data.items():
            data['typicality_llm'] = np.array([llm_lookup[c] for c in data['completions']])

        typ_sources['llm'] = {pid: d['typicality_llm'] for pid, d in all_data.items()}

        pooled_typ_llm = np.concatenate([d['typicality_llm'] for d in all_data.values()])
        print(f"  LLM typicality range: [{pooled_typ_llm.min():.2f}, {pooled_typ_llm.max():.2f}]")

    # METHOD_ORDER will be set dynamically after first apply_corrections call
    METHOD_ORDER = None

    # ---------------------------------------------------------------
    # Leave-one-out: fit on task i, evaluate on all others
    # ---------------------------------------------------------------
    loo_results = {}

    for typ_name, typ_per_prompt in typ_sources.items():
        print(f"\n{'='*70}")
        print(f"LEAVE-ONE-OUT ANALYSIS — Typicality source: {typ_name}")
        print(f"{'='*70}")

        loo_results[typ_name] = defaultdict(list)

        for fit_idx, fit_pid in enumerate(prompt_ids):
            fit_data = {
                'gen': all_data[fit_pid]['gen_score'],
                'typ': typ_per_prompt[fit_pid],
                'len': all_data[fit_pid]['lengths'],
                'val': all_data[fit_pid]['val_score'],
                'labels': all_data[fit_pid]['labels'],
            }

            if fit_idx == 0:
                print(f"\nExample fitted coefficients (fit on task {fit_pid}):")
                corrected_ex, coeffs_ex = apply_corrections(
                    fit_data['gen'], fit_data['typ'], fit_data['len'],
                    val_scores=fit_data['val'], labels=fit_data['labels'],
                    fit_data=None)
                for method, val in coeffs_ex.items():
                    if isinstance(val, np.ndarray):
                        print(f"  {method}: {val}")
                    elif isinstance(val, dict):
                        parts = [f"{k}={v:.4f}" for k, v in val.items()]
                        print(f"  {method}: {', '.join(parts)}")
                    else:
                        print(f"  {method}: b={val:.4f}")

            for eval_pid in prompt_ids:
                if eval_pid == fit_pid:
                    continue

                eval_data = all_data[eval_pid]
                eval_typ = typ_per_prompt[eval_pid]

                corrected, _ = apply_corrections(
                    eval_data['gen_score'], eval_typ, eval_data['lengths'],
                    val_scores=eval_data['val_score'],
                    labels=eval_data['labels'],
                    fit_data=fit_data
                )

                for method_name, corrected_gen in corrected.items():
                    metrics = compute_metrics(
                        corrected_gen, eval_data['val_score'], eval_data['labels']
                    )
                    metrics['fit_prompt'] = fit_pid
                    metrics['eval_prompt'] = eval_pid
                    loo_results[typ_name][method_name].append(metrics)

        if METHOD_ORDER is None:
            METHOD_ORDER = list(loo_results[typ_name].keys())

        # Print summary table
        print(f"\nAggregated LOO results (mean over {n_prompts}x{n_prompts-1} = "
              f"{n_prompts*(n_prompts-1)} fit-eval pairs):")
        print(f"{'Method':<22} {'Corr':>8} {'Corr+':>8} {'Corr-':>8} "
              f"{'GenROC':>8} {'ValROC':>8}")
        print("-" * 70)

        for method_name in METHOD_ORDER:
            entries = loo_results[typ_name][method_name]

            def agg(key):
                vals = [e[key] for e in entries if not np.isnan(e[key])]
                return np.mean(vals) if vals else np.nan

            print(f"{method_name:<22} {agg('corr')*100:>7.1f}% {agg('corr_pos')*100:>7.1f}% "
                  f"{agg('corr_neg')*100:>7.1f}% {agg('gen_roc')*100:>7.1f}% "
                  f"{agg('val_roc')*100:>7.1f}%")

    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------
    plot_path = save_plot or os.path.join(outputs_dir, 'typicality_correction_loo_hypernym.png')
    _make_loo_violin_plots(loo_results, prompt_ids, METHOD_ORDER, plot_path)

    # Build results DataFrame
    rows = []
    for typ_name in loo_results:
        for method_name in METHOD_ORDER:
            for e in loo_results[typ_name][method_name]:
                rows.append({
                    'typ_source': typ_name,
                    'method': method_name,
                    'fit_prompt': e['fit_prompt'],
                    'eval_prompt': e['eval_prompt'],
                    'corr': e['corr'],
                    'corr_pos': e['corr_pos'],
                    'corr_neg': e['corr_neg'],
                    'gen_roc': e['gen_roc'],
                    'val_roc': e['val_roc'],
                    'val_acc': e['val_acc'],
                })
    return pd.DataFrame(rows)


def run_ifeval_analysis(outputs_dir, compute_llm_typ=False, cache_dir=None,
                        save_plot=None, self_typicality=False):
    """Run leave-one-out typicality correction analysis on ifeval data.

    Same structure as run_hypernym_analysis but for ifeval tasks.
    Each ifeval prompt is treated as one "task" in the LOO.
    LOO: fit on prompt i, evaluate on all other prompts.
    """

    files = discover_ifeval_files(outputs_dir, self_typicality=self_typicality)
    if not files:
        print("No ifeval files found!")
        return None

    print(f"Found {len(files)} ifeval prompt files")
    for pid, fp in files:
        print(f"  {pid}: {os.path.basename(fp)}")

    # Load all data
    all_data = {}
    all_completions = []
    for pid, filepath in files:
        data = load_ifeval_csv(filepath)
        all_data[pid] = data
        all_completions.extend(data['completions'].tolist())

    prompt_ids = list(all_data.keys())
    n_prompts = len(prompt_ids)

    # Summary stats
    all_gen = np.concatenate([d['gen_score'] for d in all_data.values()])
    all_typ = np.concatenate([d['typicality_gpt2'] for d in all_data.values()])
    all_len = np.concatenate([d['lengths'] for d in all_data.values()])
    all_labels = np.concatenate([d['labels'] for d in all_data.values()])

    print(f"\nTotal data: {len(all_gen)} examples across {n_prompts} ifeval prompts")
    print(f"  Label balance: {int(all_labels.sum())} pos / {int((1-all_labels).sum())} neg")
    print(f"  Gen score range: [{all_gen.min():.2f}, {all_gen.max():.2f}]")
    print(f"  Typicality range: [{all_typ.min():.2f}, {all_typ.max():.2f}]")
    print(f"  Length range: [{all_len.min():.0f}, {all_len.max():.0f}]")

    # Typicality sources
    typ_sources = {'gpt2': {pid: d['typicality_gpt2'] for pid, d in all_data.items()}}

    # Compute LLM-self typicality if requested
    if compute_llm_typ:
        cache_file = os.path.join(cache_dir, 'llm_typicality_ifeval.csv') if cache_dir else None
        unique_completions = list(set(all_completions))
        llm_scores = compute_llm_typicality(unique_completions, cache_file=cache_file)
        llm_lookup = dict(zip(unique_completions, llm_scores))

        for pid, data in all_data.items():
            data['typicality_llm'] = np.array([llm_lookup[c] for c in data['completions']])

        typ_sources['llm'] = {pid: d['typicality_llm'] for pid, d in all_data.items()}

        pooled_typ_llm = np.concatenate([d['typicality_llm'] for d in all_data.values()])
        print(f"  LLM typicality range: [{pooled_typ_llm.min():.2f}, {pooled_typ_llm.max():.2f}]")

    # METHOD_ORDER will be set dynamically after first apply_corrections call
    METHOD_ORDER = None

    # ---------------------------------------------------------------
    # Leave-one-out: fit on prompt i, evaluate on all others
    # ---------------------------------------------------------------
    loo_results = {}

    for typ_name, typ_per_prompt in typ_sources.items():
        print(f"\n{'='*70}")
        print(f"LEAVE-ONE-OUT ANALYSIS — Typicality source: {typ_name}")
        print(f"{'='*70}")

        loo_results[typ_name] = defaultdict(list)

        for fit_idx, fit_pid in enumerate(prompt_ids):
            fit_data = {
                'gen': all_data[fit_pid]['gen_score'],
                'typ': typ_per_prompt[fit_pid],
                'len': all_data[fit_pid]['lengths'],
                'val': all_data[fit_pid]['val_score'],
                'labels': all_data[fit_pid]['labels'],
            }

            if fit_idx == 0:
                print(f"\nExample fitted coefficients (fit on {fit_pid}):")
                corrected_ex, coeffs_ex = apply_corrections(
                    fit_data['gen'], fit_data['typ'], fit_data['len'],
                    val_scores=fit_data['val'], labels=fit_data['labels'],
                    fit_data=None)
                for method, val in coeffs_ex.items():
                    if isinstance(val, np.ndarray):
                        print(f"  {method}: {val}")
                    elif isinstance(val, dict):
                        parts = [f"{k}={v:.4f}" for k, v in val.items()]
                        print(f"  {method}: {', '.join(parts)}")
                    else:
                        print(f"  {method}: b={val:.4f}")

            for eval_pid in prompt_ids:
                if eval_pid == fit_pid:
                    continue

                eval_data = all_data[eval_pid]
                eval_typ = typ_per_prompt[eval_pid]

                corrected, _ = apply_corrections(
                    eval_data['gen_score'], eval_typ, eval_data['lengths'],
                    val_scores=eval_data['val_score'],
                    labels=eval_data['labels'],
                    fit_data=fit_data
                )

                for method_name, corrected_gen in corrected.items():
                    metrics = compute_metrics(
                        corrected_gen, eval_data['val_score'], eval_data['labels']
                    )
                    metrics['fit_prompt'] = fit_pid
                    metrics['eval_prompt'] = eval_pid
                    loo_results[typ_name][method_name].append(metrics)

        if METHOD_ORDER is None:
            METHOD_ORDER = list(loo_results[typ_name].keys())

        # Print summary table
        print(f"\nAggregated LOO results (mean over {n_prompts}x{n_prompts-1} = "
              f"{n_prompts*(n_prompts-1)} fit-eval pairs):")
        print(f"{'Method':<22} {'Corr':>8} {'Corr+':>8} {'Corr-':>8} "
              f"{'GenROC':>8} {'ValROC':>8}")
        print("-" * 70)

        for method_name in METHOD_ORDER:
            entries = loo_results[typ_name][method_name]

            def agg(key):
                vals = [e[key] for e in entries if not np.isnan(e[key])]
                return np.mean(vals) if vals else np.nan

            print(f"{method_name:<22} {agg('corr')*100:>7.1f}% {agg('corr_pos')*100:>7.1f}% "
                  f"{agg('corr_neg')*100:>7.1f}% {agg('gen_roc')*100:>7.1f}% "
                  f"{agg('val_roc')*100:>7.1f}%")

    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------
    plot_path = save_plot or os.path.join(outputs_dir, 'typicality_correction_loo_ifeval.png')
    _make_loo_violin_plots(loo_results, prompt_ids, METHOD_ORDER, plot_path)

    # Build results DataFrame
    rows = []
    for typ_name in loo_results:
        for method_name in METHOD_ORDER:
            for e in loo_results[typ_name][method_name]:
                rows.append({
                    'typ_source': typ_name,
                    'method': method_name,
                    'fit_prompt': e['fit_prompt'],
                    'eval_prompt': e['eval_prompt'],
                    'corr': e['corr'],
                    'corr_pos': e['corr_pos'],
                    'corr_neg': e['corr_neg'],
                    'gen_roc': e['gen_roc'],
                    'val_roc': e['val_roc'],
                    'val_acc': e['val_acc'],
                })
    return pd.DataFrame(rows)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze typicality correction methods')
    parser.add_argument('--task', type=str, default='plausibleqa',
                       choices=['plausibleqa', 'hypernym', 'ifeval'],
                       help='Task to analyze')
    parser.add_argument('--outputs-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs'),
                       help='Directory containing score CSVs')
    parser.add_argument('--compute-llm-typicality', action='store_true',
                       help='Compute P_LLM(y|null) using the scoring model (needs GPU)')
    parser.add_argument('--cache-dir', type=str,
                       default=str(Path(__file__).parent.parent / 'outputs' / 'typicality_cache'),
                       help='Directory to cache computed typicality scores')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save detailed results CSV')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path for output plots (base name, .png)')
    parser.add_argument('--self-typicality', action='store_true',
                       help='Use self-typicality score CSVs (self- prefix) instead of GPT-2 ones')
    parser.add_argument('--insample', action='store_true',
                       help='Run in-sample diagnostic (fit+eval on same prompt)')
    args = parser.parse_args()

    if args.self_typicality and args.compute_llm_typicality:
        print("Warning: --compute-llm-typicality is redundant with --self-typicality "
              "(CSV already contains model's own typicality). Skipping LLM typicality computation.")
        args.compute_llm_typicality = False

    if args.insample:
        run_insample_diagnostic(args.outputs_dir, self_typicality=args.self_typicality)
        return

    if args.task == 'plausibleqa':
        results_df = run_plausibleqa_analysis(
            args.outputs_dir,
            compute_llm_typ=args.compute_llm_typicality,
            cache_dir=args.cache_dir,
            save_plot=args.save_plot,
            self_typicality=args.self_typicality,
        )
    elif args.task == 'hypernym':
        results_df = run_hypernym_analysis(
            args.outputs_dir,
            compute_llm_typ=args.compute_llm_typicality,
            cache_dir=args.cache_dir,
            save_plot=args.save_plot,
            self_typicality=args.self_typicality,
        )
    elif args.task == 'ifeval':
        results_df = run_ifeval_analysis(
            args.outputs_dir,
            compute_llm_typ=args.compute_llm_typicality,
            cache_dir=args.cache_dir,
            save_plot=args.save_plot,
            self_typicality=args.self_typicality,
        )

    if args.save_results and results_df is not None:
        results_df.to_csv(args.save_results, index=False)
        print(f"\nDetailed results saved to {args.save_results}")


if __name__ == '__main__':
    main()
