#!/usr/bin/env python3
"""Compute ROC-G and Pearson correlation for base (non-finetuned) models.

Reads per-example score CSVs from outputs/, computes per-sub-task metrics,
then macro-averages across sub-tasks (matching the dashboard).

Scores computed:
  - Raw:  gen_score  as predictor
  - PMI:  gen_score_typcorr from self- files
  - Neg:  gen_score_typcorr from neg- files

Outputs a LaTeX table matching the format in the paper.

Usage:
    python scripts/compute_base_model_table.py v6-google_gemma-2-9b-it
    python scripts/compute_base_model_table.py v6-google_gemma-2-2b
    python scripts/compute_base_model_table.py --pooled v6-google_gemma-2-9b-it  # pool instead of macro-avg
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

OUTPUTS_DIR = "outputs"

TASK_FAMILIES = {
    "AmbigQA":     "ambigqa-",
    "PlausibleQA": "plausibleqa-",
    "Hypernymy":   "hypernym-",
    "IFEval":      "ifeval-",
}

METHODS = ["Raw", "PMI", "Neg"]


def extract_subtask(filepath, task_prefix):
    """Extract the sub-task name (e.g. 'hypernym-cars') from a filepath."""
    basename = os.path.basename(filepath)
    idx = basename.find(task_prefix)
    if idx < 0:
        return basename
    rest = basename[idx:]
    # Sub-task ends at _test_ or _train_
    for sep in ["_test_", "_train_"]:
        pos = rest.find(sep)
        if pos >= 0:
            return rest[:pos]
    return rest


def find_base_model_files(model_slug, task_prefix, correction_type, include_eos=False):
    """Find per-example CSVs for the base model (no delta/epoch in filename).

    Excludes _eos_ files by default. Deduplicates by sub-task, keeping newest.
    """
    prefix = "neg" if correction_type == "neg" else "self"
    pattern = f"{OUTPUTS_DIR}/scores_{prefix}-{model_slug}_{task_prefix}*_test_*.csv"
    all_files = glob.glob(pattern)
    base_files = [f for f in all_files if "delta" not in f and "epoch" not in f]

    if not include_eos:
        base_files = [f for f in base_files if "_eos_" not in f]

    by_subtask = {}
    for f in base_files:
        subtask = extract_subtask(f, task_prefix)
        if subtask not in by_subtask:
            by_subtask[subtask] = []
        by_subtask[subtask].append(f)

    deduped = []
    for subtask in sorted(by_subtask.keys()):
        files = sorted(by_subtask[subtask])
        deduped.append(files[-1])
    return deduped


def read_per_subtask(files):
    """Read CSVs, returning list of (filename, DataFrame) pairs."""
    result = []
    for f in files:
        df = pd.read_csv(f)
        result.append((f, df))
    return result


def get_ground_truth(df):
    """Extract binary ground truth labels."""
    if "gpt4_ground_truth" in df.columns:
        col = "gpt4_ground_truth"
    elif "correct" in df.columns:
        col = "correct"
    else:
        raise ValueError(f"No ground truth column found. Columns: {list(df.columns)}")
    labels = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})
    return labels


def compute_metrics(gen_scores, val_scores, labels):
    """Compute ROC-G and Pearson correlation with val_score."""
    valid = np.isfinite(gen_scores) & np.isfinite(val_scores) & np.isfinite(labels)
    g = gen_scores[valid].values
    v = val_scores[valid].values
    y = labels[valid].values

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    try:
        roc = roc_auc_score(y, g)
    except Exception:
        roc = np.nan

    try:
        corr, _ = pearsonr(g, v)
    except Exception:
        corr = np.nan

    return roc, corr


def compute_macro_metrics(file_list, gen_col):
    """Compute per-sub-task ROC and corr, then macro-average."""
    rocs, corrs = [], []
    for fpath, df in file_list:
        labels = get_ground_truth(df)
        val = df["val_score"].astype(float)
        gen = df[gen_col].astype(float)
        roc, corr = compute_metrics(gen, val, labels)
        if not np.isnan(roc):
            rocs.append(roc)
        if not np.isnan(corr):
            corrs.append(corr)
    mean_roc = np.mean(rocs) if rocs else np.nan
    mean_corr = np.mean(corrs) if corrs else np.nan
    return mean_roc, mean_corr


def compute_pooled_metrics(file_list, gen_col):
    """Pool all examples across sub-tasks, compute single ROC and corr."""
    dfs = [df for _, df in file_list]
    if not dfs:
        return np.nan, np.nan
    pooled = pd.concat(dfs, ignore_index=True)
    labels = get_ground_truth(pooled)
    val = pooled["val_score"].astype(float)
    gen = pooled[gen_col].astype(float)
    return compute_metrics(gen, val, labels)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compute_base_model_table.py [--pooled] <model_slug> [model_slug2 ...]")
        print("Example: python scripts/compute_base_model_table.py v6-google_gemma-2-9b-it")
        sys.exit(1)

    args = sys.argv[1:]
    pooled = False
    if "--pooled" in args:
        pooled = True
        args.remove("--pooled")

    model_slugs = args
    agg_fn = compute_pooled_metrics if pooled else compute_macro_metrics
    agg_label = "POOLED" if pooled else "MACRO-AVERAGED"

    for model_slug in model_slugs:
        print(f"\n{'='*70}")
        print(f"Model: {model_slug}  ({agg_label})")
        print(f"{'='*70}\n")

        results = {}

        for task_name, task_prefix in TASK_FAMILIES.items():
            results[task_name] = {}

            neg_files = find_base_model_files(model_slug, task_prefix, "neg")
            self_files = find_base_model_files(model_slug, task_prefix, "self")

            neg_data = read_per_subtask(neg_files) if neg_files else []
            self_data = read_per_subtask(self_files) if self_files else []

            any_data = neg_data if neg_data else self_data
            if not any_data:
                print(f"  {task_name}: no files found")
                for m in METHODS:
                    results[task_name][m] = (np.nan, np.nan)
                continue

            n_examples = sum(len(df) for _, df in any_data)
            print(f"  {task_name}: neg={len(neg_files)} files, "
                  f"self={len(self_files)} files, "
                  f"n={n_examples} examples across {len(any_data)} sub-tasks")

            roc_raw, corr_raw = agg_fn(any_data, "gen_score")
            results[task_name]["Raw"] = (roc_raw, corr_raw)
            print(f"    Raw:  ROC-G={roc_raw*100:.1f}  rho={corr_raw:.2f}")

            if self_data:
                roc_pmi, corr_pmi = agg_fn(self_data, "gen_score_typcorr")
            else:
                roc_pmi, corr_pmi = np.nan, np.nan
            results[task_name]["PMI"] = (roc_pmi, corr_pmi)
            print(f"    PMI:  ROC-G={roc_pmi*100:.1f}  rho={corr_pmi:.2f}")

            if neg_data:
                roc_neg, corr_neg = agg_fn(neg_data, "gen_score_typcorr")
            else:
                roc_neg, corr_neg = np.nan, np.nan
            results[task_name]["Neg"] = (roc_neg, corr_neg)
            print(f"    Neg:  ROC-G={roc_neg*100:.1f}  rho={corr_neg:.2f}")

        print(f"\n--- LaTeX table for {model_slug} ---\n")
        tasks = list(TASK_FAMILIES.keys())
        ncols = 2 * len(tasks)
        col_spec = "l" + " | ".join(["cc"] * len(tasks))

        print(r"\begin{table}[t]")
        print(r"    \centering")
        print(r"    \begin{tabular}{" + col_spec + "}")

        header1 = "        "
        for t in tasks:
            header1 += f"& \\multicolumn{{2}}{{c|}}{{{t}}} "
        header1 = header1.rstrip("| ") + r" \\"
        print(header1)
        print(r"        \midrule")

        header2 = "        "
        for t in tasks:
            header2 += r"& ROC-G & $\rho$ "
        header2 += r"\\"
        print(header2)
        print(r"        \midrule")

        for method in METHODS:
            row = f"        {method} "
            for t in tasks:
                roc_val, corr_val = results[t][method]
                if np.isnan(roc_val):
                    row += "& -- & -- "
                else:
                    row += f"& {roc_val*100:.1f} & {corr_val:.2f} "
            row += r"\\"
            print(row)

        print(r"    \end{tabular}")
        short_name = model_slug.replace("v6-google_", "").replace("_", "-")
        print(r"    \caption{Base model results for " + short_name + ".}")
        print(r"    \label{tab:" + short_name + "}")
        print(r"\end{table}")


if __name__ == "__main__":
    main()
