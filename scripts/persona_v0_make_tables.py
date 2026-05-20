#!/usr/bin/env python3
"""
Build per-(base, metric, eval_TC) tables from the persona-v0 summary CSV.

Reads:   output-metrics/persona_v0_summary.csv
         (produced by `python scripts/summarize_scores.py --file-pattern 'scores_*persona-v0*.csv'`)

Writes:  output-metrics/persona_v0_tables_long.csv
         (one row per (base, variant, eval_TC, eval_variant, metric, mean, std))
         output-metrics/persona_v0_tables.md
         (4 metrics x 2 base models x 2 eval TC flavors = 16 markdown tables)

Also prints all tables to stdout.

All eval jobs used --base-typcorr (the trained-model evals reference the
*base instruct model* for typicality, matching offline-TC train convention).
This is noted in every table title so we never lose track of it.

Layout of each markdown table:
   rows: 9 training variants (1.SFT-lo through 9.RankAlign+negTC); a variant is
         shown only if it has data for the given (base, eval_TC) combo
         (matched-TC policy: TC-trained variants only appear for their matched flavor).
   cols: gen_score variant used at scoring time -- raw (gen_score) and tc
         (gen_score_typcorr).
   cells: "mean (std)" across the 8 persona-v0-<slug> test tasks, x 100.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO_ROOT / "output-metrics" / "persona_v0_summary.csv"
TABLES_CSV = REPO_ROOT / "output-metrics" / "persona_v0_tables_long.csv"
TABLES_MD = REPO_ROOT / "output-metrics" / "persona_v0_tables.md"

# Order of variants and the suffix in `training_config` that uniquely identifies each.
# (Suffixes are mutually distinguishable; matched against substring presence/absence.)
# 0.Base = untrained base instruct model; classified separately (not via training_config).
VARIANTS = [
    ("1.SFT-lo",                 dict(tc=None,   has_nll=True,  has_fsx=False, has_vlo=False, semi="labelonly")),
    ("2.RankAlign",              dict(tc=None,   has_nll=False, has_fsx=False, has_vlo=False, semi="semi")),
    ("3.New+fsx",                dict(tc=None,   has_nll=True,  has_fsx=True,  has_vlo=True,  semi="semi")),
    ("4.New+fsx+selfTC",         dict(tc="self", has_nll=True,  has_fsx=True,  has_vlo=True,  semi="semi")),
    ("5.RankAlign+fsx+selfTC",   dict(tc="self", has_nll=False, has_fsx=True,  has_vlo=False, semi="semi")),
    ("6.RankAlign+selfTC",       dict(tc="self", has_nll=False, has_fsx=False, has_vlo=False, semi="semi")),
    ("7.New+fsx+negTC",          dict(tc="neg",  has_nll=True,  has_fsx=True,  has_vlo=True,  semi="semi")),
    ("8.RankAlign+fsx+negTC",    dict(tc="neg",  has_nll=False, has_fsx=True,  has_vlo=False, semi="semi")),
    ("9.RankAlign+negTC",        dict(tc="neg",  has_nll=False, has_fsx=False, has_vlo=False, semi="semi")),
]


def classify_variant(training_config: str) -> str | None:
    """Map a training_config string to a variant label, or None if it doesn't fit any.

    The 4-tuple (tc, has_nll, has_fsx, has_vlo) uniquely identifies each of the
    9 variants (semi vs labelonly is implied by has_nll-only-and-no-pref vs the rest).
    We deliberately do NOT require "semi"/"labelonly" substrings because
    long variant filenames (e.g. 9b-it tc-self/tc-neg + comb + fsx + vlo + semi)
    overflow filesystem NAME_MAX and end up hashed (e.g. "...vallogodds-_c7cf10a8").
    """
    has_self = "tc-self" in training_config
    has_neg = "tc-neg" in training_config
    has_nll = "nllv1.0" in training_config and "nllg1.0" in training_config
    has_fsx = "force-same-x" in training_config
    has_vlo = "vallogodds" in training_config
    has_pref0 = "pref0.0" in training_config  # SFT marker
    has_labelonly = "labelonly" in training_config

    if has_self and has_neg:
        return None
    tc = "self" if has_self else ("neg" if has_neg else None)

    # SFT (#1) is the only labelonly variant; require pref0.0 + labelonly to lock it in.
    # All other variants are pref-only or comb on semi data.
    is_sft = has_pref0 and has_labelonly

    for label, sig in VARIANTS:
        sig_is_sft = (sig["semi"] == "labelonly")
        if sig_is_sft != is_sft:
            continue
        if (sig["tc"] == tc
                and sig["has_nll"] == has_nll
                and sig["has_fsx"] == has_fsx
                and sig["has_vlo"] == has_vlo):
            return label
    return None


def base_model_short(model: str) -> str | None:
    """Map a `model` column value to '9b-it' / '2b-it' or None.

    Handles both naming conventions:
      - Base instruct models    : `v6-google_gemma-2-Xb-it`     (single underscore)
      - Finetuned (model_path)  : `v6-google--gemma-2-Xb-it`    (double dash)
    """
    if "gemma-2-9b-it" in model:
        return "9b-it"
    if "gemma-2-2b-it" in model:
        return "2b-it"
    return None


def eval_tc_dim(row) -> str | None:
    """Group rows along the eval-TC dimension used to slice tables.

    Two slices: 'self' and 'neg'. Whether `--base-typcorr` was used is a
    separate axis (always True for finetuned rows here, always False for base
    rows since the base model IS the typicality reference).
    """
    if row["self_tc"] and not row["neg_tc"]:
        return "self"
    if row["neg_tc"] and not row["self_tc"]:
        return "neg"
    return None


METRICS = [
    ("gen_roc", "Generator ROC-AUC"),
    ("val_roc", "Validator ROC-AUC"),
    ("val_acc", "Validator accuracy (threshold 0)"),
    ("corr",    "Pearson(gen_score, val_score)"),
]


def main():
    if not SUMMARY_CSV.exists():
        sys.exit(f"Missing {SUMMARY_CSV}. Run summarize_scores.py first.")
    df = pd.read_csv(SUMMARY_CSV)

    # Two row groups:
    #   - Finetuned   : require --base-typcorr at eval (basetyp_tc=True), classify variant.
    #   - Base instr. : NOT finetuned, NOT basetyp_tc (base model IS the ref),
    #                   variant = "0.Base".
    # Both restricted to test split, raw / tc gen-score variants only.
    base_mask = (
        df["finetuned"]
        & df["basetyp_tc"]
        & (df["split"] == "test")
        & df["eval_variant"].isin(["raw", "tc"])
    )
    sub_ft = df[base_mask].copy()
    sub_ft["variant"] = sub_ft["training_config"].map(classify_variant)

    base_mask = (
        (~df["finetuned"])
        & (~df["basetyp_tc"])
        & (df["split"] == "test")
        & df["eval_variant"].isin(["raw", "tc"])
    )
    sub_base = df[base_mask].copy()
    sub_base["variant"] = "0.Base"

    sub = pd.concat([sub_ft, sub_base], ignore_index=True)
    sub["base"] = sub["model"].map(base_model_short)
    sub["eval_TC"] = sub.apply(eval_tc_dim, axis=1)
    sub = sub.dropna(subset=["base", "eval_TC", "variant"])

    # Long-format aggregate: mean / std across 8 personas per
    # (base, variant, eval_TC, eval_variant, metric).
    rows = []
    for (base, variant, eval_TC, eval_variant), grp in sub.groupby(
            ["base", "variant", "eval_TC", "eval_variant"], sort=False):
        for metric_key, _metric_label in METRICS:
            vals = grp[metric_key].dropna().values * 100  # scale to percentage points
            if len(vals) == 0:
                continue
            rows.append(dict(
                base=base, variant=variant, eval_TC=eval_TC,
                eval_variant=eval_variant, metric=metric_key,
                n_tasks=len(vals),
                mean=round(float(vals.mean()), 2),
                std=round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 2),
            ))
    long_df = pd.DataFrame(rows)
    TABLES_CSV.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(TABLES_CSV, index=False)
    print(f"Wrote {TABLES_CSV} ({len(long_df)} rows)")

    # Markdown tables: one per (base, eval_TC, metric); columns = raw, tc gen-score variants.
    variant_order = ["0.Base"] + [v for v, _ in VARIANTS]
    md_lines = []
    md_lines.append("# Persona-v0 base + trained-model results")
    md_lines.append("")
    md_lines.append("All cells: **mean (std) across the 8 persona-v0-<slug> test tasks, x 100**.")
    md_lines.append("`raw` column = gen_score; `tc` column = gen_score_typcorr.")
    md_lines.append("")
    md_lines.append("Eval flags by row:")
    md_lines.append("- `0.Base`  : `--self-typcorr` or `--neg-typcorr` only (the base model IS the typicality reference, so no `--base-typcorr`).")
    md_lines.append("- `1`–`9` finetuned rows : `--self-typcorr --base-typcorr` or `--neg-typcorr --base-typcorr`.")
    md_lines.append("")
    md_lines.append("Source CSV: `output-metrics/persona_v0_tables_long.csv`")
    md_lines.append("(derived from `output-metrics/persona_v0_summary.csv`).")
    md_lines.append("")

    for base in ["9b-it", "2b-it"]:
        for eval_TC in ["self", "neg"]:
            for metric_key, metric_label in METRICS:
                title = f"## gemma-2-{base}  {metric_label}  (eval = {eval_TC}-typcorr)"
                md_lines.append(title)
                md_lines.append("")
                md_lines.append("| variant | raw | tc |")
                md_lines.append("| --- | --- | --- |")
                for variant in variant_order:
                    cell_raw = long_df[(long_df.base == base)
                                       & (long_df.eval_TC == eval_TC)
                                       & (long_df.variant == variant)
                                       & (long_df.metric == metric_key)
                                       & (long_df.eval_variant == "raw")]
                    cell_tc = long_df[(long_df.base == base)
                                      & (long_df.eval_TC == eval_TC)
                                      & (long_df.variant == variant)
                                      & (long_df.metric == metric_key)
                                      & (long_df.eval_variant == "tc")]
                    if cell_raw.empty and cell_tc.empty:
                        continue   # variant not in this eval_TC slice (matched-TC policy)
                    def fmt(c):
                        if c.empty:
                            return "—"
                        return f"{c.iloc[0]['mean']:.2f} ({c.iloc[0]['std']:.2f})"
                    md_lines.append(f"| {variant} | {fmt(cell_raw)} | {fmt(cell_tc)} |")
                md_lines.append("")
    md_text = "\n".join(md_lines)
    TABLES_MD.write_text(md_text)
    print(f"Wrote {TABLES_MD}")
    print()
    print(md_text)


if __name__ == "__main__":
    main()
