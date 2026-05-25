#!/usr/bin/env python3
"""Build gen_roc tables for the ra-9b-it persona+member pod runs.

Reads all score CSVs from outputs_gemma4_from_pod-v7/ra9b_persona_member/
on mll, determines setting from the model_path column (or from the
symlink suffix in the filename for older-format files), and prints
tables for persona-v1 and membership (rosch).

Run on mll:
  python3 _build_ra9b_persona_member_table.py

Env vars:
  METRIC   gen_roc | val_roc | pearson | spearman    (default gen_roc)
  DATA_DIR override default search dir
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    "/datastor2/jdr/rankalign/outputs_gemma4_from_pod-v7/ra9b_persona_member"
))

METRIC = os.environ.get("METRIC", "gen_roc").lower()

# Persona eval tasks (6)
PERSONA_TASKS = [
    "persona-v1-psychopathy", "persona-v1-machiavellianism",
    "persona-v1-narcissism", "persona-v1-desire-to-create-allies",
    "persona-v1-interest-in-music", "persona-v1-interest-in-science",
]
PERSONA_TASK_LABELS = {
    "persona-v1-psychopathy": "psychopathy",
    "persona-v1-machiavellianism": "machiavellianism",
    "persona-v1-narcissism": "narcissism",
    "persona-v1-desire-to-create-allies": "desire-allies",
    "persona-v1-interest-in-music": "interest-music",
    "persona-v1-interest-in-science": "interest-sci",
}

# Rosch eval tasks (10)
ROSCH_TASKS = [
    "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
    "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vehicle",
    "rosch-vegetable", "rosch-weapon",
]

# Method labels for table rows
SETTING_LABELS = {
    "s1": "SFT labelonly 10%",
    "s2": "RankAlign",
    "s3": "New + fsx [-TC]",
    "s4": "New + PMI + fsx",
    "s7": "New + NegTC + fsx",
}

EVAL_PREFIXES = ("basetypneg-", "basetyp-", "self-", "neg-")


def _parse_eval_prefix(filename: str) -> str:
    after = filename[len("scores_"):] if filename.startswith("scores_") else filename
    for p in EVAL_PREFIXES:
        if after.startswith(p):
            return p
    return ""


def _parse_eval_task(filename: str) -> str | None:
    """Extract task name from filename pattern *_{task}_test_log-odds*.csv"""
    m = re.search(r'_((?:persona-v1|rosch)-[^_]+)_test_log-odds', filename)
    return m.group(1) if m else None


def _setting_from_model_path(model_path: str) -> str | None:
    """Determine setting from model_path column value."""
    # Symlink format: /workspace/eval_model_s4
    m = re.search(r'/eval_model_(s\d+)$', model_path)
    if m:
        return m.group(1)

    # Real path: /workspace/models2/v7-google--gemma-2-9b-it-...-fix1_merged
    # Parse the basename to determine setting.
    basename = Path(model_path).name.replace("_merged", "")

    # Check key flag substrings in the full basename
    has_fsx = "--force-same-x" in basename
    has_ppd = "--ppd" in basename
    has_vallogodds = "--vallogodds" in basename

    # TC
    if "--tc-self" in basename:
        tc = "self"
    elif "--tc-neg" in basename:
        tc = "neg"
    else:
        tc = None

    # NLL vs pref-only
    has_nll = "--nllv1.0" in basename or "--nllv1" in basename

    # labeled-only vs semi
    has_labelonly = "--labelonly0.1" in basename
    has_semi = "--semi0.1" in basename

    # Match settings:
    if has_labelonly and not has_fsx and tc is None:
        return "s1"
    if has_semi and not has_fsx and tc is None and not has_nll:
        return "s2"
    if has_semi and has_fsx and tc is None:
        return "s3"
    if has_semi and has_fsx and tc == "self":
        return "s4"
    if has_semi and has_fsx and tc == "neg":
        return "s7"

    return None


def _setting_from_filename(filename: str) -> str | None:
    """Fallback: extract setting from eval_model_sN in filename."""
    m = re.search(r'_eval_model_(s\d+)_', filename)
    return m.group(1) if m else None


def compute_gen_roc(df: pd.DataFrame, variant: str) -> float | None:
    """Compute gen_roc for a score variant ('raw', 'tc', 'lenorm', 'tc+lenorm').

    variant:
      'raw'       -> gen_score
      'tc'        -> gen_score_typcorr
      'lenorm'    -> gen_score_lenorm
      'tc+lenorm' -> gen_score_typcorr_lenorm
    """
    COL = {
        "raw": "gen_score",
        "tc": "gen_score_typcorr",
        "lenorm": "gen_score_lenorm",
        "tc+lenorm": "gen_score_typcorr_lenorm",
    }[variant]
    if COL not in df.columns:
        return None

    # Find label column
    for lcol in ("correct", "label", "gpt4_ground_truth"):
        if lcol in df.columns:
            break
    else:
        return None

    y_raw = df[lcol].astype(str).str.strip().str.lower()
    y = y_raw.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    g = df[COL].astype(float)

    mask = ~(y.isna() | g.isna())
    y, g = y[mask].astype(int), g[mask]
    if len(y) < 4 or len(set(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, g))
    except Exception:
        return None


# ============================================================
# Main: scan CSVs
# ============================================================

Record = dict  # keys: setting, dataset, task, eval_prefix, raw_roc, tc_roc, path

def scan_csvs(data_dir: Path) -> list[Record]:
    records = []
    skipped = []
    for csv_path in sorted(data_dir.glob("scores_*.csv")):
        fname = csv_path.name
        eval_prefix = _parse_eval_prefix(fname)
        task = _parse_eval_task(fname)
        if task is None:
            skipped.append(fname)
            continue
        dataset = "persona" if task.startswith("persona-v1") else "membership"

        # Read model_path from first data row (efficient — no full read needed)
        try:
            row0 = pd.read_csv(csv_path, nrows=1)
        except Exception as e:
            skipped.append(f"{fname} (read error: {e})")
            continue

        model_path = str(row0.get("model_path", pd.Series([""]), ).iloc[0]) if "model_path" in row0.columns else ""
        setting = _setting_from_model_path(model_path)
        if setting is None:
            setting = _setting_from_filename(fname)
        if setting is None:
            skipped.append(f"{fname} (unknown setting, model_path={model_path!r})")
            continue

        # Read full CSV for metrics
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            skipped.append(f"{fname} (read error: {e})")
            continue

        raw_roc = compute_gen_roc(df, "raw")
        tc_roc = compute_gen_roc(df, "tc")

        records.append({
            "setting": setting,
            "dataset": dataset,
            "task": task,
            "eval_prefix": eval_prefix,
            "raw_roc": raw_roc,
            "tc_roc": tc_roc,
            "path": str(csv_path),
        })

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} files:")
        for s in skipped[:10]:
            print(f"  {s}")

    return records


def _mean(vals: list[float | None]) -> str:
    vs = [v for v in vals if v is not None and not np.isnan(v)]
    if not vs:
        return "—"
    return f"{np.mean(vs) * 100:.1f}"


def build_table(records: list[Record], dataset: str) -> str:
    """Build a markdown table for the given dataset."""
    recs = [r for r in records if r["dataset"] == dataset]
    if not recs:
        return f"No records for {dataset}."

    tasks = PERSONA_TASKS if dataset == "persona" else ROSCH_TASKS
    task_labels = PERSONA_TASK_LABELS if dataset == "persona" else {t: t.replace("rosch-", "") for t in ROSCH_TASKS}

    settings_present = sorted(set(r["setting"] for r in recs), key=lambda s: int(s[1:]))
    all_eval_prefixes = sorted(set(r["eval_prefix"] for r in recs))

    # Columns: Raw | self-tc | neg-tc | basetyp-tc | basetypneg-tc
    # (only show columns we actually have data for)
    col_defs = [
        ("Raw",      None,          "raw"),
        ("PMI self", "self-",       "tc"),
        ("PMI base", "basetyp-",    "tc"),
        ("Neg self", "neg-",        "tc"),
        ("Neg base", "basetypneg-", "tc"),
    ]
    col_defs_present = [
        (label, pfx, variant) for label, pfx, variant in col_defs
        if pfx is None or any(r["eval_prefix"] == pfx for r in recs if r["setting"] in settings_present)
    ]

    lines = []
    lines.append(f"### {dataset.capitalize()} — gen_roc × 100 (mean across {len(tasks)} tasks)")
    lines.append("")

    header = "| Setting | Label | " + " | ".join(c[0] for c in col_defs_present) + " |"
    sep    = "| --- | --- | " + " | ".join("---" for _ in col_defs_present) + " |"
    lines.append(header)
    lines.append(sep)

    for setting in settings_present:
        label = SETTING_LABELS.get(setting, setting)
        row_recs = [r for r in recs if r["setting"] == setting]
        cells = []
        for col_label, pfx, variant in col_defs_present:
            if pfx is None:
                # Raw: average across all eval prefixes
                vals = [r["raw_roc"] for r in row_recs]
            else:
                vals = [r["tc_roc"] for r in row_recs if r["eval_prefix"] == pfx]
            cells.append(_mean(vals))
        lines.append(f"| {setting} | {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_per_task_table(records: list[Record], dataset: str, setting: str, eval_prefix: str, variant: str) -> str:
    """Per-task breakdown for one setting/column."""
    tasks = PERSONA_TASKS if dataset == "persona" else ROSCH_TASKS
    recs = {r["task"]: r for r in records
            if r["dataset"] == dataset and r["setting"] == setting
            and (eval_prefix is None or r["eval_prefix"] == eval_prefix)}
    lines = [f"  Tasks ({setting} {eval_prefix or 'raw'}):"]
    for t in tasks:
        r = recs.get(t)
        val = r["tc_roc"] if (variant == "tc" and r) else (r["raw_roc"] if r else None)
        v = f"{val * 100:.1f}" if val is not None else "—"
        lines.append(f"    {t}: {v}")
    return "\n".join(lines)


def main():
    if not DATA_DIR.is_dir():
        sys.exit(f"ERROR: DATA_DIR not found: {DATA_DIR}")

    print(f"Scanning {DATA_DIR} ...")
    records = scan_csvs(DATA_DIR)
    print(f"Found {len(records)} valid records.\n")

    if not records:
        print("No records found — check DATA_DIR and CSV formats.")
        return

    # Summary of what was found
    from collections import Counter
    setting_counts = Counter((r["dataset"], r["setting"]) for r in records)
    print("Records by (dataset, setting):")
    for (ds, s), n in sorted(setting_counts.items()):
        print(f"  {ds} {s}: {n}")
    print()

    # Print tables
    for dataset in ("persona", "membership"):
        print(build_table(records, dataset))
        print()

    # Also print available eval columns per dataset+setting
    print("=== Available eval prefixes per setting ===")
    pfx_by_ds_setting = {}
    for r in records:
        k = (r["dataset"], r["setting"])
        pfx_by_ds_setting.setdefault(k, set()).add(r["eval_prefix"])
    for k in sorted(pfx_by_ds_setting):
        print(f"  {k[0]} {k[1]}: {sorted(pfx_by_ds_setting[k])}")


if __name__ == "__main__":
    main()
