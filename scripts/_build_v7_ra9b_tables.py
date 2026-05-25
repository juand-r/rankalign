#!/usr/bin/env python3
"""Build gemma-2-9b-it v7 result tables for persona, ifeval, and membership/rosch.

Reads score CSVs from the local pod output directories:
  - outputs_gemma4_from_pod-v7/ra9b_persona_member/  (persona + rosch)
  - outputs_gemma4_from_pod-v7/ra9b_ifeval/          (ifeval)

Saves markdown tables to docs/v7_ra9b_results_<date>.md

Persona ID/OOD split:
  ID  = psychopathy, machiavellianism, narcissism (label-flipped, in-domain)
  OOD = desire-to-create-allies, interest-in-music, interest-in-science (held-out)

IFEval: all 20 prompts (1-13, 15-21) are OOD; no ID split available for v7.

Metrics computed for each (setting, task, eval_prefix):
  Raw gen_roc         = ROC-AUC of gen_score vs correct label
  PMI base gen_roc    = ROC-AUC of gen_score_typcorr (from basetyp- files)
  Neg base gen_roc    = ROC-AUC of gen_score_typcorr (from basetypneg- files)
  LenNorm gen_roc     = ROC-AUC of gen_score_lenorm
  PMI+Len gen_roc     = ROC-AUC of gen_score_typcorr_lenorm (from basetyp- files)
  Neg+Len gen_roc     = ROC-AUC of gen_score_typcorr_lenorm (from basetypneg- files)
  val_roc             = ROC-AUC of val_score vs correct label
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
PM_DIR = REPO / "outputs_gemma4_from_pod-v7" / "ra9b_persona_member"
IF_DIR = REPO / "outputs_gemma4_from_pod-v7" / "ra9b_ifeval"
DOCS_DIR = REPO / "docs"

# ─── Task definitions ──────────────────────────────────────────────────────────

PERSONA_ID = ["psychopathy", "machiavellianism", "narcissism"]
PERSONA_OOD = ["desire-to-create-allies", "interest-in-music", "interest-in-science"]
PERSONA_ALL = PERSONA_ID + PERSONA_OOD
PERSONA_TASKS = {f"persona-v1-{p}" for p in PERSONA_ALL}

ROSCH_TASKS = {
    "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
    "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vehicle",
    "rosch-vegetable", "rosch-weapon",
}

IFEVAL_PROMPTS = list(range(1, 14)) + list(range(15, 22))  # 1-13, 15-21 (no 14)
IFEVAL_TASKS = {f"ifeval-prompt_{n}" for n in IFEVAL_PROMPTS}

SETTING_LABELS = {
    "s1": "SFT labelonly 10%",
    "s2": "RankAlign",
    "s3": "New + fsx [-TC]",
    "s4": "New + PMI + fsx",
    "s7": "New + NegTC + fsx",
}
SETTINGS_ORDER = ["s1", "s2", "s3", "s4", "s7"]

# ─── Filename / model_path parsing ─────────────────────────────────────────────

_EVAL_PREFIX_ORDER = ("basetypneg-", "basetyp-", "neg-", "self-")


def _parse_eval_prefix(filename: str) -> str:
    name = Path(filename).name
    after = name[len("scores_"):] if name.startswith("scores_") else name
    for p in _EVAL_PREFIX_ORDER:
        if after.startswith(p):
            return p
    return ""


def _parse_task(filename: str) -> str | None:
    """Extract task name (e.g. 'persona-v1-psychopathy', 'rosch-bird', 'ifeval-prompt_1')."""
    name = Path(filename).name
    m = re.search(r'_((?:persona-v1|rosch|ifeval-prompt)-[^_]+(?:_\d+)?)_test_log-odds', name)
    if m:
        return m.group(1)
    # ifeval-prompt_N pattern with underscore in task name
    m2 = re.search(r'_(ifeval-prompt_\d+)_test_log-odds', name)
    if m2:
        return m2.group(1)
    return None


def _setting_from_model_path(model_path: str) -> str | None:
    """Detect setting from model_path column value."""
    # Symlink format: /workspace/eval_model_sN
    m = re.search(r'/eval_model_(s\d+)$', model_path)
    if m:
        return m.group(1)
    # Full path: parse basename flags
    basename = Path(model_path).name.replace("_merged", "")
    has_fsx = "--force-same-x" in basename
    has_labelonly = "--labelonly0.1" in basename
    has_semi = "--semi0.1" in basename
    has_nll = "--nllv1.0" in basename or "--nllv1" in basename
    if "--tc-self" in basename:
        tc = "self"
    elif "--tc-neg" in basename:
        tc = "neg"
    else:
        tc = None
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
    m = re.search(r'_eval_model_(s\d+)_', Path(filename).name)
    return m.group(1) if m else None

# ─── Metric computation ─────────────────────────────────────────────────────────

_SCORE_COLS = {
    "raw":      "gen_score",
    "tc":       "gen_score_typcorr",
    "lenorm":   "gen_score_lenorm",
    "tc+lenorm": "gen_score_typcorr_lenorm",
}
_LABEL_COLS = ("correct", "label", "gpt4_ground_truth")


def _label_series(df: pd.DataFrame) -> pd.Series | None:
    """Return 0/1 series from whichever label column is present."""
    for col in _LABEL_COLS:
        if col in df.columns:
            raw = df[col].astype(str).str.strip().str.lower()
            s = raw.map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
            if s.notna().sum() > 0:
                return s
    return None


def _roc(y: pd.Series, scores: pd.Series) -> float | None:
    mask = ~(y.isna() | scores.isna())
    y2, s2 = y[mask].astype(int), scores[mask].astype(float)
    if len(y2) < 4 or len(set(y2)) < 2:
        return None
    try:
        return float(roc_auc_score(y2, s2))
    except Exception:
        return None


def compute_metrics(df: pd.DataFrame) -> dict[str, float | None]:
    """Return dict of metric_name -> ROC-AUC (or None)."""
    y = _label_series(df)
    if y is None:
        return {}
    result: dict[str, float | None] = {}
    for variant, col in _SCORE_COLS.items():
        if col in df.columns:
            result[f"gen_roc_{variant}"] = _roc(y, df[col])
    if "val_score" in df.columns:
        result["val_roc"] = _roc(y, df["val_score"])
    return result

# ─── CSV scanning ──────────────────────────────────────────────────────────────

def _is_canonical_v7(model_path: str, task_type: str) -> bool:
    """Check whether this record is from a canonical v7 pod eval (not an old experiment).

    For ifeval and rosch: must use the eval_model_sN symlink format.
    For persona: must contain persona-v1-all in the model path.
    """
    if task_type == "persona":
        return "persona-v1-all" in model_path
    # rosch and ifeval: require the symlink format
    return bool(re.search(r'/eval_model_s\d+$', model_path))


def scan_dir(csv_dir: Path, known_tasks: set[str]) -> list[dict]:
    """Scan all score CSVs in csv_dir and return records.

    Only includes canonical v7 records (filters out old-experiment files).
    Deduplicates: for each (setting, task, eval_prefix), keeps only the newest file.
    """
    candidates: dict[tuple, list[tuple[str, Path, dict]]] = {}  # key -> [(filename, path, metrics)]

    skipped = []
    for p in sorted(csv_dir.glob("scores_*.csv")):
        task = _parse_task(p.name)
        if task is None or task not in known_tasks:
            continue
        task_type = "persona" if task.startswith("persona") else ("ifeval" if task.startswith("ifeval") else "rosch")
        eval_prefix = _parse_eval_prefix(p.name)
        try:
            row0 = pd.read_csv(p, nrows=1)
        except Exception as e:
            skipped.append(f"{p.name} (read error: {e})")
            continue
        model_path = str(row0["model_path"].iloc[0]) if "model_path" in row0.columns else ""
        if not _is_canonical_v7(model_path, task_type):
            continue
        setting = _setting_from_model_path(model_path)
        if setting is None:
            setting = _setting_from_filename(p.name)
        if setting is None or setting not in SETTINGS_ORDER:
            skipped.append(f"{p.name} (unknown setting; model_path={model_path!r})")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            skipped.append(f"{p.name} (read error: {e})")
            continue
        metrics = compute_metrics(df)
        key = (setting, task, eval_prefix)
        candidates.setdefault(key, []).append((p.name, p, metrics))

    # Deduplicate: keep newest filename for each key
    records = []
    for key, cands in candidates.items():
        setting, task, eval_prefix = key
        _, p, metrics = max(cands, key=lambda x: x[0])
        records.append({
            "setting": setting,
            "task": task,
            "eval_prefix": eval_prefix,
            "path": str(p),
            **metrics,
        })

    if skipped:
        print(f"  [warn] skipped {len(skipped)} files (first 5):")
        for s in skipped[:5]:
            print(f"    {s}")
    return records

# ─── Table building ─────────────────────────────────────────────────────────────

METRIC_COLS = [
    ("Raw",        "gen_roc_raw"),
    ("PMI base",   "gen_roc_tc"),        # from basetyp- files
    ("Neg base",   "gen_roc_tc"),        # from basetypneg- files (same column, different prefix)
    ("LenNorm",    "gen_roc_lenorm"),
    ("PMI+Len",    "gen_roc_tc+lenorm"), # from basetyp- files
    ("Neg+Len",    "gen_roc_tc+lenorm"), # from basetypneg- files
    ("ValROC",     "val_roc"),
]

# Which eval_prefix each column belongs to (None = any)
COL_PREFIX = {
    "Raw":      None,
    "PMI base": "basetyp-",
    "Neg base": "basetypneg-",
    "LenNorm":  None,
    "PMI+Len":  "basetyp-",
    "Neg+Len":  "basetypneg-",
    "ValROC":   None,
}

# For settings that only run one eval type, certain columns are N/A
# s4: only basetyp- eval → Neg base, Neg+Len are N/A
# s7: only basetypneg- eval → PMI base, PMI+Len are N/A
SETTING_NA = {
    "s4": {"Neg base", "Neg+Len"},
    "s7": {"PMI base", "PMI+Len"},
}


def _mean_roc(records: list[dict], tasks: list[str], setting: str, col_label: str, metric_col: str) -> str:
    """Compute mean gen_roc × 100 across tasks for given setting/column."""
    pfx = COL_PREFIX[col_label]
    if setting in SETTING_NA and col_label in SETTING_NA[setting]:
        return "N/A"
    recs = [r for r in records
            if r["setting"] == setting
            and r["task"] in tasks
            and (pfx is None or r["eval_prefix"] == pfx)]
    vals = [r[metric_col] for r in recs if r.get(metric_col) is not None]
    if not vals:
        return "—"
    mean = np.mean(vals) * 100
    return f"{mean:.1f}"


def _per_task_rows(records: list[dict], tasks: list[str]) -> list[dict]:
    """Return per-task rows with all metrics."""
    rows = []
    for task in tasks:
        for setting in SETTINGS_ORDER:
            for col_label, metric_col in METRIC_COLS:
                pfx = COL_PREFIX[col_label]
                recs = [r for r in records
                        if r["setting"] == setting
                        and r["task"] == task
                        and (pfx is None or r["eval_prefix"] == pfx)]
                # For Raw/LenNorm/ValROC, pick best available prefix record
                vals = [r[metric_col] for r in recs if r.get(metric_col) is not None]
                val = float(np.mean(vals)) * 100 if vals else None
                rows.append({"task": task, "setting": setting, "metric": col_label,
                             "value": val})
    return rows


def build_summary_table(records: list[dict], tasks: list[str], title: str) -> str:
    """Build a markdown table: rows=settings, cols=metrics, cells=mean ROC-AUC×100."""
    lines = [f"### {title}", ""]
    cols = [c[0] for c in METRIC_COLS]
    header = "| Setting | Label | " + " | ".join(cols) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in cols) + " |"
    lines.extend([header, sep])
    for setting in SETTINGS_ORDER:
        label = SETTING_LABELS.get(setting, setting)
        cells = [_mean_roc(records, tasks, setting, cl, mc) for cl, mc in METRIC_COLS]
        lines.append(f"| {setting} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_per_task_table(records: list[dict], tasks: list[str], col_label: str, metric_col: str, title: str) -> str:
    """Build per-task breakdown for a single metric/column: rows=tasks, cols=settings."""
    pfx = COL_PREFIX[col_label]
    lines = [f"### {title} — {col_label}", ""]
    header = "| Task | " + " | ".join(SETTINGS_ORDER) + " |"
    sep = "| --- | " + " | ".join("---" for _ in SETTINGS_ORDER) + " |"
    lines.extend([header, sep])
    for task in tasks:
        cells = []
        for setting in SETTINGS_ORDER:
            if setting in SETTING_NA and col_label in SETTING_NA[setting]:
                cells.append("N/A")
                continue
            recs = [r for r in records
                    if r["setting"] == setting
                    and r["task"] == task
                    and (pfx is None or r["eval_prefix"] == pfx)]
            vals = [r[metric_col] for r in recs if r.get(metric_col) is not None]
            cells.append(f"{np.mean(vals) * 100:.1f}" if vals else "—")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    today = date.today().strftime("%Y-%m-%d")
    out_path = DOCS_DIR / f"v7_ra9b_results_{today}.md"

    all_known = PERSONA_TASKS | ROSCH_TASKS | IFEVAL_TASKS

    print(f"Scanning {PM_DIR} ...")
    pm_records = scan_dir(PM_DIR, PERSONA_TASKS | ROSCH_TASKS)
    print(f"  → {len(pm_records)} records\n")

    print(f"Scanning {IF_DIR} ...")
    if_records = scan_dir(IF_DIR, IFEVAL_TASKS)
    print(f"  → {len(if_records)} records\n")

    # ── Summary counts ──
    from collections import Counter
    pm_cnt = Counter((r["setting"], r["task"].split("-")[0]) for r in pm_records)
    if_cnt = Counter((r["setting"],) for r in if_records)
    print("PM records by (setting, dataset-prefix):", dict(pm_cnt))
    print("IFEval records by setting:", dict(if_cnt))
    print()

    # ── Build sections ──
    sections: list[str] = [
        f"# Gemma-2-9b-it v7 Results — {today}",
        "",
        "Model: gemma-2-9b-it, trained with rankalign v7 (epoch2), evaluated with base-model typicality correction.",
        "",
        "**Columns**: gen_roc × 100 unless noted.",
        "- Raw = log P(y|x)",
        "- PMI base = (log P(y|x) − log P_base(y|x)) / len  [from basetyp- eval files]",
        "- Neg base = (log P(y|x) − log P_neg(y|x)) / len  [from basetypneg- eval files]",
        "- LenNorm = Raw / num_tokens",
        "- PMI+Len = PMI base / num_tokens",
        "- Neg+Len = Neg base / num_tokens",
        "- ValROC = ROC-AUC of validation score",
        "- N/A = this eval type was not run for this setting (s4 only runs PMI; s7 only runs Neg)",
        "",
        "---",
        "",
    ]

    # ── Persona ──
    persona_all_tasks = [f"persona-v1-{p}" for p in PERSONA_ALL]
    persona_id_tasks = [f"persona-v1-{p}" for p in PERSONA_ID]
    persona_ood_tasks = [f"persona-v1-{p}" for p in PERSONA_OOD]

    sections.append("## Persona")
    sections.append("")
    sections.append("Tasks trained on: persona-v1-all")
    sections.append("- **ID** (label-flipped, in-domain): psychopathy, machiavellianism, narcissism")
    sections.append("- **OOD** (held-out): desire-to-create-allies, interest-in-music, interest-in-science")
    sections.append("")
    sections.append(build_summary_table(pm_records, persona_all_tasks, "Persona — all 6 tasks (mean gen_roc × 100)"))
    sections.append("")
    sections.append(build_summary_table(pm_records, persona_id_tasks, "Persona — ID (3 in-domain tasks, mean gen_roc × 100)"))
    sections.append("")
    sections.append(build_summary_table(pm_records, persona_ood_tasks, "Persona — OOD (3 held-out tasks, mean gen_roc × 100)"))
    sections.append("")
    sections.append("#### Per-task breakdown (Raw gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(pm_records, persona_all_tasks, "Raw", "gen_roc_raw", "Persona"))
    sections.append("")
    sections.append("#### Per-task breakdown (PMI base gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(pm_records, persona_all_tasks, "PMI base", "gen_roc_tc", "Persona"))
    sections.append("")
    sections.append("#### Per-task breakdown (Neg base gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(pm_records, persona_all_tasks, "Neg base", "gen_roc_tc", "Persona"))
    sections.append("")
    sections.append("---")
    sections.append("")

    # ── Membership / Rosch ──
    rosch_tasks = sorted(ROSCH_TASKS)
    sections.append("## Membership / Rosch")
    sections.append("")
    sections.append("10 rosch category-typicality tasks (all OOD w.r.t. the membership-sans-rosch training split).")
    sections.append("")
    sections.append(build_summary_table(pm_records, rosch_tasks, "Rosch — all 10 tasks (mean gen_roc × 100)"))
    sections.append("")
    sections.append("#### Per-task breakdown (Raw gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(pm_records, rosch_tasks, "Raw", "gen_roc_raw", "Rosch"))
    sections.append("")
    sections.append("#### Per-task breakdown (PMI base gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(pm_records, rosch_tasks, "PMI base", "gen_roc_tc", "Rosch"))
    sections.append("")
    sections.append("#### Per-task breakdown (Neg base gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(pm_records, rosch_tasks, "Neg base", "gen_roc_tc", "Rosch"))
    sections.append("")
    sections.append("---")
    sections.append("")

    # ── IFEval ──
    ifeval_tasks = [f"ifeval-prompt_{n}" for n in IFEVAL_PROMPTS]
    sections.append("## IFEval")
    sections.append("")
    sections.append("20 prompts (1-13, 15-21; prompt 14 has no data).")
    sections.append("Training task: `ifeval-concat`. Prompts 1-21 are OOD (never appeared in training); prompts 22+ are ID but not evaluated here.")
    sections.append("")
    sections.append(build_summary_table(if_records, ifeval_tasks, "IFEval — all 20 prompts (mean gen_roc × 100)"))
    sections.append("")
    sections.append("#### Per-prompt breakdown (Raw gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(if_records, ifeval_tasks, "Raw", "gen_roc_raw", "IFEval"))
    sections.append("")
    sections.append("#### Per-prompt breakdown (PMI base gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(if_records, ifeval_tasks, "PMI base", "gen_roc_tc", "IFEval"))
    sections.append("")
    sections.append("#### Per-prompt breakdown (Neg base gen_roc × 100)")
    sections.append("")
    sections.append(build_per_task_table(if_records, ifeval_tasks, "Neg base", "gen_roc_tc", "IFEval"))
    sections.append("")

    content = "\n".join(sections)
    out_path.write_text(content)
    print(f"\nSaved: {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
