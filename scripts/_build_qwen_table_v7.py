#!/usr/bin/env python3
"""Build metric tables for Qwen wandb-rerun evaluations.

Qwen score CSVs use a simple model_short: `eval_model_qw_s{N}`.
No checkpoint-name parser needed — just direct string matching.

Supports both ifeval and rosch tasks via QWEN_TASK env var.

Env vars:
- QWEN_TASK    ∈ {ifeval-ood, ifeval-id, rosch}   default rosch
- QWEN_METRIC  ∈ {gen_roc, pearson, spearman, val_roc, val_acc}
                                                   default gen_roc

Writes:
- metrics-from-scores-rerun-wandb/qwen_{task}_{metric}_table_long.csv
- metrics-from-scores-rerun-wandb/qwen_{task}_{metric}_table_cells.csv
- prints markdown table to stdout
"""

from __future__ import annotations
import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

SEARCH_DIRS = [
    Path("/datastor2/jdr/rankalign/outputs-rerun-wandb"),
]
METRICS_DIR = REPO / "metrics-from-scores-rerun-wandb"
METRICS_DIR.mkdir(exist_ok=True)

# --- Task selection -----------------------------------------------------------

TASK_TYPE = os.environ.get("QWEN_TASK", "rosch").lower()
DATA_DIR = REPO / "data" / "fixed-prompts-ifeval"

ROSCH_TASKS = [
    "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
    "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vegetable",
    "rosch-vehicle", "rosch-weapon",
]


def _discover_prompts() -> list[int]:
    nums = []
    if DATA_DIR.is_dir():
        for f in DATA_DIR.iterdir():
            if f.name.startswith("gpt_ifeval_results_prompt_") and f.name.endswith(".jsonl"):
                s = f.name[len("gpt_ifeval_results_prompt_"):-len(".jsonl")]
                if s.isdigit():
                    nums.append(int(s))
    return sorted(nums)


_ALL_PROMPTS = _discover_prompts()

if TASK_TYPE == "rosch":
    EVAL_TASKS = ROSCH_TASKS
    TASK_LABEL = "Rosch (all OOD)"
elif TASK_TYPE == "ifeval-ood":
    EVAL_TASKS = [f"ifeval-prompt_{n}" for n in _ALL_PROMPTS if 1 <= n <= 21]
    TASK_LABEL = "IFEval OOD (prompts 1..21)"
elif TASK_TYPE == "ifeval-id":
    EVAL_TASKS = [f"ifeval-prompt_{n}" for n in _ALL_PROMPTS if n >= 22]
    TASK_LABEL = "IFEval ID (prompts 22+)"
else:
    raise SystemExit(f"QWEN_TASK must be 'rosch', 'ifeval-ood', or 'ifeval-id', got {TASK_TYPE!r}")

N_EXPECTED = len(EVAL_TASKS)
if N_EXPECTED == 0:
    raise SystemExit(f"No tasks discovered for QWEN_TASK={TASK_TYPE!r}")

# --- Metric -------------------------------------------------------------------

METRIC = os.environ.get("QWEN_METRIC", "gen_roc").lower()
SUPPORTED_METRICS = {"gen_roc", "pearson", "spearman", "val_roc", "val_acc"}
if METRIC not in SUPPORTED_METRICS:
    raise SystemExit(f"QWEN_METRIC must be one of {SUPPORTED_METRICS}, got {METRIC!r}")
METRIC_LABEL = {
    "gen_roc": "GenROC", "pearson": "Pearson(gen, val)",
    "spearman": "Spearman(gen, val)", "val_roc": "ValROC", "val_acc": "ValAcc",
}[METRIC]

# --- Column definitions (same as other builders) ------------------------------

COLUMNS = [
    ("Raw",      ["self-", "neg-", "basetyp-", "basetypneg-"], "raw"),
    ("PMI self", "self-",       "tc"),
    ("PMI base", "basetyp-",    "tc"),
    ("Neg self", "neg-",        "tc"),
    ("Neg base", "basetypneg-", "tc"),
]

# --- Methods: simple string matching on model_short ---------------------------
# s2, s3 have BOTH basetyp + basetypneg → no NA
# s4 has basetyp only → Neg cols are NA
# s7 has basetypneg only → PMI cols are NA

METHODS: list[dict] = [
    dict(num=2, label="RankAlign (s2)",
         match=lambda s: s == "eval_model_qw_s2"),
    dict(num=3, label="New + fsx [-TC] (s3)",
         match=lambda s: s == "eval_model_qw_s3"),
    dict(num=4, label="New + PMI + fsx (s4)",
         match=lambda s: s == "eval_model_qw_s4"),
    dict(num=7, label="New + NegTC + fsx (s7)",
         match=lambda s: s == "eval_model_qw_s7"),
]

NA_COLS = {
    4:  {"Neg self", "Neg base"},
    7:  {"PMI self", "PMI base"},
}

# --- Score-file matching (same logic as other builders) -----------------------

_EVAL_PREFIXES_ORDERED = ("basetypneg-", "basetyp-", "self-", "neg-")


def _extract_eval_prefix(filename: str) -> str:
    after = filename[len("scores_"):] if filename.startswith("scores_") else filename
    for p in _EVAL_PREFIXES_ORDERED:
        if after.startswith(p):
            return p
    return ""


def find_score_files(method: dict, eval_prefix: str | list[str]) -> dict[str, list[Path]]:
    if isinstance(eval_prefix, str):
        prefixes = [eval_prefix]
    else:
        prefixes = list(eval_prefix)
    matches: dict[str, list[Path]] = {t: [] for t in EVAL_TASKS}
    seen_basenames: set[str] = set()
    for pfx in prefixes:
        pattern = f"scores_{pfx}*_test_log-odds*.csv"
        for d in SEARCH_DIRS:
            if not d.is_dir():
                continue
            for p in d.glob(pattern):
                name = p.name
                if name in seen_basenames:
                    continue
                after_prefix = name[len("scores_"):]
                if pfx and not after_prefix.startswith(pfx):
                    continue
                if not pfx:
                    if any(after_prefix.startswith(x) for x in _EVAL_PREFIXES_ORDERED):
                        continue
                rest = after_prefix[len(pfx):] if pfx else after_prefix
                chosen_task = None
                for t in EVAL_TASKS:
                    if f"_{t}_test_log-odds" in rest:
                        chosen_task = t
                        break
                if chosen_task is None:
                    continue
                model_short = rest.split(f"_{chosen_task}_test_log-odds", 1)[0]
                if method["match"](model_short):
                    matches[chosen_task].append(p)
                    seen_basenames.add(name)
    return matches


def cell_value(method: dict, eval_prefix, variant: str):
    files_by_task = find_score_files(method, eval_prefix)
    vals = []
    for task in EVAL_TASKS:
        candidates = files_by_task.get(task, [])
        if not candidates:
            continue
        path = sorted(candidates)[-1]
        try:
            df = load_scores(path)
            metrics = compute_all_metrics(df)
        except Exception:
            continue
        if variant not in metrics:
            continue
        v = metrics[variant].get(METRIC)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        vals.append((task, float(v), path.name))
    n = len(vals)
    if n == 0:
        return dict(mean=None, se=None, sd=None, n=0, vals=[])
    arr = np.array([v for _, v, _ in vals], dtype=float) * 100.0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    return dict(mean=mean, se=se, sd=sd, n=n, vals=vals)


def fmt_cell(c: dict, expected_n: int) -> str:
    if c["n"] == 0:
        return "—"
    if c["se"] is None or math.isnan(c["se"]):
        s = f"{c['mean']:.2f}"
    else:
        s = f"{c['mean']:.2f} ± {c['se']:.2f}"
    if c["n"] != expected_n:
        s += f" (n={c['n']})"
    return s


# --- Main ---------------------------------------------------------------------

def main():
    long_rows = []
    cell_rows = []
    table_rows = []

    for m in METHODS:
        cells = {}
        for col_label, eval_prefix, variant in COLUMNS:
            if col_label in NA_COLS.get(m["num"], set()):
                cells[col_label] = "---"
                cell_rows.append(dict(
                    method_num=m["num"], method=m["label"],
                    column=col_label, mean=None, sd=None, se=None, n=0, note="NA per template",
                ))
                continue
            c = cell_value(m, eval_prefix, variant)
            cells[col_label] = fmt_cell(c, N_EXPECTED)
            cell_rows.append(dict(
                method_num=m["num"], method=m["label"],
                column=col_label, mean=c["mean"], sd=c["sd"], se=c["se"], n=c["n"],
                note=("OK" if c["n"] == N_EXPECTED else f"missing {N_EXPECTED - c['n']}/{N_EXPECTED}"),
            ))
            for task, val, fn in c["vals"]:
                long_rows.append(dict(
                    method_num=m["num"], method=m["label"],
                    column=col_label, eval_prefix=eval_prefix, variant=variant,
                    task=task, value=val, file=fn,
                ))
        table_rows.append(dict(num=m["num"], label=m["label"], **cells))

    long_csv = METRICS_DIR / f"qwen_{TASK_TYPE}_{METRIC}_table_long.csv"
    cells_csv = METRICS_DIR / f"qwen_{TASK_TYPE}_{METRIC}_table_cells.csv"
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_csv, index=False)

    print(f"\nQwen {TASK_LABEL} [v7/fix1] {METRIC_LABEL} × 100 — mean ± SE across {N_EXPECTED} tasks")
    print(f"Source: outputs-rerun-wandb/\n")
    header = ["Method"] + [c[0] for c in COLUMNS]
    sep = ["---"] * len(header)
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(sep) + " |")
    for r in table_rows:
        cells_display = [str(r[c[0]]) for c in COLUMNS]
        print(f"| {r['num']} {r['label']} | " + " | ".join(cells_display) + " |")

    print(f"\nCSVs:\n- [{long_csv.relative_to(REPO)}]({long_csv.relative_to(REPO)})")
    print(f"- [{cells_csv.relative_to(REPO)}]({cells_csv.relative_to(REPO)})")


if __name__ == "__main__":
    main()
