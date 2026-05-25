#!/usr/bin/env python3
"""Split the persona-v1 v7 GenROC table by ID / OOD persona type.

Reads the long-format CSVs produced by `_build_persona_v1_table_v7.py`:
    metrics-from-scores/persona_v1_v7_gemma-2-2b-it_all_gen_roc_table_long.csv
    metrics-from-scores/persona_v1_v7_gemma-2-9b-it_all_gen_roc_table_long.csv

Splits per the persona-v1 build report (data/persona/v1/_BUILD_REPORT.json):
    ID  = psychopathy, machiavellianism, narcissism      (3 tasks)
    OOD = desire-to-create-allies, interest-in-music,
          interest-in-science                             (3 tasks)

For each (model, method, column), reports mean ± SE across the 3 tasks in
each subset. Cells are gen_roc × 100 (consistent with the v7 table).
"""
import sys
from pathlib import Path
import pandas as pd

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
LONG = {
    "gemma-2-2b-it": REPO / "metrics-from-scores" / "persona_v1_v7_gemma-2-2b-it_all_gen_roc_table_long.csv",
    "gemma-2-9b-it": REPO / "metrics-from-scores" / "persona_v1_v7_gemma-2-9b-it_all_gen_roc_table_long.csv",
}
ID_TASKS = {
    "persona-v1-psychopathy",
    "persona-v1-machiavellianism",
    "persona-v1-narcissism",
}
OOD_TASKS = {
    "persona-v1-desire-to-create-allies",
    "persona-v1-interest-in-music",
    "persona-v1-interest-in-science",
}
COL_ORDER = ["Raw", "PMI self", "PMI base", "Neg self", "Neg base"]
ROW_ORDER = [
    (0,  "Base"),
    (1,  "SFT labelonly 10%"),
    (2,  "RankAlign"),
    (3,  "New + fsx [-TC]"),
    (4,  "New + PMI + fsx"),
    (5,  "RA + PMI + fsx [-NLL]"),
    (6,  "RA + PMI [+TC]"),
    (11, "New + PMI [-fsx]"),
    (7,  "New + NegTC + fsx"),
    (8,  "RA + NegTC + fsx [-NLL]"),
    (9,  "RA + NegTC [+TC]"),
    (12, "New + NegTC [-fsx]"),
]


def fmt_cell(values):
    """Mean ± SE over `values` (length 3 expected). Returns string or '—'."""
    vs = [v for v in values if pd.notna(v)]
    if not vs:
        return "—"
    n = len(vs)
    if n == 1:
        return f"{100*vs[0]:.2f} ± —"
    mean = sum(vs) / n
    var = sum((v - mean) ** 2 for v in vs) / (n - 1)
    se = (var ** 0.5) / (n ** 0.5)
    return f"{100*mean:.2f} ± {100*se:.2f}"


def render_table(df, subset_tasks, subset_label, model_label):
    sub = df[df["task"].isin(subset_tasks)]
    print(f"### {model_label} — {subset_label} (N = {len(subset_tasks)} tasks)")
    print()
    print("| Method | " + " | ".join(COL_ORDER) + " |")
    print("|---|" + "|".join(["---"] * len(COL_ORDER)) + "|")
    for num, label in ROW_ORDER:
        cells = []
        for col in COL_ORDER:
            mask = (sub["method_num"] == num) & (sub["column"] == col)
            cell_df = sub[mask]
            # Each row in cell_df has one value per task. Some methods are
            # not applicable for some columns (Neg* for self-TC trained, etc).
            if cell_df.empty:
                cells.append("---" if col not in ("Raw",) else "—")
                continue
            values = cell_df["value"].tolist()
            cells.append(fmt_cell(values))
        print(f"| {num} {label} | " + " | ".join(cells) + " |")
    print()


def main():
    print("# Persona-v1 GenROC table v7 — ID vs OOD split")
    print()
    from datetime import datetime, timezone
    print(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print()
    print("**Trained on:** `persona-v1` (with the 3 ID personas' labels flipped per the build report).")
    print("**Eval on:** 6 persona-v1 test tasks split into ID and OOD per `data/persona/v1/_BUILD_REPORT.json`.")
    print()
    print("- **ID** (3 tasks, label-flipped): psychopathy, machiavellianism, narcissism")
    print("- **OOD** (3 tasks, labels unchanged): desire-to-create-allies, interest-in-music, interest-in-science")
    print()
    print("**Cells:** `gen_roc × 100 ± SE` over the 3 tasks in the subset. `---` = trained-TC rule (column not applicable). `—` = no data.")
    print()
    for model, path in LONG.items():
        if not path.is_file():
            print(f"## {model}\n\n(missing long CSV: {path})\n")
            continue
        df = pd.read_csv(path)
        # `column` is one of Raw/PMI self/PMI base/Neg self/Neg base.
        # `method_num` is the integer setting label.
        # `task` is e.g. persona-v1-psychopathy.
        # Drop any non-task rows (defensive).
        df = df[df["task"].astype(str).str.startswith("persona-v1-")]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        print(f"## {model}")
        print()
        render_table(df, ID_TASKS, "ID", model)
        render_table(df, OOD_TASKS, "OOD", model)


if __name__ == "__main__":
    main()
