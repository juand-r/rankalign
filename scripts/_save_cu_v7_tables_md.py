#!/usr/bin/env python3
"""Save humaneval correct-upper v7 tables as a markdown snapshot in docs/."""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
BUILDER = REPO / "scripts/_build_humaneval_cu_v7_table.py"
DOCS = REPO / "docs"
METRICS = ["gen_roc", "pearson", "spearman", "val_roc", "val_acc"]

METRIC_LABELS = {
    "gen_roc":  "GenROC × 100 (mean ± SE across problems)",
    "pearson":  "Pearson(gen, val) × 100",
    "spearman": "Spearman(gen, val) × 100",
    "val_roc":  "ValROC × 100",
    "val_acc":  "ValAcc × 100",
}

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M CT")
date_str = now.strftime("%Y-%m-%d")

lines = [
    f"# humaneval-v2.1correct-upper × gemma-4-31B-it — v7 epoch2 tables",
    f"",
    f"Snapshot: **{timestamp}**",
    f"",
    f"Model: gemma-4-31B-it, trained on humaneval-v2.1correct-upper-all, epoch 2.",
    f"Columns = scoring method at eval time (Raw / PMI base / Neg base etc.).",
    f"Rows = training method. Cells with `(n=N)` are partial — N < 82 problems scored.",
    f"`—` = no CSVs on disk yet. `---` = N/A per eval policy.",
    f"",
    f"To refresh: `python scripts/_save_cu_v7_tables_md.py`",
    f"",
]

for metric in METRICS:
    env = {**os.environ, "HUMANEVAL_METRIC": metric}
    result = subprocess.run(
        [PYTHON, str(BUILDER)],
        env=env,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()

    # Extract just the markdown table (lines starting with |)
    table_lines = [l for l in output.split("\n") if l.startswith("|")]
    # Extract header line (first non-empty, non-CSV line)
    header_lines = [l for l in output.split("\n") if l.startswith("Humaneval")]

    lines.append(f"## {METRIC_LABELS.get(metric, metric)}")
    lines.append("")
    if header_lines:
        lines.append(f"*{header_lines[0].strip()}*")
        lines.append("")
    if table_lines:
        lines.extend(table_lines)
    else:
        lines.append("*(no data)*")
    lines.append("")

# Write file
out_path = DOCS / f"humaneval_cu_v7_tables_{date_str}.md"
out_path.write_text("\n".join(lines) + "\n")
print(f"Saved: {out_path}")
print(f"  ({len(lines)} lines)")
