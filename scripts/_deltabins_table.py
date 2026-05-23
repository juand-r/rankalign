#!/usr/bin/env python3
"""
Build delta-bins sweep results tables for persona-v1.

Walks scores_basetyp-v7-*--fix1*persona-v1-*test_log-odds_tc_*.csv files in
the configured outputs dir, groups by (base_model, delta_bins, split, task),
computes metrics, then aggregates mean +/- stderr across 3 tasks per split.

Splits:
  ID  -> psychopathy, machiavellianism, narcissism
  OOD -> desire-to-create-allies, interest-in-music, interest-in-science

For each (base_model, split) -> a table:
  rows = delta_bins value (5 / 10 / 30 / 50 / 100, sorted ascending)
  cols = gen_roc(raw), gen_roc(tc), val_acc, val_roc, pearson(raw), pearson(tc)
  cells = mean +/- stderr across 3 tasks

Usage:
  python scripts/_deltabins_table.py [--outputs-dir DIR] [--out FILE]
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from glob import glob
from math import sqrt
from pathlib import Path

import numpy as np

# Make summarize_scores_file importable.
sys.path.insert(0, str(Path(__file__).parent))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

ID_PERSONAS = {"psychopathy", "machiavellianism", "narcissism"}
OOD_PERSONAS = {
    "desire-to-create-allies",
    "interest-in-music",
    "interest-in-science",
}

# Filename pattern for variant 4 (--tc-self) fix1 evals at log-odds + TC.
# Long model names get truncated by the eval pipeline at the model name and
# replaced with a short hex hash. So the suffix after delta+epoch can be
# arbitrarily truncated. We just need {base, delta, epoch, persona, date}.
FILENAME_RE = re.compile(
    r'^scores_basetyp-v7-(?P<base>google--gemma-2-(?:2b|2b-it|9b-it))'
    r'-delta(?P<delta>[0-9.]+)-epoch(?P<epoch>\d+)--persona-v1-all'
    r'--d2g--random--alpha1\.0--tc-self--full-completion--nllv1\.0'
    r'--nllg1\.0--.*'  # may be truncated/hashed here
    r'_persona-v1-(?P<persona>[a-z-]+?)_test_log-odds_tc_'
    r'(?P<date>\d+)\.csv$'
)


def load_bins_lookup(auto_delta_csv: Path) -> dict:
    """Returns {(base_model, delta_rounded_6dp): bins} from auto_delta_log.csv."""
    lookup = {}
    if not auto_delta_csv.is_file():
        return lookup
    with open(auto_delta_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                model = row["model"]
                bins = int(row["delta_bins"])
                delta = float(row["delta_used"])
            except (KeyError, ValueError):
                continue
            lookup[(model, f"{delta:.6f}")] = bins
    return lookup


def collect_files(outputs_dir: Path):
    """Group score files into {(base_model, delta_str, persona) -> path}."""
    files = sorted(glob(str(outputs_dir / "scores_basetyp-v7-*persona-v1*tc_*.csv")))
    out = {}
    for path in files:
        name = Path(path).name
        m = FILENAME_RE.match(name)
        if not m:
            continue
        base_repl = m.group("base")
        base_model = base_repl.replace("--", "/")
        delta_str = m.group("delta")
        persona = m.group("persona")
        key = (base_model, delta_str, persona)
        # If multiple dates, prefer the latest one
        if key not in out or name > Path(out[key]).name:
            out[key] = path
    return out


def aggregate_metrics(values):
    """Mean +/- standard error across a list of metric values."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, float("nan"), n
    sem = float(arr.std(ddof=1) / sqrt(n))
    return mean, sem, n


def fmt_cell(mean, sem, decimals=3):
    if mean != mean:  # NaN
        return "n/a"
    if sem != sem:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}\u00b1{sem:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        default="/datastor2/jdr/rankalign/outputs",
        type=Path,
    )
    parser.add_argument(
        "--auto-delta-log",
        default="/datastor2/jdr/rankalign/models2/auto_delta_log.csv",
        type=Path,
    )
    parser.add_argument("--out", default="-", type=str,
                        help="Output markdown file (or '-' for stdout)")
    args = parser.parse_args()

    bins_lookup = load_bins_lookup(args.auto_delta_log)
    file_map = collect_files(args.outputs_dir)

    # Compute per-(base, delta, persona, variant) metric dicts.
    # rows[(base, delta, persona)] = {"raw": metrics, "tc": metrics}
    rows = {}
    for (base, delta_str, persona), path in file_map.items():
        try:
            df = load_scores(path)
            metrics = compute_all_metrics(df)
        except Exception as exc:  # noqa: BLE001
            print(f"# WARN: could not parse {path}: {exc}", file=sys.stderr)
            continue
        rows[(base, delta_str, persona)] = metrics

    # Build {(base, bins, split) -> per-task metric values} aggregations.
    # We store raw (gen_score), tc (gen_score_typcorr), and val_*.
    METRIC_FIELDS = ["gen_roc", "val_roc", "val_acc", "pearson"]
    # variant key in metrics dict: "raw" or "tc"
    agg = defaultdict(list)  # [(base,bins,split,variant,metric)] -> list

    for (base, delta_str, persona), m in rows.items():
        if persona in ID_PERSONAS:
            split = "ID"
        elif persona in OOD_PERSONAS:
            split = "OOD"
        else:
            print(f"# WARN: unknown persona {persona}, skipping", file=sys.stderr)
            continue

        delta_rounded = f"{float(delta_str):.6f}"
        bins = bins_lookup.get((base, delta_rounded), None)
        if bins is None:
            print(f"# WARN: no bins lookup for {base} delta={delta_rounded}", file=sys.stderr)
            continue

        for variant in ("raw", "tc"):
            if variant not in m:
                continue
            mv = m[variant]
            for field in METRIC_FIELDS:
                agg[(base, bins, split, variant, field)].append(mv.get(field))

    # Now produce 6 tables: 3 bases x 2 splits.
    out_lines = []
    out_lines.append("# Delta-bins sweep results (persona-v1, variant 4: comb + --self-typcorr + --log-odds)")
    out_lines.append("")
    out_lines.append("Eval flags: `--self-typcorr --base-typcorr --base-model <BASE> --log-odds`, disc-shots=few.")
    out_lines.append("")
    out_lines.append("Cells: mean ± stderr across 3 tasks per split.")
    out_lines.append("")
    out_lines.append("- ID = psychopathy, machiavellianism, narcissism")
    out_lines.append("- OOD = desire-to-create-allies, interest-in-music, interest-in-science")
    out_lines.append("- raw = gen_score (no TC); tc = gen_score_typcorr (eval-time self-TC)")
    out_lines.append("")

    bases_present = sorted({b for (b, _, _, _, _) in agg.keys()})
    splits = ["ID", "OOD"]
    bins_order = sorted({bins for (_, bins, _, _, _) in agg.keys()})

    for base in bases_present:
        for split in splits:
            out_lines.append(f"## {base} ({split})")
            out_lines.append("")
            cols = [
                ("gen_roc", "raw"),
                ("gen_roc", "tc"),
                ("val_acc", "raw"),
                ("val_roc", "raw"),
                ("pearson", "raw"),
                ("pearson", "tc"),
            ]
            header = "| bins |"
            for field, var in cols:
                header += f" {field}({var}) |"
            out_lines.append(header)
            out_lines.append("|------|" + "------|" * len(cols))
            for bins in bins_order:
                line = f"| {bins} |"
                any_data = False
                for field, var in cols:
                    vals = agg.get((base, bins, split, var, field), [])
                    mean, sem, n = aggregate_metrics(vals)
                    if n > 0:
                        any_data = True
                    line += f" {fmt_cell(mean, sem)} |"
                if any_data:
                    out_lines.append(line)
                else:
                    # No data for this row, skip silently (table stays clean).
                    pass
            out_lines.append("")

    # Coverage report at end
    out_lines.append("## Coverage")
    out_lines.append("")
    coverage = defaultdict(set)
    for (base, bins, split, variant, field), values in agg.items():
        n_present = sum(1 for v in values if v is not None and not np.isnan(v))
        if n_present > 0 and field == "gen_roc" and variant == "raw":
            coverage[(base, bins)].add(split)
    for base in bases_present:
        for bins in bins_order:
            splits_have = coverage.get((base, bins), set())
            tag = ", ".join(sorted(splits_have)) or "(no data)"
            out_lines.append(f"- {base} bins={bins}: {tag}")

    text = "\n".join(out_lines) + "\n"
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"Wrote table to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
