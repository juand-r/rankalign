#!/usr/bin/env python3
"""
Build the main9 results table for persona-v1 (9 models: 3 bases x 3 variants).

Filters score CSVs to bins=10 main9 runs only (by delta value matching the
auto_delta_log entry for bins=10 per base).

For each (base_model, variant, eval_tc_kind, split), aggregates metrics
across personas (mean +/- stderr).

Variants:
  3 (New)         -> trained without --tc-self / --tc-neg, eval'd with both:
                     - self+base (filename prefix `basetyp-`)
                     - neg+base (filename prefix `basetypneg-`)
  4 (New+selfTC)  -> trained with --tc-self, eval'd self+base only.
  7 (New+negTC)   -> trained with --tc-neg, eval'd neg+base only.
"""
import csv
import re
import sys
from collections import defaultdict
from glob import glob
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

ID_PERSONAS = {"psychopathy", "machiavellianism", "narcissism"}
OOD_PERSONAS = {"desire-to-create-allies", "interest-in-music",
                "interest-in-science"}

# Filename prefixes for the eval-kind:
#   basetyp-     -> --self-typcorr + --base-typcorr  (i.e. "self+base")
#   basetypneg-  -> --neg-typcorr  + --base-typcorr  (i.e. "neg+base")
# Note: the eval prefix logic in run_eval_semi.sh is a precedence ladder, so
# "basetypneg-" only appears when both --base + --neg flags are passed
# (we do that for variant 3 neg+base and variant 7 neg+base).

# We look for v7-*-fix1[_merged] dirs whose delta == bins=10 auto-delta
# for the corresponding base.
FILENAME_RE = re.compile(
    r'^scores_(?P<eval_pfx>basetyp|basetypneg)-v7-'
    r'(?P<base>google--gemma-2-(?:2b|2b-it|9b-it))'
    r'-delta(?P<delta>[0-9.]+)-epoch(?P<epoch>\d+)--persona-v1-all'
    r'--d2g--random--alpha1\.0'
    r'(?P<tc>(?:--tc-self|--tc-neg)?)'
    r'--full-completion--nllv1\.0--nllg1\.0--.*'  # may be truncated
    r'_persona-v1-(?P<persona>[a-z-]+?)_test_log-odds_tc_'
    r'(?P<date>\d+)\.csv$'
)


def load_bins10_deltas(auto_delta_csv: Path) -> dict[str, float]:
    """Return {base_model: delta_used_for_bins=10}."""
    out = {}
    if not auto_delta_csv.is_file():
        return out
    with open(auto_delta_csv) as f:
        for row in csv.DictReader(f):
            try:
                bins = int(row["delta_bins"])
                if bins != 10:
                    continue
                model = row["model"]
                delta = float(row["delta_used"])
            except (KeyError, ValueError):
                continue
            # Last write wins (assumes auto-delta is deterministic per base/task).
            out[model] = delta
    return out


def parse_variant_kind(tc_part: str, eval_pfx: str) -> tuple[int, str]:
    """Return (variant_number, eval_kind) where:
      variant_number in {3, 4, 7}
      eval_kind in {"self+base", "neg+base"}.
    """
    if tc_part == "--tc-self":
        return 4, "self+base" if eval_pfx == "basetyp" else None
    if tc_part == "--tc-neg":
        return 7, "neg+base" if eval_pfx == "basetypneg" else None
    # Variant 3: no --tc-* in dirname; eval is both self+base and neg+base
    if eval_pfx == "basetyp":
        return 3, "self+base"
    if eval_pfx == "basetypneg":
        return 3, "neg+base"
    return None, None


def aggregate_metrics(values):
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
    if mean != mean:
        return "n/a"
    if sem != sem:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}\u00b1{sem:.{decimals}f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir",
                    default="/datastor2/jdr/rankalign/outputs", type=Path)
    ap.add_argument("--auto-delta-log",
                    default="/datastor2/jdr/rankalign/models2/auto_delta_log.csv",
                    type=Path)
    ap.add_argument("--out", default="-", type=str)
    args = ap.parse_args()

    bins10 = load_bins10_deltas(args.auto_delta_log)
    if not bins10:
        print("ERROR: could not load bins=10 deltas from auto_delta_log",
              file=sys.stderr)
        sys.exit(1)

    files = sorted(glob(str(args.outputs_dir / "scores_basetyp*-v7-*persona-v1*tc_*.csv")))

    # rows[(base, variant, eval_kind, persona)] = metrics dict
    rows = {}
    skipped_wrong_delta = 0
    skipped_unparsable = 0
    for path in files:
        name = Path(path).name
        m = FILENAME_RE.match(name)
        if not m:
            skipped_unparsable += 1
            continue
        base_repl = m.group("base")
        base_model = base_repl.replace("--", "/")
        delta_str = m.group("delta")
        try:
            delta = float(delta_str)
        except ValueError:
            continue
        target_delta = bins10.get(base_model)
        if target_delta is None:
            continue
        # Tolerate small float-formatting differences.
        if abs(delta - target_delta) / max(abs(target_delta), 1e-9) > 1e-6:
            skipped_wrong_delta += 1
            continue

        eval_pfx = m.group("eval_pfx")
        tc_part = m.group("tc")
        variant, eval_kind = parse_variant_kind(tc_part, eval_pfx)
        if variant is None or eval_kind is None:
            continue
        persona = m.group("persona")

        try:
            df = load_scores(path)
            metrics = compute_all_metrics(df)
        except Exception as exc:
            print(f"# WARN: skipping {name}: {exc}", file=sys.stderr)
            continue

        rows[(base_model, variant, eval_kind, persona)] = metrics

    # Build aggregations.
    METRIC_FIELDS = ["gen_roc", "val_roc", "val_acc", "pearson"]
    # agg[(base, variant, eval_kind, split, variant_inner, metric)] -> list
    agg = defaultdict(list)

    for (base, variant, eval_kind, persona), m in rows.items():
        if persona in ID_PERSONAS:
            split = "ID"
        elif persona in OOD_PERSONAS:
            split = "OOD"
        else:
            continue
        for v_inner in ("raw", "tc"):
            if v_inner not in m:
                continue
            mv = m[v_inner]
            for fld in METRIC_FIELDS:
                agg[(base, variant, eval_kind, split, v_inner, fld)].append(mv.get(fld))

    # Generate per-base tables.
    bases = sorted({k[0] for k in agg.keys()})
    out_lines = []
    out_lines.append("# Main 9-job fix1 results (persona-v1, DELTA_BINS=10)")
    out_lines.append("")
    out_lines.append("Trained with `ranking_loss_ref_fix.py`, `--delta-bins 10`, disc-shots=few,")
    out_lines.append("`--no-force-same-x`, semi-supervised 0.1, log-odds.")
    out_lines.append("")
    out_lines.append("Eval flags: `--<self|neg>-typcorr --base-typcorr --base-model <BASE> --log-odds`,")
    out_lines.append("disc-shots=few, log-odds metric, no length normalization.")
    out_lines.append("")
    out_lines.append("Cells: mean ± stderr across 3 personas per split.")
    out_lines.append("")
    out_lines.append("- ID = psychopathy, machiavellianism, narcissism")
    out_lines.append("- OOD = desire-to-create-allies, interest-in-music, interest-in-science")
    out_lines.append("- raw = `gen_score` (no eval-time TC); tc = `gen_score_typcorr` (eval-time TC)")
    out_lines.append("")

    # row labels mapped from (variant, eval_kind):
    #   (3, "self+base") -> "#3.New / self+base"
    #   (3, "neg+base")  -> "#3.New / neg+base"
    #   (4, "self+base") -> "#4.New+selfTC / self+base"
    #   (7, "neg+base")  -> "#7.New+negTC / neg+base"
    ROW_ORDER = [
        (3, "self+base", "#3.New / self+base"),
        (3, "neg+base",  "#3.New / neg+base"),
        (4, "self+base", "#4.New+selfTC / self+base"),
        (7, "neg+base",  "#7.New+negTC / neg+base"),
    ]
    splits = ["ID", "OOD"]
    cols = [
        ("gen_roc", "raw", "gen_roc(raw)"),
        ("gen_roc", "tc",  "gen_roc(tc)"),
        ("val_acc", "raw", "val_acc"),
        ("val_roc", "raw", "val_roc"),
        ("pearson", "raw", "pearson(raw)"),
        ("pearson", "tc",  "pearson(tc)"),
    ]

    for base in bases:
        for split in splits:
            out_lines.append(f"## {base} ({split})")
            out_lines.append("")
            header = "| variant / eval |"
            for _, _, name in cols:
                header += f" {name} |"
            out_lines.append(header)
            out_lines.append("|" + "------|" * (len(cols) + 1))
            for variant, eval_kind, label in ROW_ORDER:
                line = f"| {label} |"
                any_data = False
                for fld, v_inner, _ in cols:
                    vals = agg.get((base, variant, eval_kind, split, v_inner, fld), [])
                    mean, sem, n = aggregate_metrics(vals)
                    if n > 0:
                        any_data = True
                    line += f" {fmt_cell(mean, sem)} |"
                if any_data:
                    out_lines.append(line)
                else:
                    line_na = f"| {label} |" + " n/a |" * len(cols)
                    out_lines.append(line_na)
            out_lines.append("")

    out_lines.append("## Coverage")
    out_lines.append("")
    coverage = defaultdict(set)
    for (base, variant, eval_kind, split, v_inner, fld), values in agg.items():
        if v_inner != "raw" or fld != "gen_roc":
            continue
        n_present = sum(1 for v in values if v is not None and not np.isnan(v))
        if n_present > 0:
            coverage[(base, variant, eval_kind)].add(f"{split}({n_present})")
    for base in bases:
        for variant, eval_kind, label in ROW_ORDER:
            cov = coverage.get((base, variant, eval_kind), set())
            tag = ", ".join(sorted(cov)) or "(no data)"
            out_lines.append(f"- {base} {label}: {tag}")

    out_lines.append("")
    if skipped_wrong_delta or skipped_unparsable:
        out_lines.append(
            f"Filter stats: {skipped_wrong_delta} files skipped (wrong delta, "
            f"e.g. sweep models); {skipped_unparsable} files unparsable.")

    text = "\n".join(out_lines) + "\n"
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"Wrote table to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
