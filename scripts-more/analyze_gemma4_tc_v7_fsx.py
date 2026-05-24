#!/usr/bin/env python3
"""Aggregate gemma-4-31B-it humaneval v7 (fsx) trained-model score files.

Compares three new fsx settings (v7) against three v6 RankAlign baselines:

  v7 settings (new, this run):
    New+fsx       = s3  evaluated with self-TC + neg-TC
    New+fsx+tc    = s4  evaluated with self-TC
    New+fsx+negtc = s7  evaluated with neg-TC

  v6 baselines (previous run, already on mll):
    RankAlign       = #2  evaluated with self-TC + neg-TC
    RankAlign+tc    = #6  evaluated with self-TC
    RankAlign+negtc = #9  evaluated with neg-TC

Works for humaneval-v2.1correct-upper OR humaneval-v2.1correct-multi.

Score filename conventions
--------------------------
v7 adapters are saved as absolute paths.  eval_by_claude.py calls
build_model_short() which (for abs paths) uses the basename, preserving
double-hyphens, and truncates to 160 chars (151 + '_' + 8-hex MD5 if longer).

  Full adapter basename example (s4, correct-upper):
    v7-google--gemma-4-31B-it-delta0.12-epoch2--humaneval-v2.1correct-upper-
    all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--
    force-same-x--ppd--vallogodds--semi0.1--fix1   (189 chars → truncated)

  Distinguishing features visible before the 151-char cut:
    v7 vs v6 : "v7-google--" vs "v6-google--"
    s4 (tc-self) : "--tc-self--" present
    s7 (tc-neg)  : "--tc-neg--"  present
    s3 (no tc)   : neither, but "--nllv1.0--" present

v6 baselines use the same double-hyphen convention.

Usage:
  python scripts-more/analyze_gemma4_tc_v7_fsx.py \\
      --scripts-dir /datastor1/jdr/gv-gap/rankalign/scripts \\
      --scores-v6-dir /datastor1/jdr/gv-gap/rankalign/outputs_gemma4_from_pod \\
      --scores-v7-dir /datastor2/jdr/rankalign/outputs_gemma4_from_pod/correct_upper_s3s4s7 \\
      --dataset correct-upper

  For correct-multi:
      --scores-v7-dir /datastor2/jdr/rankalign/outputs_gemma4_from_pod/correct_multi_s3s4s7 \\
      --dataset correct-multi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_SCRIPTS_DIR = "/datastor1/jdr/gv-gap/rankalign/scripts"
DEFAULT_SCORES_V6_DIR = "/datastor1/jdr/gv-gap/rankalign/outputs_gemma4_from_pod"
DEFAULT_SCORES_V7_DIR = "/datastor2/jdr/rankalign/outputs_gemma4_from_pod"

METRIC_COLS = ["gen_roc", "val_roc", "val_acc", "pearson", "spearman"]
VARIANT_ORDER = ["raw", "tc", "lenorm", "tc+lenorm"]

# Group display order for tables
SELF_GROUPS = [
    "RankAlign_self",
    "RankAlign+tc_self",
    "New+fsx_self",
    "New+fsx+tc_self",
]
NEG_GROUPS = [
    "RankAlign_neg",
    "RankAlign+negtc_neg",
    "New+fsx_neg",
    "New+fsx+negtc_neg",
]


def classify_v7(name: str) -> str | None:
    """Map a v7 score CSV filename to a group, or None if not a v7 fsx file."""
    if "v7-google--" not in name:
        return None

    if name.startswith("scores_basetypneg-"):
        eval_side = "neg"
    elif name.startswith("scores_basetyp-"):
        eval_side = "self"
    else:
        return None

    # Training TC present in adapter name before the 151-char truncation point.
    if "--tc-self--" in name:
        setting = "New+fsx+tc"
    elif "--tc-neg--" in name:
        setting = "New+fsx+negtc"
    elif "--nllv1.0--" in name:
        # No TC in training, but nllv1.0/nllg1.0 flags confirm it's a v7 fsx setting.
        setting = "New+fsx"
    else:
        return None  # unrecognised v7 variant

    return f"{setting}_{eval_side}"


def classify_v6(name: str) -> str | None:
    """Map a v6 (RankAlign baseline) score CSV filename to a group, or None."""
    if "v6-google--" not in name:
        return None

    if name.startswith("scores_basetypneg-"):
        eval_side = "neg"
        if "--tc-neg--" in name:
            return "RankAlign+negtc_neg"
        return "RankAlign_neg"

    if name.startswith("scores_basetyp-"):
        eval_side = "self"
        if "--tc-self--" in name:
            return "RankAlign+tc_self"
        return "RankAlign_self"

    return None


def classify_base(name: str) -> str | None:
    """Map an untrained gemma-4-31B-it base-model score CSV to a group, or None.

    Base model HF path "google/gemma-4-31B-it" → model_short "v6-google_gemma-4-31B-it".
    (The 'v6-' prefix is added for all HF-style paths regardless of actual version.)
    """
    if name.startswith("scores_basetypneg-v6-google_gemma-4-31B-it"):
        return "Base_neg"
    if name.startswith("scores_basetyp-v6-google_gemma-4-31B-it"):
        return "Base_self"
    # Also accept the older self-/neg- prefix (no base-typicality) from
    # pre-base-typicality evals.
    if name.startswith("scores_self-v6-google_gemma-4-31B-it"):
        return "Base_self"
    if name.startswith("scores_neg-v6-google_gemma-4-31B-it"):
        return "Base_neg"
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripts-dir", default=DEFAULT_SCRIPTS_DIR,
                    help="rankalign scripts/ dir (for summarize_scores_file import)")
    ap.add_argument("--scores-v6-dir", default=DEFAULT_SCORES_V6_DIR,
                    help="dir with v6 RankAlign baseline score CSVs")
    ap.add_argument("--scores-v7-dir", default=DEFAULT_SCORES_V7_DIR,
                    help="dir with v7 New+fsx score CSVs from current run")
    ap.add_argument("--base-scores-dir", default=None,
                    help="dir with untrained base-model score CSVs "
                         "(default: <scores-v6-dir>/_base_model_eval)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for CSVs and analysis text "
                         "(default: <scores-v7-dir>/_analysis)")
    ap.add_argument("--dataset", default="correct-upper",
                    choices=["correct-upper", "correct-multi"],
                    help="which dataset to restrict to (filters by task name in filename)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scores_v6_dir = Path(args.scores_v6_dir)
    scores_v7_dir = Path(args.scores_v7_dir)
    out_dir = Path(args.out_dir) if args.out_dir else scores_v7_dir / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.scripts_dir)
    from summarize_scores_file import summarize_files  # noqa: E402

    base_dir = (Path(args.base_scores_dir) if args.base_scores_dir
                else scores_v6_dir / "_base_model_eval")

    # Only keep files that reference the selected dataset.
    dataset_token = f"humaneval-v2.1{args.dataset}-"

    def matches_dataset(fname: str) -> bool:
        return dataset_token in fname

    groups: dict[str, list[Path]] = {}
    unclassified: list[str] = []

    # --- v6 baselines ---
    if scores_v6_dir.is_dir():
        for p in sorted(scores_v6_dir.glob("scores_*.csv")):
            if not matches_dataset(p.name):
                continue
            g = classify_v6(p.name)
            if g is None:
                unclassified.append(f"v6: {p.name}")
                continue
            groups.setdefault(g, []).append(p)
    else:
        print(f"NOTE: v6 scores dir not found ({scores_v6_dir}); baseline rows omitted",
              file=sys.stderr)

    # --- v7 new fsx settings ---
    if scores_v7_dir.is_dir():
        for p in sorted(scores_v7_dir.glob("scores_*.csv")):
            if not matches_dataset(p.name):
                continue
            g = classify_v7(p.name)
            if g is None:
                unclassified.append(f"v7: {p.name}")
                continue
            groups.setdefault(g, []).append(p)
    else:
        print(f"NOTE: v7 scores dir not found ({scores_v7_dir}); new-fsx rows omitted",
              file=sys.stderr)

    # --- base model ---
    if base_dir.is_dir():
        for p in sorted(base_dir.glob("scores_*.csv")):
            if not matches_dataset(p.name):
                continue
            g = classify_base(p.name)
            if g is None:
                unclassified.append(f"base: {p.name}")
                continue
            groups.setdefault(g, []).append(p)
    else:
        print(f"NOTE: base dir not found ({base_dir}); Base row omitted",
              file=sys.stderr)

    if unclassified:
        print(f"\nUNCLASSIFIED ({len(unclassified)}):", file=sys.stderr)
        for u in unclassified[:20]:
            print(f"  {u}", file=sys.stderr)
        if len(unclassified) > 20:
            print(f"  ... and {len(unclassified) - 20} more", file=sys.stderr)

    print(f"\nDataset: {args.dataset}")
    print("Group file counts:")
    for g, paths in sorted(groups.items()):
        print(f"  {g:<24} {len(paths)}")

    if not groups:
        print("No groups found — check score directories and filenames.", file=sys.stderr)
        sys.exit(1)

    # Summarise each group
    per_task_frames = []
    for g, paths in sorted(groups.items()):
        df = summarize_files(paths, quiet=True)
        df.insert(0, "group", g)
        per_task_frames.append(df)
    per_task = pd.concat(per_task_frames, ignore_index=True)

    per_task_csv = out_dir / f"per_task_metrics_{args.dataset}.csv"
    per_task.to_csv(per_task_csv, index=False)
    print(f"\nPer-task metrics ({len(per_task)} rows) → {per_task_csv}")

    agg = (
        per_task.groupby(["group", "variant"])[METRIC_COLS]
        .agg(["mean", "std", "count"])
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    summary_csv = out_dir / f"summary_mean_std_{args.dataset}.csv"
    agg.to_csv(summary_csv, index=False)
    print(f"Summary mean/std ({len(agg)} rows) → {summary_csv}")

    present = set(per_task.group.unique())

    def row(group: str, variant: str) -> dict:
        r = per_task[(per_task.group == group) & (per_task.variant == variant)]
        if len(r) == 0:
            return {m: (float("nan"), float("nan")) for m in METRIC_COLS} | {"n": 0}
        return (
            {m: (r[m].mean(), r[m].std()) for m in METRIC_COLS}
            | {"n": len(r)}
        )

    def fmt(val: tuple[float, float]) -> str:
        m, s = val
        if m != m:  # NaN
            return "    N/A   "
        return f"{m:7.4f} ±{s:6.4f}"

    sep = "=" * 82

    def table(title: str, group_list: list[str]) -> None:
        print(f"\n{sep}\n{title}\n{sep}")
        hdr = (f"{'group':<24} {'variant':<11} "
               f"{'gen_roc':>22} {'pearson':>20} {'val_acc':>16} {'val_roc':>16}")
        print(hdr)
        for g in group_list:
            if g not in present:
                print(f"  {g:<24}  (missing)")
                continue
            for v in VARIANT_ORDER:
                m = row(g, v)
                print(
                    f"  {g:<24} {v:<11} "
                    f"{fmt(m['gen_roc'])}  {fmt(m['pearson'])}  "
                    f"{fmt(m['val_acc'])}  {fmt(m['val_roc'])}"
                )

    def delta_table(title: str, base_g: str, new_gs: list[str]) -> None:
        print(f"\n{sep}\n{title}\n{sep}")
        for v in VARIANT_ORDER:
            base = row(base_g, v)
            print(f"\n  variant={v}")
            print(f"  {'group':<24}  Δgen_roc  Δpearson  Δval_acc  Δval_roc")
            for g in new_gs:
                if g not in present or base_g not in present:
                    print(f"  {g:<24}  (missing)")
                    continue
                m = row(g, v)
                dg = m["gen_roc"][0] - base["gen_roc"][0]
                dp = m["pearson"][0] - base["pearson"][0]
                da = m["val_acc"][0] - base["val_acc"][0]
                dr = m["val_roc"][0] - base["val_roc"][0]
                print(f"  {g:<24}  "
                      f"{dg:+.4f}    {dp:+.4f}    {da:+.4f}    {dr:+.4f}")

    table(f"SELF-TC side ({args.dataset})", SELF_GROUPS + ["Base_self"])
    table(f"NEG-TC side ({args.dataset})", NEG_GROUPS + ["Base_neg"])

    delta_table(
        f"DELTAS vs RankAlign_self — self-TC side ({args.dataset})",
        "RankAlign_self",
        ["RankAlign+tc_self", "New+fsx_self", "New+fsx+tc_self"],
    )
    delta_table(
        f"DELTAS vs RankAlign_neg — neg-TC side ({args.dataset})",
        "RankAlign_neg",
        ["RankAlign+negtc_neg", "New+fsx_neg", "New+fsx+negtc_neg"],
    )

    # Pairwise: does New+fsx beat RankAlign+tc? Does New+fsx+tc beat RankAlign+tc?
    if "RankAlign+tc_self" in present and any(
        g in present for g in ("New+fsx_self", "New+fsx+tc_self")
    ):
        delta_table(
            f"DELTAS vs RankAlign+tc_self — best prior self-TC baseline ({args.dataset})",
            "RankAlign+tc_self",
            ["New+fsx_self", "New+fsx+tc_self"],
        )
    if "RankAlign+negtc_neg" in present and any(
        g in present for g in ("New+fsx_neg", "New+fsx+negtc_neg")
    ):
        delta_table(
            f"DELTAS vs RankAlign+negtc_neg — best prior neg-TC baseline ({args.dataset})",
            "RankAlign+negtc_neg",
            ["New+fsx_neg", "New+fsx+negtc_neg"],
        )


if __name__ == "__main__":
    main()
