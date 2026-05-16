#!/usr/bin/env python3
"""Aggregate gemma-4-31B-it humaneval-v2.1correct-upper trained-model score
files into the priority-1 TC comparison (docs/IMPORTANT-RESEARCH-PLAN.md §3).

Four groups (one score file per task per group):
  RankAlign_self      = #2 baseline, self-TC + base-typicality eval
  RankAlign+tc_self   = #6,          self-TC + base-typicality eval
  RankAlign_neg       = #2 baseline, neg-TC  + base-typicality eval
  RankAlign+negtc_neg = #9,          neg-TC  + base-typicality eval

Question: does #6 beat #2 (self side) and #9 beat #2 (neg side) on
Gen AUROC (primary), pearson correlation, val_acc, val_roc — read off the
matching TC'd gen column.

Group membership is decided purely from the score filename prefix/infix
(see docs/score_filename_convention.md):
  scores_basetyp-...      -> self-TC eval ; "tc-self" in name => #6 else #2
  scores_basetypneg-...   -> neg-TC  eval ; "tc-neg"  in name => #9 else #2

Usage (paths default to the mll layout used for the May-2026 run):
  python scripts-more/analyze_gemma4_tc.py \
      --scripts-dir /datastor1/jdr/gv-gap/rankalign/scripts \
      --scores-dir  /datastor1/jdr/gv-gap/rankalign/outputs_gemma4_from_pod \
      --out-dir     /datastor2/jdr/outputs_gemma4_from_pod/_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_SCRIPTS_DIR = "/datastor1/jdr/gv-gap/rankalign/scripts"
DEFAULT_SCORES_DIR = "/datastor1/jdr/gv-gap/rankalign/outputs_gemma4_from_pod"

METRIC_COLS = ["gen_roc", "val_roc", "val_acc", "pearson", "spearman"]
VARIANT_ORDER = ["raw", "tc", "lenorm", "tc+lenorm"]


def classify(name: str) -> str | None:
    """Map a trained-model score filename to one of the four groups (or None)."""
    if name.startswith("scores_basetypneg-"):
        return "RankAlign+negtc_neg" if "tc-neg" in name else "RankAlign_neg"
    if name.startswith("scores_basetyp-"):
        return "RankAlign+tc_self" if "tc-self" in name else "RankAlign_self"
    return None


def classify_base(name: str) -> str | None:
    """Map an untrained gemma-4-31B-it base-model score filename to a group.

    For the base model, --self-typicality / --neg-typicality use the base
    model itself as the unconditional-logP reference — identical to the
    trained models' --base-typicality --base-model google/gemma-4-31B-it.
    So these are the methodologically correct Base row.
    """
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
    ap.add_argument("--scores-dir", default=DEFAULT_SCORES_DIR,
                    help="dir containing the trained-model scores_*.csv files")
    ap.add_argument("--base-scores-dir", default=None,
                    help="dir with untrained gemma-4-31B-it base self-/neg- "
                         "score files (default: <scores-dir>/_base_model_eval)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for CSVs (default: <scores-dir>/_analysis)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scores_dir = Path(args.scores_dir)
    out_dir = Path(args.out_dir) if args.out_dir else scores_dir / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.scripts_dir)
    from summarize_scores_file import summarize_files  # noqa: E402

    base_dir = (Path(args.base_scores_dir) if args.base_scores_dir
                else scores_dir / "_base_model_eval")

    groups: dict[str, list[Path]] = {}
    for p in sorted(scores_dir.glob("scores_*.csv")):
        g = classify(p.name)
        if g is None:
            print(f"UNCLASSIFIED: {p.name}", file=sys.stderr)
            continue
        groups.setdefault(g, []).append(p)

    if base_dir.is_dir():
        for p in sorted(base_dir.glob("scores_*.csv")):
            g = classify_base(p.name)
            if g is None:
                print(f"UNCLASSIFIED (base): {p.name}", file=sys.stderr)
                continue
            groups.setdefault(g, []).append(p)
    else:
        print(f"NOTE: base dir not found ({base_dir}); Base row omitted",
              file=sys.stderr)

    print("Group file counts:")
    for g, paths in sorted(groups.items()):
        print(f"  {g:<22} {len(paths)}")

    per_task_frames = []
    for g, paths in sorted(groups.items()):
        df = summarize_files(paths, quiet=True)
        df.insert(0, "group", g)
        per_task_frames.append(df)
    per_task = pd.concat(per_task_frames, ignore_index=True)
    per_task_csv = out_dir / "per_task_metrics.csv"
    per_task.to_csv(per_task_csv, index=False)
    print(f"\nPer-task metrics ({len(per_task)} rows) -> {per_task_csv}")

    agg = (
        per_task.groupby(["group", "variant"])[METRIC_COLS]
        .agg(["mean", "std", "count"])
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    summary_csv = out_dir / "summary_mean_std.csv"
    agg.to_csv(summary_csv, index=False)
    print(f"Summary mean/std ({len(agg)} rows) -> {summary_csv}")

    def row(group: str, variant: str) -> dict:
        r = per_task[(per_task.group == group) & (per_task.variant == variant)]
        return {
            "gen_roc": (r.gen_roc.mean(), r.gen_roc.std()),
            "pearson": (r.pearson.mean(), r.pearson.std()),
            "val_acc": (r.val_acc.mean(), r.val_acc.std()),
            "val_roc": (r.val_roc.mean(), r.val_roc.std()),
            "n": len(r),
        }

    def block(title: str, base: str, mod: str) -> None:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        hdr = (f"{'variant':<11} {'gen_roc (mean±std)':>22} "
               f"{'pearson':>20} {'val_acc':>16} {'val_roc':>16}")
        for label, grp in [(f"{base} [#2 RankAlign]", base), (mod, mod)]:
            print(f"\n-- {label} --")
            print(hdr)
            for v in VARIANT_ORDER:
                m = row(grp, v)
                print(
                    f"{v:<11} "
                    f"{m['gen_roc'][0]:8.4f} ± {m['gen_roc'][1]:6.4f}    "
                    f"{m['pearson'][0]:7.4f} ± {m['pearson'][1]:6.4f}  "
                    f"{m['val_acc'][0]:6.4f} ±{m['val_acc'][1]:6.4f}  "
                    f"{m['val_roc'][0]:6.4f} ±{m['val_roc'][1]:6.4f}"
                )
        print(f"\n-- delta ({mod} − {base}) --")
        for v in VARIANT_ORDER:
            a, b = row(base, v), row(mod, v)
            print(
                f"{v:<11} "
                f"Δgen_roc={b['gen_roc'][0]-a['gen_roc'][0]:+.4f}   "
                f"Δpearson={b['pearson'][0]-a['pearson'][0]:+.4f}   "
                f"Δval_acc={b['val_acc'][0]-a['val_acc'][0]:+.4f}   "
                f"Δval_roc={b['val_roc'][0]-a['val_roc'][0]:+.4f}"
            )

    block("SELF-TC:  does RankAlign+tc (#6) beat RankAlign (#2)?",
          "RankAlign_self", "RankAlign+tc_self")
    block("NEG-TC:   does RankAlign+negtc (#9) beat RankAlign (#2)?",
          "RankAlign_neg", "RankAlign+negtc_neg")

    # ---- Requested tables: rows Base/RankAlign/RankAlign+TC, cols raw/tc ----
    def requested_table(title: str, base_g: str, ra_g: str, tc_g: str) -> None:
        present = set(per_task.group.unique())
        rows = [("Base", base_g), ("RankAlign", ra_g), ("RankAlign+TC", tc_g)]
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print("gen_roc (Gen AUROC), mean ± std over 82 tasks")
        print(f"{'':<14} {'raw':>16} {'tc':>16}")
        for label, g in rows:
            if g not in present:
                print(f"{label:<14} {'(missing: '+g+')':>33}")
                continue
            rr, tt = row(g, "raw"), row(g, "tc")
            print(f"{label:<14} "
                  f"{rr['gen_roc'][0]:7.4f} ±{rr['gen_roc'][1]:6.4f} "
                  f"{tt['gen_roc'][0]:7.4f} ±{tt['gen_roc'][1]:6.4f}")
        print("\npearson | val_acc | val_roc (val_* identical across raw/tc)")
        print(f"{'':<14} {'raw pearson':>14} {'tc pearson':>14} "
              f"{'val_acc':>10} {'val_roc':>10}")
        for label, g in rows:
            if g not in present:
                continue
            rr, tt = row(g, "raw"), row(g, "tc")
            print(f"{label:<14} "
                  f"{rr['pearson'][0]:14.4f} {tt['pearson'][0]:14.4f} "
                  f"{tt['val_acc'][0]:10.4f} {tt['val_roc'][0]:10.4f}")

    requested_table("TABLE 1 — self-TC (eval: --self-typicality --base-typicality)",
                    "Base_self", "RankAlign_self", "RankAlign+tc_self")
    requested_table("TABLE 2 — neg-TC (eval: --neg-typicality --base-typicality)",
                    "Base_neg", "RankAlign_neg", "RankAlign+negtc_neg")


if __name__ == "__main__":
    main()
