#!/usr/bin/env python3
"""Rename qwen wandb-rerun scores_ files from the broken short-symlink model field
(`eval_model_qw_<setting>`) to the canonical abbreviated v7 descriptor that the rest
of the pipeline (checkpoint_name_parser / _build_*_table_v7.py) expects.

WHY: run_qwen35_cell_mll.sbatch symlinked each eval model to a generic short path
`eval_model_qw_<setting>` (to dodge the 255-char filename cap), so eval_by_claude.py
wrote `scores_<mode>-eval_model_qw_s2_<task>_...csv` — losing model/delta/epoch/method
and breaking the v7 parser. The symlink TARGET is a fully-parseable v7 dir name; the
canonical short form is `to_hf_repo_name(parse_checkpoint_name(target_basename), prefix='')`
(e.g. v7-gemma-2-9b-it-d1.94-e2-ifeval-concat-all-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1).

This maps each scores file's (setting, task-dataset) -> the epoch2 model dir it was
evaluated against, computes the abbreviated name, and renames in place.

Usage:
    python _rename_qwen_rerun_scores.py --outputs DIR --models DIR [--apply]
Default is DRY-RUN. Idempotent: files already in canonical form are skipped.
"""
import argparse
import glob
import os
import re
import sys

# Per-setting DIR_SUFFIX from run_qwen35_cell_mll.sbatch (ground truth, stable).
SUFFIX = {
    "s1": "--full-completion--pref0.0--nllv1.0--nllg1.0--vallogodds--labelonly0.1--fix1",
    "s2": "--full-completion--vallogodds--semi0.1--fix1",
    "s3": "--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1",
    "s4": "--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1",
    "s7": "--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1",
}


def task_dset(fname: str) -> str:
    if "rosch" in fname:
        return "membership-sans-rosch-v0"
    if "ifeval" in fname:
        return "ifeval-concat"
    raise ValueError(f"cannot determine task dataset from: {fname}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="dir with scores_*eval_model_qw_*.csv")
    ap.add_argument("--models", required=True, help="dir with v7-Qwen--*_merged model dirs")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry-run)")
    args = ap.parse_args()

    # import parser from repo src/
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "src"))
    from checkpoint_name_parser import parse_checkpoint_name, to_hf_repo_name

    abbrev_cache: dict[tuple[str, str], str] = {}

    def resolve_abbrev(setting: str, dset: str) -> str:
        key = (setting, dset)
        if key in abbrev_cache:
            return abbrev_cache[key]
        pat = os.path.join(
            args.models,
            f"v7-Qwen--Qwen3.5-9B-delta*-epoch2--{dset}-all--d2g--random--alpha1.0{SUFFIX[setting]}_merged",
        )
        hits = sorted(glob.glob(pat))
        if len(hits) != 1:
            raise RuntimeError(f"expected 1 model dir for {setting}/{dset}, got {len(hits)}: {pat}")
        ab = to_hf_repo_name(parse_checkpoint_name(os.path.basename(hits[0])), prefix="")
        abbrev_cache[key] = ab
        return ab

    files = sorted(glob.glob(os.path.join(args.outputs, "scores_*eval_model_qw_*.csv")))
    print(f"found {len(files)} files to rename (dir={args.outputs})")
    renamed = skipped = 0
    for f in files:
        base = os.path.basename(f)
        m = re.search(r"eval_model_qw_(s\d+)", base)
        if not m:
            continue
        setting = m.group(1)
        ab = resolve_abbrev(setting, task_dset(base))
        new_base = base.replace(f"eval_model_qw_{setting}", ab)
        new_path = os.path.join(args.outputs, new_base)
        if os.path.exists(new_path):
            print(f"  SKIP (target exists): {new_base}")
            skipped += 1
            continue
        print(f"  {base}\n   -> {new_base}")
        if args.apply:
            os.rename(f, new_path)
        renamed += 1
    print(f"\n{'RENAMED' if args.apply else 'WOULD RENAME'} {renamed}; skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
