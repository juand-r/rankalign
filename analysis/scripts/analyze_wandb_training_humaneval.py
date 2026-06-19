#!/usr/bin/env python3
"""Q2 training diagnostics from WandB for the HumanEval runs (gemma-4 + qwen-3.5,
upper + multi). Adapts analyze_wandb_training.py: same project (juand-r/rankalign),
same metric keys and plotting (analyze_runs), but discovers run IDs by the HumanEval
run-name scheme and groups into the 4 model x dataset panels.

Run names (from the launchers): g4-{cu,cm}-s{1,2,3,4,7,13} and q35-{cu,cm}-s{...}.
  mll_g4_train_eval.sbatch / _genctx.sbatch -> --wandb_run_name "g4-${WTAG}-s${S}"
  mll_qwen35_he_train_eval.sbatch           -> "q35-${WTAG}-s${S}"

Run ON mll with a venv that has wandb + matplotlib + pandas, with WANDB_API_KEY exported.
Writes plots to analysis/plots/ and summary CSVs to analysis/tables/ (per group), same as
the original. Reuses analyze_runs/pull_run_history from analyze_wandb_training.py.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wandb  # noqa: E402
from analyze_wandb_training import analyze_runs  # noqa: E402  (reuse plotting verbatim)

PROJECT = "juand-r/rankalign"
MODEL_LABEL = {"g4": "gemma-4-31b", "q35": "qwen-3.5-9b"}
DS_LABEL = {"cu": "upper", "cm": "multi"}
NAME_RE = re.compile(r"^(g4|q35)-(cu|cm)-s(\d+)$")


def discover():
    api = wandb.Api()
    runs = api.runs(PROJECT, order="-created_at", per_page=300)
    groups = {}  # (model,ds) -> {setting_token: run_id}
    found = []
    for r in runs:
        m = NAME_RE.match(r.name or "")
        if not m:
            continue
        model, ds, snum = m.group(1), m.group(2), m.group(3)
        key = (model, ds)
        setting = f"s{snum}"
        # newest first (order=-created_at): keep the first (latest) per (group,setting)
        groups.setdefault(key, {})
        if setting not in groups[key]:
            groups[key][setting] = r.id
            found.append((r.name, r.state, r.id))
    print(f"discovered {len(found)} HumanEval runs:")
    for name, state, rid in sorted(found):
        print(f"  {name:14} {state:10} {rid}")
    return groups


def main():
    print("=" * 70)
    print("  Q2 WANDB DIAGNOSTICS — HumanEval (gemma-4 + qwen-3.5, upper + multi)")
    print("=" * 70)
    groups = discover()
    if not groups:
        print("NO HumanEval runs matched g4/q35-{cu,cm}-sN. Check project/names.")
        return
    for (model, ds), runs_dict in sorted(groups.items()):
        label = f"{MODEL_LABEL[model]} {DS_LABEL[ds]}"
        print(f"\n--- {label} : settings {sorted(runs_dict)} ---")
        analyze_runs(runs_dict, label)
    print("\nDone.")


if __name__ == "__main__":
    main()
