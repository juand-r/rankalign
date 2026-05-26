#!/usr/bin/env python3
"""Aggregate Qwen3.5-9B base evals for IFEval OOD and Hyponymy.

For each (task, eval-prefix) compute gen_roc per task, then mean ± SE across
tasks. Also reports pearson and spearman.

Eval prefixes for base model:
  - self-  -> "PMI self"  (= PMI base for base-model evals)
  - neg-   -> "Neg self"  (= Neg base for base-model evals)
  - Raw    -> taken from EITHER prefix (variant='raw' is identical across)

Tasks:
  - IFEval OOD: prompts 1..13, 15..21 (20 tasks)
  - Hyponymy:   10 rosch tasks
"""
from __future__ import annotations
import sys
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
sys.path.insert(0, str(REPO / "scripts"))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

OUT_DIR = Path("/datastor2/jdr/rankalign/outputs")

OOD_PROMPTS = list(range(1, 14)) + list(range(15, 22))  # 20 prompts
ROSCH_TASKS = [
    "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
    "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vegetable",
    "rosch-vehicle", "rosch-weapon",
]


def gather(task_paths_by_prefix):
    """For each prefix, compute per-task metrics, then aggregate."""
    rows = []
    for prefix, paths in task_paths_by_prefix.items():
        per_task = {"raw": [], "tc": []}
        per_task_pe = {"raw": [], "tc": []}
        per_task_sp = {"raw": [], "tc": []}
        for p in paths:
            df = load_scores(p)
            m = compute_all_metrics(df)
            for variant in ("raw", "tc"):
                if variant in m:
                    per_task[variant].append(m[variant]["gen_roc"])
                    per_task_pe[variant].append(m[variant]["pearson"])
                    per_task_sp[variant].append(m[variant]["spearman"])
        for variant in ("raw", "tc"):
            vals = np.array([v for v in per_task[variant] if v is not None and not np.isnan(v)])
            pes = np.array([v for v in per_task_pe[variant] if v is not None and not np.isnan(v)])
            sps = np.array([v for v in per_task_sp[variant] if v is not None and not np.isnan(v)])
            if vals.size == 0:
                continue
            rows.append(dict(
                prefix=prefix,
                variant=variant,
                n=int(vals.size),
                gen_roc=float(vals.mean() * 100),
                gen_roc_se=float(vals.std(ddof=1) / np.sqrt(vals.size) * 100) if vals.size > 1 else 0.0,
                pearson=float(pes.mean()) if pes.size > 0 else float("nan"),
                spearman=float(sps.mean()) if sps.size > 0 else float("nan"),
            ))
    return pd.DataFrame(rows)


def collect_prefix_paths(model_pattern: str, task_names: list[str]):
    by_prefix: dict[str, list[Path]] = {"self-": [], "neg-": []}
    for prefix in ("self-", "neg-"):
        for tn in task_names:
            pat = f"scores_{prefix}{model_pattern}_{tn}_test_log-odds_tc_*.csv"
            matches = sorted(glob(str(OUT_DIR / pat)))
            if matches:
                by_prefix[prefix].append(Path(matches[-1]))
    return by_prefix


print("=" * 72)
print("QWEN3.5-9B × IFEVAL OOD (20 prompts)")
print("=" * 72)
qwen_ifeval_tasks = [f"ifeval-prompt_{n}" for n in OOD_PROMPTS]
by_pre = collect_prefix_paths("v6-Qwen_Qwen3.5-9B", qwen_ifeval_tasks)
for k, v in by_pre.items():
    print(f"  {k}: {len(v)} files")
df1 = gather(by_pre)
print(df1.to_string(index=False))

print()
print("=" * 72)
print("QWEN3.5-9B × ROSCH (10 tasks)")
print("=" * 72)
by_pre = collect_prefix_paths("v6-Qwen_Qwen3.5-9B", ROSCH_TASKS)
for k, v in by_pre.items():
    print(f"  {k}: {len(v)} files")
df2 = gather(by_pre)
print(df2.to_string(index=False))
