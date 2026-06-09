#!/usr/bin/env python3
"""Aggregate the SFT epoch-vs-validator check (scores in outputs-epoch-check/).

For each epoch (e0/e1/e2) of the qwen SFT IFEval-OOD eval, compute mean +/- SE over
the 20 OOD prompts of: gen_roc (tc variant) and the eval-TC-independent validator
metrics val_roc / val_acc. Validator metrics are deduped per prompt (self-/neg- give
the same validator score).

Run on mll:  source ~/venvs/venv_lexcons/bin/activate && python scripts/_aggregate_epoch_check.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from summarize_scores_file import compute_all_metrics, load_scores  # noqa: E402

SCORES_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "outputs-epoch-check"
_EPOCH = re.compile(r"-e([012])-")
_PROMPT = re.compile(r"ifeval-prompt_(\d+)_")


def main() -> None:
    files = sorted(glob.glob(str(SCORES_DIR / "scores_*ifeval-prompt_*.csv")))
    if not files:
        print(f"no score files yet in {SCORES_DIR}")
        return
    # per epoch: gen_roc list (one per file/prompt); val deduped per prompt
    gen: dict[str, list[float]] = {}
    val_roc: dict[str, dict[int, float]] = {}
    val_acc: dict[str, dict[int, float]] = {}
    for fp in files:
        name = os.path.basename(fp)
        em = _EPOCH.search(name)
        pm = _PROMPT.search(name)
        if not em or not pm:
            continue
        ep, prompt = f"e{em.group(1)}", int(pm.group(1))
        try:
            mets = compute_all_metrics(load_scores(fp))
        except Exception as e:  # noqa: BLE001 - report and skip a bad file, do not hide
            print(f"  SKIP {name}: {e}", file=sys.stderr)
            continue
        tc = mets.get("tc")
        if tc is not None and not np.isnan(tc["gen_roc"]):
            gen.setdefault(ep, []).append(tc["gen_roc"])
        raw = mets.get("raw")
        if raw is not None:
            if not np.isnan(raw["val_roc"]):
                val_roc.setdefault(ep, {})[prompt] = raw["val_roc"]
            if not np.isnan(raw["val_acc"]):
                val_acc.setdefault(ep, {})[prompt] = raw["val_acc"]

    def stat(vals: list[float]) -> str:
        if not vals:
            return "  n/a"
        m = float(np.mean(vals)) * 100
        se = (float(np.std(vals, ddof=1) / np.sqrt(len(vals))) * 100) if len(vals) > 1 else 0.0
        return f"{m:5.1f} ± {se:4.1f} (n={len(vals)})"

    print("qwen SFT IFEval-OOD — validator vs epoch (venv qwen35, held constant)")
    print("reference: original pod eval val_roc = 57.5 ; rerun epoch2 (earlier) = 82.8\n")
    print(f"{'epoch':6}{'gen_roc':>22}{'val_roc':>22}{'val_acc':>22}")
    for ep in ("e0", "e1", "e2"):
        print(f"{ep:6}{stat(gen.get(ep, [])):>22}"
              f"{stat(list(val_roc.get(ep, {}).values())):>22}"
              f"{stat(list(val_acc.get(ep, {}).values())):>22}")


if __name__ == "__main__":
    main()
