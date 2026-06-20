#!/usr/bin/env python3
"""Pre-flight check for the HumanEval held-out val-loss jobs (run on mll, in the model venv).

Verifies, WITHOUT loading any model:
  1. peft + transformers import (LoRA loading path)
  2. compute_val_loss_he imports cleanly (script is deployable)
  3. the per-problem TEST tasks aggregate into a non-empty pos/neg held-out set for a dataset

Usage:  python scripts/verify_he_valloss_env.py upper   # or: multi
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ds = sys.argv[1] if len(sys.argv) > 1 else "upper"

import peft
import transformers
print(f"peft={peft.__version__}  transformers={transformers.__version__}")

import compute_val_loss_he as cv  # noqa: E402  (also triggers task registration)
print("compute_val_loss_he import: OK")

L_test, cfg = cv._aggregate_perproblem_test(ds)
pos = sum(1 for it in L_test if cfg["get_label"](it) == "yes")
neg = sum(1 for it in L_test if cfg["get_label"](it) == "no")
print(f"[{ds}] held-out pool: {len(L_test)} items  ->  {pos} pos / {neg} neg")
assert pos > 0 and neg > 0, "empty positive/negative class — aggregation broken"

# sanity: a generator + discriminator prompt build without error
item = L_test[0]
g = cfg["make_prompt"](item, style="generator", shots="zero")
d = cfg["make_prompt"](item, style="discriminator", shots="zero")
print(f"gen prompt chars={len(g.prompt)} completion chars={len(g.completion)}; disc prompt chars={len(d.prompt)}")
print("VERIFY OK")
