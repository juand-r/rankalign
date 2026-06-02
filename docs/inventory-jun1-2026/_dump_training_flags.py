#!/usr/bin/env python3
"""Run ON mll. Dump per-training-log flag fields as JSONL to stdout.

Reads /datastor2/jdr/rankalign/models2/training_run_logs/*.json and emits one
JSON line per file with the fields needed to classify the setting.
"""
import glob
import json
import os

LOGDIR = "/datastor2/jdr/rankalign/models2/training_run_logs"
for f in sorted(glob.glob(os.path.join(LOGDIR, "*.json"))):
    try:
        d = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"file": os.path.basename(f), "error": str(e)}))
        continue
    fl = d.get("flags", {}) or {}
    rec = {
        "file": os.path.basename(f),
        "model": d.get("model"),
        "task": d.get("task"),
        "fsx": fl.get("force_same_x"),
        "vlo": fl.get("validator_log_odds"),
        "self_tc": fl.get("self_typicality"),
        "neg_tc": fl.get("neg_typicality"),
        "semi": fl.get("semi_supervised"),
        "lo": fl.get("labeled_only"),
        "delta_arg": fl.get("delta_arg"),
        "delta_bins": fl.get("delta_bins"),
        "cft": (d.get("consistency_ft") is not None),
        "n_items_total": d.get("n_items_total"),
        "total_sampled": d.get("total_sampled"),
    }
    print(json.dumps(rec))
