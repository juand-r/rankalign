#!/usr/bin/env python3
"""Run ON mll. Dump one JSONL line per local wandb run dir under /datastor2/jdr.

Reads each run-*/files/wandb-metadata.json `args` list to recover model/task/flags,
so we can map wandb runs -> setting. Emits to stdout.
"""
import glob
import json
import os

dirs = []
for base in ["/datastor2/jdr/rankalign/wandb", "/datastor2/jdr/rankalign/scripts/wandb"]:
    dirs += glob.glob(os.path.join(base, "run-*"))

for d in sorted(dirs):
    meta = os.path.join(d, "files", "wandb-metadata.json")
    rec = {"run_dir": d.replace("/datastor2/jdr/rankalign/", ""), "date": os.path.basename(d)[4:12]}
    try:
        m = json.load(open(meta))
        args = m.get("args", [])
        rec["args"] = args

        def val(flag):
            return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None
        rec["model"] = val("--model")
        rec["task"] = val("--task")
        rec["delta"] = val("--delta")
        rec["delta_bins"] = val("--delta-bins")
        rec["pref"] = val("--preference_loss_weight")
        rec["nllv"] = val("--nll_validator_weight")
        rec["nllg"] = val("--nll_generator_weight")
        rec["fsx"] = "--force-same-x" in args
        rec["ppd"] = "--per-prompt-delta" in args
        rec["vlo"] = "--validator-log-odds" in args
        rec["self_tc"] = "--self-typicality" in args
        rec["neg_tc"] = "--neg-typicality" in args
        rec["cft"] = "--consistency-ft" in args
        rec["semi"] = val("--semi-supervised")
        rec["lo"] = val("--labeled-only")
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)
    print(json.dumps(rec))
