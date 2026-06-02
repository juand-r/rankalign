import wandb, json
api = wandb.Api(timeout=120)
TASKS = ["ifeval-concat", "membership-sans-rosch-v0", "humaneval-v2.1correct-upper"]
runs = api.runs("juand-r/rankalign", filters={"config.task": {"$in": TASKS}}, per_page=500)
out = []
for r in runs:
    c = r.config or {}
    rec = {"name": r.name, "id": r.id, "url": r.url, "state": r.state,
           "created": (r.created_at or "")[:10], "task": c.get("task"),
           "model": (c.get("model") or "?").split("/")[-1],
           "pref": c.get("preference_loss_weight"), "nllv": c.get("nll_validator_weight"),
           "nllg": c.get("nll_generator_weight"), "self_tc": c.get("self_typicality"),
           "neg_tc": c.get("neg_typicality"), "lo": c.get("labeled_only"),
           "semi": c.get("semi_supervised"), "delta": c.get("delta")}
    try:
        md = r.metadata
        args = (md or {}).get("args", [])
        rec["fsx"] = "--force-same-x" in args
        rec["ppd"] = "--per-prompt-delta" in args
        rec["vlo"] = "--validator-log-odds" in args
        rec["cft"] = "--consistency-ft" in args
        rec["delta_bins"] = ("--delta-bins" in args)
        rec["args"] = " ".join(args)
    except Exception as e:
        rec["args_err"] = str(e)
    out.append(rec)
json.dump(out, open("/tmp/_wandb_paper.json", "w"))
print("paper-task runs:", len(out))
