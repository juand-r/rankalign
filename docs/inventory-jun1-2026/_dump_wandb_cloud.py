import wandb, json
from collections import Counter
api = wandb.Api(timeout=120)
runs = api.runs("juand-r/rankalign", per_page=500)
out=[]
for r in runs:
    c = r.config or {}
    out.append({"name": r.name, "task": c.get("task","?"), "model": c.get("model","?"),
                "created": (r.created_at or "")[:10], "state": r.state,
                "pref": c.get("preference_loss_weight"), "nllv": c.get("nll_validator_weight"),
                "delta": c.get("delta")})
json.dump(out, open("/tmp/_wandb_cloud.json","w"))
print("TOTAL", len(out))
bt=Counter(o["task"] for o in out); bm=Counter(o["model"].split("/")[-1] for o in out)
print("by task:", dict(bt))
print("by (task,model):")
tm=Counter((o["task"], o["model"].split("/")[-1]) for o in out)
for k,v in sorted(tm.items()): print(f"   {v:4d}  {k}")
