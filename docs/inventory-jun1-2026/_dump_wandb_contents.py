import wandb, json
api = wandb.Api(timeout=120)
# a representative FINISHED run (gemma-2-9b-it membership s2, delta-bins)
r = api.run("juand-r/rankalign/mowgknkz")
print("=== RUN", r.name, "| state", r.state, "===")
print("\n-- config keys (%d) --" % len(r.config)); print(sorted(r.config.keys()))
print("\n-- summary keys (final values + wandb internals) --")
sk = [k for k in r.summary.keys()]
print(sorted(sk))
print("\n-- a few summary values --")
for k in ["epoch/avg_loss","train/loss","train/preference_loss","_runtime","_step","_timestamp"]:
    if k in r.summary: print(f"   {k} = {r.summary[k]}")
print("\n-- history (logged time-series) columns --")
try:
    h = r.history(samples=3)
    print(list(h.columns))
    print("   n history rows (approx):", r.lastHistoryStep)
except Exception as e:
    print("history err", e)
print("\n-- FILES stored in the run --")
for f in r.files():
    print(f"   {f.name}  ({f.size} bytes)")
