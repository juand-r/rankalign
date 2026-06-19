"""Build per-metric HumanEval TEST tables from the harvest per-file metric CSVs.

One LaTeX table PER METRIC (gen ROC, val ROC, val acc, Pearson), each showing the
**raw** and **tc** gen variants, for ALL systems (gemma-4-31b & qwen-3.5-9b ×
base/s1/s2/s3/s4/s7/s13 × eval modes), both upper+multi. Values ×100, 1 decimal,
midrules between (model, dataset) blocks.

Source: {qwen,gemma_cu,gemma_cm}_he_perfile_metrics.csv (from scripts/harvest_he_runs.sh).
NB: val ROC / val acc are validator-based (gen-variant-independent) -> their raw and tc
columns are identical by construction; shown for uniformity.
"""
import re
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["qwen_he_perfile_metrics.csv", "gemma_cu_perfile_metrics.csv", "gemma_cm_perfile_metrics.csv"]
METRICS = [("gen_roc", "gen ROC"), ("val_roc", "val ROC"), ("val_acc", "val acc"), ("pearson", "Pearson")]
VARIANTS = ["raw", "tc"]   # user: raw + tc only (no lenorm)

def parse(fname: str) -> dict:
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", fname)
    pref = m.group(1) if m else "?"
    is_base = ("v6-" in fname) and ("v7-" not in fname)
    if "Qwen3.5-9B" in fname or "Qwen_Qwen3" in fname or "Qwen--Qwen3" in fname:
        model = "qwen-3.5-9b" + (" (base)" if is_base else "")
        mkey = "qwen"
    elif "gemma-4-31B-it" in fname or "gemma-4-31b" in fname:
        model = "gemma-4-31b" + (" (base)" if is_base else "")
        mkey = "gemma"
    else:
        model, mkey = "?", "?"
    dataset = "upper" if "correct-upper" in fname else ("multi" if "correct-multi" in fname else "?")
    # setting via recipe tokens (works for abbreviated AND full names)
    if is_base:
        setting = "base"
    elif ("tcs" in fname) or ("tc-self" in fname):
        setting = "s4"
    elif ("tcn" in fname) or ("tc-neg" in fname):
        setting = "s7"
    elif "cft" in fname:
        setting = "s13"
    elif ("fsx" in fname) or ("force-same-x" in fname):
        setting = "s3"
    elif ("lo0.1" in fname) or ("labelonly" in fname) or re.search(r"-p0-", fname) or ("pref0" in fname):
        setting = "s1"
    else:
        setting = "s2"
    # eval mode + base-typ from the prefix
    typ = "neg" if pref in ("neg", "basetypneg") else "self"
    bt = "base-typ" if pref in ("basetyp", "basetypneg") else "no-base"
    return dict(model=model, mkey=mkey, dataset=dataset, setting=setting, eval=f"{typ}/{bt}")

# --- load + parse ---
rows = []
for fn in FILES:
    p = HERE / fn
    if not p.exists():
        print(f"WARN missing {fn}"); continue
    df = pd.read_csv(p)
    meta = df["file"].apply(parse).apply(pd.Series)
    rows.append(pd.concat([meta, df], axis=1))
allm = pd.concat(rows, ignore_index=True)
allm = allm[allm["variant"].isin(VARIANTS)].copy()

# Drop anything that isn't humaneval (the qwen dir also holds the earlier ifeval rerun).
_n0 = len(allm)
allm = allm[allm["file"].str.contains("humaneval", na=False)].copy()
allm = allm[allm["dataset"].isin(["upper", "multi"])].copy()
print(f"Filtered to humaneval upper/multi: {len(allm)} of {_n0} variant-rows kept")

# sanity: any humaneval rows still unparsed?
bad = allm[(allm["model"] == "?") | (allm["setting"] == "?") | (allm["dataset"] == "?")]
if len(bad):
    print(f"WARNING: {len(bad)} unparsed humaneval rows; sample files:")
    for f in bad["file"].head(8):
        print("   ", f)
else:
    print("all humaneval rows parsed cleanly.")

# --- aggregate: mean over the per-task files, per (model,dataset,setting,eval,variant) ---
gcols = ["model", "mkey", "dataset", "setting", "eval", "variant"]
agg = (allm.groupby(gcols, as_index=False)
       .agg(n=("gen_roc", "size"),
            gen_roc=("gen_roc", "mean"), val_roc=("val_roc", "mean"),
            val_acc=("val_acc", "mean"), pearson=("pearson", "mean")))
agg.to_csv(HERE / "humaneval_TEST_metrics_aggregated.csv", index=False)
print(f"\nWrote humaneval_TEST_metrics_aggregated.csv ({len(agg)} groups)")

# ordering
mod_order = {"gemma-4-31b": 0, "gemma-4-31b (base)": 1, "qwen-3.5-9b": 2, "qwen-3.5-9b (base)": 3}
set_order = {"base": 0, "s1": 1, "s2": 2, "s3": 3, "s4": 4, "s7": 5, "s13": 6}
ds_order = {"upper": 0, "multi": 1}
eval_order = {"self/base-typ": 0, "self/no-base": 1, "neg/base-typ": 2, "neg/no-base": 3}

def esc(s):  # latex-escape underscores in display strings
    return str(s).replace("_", r"\_")

def build_metric_table(metric_key, metric_name):
    # pivot variant -> columns (raw, tc) for this metric
    piv = agg.pivot_table(index=["model", "mkey", "dataset", "setting", "eval"],
                          columns="variant", values=metric_key).reset_index()
    for v in VARIANTS:
        if v not in piv.columns:
            piv[v] = float("nan")
    piv["_m"] = piv["model"].map(lambda x: mod_order.get(x, 9))
    piv["_d"] = piv["dataset"].map(lambda x: ds_order.get(x, 9))
    piv["_s"] = piv["setting"].map(lambda x: set_order.get(x, 9))
    piv["_e"] = piv["eval"].map(lambda x: eval_order.get(x, 9))
    piv = piv.sort_values(["_m", "_d", "_s", "_e"])
    header = "Model & Data & Setting & Eval & raw & tc " + r"\\"
    body, prev = [], None
    for _, r in piv.iterrows():
        key = (r["model"], r["dataset"])
        if prev is not None and key != prev:
            body.append(r"\midrule")
        raw = f"{r['raw']*100:.1f}" if pd.notna(r["raw"]) else "--"
        tc = f"{r['tc']*100:.1f}" if pd.notna(r["tc"]) else "--"
        body.append(f"{esc(r['model'])} & {r['dataset']} & {r['setting']} & {esc(r['eval'])} & {raw} & {tc} " + r"\\")
        prev = key
    tex = (
        "\\begin{table}[ht]\\centering\\small\n"
        f"\\caption{{HumanEval TEST {metric_name} ($\\times100$), raw vs tc gen variant. "
        "Systems: gemma-4-31b \\& qwen-3.5-9b $\\times$ base/s1/s2/s3/s4/s7/s13, both datasets.}}\n"
        f"\\label{{tab:he_test_{metric_key}}}\n"
        "\\begin{tabular}{llll rr}\n\\toprule\n" + header + "\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    out = HERE / f"humaneval_TEST_table_{metric_key}.tex"
    out.write_text(tex)
    print(f"  wrote {out.name} ({len(piv)} rows)")

print("\n=== building 4 per-metric tables (raw + tc) ===")
for mk, mn in METRICS:
    build_metric_table(mk, mn)
print("done.")
