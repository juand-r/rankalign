"""Aggregate per-file humaneval metrics (from summarize_scores_file.py) into
per-(model, dataset, setting, eval-mode) tables. Metrics: gen ROC, val ROC,
val acc, Pearson (gen vs val). Source CSVs produced on mll 2026-06-16.

Models: qwen3.5-9b trained (v7), qwen3.5-9b BASE (v6, untrained), gemma-4-31b-it trained (v7).
Settings: s2 = RankAlign, s4 = New+fsx+self-tc (FLORA-PMI), base = untrained.
Eval prefix -> (typicality mode, base-correction):
  basetyp = self-typ + base-typ ; self = self-typ no-base ; basetypneg = neg-typ + base-typ ; neg = neg-typ no-base
"""
import re
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
FILES = {
    "qwen": HERE / "qwen_he_perfile_metrics.csv",
    "gemma_cu": HERE / "gemma_cu_perfile_metrics.csv",
    "gemma_cm": HERE / "gemma_cm_perfile_metrics.csv",
}

def parse(fname: str) -> dict:
    # eval prefix
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", fname)
    prefix = m.group(1) if m else "?"
    # model
    if "v7-Qwen" in fname or re.search(r"v7-Qwen3\.5-9B", fname):
        model = "qwen3.5-9b (trained)"
    elif re.search(r"v6-Qwen|Qwen_Qwen3\.5-9B", fname) or ("Qwen3.5-9B" in fname and "v7" not in fname):
        model = "qwen3.5-9b (BASE)"
    elif "gemma-4-31B-it" in fname or "gemma-4" in fname:
        model = "gemma-4-31b-it (trained)"
    else:
        model = "?"
    # dataset
    if "correct-upper" in fname:
        dataset = "upper"
    elif "correct-multi" in fname:
        dataset = "multi"
    else:
        dataset = "?"
    # setting: s4 has tc-self (gemma) or tcs (qwen abbrev); base model = base; else s2
    if model == "qwen3.5-9b (BASE)":
        setting = "base"
    elif "tc-self" in fname or re.search(r"-tcs-", fname):
        setting = "s4"
    else:
        setting = "s2"
    return dict(model=model, dataset=dataset, setting=setting, eval=prefix)

rows = []
for tag, path in FILES.items():
    df = pd.read_csv(path)
    meta = df["file"].apply(parse).apply(pd.Series)
    rows.append(pd.concat([meta, df], axis=1))
allm = pd.concat(rows, ignore_index=True)

# Aggregate: mean over the per-task files, per (model,dataset,setting,eval,variant)
agg = (allm.groupby(["model", "dataset", "setting", "eval", "variant"], as_index=False)
       .agg(n_files=("gen_roc", "size"),
            gen_roc=("gen_roc", "mean"),
            val_roc=("val_roc", "mean"),
            val_acc=("val_acc", "mean"),
            pearson=("pearson", "mean")))
agg = agg.round(4)
agg.to_csv(HERE / "humaneval_metrics_aggregated.csv", index=False)
print(f"Wrote humaneval_metrics_aggregated.csv ({len(agg)} groups)")

# Focused table: gen 'tc' variant (typicality-corrected), per model/dataset/setting/eval
focus = agg[agg["variant"] == "tc"].copy()
order = {"gemma-4-31b-it (trained)": 0, "qwen3.5-9b (trained)": 1, "qwen3.5-9b (BASE)": 2}
focus["mo"] = focus["model"].map(order)
focus = focus.sort_values(["mo", "dataset", "setting", "eval"]).drop(columns="mo")
print("\n=== FOCUSED (gen variant = tc) ===")
print(focus[["model","dataset","setting","eval","n_files","gen_roc","val_roc","val_acc","pearson"]].to_string(index=False))

# LaTeX
lt = focus[["model","dataset","setting","eval","gen_roc","val_roc","val_acc","pearson"]].copy()
lt.columns = ["Model","Data","Setting","Eval","gen ROC","val ROC","val acc","Pearson"]
# Scale the four metric columns to a 0-100 scale (xx.xx), as numbers, before formatting.
for _c in ["gen ROC", "val ROC", "val acc", "Pearson"]:
    lt[_c] = lt[_c] * 100.0
with open(HERE / "humaneval_metrics_table.tex", "w") as f:
    f.write("% qwen3.5-9b + gemma-4-31b-it humaneval metrics (gen variant = tc). Metrics x100. Generated 2026-06-16.\n")
    f.write(lt.to_latex(index=False, float_format="%.1f", longtable=True,
                        caption="HumanEval typicality metrics (gen variant = tc), x100: gen ROC, val ROC, val acc, Pearson(gen,val).",
                        label="tab:he_metrics"))
print("\nWrote humaneval_metrics_table.tex")
