"""Build per-metric HumanEval TRAIN-SET per-epoch-dynamics tables.

Mirrors docs/he_harvest_2026-06-17/build_he_metric_tables.py but adds an EPOCH axis
(base / ep0 / ep1 / ep2) — the test tables only used epoch2. One LaTeX table PER METRIC
(gen ROC, val ROC, val acc, Pearson), each showing the **raw** and **tc** gen variants,
for gemma-4-31b s2 (RankAlign) and s4 (self-tc) across epochs, on both
humaneval-v2.1correct-upper and -multi. Scores from the --train split, N=50 stratified
(25 pos / 25 neg) candidates per problem. Values x100, 1 decimal, means over the 82
problems, midrules between (dataset) blocks.

Source: trainset_he_perfile_metrics.csv (from scripts/harvest_he_trainset.sh).
NB: val ROC / val acc are validator-based (gen-variant-independent) -> their raw and tc
columns are identical by construction; shown for uniformity.
"""
import re
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "trainset_he_perfile_metrics.csv"
METRICS = [("gen_roc", "gen ROC"), ("val_roc", "val ROC"), ("val_acc", "val acc"), ("pearson", "Pearson")]
VARIANTS = ["raw", "tc"]   # user: raw + tc only (no lenorm)


def parse(fname: str) -> dict:
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", fname)
    pref = m.group(1) if m else "?"
    is_base = ("v6-" in fname) and ("-v7-" not in fname)
    # epoch axis
    if is_base:
        epoch = "base"
    elif re.search(r"epoch0|-e0-", fname):
        epoch = "ep0"
    elif re.search(r"epoch1|-e1-", fname):
        epoch = "ep1"
    elif re.search(r"epoch2|-e2-", fname):
        epoch = "ep2"
    else:
        epoch = "?"
    # all train-set files here are gemma-4-31b
    model = "gemma-4-31b"
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
    typ = "neg" if pref in ("neg", "basetypneg") else "self"
    bt = "base-typ" if pref in ("basetyp", "basetypneg") else "no-base"
    return dict(model=model, dataset=dataset, setting=setting, epoch=epoch, eval=f"{typ}/{bt}")


# --- load + parse ---
df = pd.read_csv(SRC)
meta = df["file"].apply(parse).apply(pd.Series)
allm = pd.concat([meta, df], axis=1)
allm = allm[allm["variant"].isin(VARIANTS)].copy()
allm = allm[allm["dataset"].isin(["upper", "multi"])].copy()

bad = allm[(allm["setting"] == "?") | (allm["epoch"] == "?") | (allm["dataset"] == "?")]
if len(bad):
    print(f"WARNING: {len(bad)} unparsed rows; sample files:")
    for f in bad["file"].head(8):
        print("   ", f)
else:
    print("all train-set rows parsed cleanly.")

# only s2 / s4 / base are expected in this run
keep = allm["setting"].isin(["base", "s2", "s4"])
dropped = allm[~keep]["setting"].value_counts().to_dict()
if dropped:
    print(f"NOTE dropping non-{{base,s2,s4}} settings present in dir: {dropped}")
allm = allm[keep].copy()

# --- aggregate: mean over per-task files, per (dataset,setting,epoch,eval,variant) ---
gcols = ["model", "dataset", "setting", "epoch", "eval", "variant"]
agg = (allm.groupby(gcols, as_index=False)
       .agg(n=("gen_roc", "size"),
            gen_roc=("gen_roc", "mean"), val_roc=("val_roc", "mean"),
            val_acc=("val_acc", "mean"), pearson=("pearson", "mean")))
agg.to_csv(HERE / "humaneval_TRAIN_metrics_aggregated.csv", index=False)
print(f"\nWrote humaneval_TRAIN_metrics_aggregated.csv ({len(agg)} groups)")

# ordering
ds_order = {"upper": 0, "multi": 1}
set_order = {"base": 0, "s2": 1, "s4": 2}
epoch_order = {"base": 0, "ep0": 1, "ep1": 2, "ep2": 3}
eval_order = {"self/base-typ": 0, "self/no-base": 1, "neg/base-typ": 2, "neg/no-base": 3}


def esc(s):
    return str(s).replace("_", r"\_")


def build_metric_table(metric_key, metric_name):
    piv = agg.pivot_table(index=["model", "dataset", "setting", "epoch", "eval"],
                          columns="variant", values=metric_key).reset_index()
    for v in VARIANTS:
        if v not in piv.columns:
            piv[v] = float("nan")
    piv["_d"] = piv["dataset"].map(lambda x: ds_order.get(x, 9))
    piv["_s"] = piv["setting"].map(lambda x: set_order.get(x, 9))
    piv["_p"] = piv["epoch"].map(lambda x: epoch_order.get(x, 9))
    piv["_e"] = piv["eval"].map(lambda x: eval_order.get(x, 9))
    piv = piv.sort_values(["_d", "_s", "_p", "_e"])
    header = "Data & Setting & Epoch & Eval & raw & tc " + r"\\"
    body, prev = [], None
    for _, r in piv.iterrows():
        key = (r["dataset"],)
        if prev is not None and key != prev:
            body.append(r"\midrule")
        raw = f"{r['raw']*100:.1f}" if pd.notna(r["raw"]) else "--"
        tc = f"{r['tc']*100:.1f}" if pd.notna(r["tc"]) else "--"
        body.append(f"{r['dataset']} & {r['setting']} & {r['epoch']} & {esc(r['eval'])} & {raw} & {tc} " + r"\\")
        prev = key
    cap = (f"HumanEval TRAIN-set {metric_name} ($\\times100$) per-epoch dynamics, raw vs tc gen "
           "variant. gemma-4-31b, s2 (RankAlign) \\& s4 (self-tc), base/ep0/ep1/ep2, both datasets. "
           "Means over 82 problems, N=50 stratified candidates each.")
    tex = (
        "{\\small\n\\begin{longtable}{llll rr}\n"
        f"\\caption{{{cap}}}\\label{{tab:he_train_{metric_key}}}\\\\\n"
        "\\toprule\n" + header + "\n\\midrule\n\\endfirsthead\n"
        "\\multicolumn{6}{l}{\\itshape (continued)}\\\\\n\\toprule\n" + header + "\n\\midrule\n\\endhead\n"
        "\\bottomrule\n\\endfoot\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{longtable}\n}\n"
    )
    out = HERE / f"humaneval_TRAIN_table_{metric_key}.tex"
    out.write_text(tex)
    print(f"  wrote {out.name} ({len(piv)} rows)")


print("\n=== building 4 per-metric train-set tables (raw + tc) ===")
for mk, mn in METRICS:
    build_metric_table(mk, mn)
print("done.")
