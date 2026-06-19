"""Build LaTeX table fragments for the HumanEval training-dynamics report
(analysis/report_humaneval.tex), mirroring analysis/report.tex.

Reproducible source of the numbers in the report. Reads the already-harvested aggregated
metric CSVs (no mll access needed):
  TEST : docs/he_harvest_2026-06-17/humaneval_TEST_metrics_aggregated.csv
  TRAIN: docs/he_trainset_2026-06-19/humaneval_TRAIN_metrics_aggregated.csv  (gemma s2/s4 only)

Emits into analysis/tables/:
  he_q1_test.tex          -- Q1 main comparison: gen ROC (tc) per (model,dataset) x setting,
                             each method shown in its natural eval mode (RA/s2 -> Neg base;
                             s4 -> PMI base; s7 -> Neg base; s1/s3 -> PMI base; base -> self).
  he_traintest_gemma.tex  -- gemma s2 vs s4 train-vs-test gen ROC (tc), both datasets (overfitting).

Metric = gen_roc, variant = tc, x100, 1 decimal. Eval-mode labels in the aggregated CSVs:
  self/no-base, self/base-typ, neg/no-base, neg/base-typ.
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TEST = REPO / "docs/he_harvest_2026-06-17/humaneval_TEST_metrics_aggregated.csv"
TRAIN = REPO / "docs/he_trainset_2026-06-19/humaneval_TRAIN_metrics_aggregated.csv"
OUT = REPO / "analysis/tables"
OUT.mkdir(parents=True, exist_ok=True)

# the eval mode each method is presented in (base-typicality convention, matching
# rerun_only_tables / the cell the effect lives in)
NATURAL_MODE = {"base": "self/no-base", "s1": "self/base-typ", "s2": "neg/base-typ",
                "s3": "neg/base-typ", "s4": "self/base-typ", "s7": "neg/base-typ", "s13": "self/base-typ"}
SETTINGS = ["base", "s1", "s2", "s3", "s4", "s7", "s13"]
SETTING_LABEL = {"base": "Base", "s1": "SFT", "s2": "RankAlign", "s3": "New+fsx",
                 "s4": "FLORA-PMI (self-TC)", "s7": "FLORA-Neg (neg-TC)", "s13": "Consistency FT"}


def test_val(df, mkey, ds, setting, mode, metric="gen_roc"):
    r = df[(df.mkey == mkey) & (df.dataset == ds) & (df.setting == setting)
           & (df["eval"] == mode) & (df.variant == "tc")]
    return None if r.empty else round(float(r[metric].iloc[0]) * 100, 1)


def build_q1(df):
    lines = [r"\begin{tabular}{ll" + "c" * len(SETTINGS) + "}", r"\toprule",
             "Model & Data & " + " & ".join(SETTING_LABEL[s] for s in SETTINGS) + r" \\",
             r"\midrule"]
    for mkey, mlab in [("gemma", "gemma-4-31b"), ("qwen", "qwen-3.5-9b")]:
        for ds in ["upper", "multi"]:
            cells = []
            for s in SETTINGS:
                v = test_val(df, mkey, ds, s, NATURAL_MODE[s])
                cells.append("---" if v is None else f"{v}")
            lines.append(f"{mlab} & {ds} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "he_q1_test.tex").write_text("\n".join(lines) + "\n")
    print("wrote he_q1_test.tex")


def build_traintest(test_df, train_df):
    # gemma s2 (RankAlign) vs s4 (self-TC), epoch2, gen ROC tc.
    # s2 natural mode neg/base-typ; s4 self/base-typ. Train CSV uses same eval labels.
    def train_val(mkey_model, ds, setting, mode):
        r = train_df[(train_df.model == mkey_model) & (train_df.dataset == ds)
                     & (train_df.setting == setting) & (train_df.epoch == "ep2")
                     & (train_df["eval"] == mode) & (train_df.variant == "tc")]
        return None if r.empty else round(float(r["gen_roc"].iloc[0]) * 100, 1)
    rows = []
    for ds in ["upper", "multi"]:
        for s, mode in [("s2", "neg/base-typ"), ("s4", "self/base-typ")]:
            tr = train_val("gemma-4-31b", ds, s, mode)
            te = test_val(test_df, "gemma", ds, s, mode)
            gap = None if (tr is None or te is None) else round(tr - te, 1)
            rows.append((ds, SETTING_LABEL[s], tr, te, gap))
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Data & Setting & Train & Test & $\Delta$(train$-$test) \\", r"\midrule"]
    for ds, lab, tr, te, gap in rows:
        f = lambda x: "---" if x is None else f"{x}"
        lines.append(f"{ds} & {lab} & {f(tr)} & {f(te)} & {f(gap)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "he_traintest_gemma.tex").write_text("\n".join(lines) + "\n")
    print("wrote he_traintest_gemma.tex")


# ---- per-candidate perfile metrics (have spearman + per-problem rows for SE) -------------
import re
import numpy as np
HARVEST = REPO / "docs/he_harvest_2026-06-17"
PERFILE = ["qwen_he_perfile_metrics.csv", "gemma_cu_perfile_metrics.csv", "gemma_cm_perfile_metrics.csv"]


def _parse(fname):
    m = re.match(r"scores_(basetypneg|basetyp|self|neg)-", fname); pref = m.group(1) if m else "?"
    is_base = ("v6-" in fname) and ("-v7-" not in fname)
    mkey = "qwen" if ("Qwen3.5-9B" in fname or "Qwen--Qwen3" in fname or "Qwen_Qwen3" in fname) else ("gemma" if "gemma-4-31" in fname.lower() else "?")
    ds = "upper" if "correct-upper" in fname else ("multi" if "correct-multi" in fname else "?")
    if is_base: s = "base"
    elif ("tcs" in fname) or ("tc-self" in fname): s = "s4"
    elif ("tcn" in fname) or ("tc-neg" in fname): s = "s7"
    elif "cft" in fname: s = "s13"
    elif ("fsx" in fname) or ("force-same-x" in fname): s = "s3"
    elif ("lo0.1" in fname) or ("labelonly" in fname) or re.search(r"-p0-", fname) or ("pref0" in fname): s = "s1"
    else: s = "s2"
    typ = "neg" if pref in ("neg", "basetypneg") else "self"
    bt = "base-typ" if pref in ("basetyp", "basetypneg") else "no-base"
    return mkey, ds, s, f"{typ}/{bt}"


def load_perfile():
    rows = []
    for fn in PERFILE:
        df = pd.read_csv(HARVEST / fn)
        df = df[df["file"].str.contains("humaneval", na=False)].copy()
        meta = df["file"].apply(lambda f: pd.Series(_parse(f), index=["mkey", "ds", "setting", "evalmode"]))
        rows.append(pd.concat([meta, df], axis=1))
    return pd.concat(rows, ignore_index=True)


def _ms(perfile, mkey, ds, setting, mode, col):
    sub = perfile[(perfile.mkey == mkey) & (perfile.ds == ds) & (perfile.setting == setting)
                  & (perfile.evalmode == mode) & (perfile.variant == "tc")]
    if sub.empty: return None
    v = sub[col].to_numpy(dtype=float); v = v[~np.isnan(v)]
    if len(v) == 0: return None
    return v.mean() * 100, (v.std(ddof=1) / np.sqrt(len(v)) * 100 if len(v) > 1 else float("nan"))


# eval-mode label used in perfile (no-base/base-typ) for each setting's natural mode
NAT_PF = {"base": "self/no-base", "s1": "self/base-typ", "s2": "neg/base-typ",
          "s3": "neg/base-typ", "s4": "self/base-typ", "s7": "neg/base-typ", "s13": "self/base-typ"}


def build_corr_table(perfile, col, outname):
    def cell(mk, ds, s):
        r = _ms(perfile, mk, ds, s, NAT_PF[s], col)
        return "---" if r is None else f"{r[0]:.1f}\\stdv{{{r[1]:.1f}}}" if not np.isnan(r[1]) else f"{r[0]:.1f}"
    lines = [r"\begin{tabular}{ll" + "c" * len(SETTINGS) + "}", r"\toprule",
             "Model & Data & " + " & ".join(SETTING_LABEL[s] for s in SETTINGS) + r" \\", r"\midrule"]
    for mk, ml in [("gemma", "gemma-4-31b"), ("qwen", "qwen-3.5-9b")]:
        for ds in ["upper", "multi"]:
            lines.append(f"{ml} & {ds} & " + " & ".join(cell(mk, ds, s) for s in SETTINGS) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (OUT / outname).write_text("\n".join(lines) + "\n")
    print("wrote", outname)


def build_tc_compare(perfile):
    # s3 (no-TC) vs s4 (self-TC) in self mode; s3 vs s7 (neg-TC) in neg mode. gen ROC tc, base-typ.
    def g(mk, ds, s, mode):
        r = _ms(perfile, mk, ds, s, mode, "gen_roc"); return None if r is None else r[0]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Data & s3 (self) & s4 (+self-TC) & s3 (neg) & s7 (+neg-TC) \\", r"\midrule"]
    for mk, ml in [("gemma", "gemma-4-31b"), ("qwen", "qwen-3.5-9b")]:
        for ds in ["upper", "multi"]:
            s3s = g(mk, ds, "s3", "self/base-typ"); s4 = g(mk, ds, "s4", "self/base-typ")
            s3n = g(mk, ds, "s3", "neg/base-typ"); s7 = g(mk, ds, "s7", "neg/base-typ")
            def d(a, b): return "" if (a is None or b is None) else f" ({'+' if b-a>=0 else ''}{b-a:.1f})"
            f = lambda x: "---" if x is None else f"{x:.1f}"
            lines.append(f"{ml} & {ds} & {f(s3s)} & {f(s4)}{d(s3s,s4)} & {f(s3n)} & {f(s7)}{d(s3n,s7)} " + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
    (OUT / "he_tc_compare_test.tex").write_text("\n".join(lines) + "\n")
    print("wrote he_tc_compare_test.tex")


def main():
    test_df = pd.read_csv(TEST)
    train_df = pd.read_csv(TRAIN)
    build_q1(test_df)
    build_traintest(test_df, train_df)
    perfile = load_perfile()
    build_corr_table(perfile, "spearman", "he_spearman_test.tex")
    build_corr_table(perfile, "pearson", "he_pearson_test.tex")
    build_tc_compare(perfile)
    print("done.")


if __name__ == "__main__":
    main()
