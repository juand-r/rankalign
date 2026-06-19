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


def main():
    test_df = pd.read_csv(TEST)
    train_df = pd.read_csv(TRAIN)
    build_q1(test_df)
    build_traintest(test_df, train_df)
    print("done.")


if __name__ == "__main__":
    main()
