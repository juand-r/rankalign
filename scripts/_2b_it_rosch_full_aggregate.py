#!/usr/bin/env python3
"""Aggregate 2b-it × rosch v7 evals into a markdown doc — fresh from disk
(does NOT rely on metrics-from-scores/rosch_v7_2b-it_*.csv since those may
be stale w.r.t. today's evals).
"""
from __future__ import annotations
import sys
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
sys.path.insert(0, str(REPO / "scripts"))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

OUT = Path("/datastor2/jdr/rankalign/outputs")

ROSCH_TASKS = [
    "rosch-bird", "rosch-carpenters-tool", "rosch-clothing", "rosch-fruit",
    "rosch-furniture", "rosch-sport", "rosch-toy", "rosch-vegetable",
    "rosch-vehicle", "rosch-weapon",
]

# (method_num, label, fingerprint, epoch_used)
SETTINGS = [
    (0,  "Base",                    "v6-google_gemma-2-2b-it",                                                                                                       "—"),
    (1,  "SFT labelonly 10%",       "__MISSING__",                                                                                                                   "—"),
    (2,  "RankAlign",               "v7-google--gemma-2-2b-it-delta0.85-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1", "e2"),
    (3,  "New + fsx [-TC]",         "v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1",                                          "e2"),
    (4,  "New + PMI + fsx",         "v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcs-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1",                                      "e2"),
    (7,  "New + NegTC + fsx",       "v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcn-nv1-ng1-vlo-fsx-ppd-sm0.1-fix1",                                      "e2"),
    (11, "New + PMI [-fsx]",        "v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcs-nv1-ng1-vlo-sm0.1-fix1",                                              "e2"),
    (12, "New + NegTC [-fsx]",      "v7-gemma-2-2b-it-d1.74-e2-membership-sans-rosch-v0-all-tcn-nv1-ng1-vlo-sm0.1-fix1",                                              "e2"),
    (13, "SFT + CFT",               "v7-gemma-2-2b-it-d0.81-e2-membership-sans-rosch-v0-all-p0-nv1-ng1-cft-lo0.1-fix1",                                               "e2"),
]

COLS = [
    ("Raw",       "self-",       "raw"),
    ("PMI self",  "self-",       "tc"),
    ("PMI base",  "basetyp-",    "tc"),
    ("Neg self",  "neg-",        "tc"),
    ("Neg base", "basetypneg-", "tc"),
]

_CACHE: dict[Path, dict] = {}


def find_csv(prefix: str, model_fp: str, task: str) -> Path | None:
    if model_fp == "__MISSING__":
        return None
    pattern = f"scores_{prefix}{model_fp}_{task}_test_log-odds_tc_*.csv"
    matches = sorted(glob(str(OUT / pattern)))
    return Path(matches[-1]) if matches else None


def metrics_for(p: Path) -> dict:
    if p not in _CACHE:
        try:
            df = load_scores(p)
            _CACHE[p] = compute_all_metrics(df)
        except Exception:
            _CACHE[p] = {}
    return _CACHE[p]


def cell(model_fp: str, prefix: str, variant: str, metric_key: str, scale: bool):
    vals = []
    for task in ROSCH_TASKS:
        p = find_csv(prefix, model_fp, task)
        if p is None:
            continue
        m = metrics_for(p)
        if variant in m and not np.isnan(m[variant].get(metric_key, np.nan)):
            vals.append(m[variant][metric_key])
    if not vals:
        return None, None, 0
    arr = np.array(vals, dtype=float)
    mu = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    if scale:
        mu *= 100
        se *= 100
    return mu, se, int(arr.size)


def fmt_cell(mu, se, n, scale: bool, expected_n=10):
    if mu is None:
        return "--"
    if scale:
        body = f"{mu:.1f} ± {se:.1f}"
    else:
        body = f"{mu:.3f} ± {se:.3f}"
    if n < expected_n:
        body += f" ({n}/10)"
    return body


def aggregate_metric(metric_key: str, scale: bool) -> pd.DataFrame:
    rows = []
    for num, label, fp, ep in SETTINGS:
        row = {"#": num, "Setting": label, "Epoch": ep}
        for col_label, prefix, variant in COLS:
            if num == 0:
                if col_label == "Raw":
                    use_prefix = "self-"
                elif col_label in ("PMI self", "PMI base"):
                    use_prefix = "self-"
                elif col_label in ("Neg self", "Neg base"):
                    use_prefix = "neg-"
                else:
                    use_prefix = prefix
                mu, se, n = cell(fp, use_prefix, variant, metric_key, scale)
            else:
                mu, se, n = cell(fp, prefix, variant, metric_key, scale)
                if mu is None and col_label == "Raw":
                    for fb in ("basetyp-", "neg-", "basetypneg-"):
                        mu, se, n = cell(fp, fb, variant, metric_key, scale)
                        if mu is not None:
                            break
            row[col_label] = fmt_cell(mu, se, n, scale)
        rows.append(row)
    return pd.DataFrame(rows)


def render_md(df: pd.DataFrame, metric_label: str) -> str:
    cols = ["#", "Setting", "Epoch"] + [c for c, _, _ in COLS]
    lines = [f"### {metric_label}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    blocks = []
    for metric_key, label, scale in [
        ("gen_roc",  "GenROC × 100",  True),
        ("val_roc",  "ValROC × 100",  True),
        ("val_acc",  "ValAcc × 100",  True),
        ("pearson",  "Pearson r",     False),
        ("spearman", "Spearman r",    False),
    ]:
        df = aggregate_metric(metric_key, scale)
        blocks.append(render_md(df, label))

    n_unique_csvs = len(_CACHE)
    header = (
        "# Gemma-2-2b-it × Rosch — full v7 aggregation (fresh from disk)\n"
        "\n"
        f"_Aggregated {n_unique_csvs} unique CSV files. Cells: mean ± SE across the "
        "10 rosch tasks._\n"
        "\n"
        "**Eval-column convention:** `Raw` = log P(y|x); `PMI {self,base}` = "
        "tc-correction using {fine-tuned, base} model logprobs as the typicality "
        "reference; `Neg {self,base}` = same with the negated-prompt alternative. "
        "For the **Base** row only, `PMI self == PMI base` and "
        "`Neg self == Neg base` since the base model IS its own self.\n"
        "\n"
        "**Settings table:**\n"
        "\n"
        "| # | Setting | Train fingerprint | Epoch | Note |\n"
        "|---|---|---|---|---|\n"
    )
    for num, label, fp, ep in SETTINGS:
        if fp == "__MISSING__":
            note = "no eval CSVs on disk — never trained or evaluated"
            short = "—"
        else:
            note = ""
            short = fp[:64] + "…" if len(fp) > 64 else fp
        header += f"| {num} | {label} | `{short}` | {ep} | {note} |\n"
    header += "\n"

    body = "\n\n".join(blocks)
    full = header + body + "\n"

    if args.out:
        Path(args.out).write_text(full)
        print(f"Wrote {len(full)} chars to {args.out}")
    else:
        print(full)


if __name__ == "__main__":
    main()
