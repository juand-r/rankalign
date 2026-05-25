#!/usr/bin/env python3
"""Build morning-results-format tables for the POD eval runs (v7 only).

Covers the two model families evaluated on RunPod and downloaded locally:
  - Qwen3.5-9B
  - gemma-2-9b-it

All current pod eval data is **delta-bins 10** (the fixed delta=0.15 / "v7b" batch
is still training — its eval results are not in yet, shown as "soon"). For each
(model x eval-set) there can be up to three groups:

  1. delta-bins 10 · eval_model_sN (canonical) -- the primary eval. Evaluated via
     a downloaded model symlinked as /workspace/eval_model_sN. The symlink hides
     the training delta, but it was recovered from the source HF repos' commit
     messages (see EVAL_MODEL_IFEVAL_DELTA). 20 OOD prompts + ID prompts.
  2. delta-bins 10 · named deltaX.XX (earlier on-pod eval) -- an earlier/partial
     eval whose filename encodes the delta (delta0.96/1.89/1.93/1.94/2.49/...).
     Shown BELOW the canonical one for comparison ("just in case").
  3. delta 0.15 (fixed) · v7b -- the fixed-delta batch. Training pods confirmed
     `--delta 0.15`. Eval mostly not started yet -> "soon".

Eval sets:
  - ifeval OOD  : prompts 1-21 (held out entirely)
  - ifeval ID   : prompts 22-109 (50% of completions held out)
  - persona ID  : psychopathy, machiavellianism, narcissism
  - persona OOD : desire-to-create-allies, interest-in-music, interest-in-science
  - rosch       : 10 cross-categorization tasks

v6 files excluded (by model_path). Output: docs/pod-results-<metric>-<date>.md, one
per metric, same column layout as docs/morning-results-*.md:
  Setting | Train | Raw | basetyp- (PMI base) | self- (PMI self) | basetypneg- (Neg base) | neg- (Neg self)

Blank cells -> "--". Cells whose eval is actively in flight -> "soon".
Reuses summarize_scores_file.compute_all_metrics for canonical metric math.

Usage:
    python scripts/_build_pod_morning_results.py --report   # classification only
    python scripts/_build_pod_morning_results.py            # write markdown
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_scores_file import load_scores, compute_all_metrics  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SCAN_DIRS = [
    REPO / "outputs_gemma4_from_pod-v7" / "qw35_ifeval",
    REPO / "outputs_gemma4_from_pod-v7" / "ra9b_ifeval",
    REPO / "outputs_gemma4_from_pod-v7" / "qw35_persona_member",
    REPO / "outputs_gemma4_from_pod-v7" / "ra9b_persona_member",
    REPO / "outputs_gemma4_from_pod-v7b" / "v7b_persona_member",
    REPO / "outputs_gemma4_from_pod-v7b" / "v7b_ifeval",
]
DOCS_DIR = REPO / "docs"

METRICS = ["gen_roc", "pearson", "spearman", "val_roc", "val_acc"]
METRIC_NAMES = {
    "gen_roc": "GenROC", "pearson": "Pearson(gen, val)", "spearman": "Spearman(gen, val)",
    "val_roc": "ValROC", "val_acc": "ValAcc",
}

# Eval-group labels (delta regime · eval source).
G_EVALMODEL = "delta-bins 10 · eval_model_sN (canonical)"
G_NAMED = "delta-bins 10 · named deltaX.XX (earlier on-pod eval)"
G_V7B = "delta 0.15 (fixed) · v7b"
GROUP_ORDER = [G_EVALMODEL, G_NAMED, G_V7B]

# Provenance recovered 2026-05-25 from the source HF repos' commit messages.
# upload_v7_model.py strips the delta from the repo *name* but preserves the
# original checkpoint dir in the commit message ("Upload rankalign v7 checkpoint:
# v7-google--gemma-2-9b-it-delta<X>-epoch2--ifeval-concat-all--..._merged").
# Source: TAUR-dev/rankalign-v7-gemma2-9b-it-ifeval-sN-ep2 commit history.
EVAL_MODEL_IFEVAL_DELTA = {  # gemma-2-9b-it ifeval eval_model_sN
    "s1": "1.89", "s2": "1.93", "s3": "1.94", "s4": "1.94", "s7": "1.94",
}

PERSONA_ID = ["psychopathy", "machiavellianism", "narcissism"]
PERSONA_OOD = ["desire-to-create-allies", "interest-in-music", "interest-in-science"]

SETTING_LABELS = [
    (0, "Base", None),
    (1, "SFT labelonly 10%", "s1"),
    (2, "RankAlign", "s2"),
    (3, "New + fsx [-TC]", "s3"),
    (4, "New + PMI + fsx", "s4"),
    (5, "RA + PMI + fsx [-NLL]", "s5"),
    (6, "RA + PMI [+TC]", "s6"),
    (7, "New + NegTC + fsx", "s7"),
    (8, "RA + NegTC + fsx [-NLL]", "s8"),
    (9, "RA + NegTC [+TC]", "s9"),
    (11, "New + PMI [-fsx]", "s11"),
    (12, "New + NegTC [-fsx]", "s12"),
    (13, "SFT + CFT", "s13"),
]

COLUMNS = [
    ("Raw", None, "raw"),
    ("basetyp- (PMI base)", "basetyp-", "tc"),
    ("self- (PMI self)", "self-", "tc"),
    ("basetypneg- (Neg base)", "basetypneg-", "tc"),
    ("neg- (Neg self)", "neg-", "tc"),
]

SETTING_NA = {
    "s4": {"basetypneg- (Neg base)", "neg- (Neg self)"},
    "s7": {"basetyp- (PMI base)", "self- (PMI self)"},
}

_PREFIX_ORDER = ("basetypneg-", "basetyp-", "neg-", "self-")

# Cells whose eval is actively in flight -> render "soon" when absent.
# The gemma ifeval ID eval running now uses the delta-bins eval_model_sN models.
IN_FLIGHT = {
    ("gemma-2-9b-it", G_EVALMODEL, "ifeval ID"): {"s1", "s2", "s3", "s4", "s7"},
}


def is_v6(mp: str) -> bool:
    return "v6-" in mp


def family(mp: str, dirpath: str) -> str | None:
    if "Qwen3.5-9B" in mp:
        return "Qwen3.5-9B"
    if "gemma-2-9b-it" in mp:
        return "gemma-2-9b-it"
    if re.search(r"/eval_model_s\d+$", mp):
        return "Qwen3.5-9B" if "qw35" in dirpath else "gemma-2-9b-it"
    return None  # 2b / 2b-it / gemma-4 / unknown -> excluded


def group_of(mp: str, dirpath: str) -> str:
    """Classify the eval group (delta regime · source)."""
    if re.search(r"/eval_model_s\d+$", mp):
        # symlink: v7b dir -> fixed delta 0.15 batch; v7 dir -> delta-bins canonical
        return G_V7B if "v7b" in dirpath else G_EVALMODEL
    m = re.search(r"delta([0-9.]+)", mp) or re.search(r"-d([0-9.]+)-", mp)
    d = m.group(1).rstrip(".") if m else ""
    if d == "0.15":
        return G_V7B
    return G_NAMED  # named non-0.15 -> delta-bins, earlier on-pod eval


def setting_of(mp: str, fname: str) -> str | None:
    m = re.search(r"/eval_model_(s\d+)$", mp)
    if m:
        return m.group(1)
    m = re.search(r"_eval_model_(s\d+)_", fname)
    if m:
        return m.group(1)
    b = Path(mp).name.replace("_merged", "")
    fsx = "force-same-x" in b
    lo = "labelonly0.1" in b
    semi = "semi0.1" in b
    nll = "nllv1" in b
    cft = "--cft--" in b or "-cft-" in b
    tc = "self" if ("tc-self" in b or "-tcs-" in b) else ("neg" if ("tc-neg" in b or "-tcn-" in b) else None)
    if cft:
        return "s13"
    if lo and not fsx and tc is None:
        return "s1"
    if semi and not fsx and tc is None and not nll:
        return "s2"
    if semi and fsx and tc is None:
        return "s3"
    if semi and fsx and tc == "self":
        return "s4"
    if semi and fsx and tc == "neg":
        return "s7"
    return None


def eval_prefix(fname: str) -> str:
    after = fname[len("scores_"):] if fname.startswith("scores_") else fname
    for p in _PREFIX_ORDER:
        if after.startswith(p):
            return p
    return ""


def parse_task(fname: str):
    m = re.search(r"_ifeval-prompt_(\d+)_test_log-odds", fname)
    if m:
        n = int(m.group(1))
        return ("ifeval OOD" if n <= 21 else "ifeval ID"), f"prompt_{n}"
    m = re.search(r"_(persona-v1-[^_]+)_test_log-odds", fname)
    if m:
        p = m.group(1).replace("persona-v1-", "")
        if p in PERSONA_ID:
            return "persona ID", p
        if p in PERSONA_OOD:
            return "persona OOD", p
        return None, None
    m = re.search(r"_(rosch-[^_]+)_test_log-odds", fname)
    if m:
        return "rosch", m.group(1)
    return None, None


def scan():
    """records[(family, group, eval_set, setting, prefix, task)] = metrics dict,
    deduped to the newest filename per key."""
    raw = defaultdict(list)
    skipped = defaultdict(int)
    for d in SCAN_DIRS:
        if not d.is_dir():
            continue
        dirpath = str(d)
        for p in sorted(d.glob("scores_*.csv")):
            try:
                head = pd.read_csv(p, nrows=1)
            except Exception:
                skipped["read_head"] += 1
                continue
            mp = str(head["model_path"].iloc[0]) if "model_path" in head.columns else ""
            if is_v6(mp):
                skipped["v6"] += 1
                continue
            fam = family(mp, dirpath)
            if fam is None:
                skipped["other_family"] += 1
                continue
            grp = group_of(mp, dirpath)
            st = setting_of(mp, p.name)
            if st is None:
                skipped["no_setting"] += 1
                continue
            eset, task = parse_task(p.name)
            if eset is None:
                skipped["no_task"] += 1
                continue
            pfx = eval_prefix(p.name)
            try:
                df = load_scores(p)
                metrics = compute_all_metrics(df)
            except Exception:
                skipped["metric_err"] += 1
                continue
            raw[(fam, grp, eset, st, pfx, task)].append((p.name, metrics))

    records = {}
    for key, lst in raw.items():
        _, metrics = max(lst, key=lambda x: x[0])  # newest filename
        records[key] = metrics
    return records, dict(skipped)


def cell_values(records, fam, grp, eset, setting, col_header, variant, metric):
    pfx = dict((c[0], c[1]) for c in COLUMNS)[col_header]
    vals = []
    seen = set()
    for (f, g, e, s, p, t), m in records.items():
        if (f, g, e, s) != (fam, grp, eset, setting):
            continue
        if pfx is not None and p != pfx:
            continue
        if variant not in m:
            continue
        v = m[variant].get(metric)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if pfx is None:  # Raw is prefix-agnostic: count each task once
            if t in seen:
                continue
            seen.add(t)
        vals.append(float(v))
    return vals


def fmt(vals, scale=100.0):
    if not vals:
        return None
    mean = np.mean(vals) * scale
    if len(vals) > 1:
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) * scale
        return f"{mean:.1f} ± {se:.1f}"
    return f"{mean:.1f}"


def build_table(records, fam, grp, eset, metric):
    lines = []
    head = ["Setting", "Train"] + [c[0] for c in COLUMNS]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    inflight = IN_FLIGHT.get((fam, grp, eset), set())
    any_data = False
    for mn, label, sN in SETTING_LABELS:
        cells = []
        train = "(base model)" if mn == 0 else ""
        row_has = False
        for col_header, _pfx, variant in COLUMNS:
            if sN in SETTING_NA and col_header in SETTING_NA[sN]:
                cells.append("N/A")
                continue
            vals = cell_values(records, fam, grp, eset, sN, col_header, variant, metric) if sN else []
            s = fmt(vals)
            if s is None:
                cells.append("soon" if sN in inflight else "--")
            else:
                cells.append(s)
                row_has = True
                any_data = True
        if sN and row_has:
            train = "✓"
        elif sN and sN in inflight:
            train = "soon"
        elif mn != 0:
            train = "–"
        lines.append(f"| {mn} {label} | {train} | " + " | ".join(cells) + " |")
    return "\n".join(lines), any_data


def build_doc(records, metric):
    today = datetime.now(timezone.utc).strftime("%FT%TZ")
    name = METRIC_NAMES[metric]
    out = [
        f"# Pod Results — {name} — {today}",
        "",
        f"All cells: **{name} × 100**, mean ± SE across the eval-task split for that section "
        "(no ± when a single task). Scope: **v7 only** (v6 excluded), models **Qwen3.5-9B** and "
        "**gemma-2-9b-it**, evaluated on RunPod and downloaded locally.",
        "",
        "**Groups** (per model × eval-set, shown in this order):",
        f"1. **{G_EVALMODEL}** — the primary eval (downloaded model symlinked as `eval_model_sN`; "
        "20 OOD prompts + ID). Training delta recovered from the source HF repo commit messages: "
        "gemma-2-9b-it ifeval s1=1.89, s2=1.93, s3/s4/s7=1.94 (all delta-bins 10). Qwen experiments "
        "are delta-bins 10 throughout (per design).",
        f"2. **{G_NAMED}** — an earlier/partial eval whose filename encodes the delta. Shown for "
        "comparison; differs from the canonical eval mainly because it covers fewer prompts.",
        f"3. **{G_V7B}** — the fixed delta=0.15 batch (v7b training pods confirmed `--delta 0.15`). "
        "Eval mostly not started -> `soon`.",
        "",
        "**Columns** = scoring method at eval time. Raw = log P(y|x); basetyp-/self- = PMI vs base/self; "
        "basetypneg-/neg- = Neg vs base/self. `N/A` = variant not run for that setting "
        "(s4 = basetyp+self only; s7 = basetypneg+neg only).",
        "",
        "**Cells:** `--` = no data · `soon` = eval in flight. Train: ✓ data present · soon in flight · – not run.",
        "",
        "> ifeval **OOD** = prompts 1–21 (fully held out); **ID** = prompts 22–109 (50% completions held "
        "out). This is the data split, independent of delta.",
        "",
    ]
    present = {(f, g, e) for (f, g, e, s, p, t) in records}
    families = ["gemma-2-9b-it", "Qwen3.5-9B"]
    eval_sets = ["ifeval OOD", "ifeval ID", "persona ID", "persona OOD", "rosch"]
    for fam in families:
        for eset in eval_sets:
            for grp in GROUP_ORDER:
                inflight = IN_FLIGHT.get((fam, grp, eset), set())
                if (fam, grp, eset) not in present and not inflight:
                    continue
                table, any_data = build_table(records, fam, grp, eset, metric)
                if not any_data and not inflight:
                    continue
                out.append(f"## {fam} × {eset} — {grp}")
                if grp == G_EVALMODEL and eset.startswith("ifeval") and fam == "gemma-2-9b-it":
                    out.append("")
                    out.append("> Per-setting training delta (HF-commit provenance): "
                               "s1=1.89, s2=1.93, s3/s4/s7=1.94 — all delta-bins 10.")
                out.append("")
                out.append(table)
                out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print classification summary only")
    args = ap.parse_args()

    records, skipped = scan()
    print(f"Scanned -> {len(records)} deduped records. Skipped: {skipped}", file=sys.stderr)

    if args.report:
        from collections import Counter
        c = Counter((f, g, e, s) for (f, g, e, s, p, t) in records)
        for k in sorted(c, key=lambda x: (x[0], x[2], x[1], x[3])):
            print(f"  {k}  x{c[k]}")
        return

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    for metric in METRICS:
        doc = build_doc(records, metric)
        outp = DOCS_DIR / f"pod-results-{metric.replace('_', '')}-{today}.md"
        outp.write_text(doc)
        print(f"wrote {outp.relative_to(REPO)}")


if __name__ == "__main__":
    main()
