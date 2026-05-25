#!/usr/bin/env python3
"""Build per-metric morning results markdown files. One file per metric.
Each file groups by (model x dataset) and shows a table with:
  Setting | Train | Raw | basetyp (PMI base) | self (PMI self) | basetypneg (Neg base) | neg (Neg self)

Cells are filled with `mean ± SE` (×100 already from the table_cells.csv).
"""
import csv, os, re, subprocess
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
METRICS_DIR = REPO / "metrics-from-scores"
MODELS_DIR = Path("/datastor2/jdr/rankalign/models2")
JOB_LOG = REPO / "overnight" / "_overnight_jobids.txt"

DOC_OUT_DIR = REPO / "docs"

METRICS = ["gen_roc", "pearson", "spearman", "val_roc", "val_acc"]
METRIC_NAMES = {
    "gen_roc": "GenROC",
    "pearson": "Pearson(gen, val)",
    "spearman": "Spearman(gen, val)",
    "val_roc": "ValROC",
    "val_acc": "ValAcc",
}

# (model, dataset_label, csv_path_template, train_dataset_key)
# `train_dataset_key` is what we look up in disk/queue to decide the Train col;
# it is also what we use to gate "has the model been trained at all?".
SOURCES = [
    # rosch — eval set (Rosch 2b cross-categorization tasks).
    # The training data is `membership-sans-rosch-v0`; every "membership"
    # row below is therefore "trained on membership-sans-rosch, evaluated on
    # the 10 held-out rosch tasks (bird, carpenters-tool, clothing, fruit,
    # furniture, sport, toy, vegetable, vehicle, weapon)".
    ("gemma-2-2b",       "membership (eval = rosch, all 10 tasks)", "rosch_v7_2b_{m}_table_cells.csv",          "membership"),
    ("gemma-2-2b-it",    "membership (eval = rosch, all 10 tasks)", "rosch_v7_2b-it_{m}_table_cells.csv",       "membership"),
    ("gemma-2-9b-it",    "membership (eval = rosch, all 10 tasks)", "rosch_v7_9b-it_{m}_table_cells.csv",       "membership"),
    # persona — all 6 personas
    ("gemma-2-2b",       "persona (all 6 personas)",     "persona_v1_v7_gemma-2-2b_all_{m}_table_cells.csv",    "persona"),
    ("gemma-2-2b-it",    "persona (all 6 personas)",     "persona_v1_v7_gemma-2-2b-it_all_{m}_table_cells.csv", "persona"),
    ("gemma-2-9b-it",    "persona (all 6 personas)",     "persona_v1_v7_gemma-2-9b-it_all_{m}_table_cells.csv", "persona"),
    # persona — ID split (3 in-domain personas: psychopathy, machiavellianism, narcissism)
    ("gemma-2-2b",       "persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)", "persona_v1_v7_gemma-2-2b_id_{m}_table_cells.csv",    "persona"),
    ("gemma-2-2b-it",    "persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)", "persona_v1_v7_gemma-2-2b-it_id_{m}_table_cells.csv", "persona"),
    ("gemma-2-9b-it",    "persona ID (3 in-domain: psychopathy, machiavellianism, narcissism)", "persona_v1_v7_gemma-2-9b-it_id_{m}_table_cells.csv", "persona"),
    # persona — OOD split (3 held-out personas)
    ("gemma-2-2b",       "persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)", "persona_v1_v7_gemma-2-2b_ood_{m}_table_cells.csv",    "persona"),
    ("gemma-2-2b-it",    "persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)", "persona_v1_v7_gemma-2-2b-it_ood_{m}_table_cells.csv", "persona"),
    ("gemma-2-9b-it",    "persona OOD (3 held-out: desire-to-create-allies, interest-in-music, interest-in-science)", "persona_v1_v7_gemma-2-9b-it_ood_{m}_table_cells.csv", "persona"),
    # ifeval — only gemma-2-9b-it has been trained on ifeval-concat-all.
    # IMPORTANT: the existing ifeval table_cells CSVs are built by the legacy
    # `_build_ifeval_ood_table.py`, which hardcodes
    # `CP = v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all_d2g_random_alpha1.0`.
    # All cells therefore come from a v6 (pre-fix1) training run, NOT the v7
    # ifeval-concat-all models we have on disk now. The v7 ifeval × s13 trains
    # are still in flight; v7 evals for s1-s9 have not been launched. Treat
    # these rows as legacy reference, not as a v7 result.
    # Filename quirk: gen_roc CSV omits the metric infix.
    ("gemma-2-9b-it",    "ifeval ID  **[v6 LEGACY]** (n=79 prompts \u2265 22; held-out 50% of completions for each)", "{ifevalprefix_id}",  "ifeval"),
    ("gemma-2-9b-it",    "ifeval OOD **[v6 LEGACY]** (n=20 fully held-out prompts: prompt_1..13, 15..21)",            "{ifevalprefix_ood}", "ifeval"),
    # humaneval
    ("gemma-4-31B-it",   "humaneval",  "humaneval_v2.1correct-upper_g4-31B-it_{m}_table_cells.csv", "humaneval"),
]

# Map row labels in CSVs to (setting_name, sN). The table builders use these
# stable labels for every metric. method_num 0 is Base; 1-13 are settings.
# Note: 8/9 are NegTC analogs of 5/6; we render them even if some sources
# (ifeval, humaneval) don't have s13 data — empty rows are fine.
SETTING_LABELS = [
    (0,  "Base",                       None),
    (1,  "SFT labelonly 10%",          "s1"),
    (2,  "RankAlign",                  "s2"),
    (3,  "New + fsx [-TC]",            "s3"),
    (4,  "New + PMI + fsx",            "s4"),
    (5,  "RA + PMI + fsx [-NLL]",      "s5"),
    (6,  "RA + PMI [+TC]",             "s6"),
    (7,  "New + NegTC + fsx",          "s7"),
    (8,  "RA + NegTC + fsx [-NLL]",    "s8"),
    (9,  "RA + NegTC [+TC]",           "s9"),
    (11, "New + PMI [-fsx]",           "s11"),
    (12, "New + NegTC [-fsx]",         "s12"),
    (13, "SFT + CFT",                  "s13"),
]

# Columns in our final table (5 cells per row) and how they map to the
# CSV `column` field.
COL_MAP = [
    ("Raw",                       "Raw"),
    ("basetyp- (PMI base)",       "PMI base"),
    ("self- (PMI self)",          "PMI self"),
    ("basetypneg- (Neg base)",    "Neg base"),
    ("neg- (Neg self)",           "Neg self"),
]


# -----------------------------------------------------------------------
# Disk + queue scan (reused logic from _morning_status.py, abridged).

DATASETS_FULL = {
    "membership": "membership-sans-rosch-v0",
    "persona":    "persona-v1",
    "ifeval":     "ifeval-concat",
    "humaneval":  "humaneval-v2.1correct-upper",
}


def classify_dir(d: str):
    if "_merged" in d: return None
    if not d.startswith("v7-google--"): return None
    m_ep = re.search(r"-epoch(\d+)--", d)
    if not m_ep: return None
    epoch = int(m_ep.group(1))
    model = None
    for cand in ["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-4-31B-it"]:
        if f"v7-google--{cand}-delta" in d:
            model = cand; break
    if not model: return None
    ds = None
    for short, full in DATASETS_FULL.items():
        if f"--{full}-all--" in d: ds = short; break
    if not ds: return None
    if "--cft--" in d and "--labelonly0.1--" in d and "--pref0.0--" in d: s="s13"
    elif "--labelonly0.1--" in d and "--pref0.0--" in d: s="s1"
    elif "--tc-self--" in d and "--force-same-x--" in d and "--vallogodds--" in d: s="s4"
    elif "--tc-self--" in d and "--force-same-x--" in d: s="s5"
    elif "--tc-self--" in d and "--vallogodds--" in d: s="s11"
    elif "--tc-self--" in d: s="s6"
    elif "--tc-neg--" in d and "--force-same-x--" in d: s="s7"
    elif "--tc-neg--" in d: s="s12"
    elif "--force-same-x--" in d: s="s3"
    else: s="s2"
    return (model, ds, s, epoch)


def disk_coverage():
    cov = {}
    if not MODELS_DIR.is_dir(): return cov
    for entry in MODELS_DIR.iterdir():
        cls = classify_dir(entry.name)
        if not cls: continue
        model, ds, s, ep = cls
        key = (ds, model, s)
        cov[key] = max(cov.get(key, -1), ep)
    return cov


def parse_label(label):
    m = re.search(r"-s(\d+)$", label)
    if not m: return None
    setting = f"s{m.group(1)}"
    rest = label[: m.start()]
    for ds in ("persona", "membership", "ifeval", "humaneval"):
        if rest.startswith(ds + "-"): return ds, rest[len(ds)+1:], setting
    return None


def build_jobid_index():
    idx = {}
    if not JOB_LOG.is_file(): return idx
    for line in JOB_LOG.read_text(errors="replace").splitlines():
        m = re.search(r"\s+(\S+)\s+TRAIN=(\d+)\b", line)
        if not m: continue
        label, jid = m.group(1), m.group(2)
        label = re.sub(r"-evaltc-(self|neg)(-EVALONLY)?$", "", label)
        p = parse_label(label)
        if p: idx[jid] = p
    return idx


def queue_status():
    out = subprocess.run(
        ["squeue", "-u", "jdr", "-h", "-o", "%i %t %M %L %j"],
        capture_output=True, text=True
    ).stdout.splitlines()
    idx = build_jobid_index()
    rows = {}
    for line in out:
        parts = line.split(maxsplit=4)
        if len(parts) < 5: continue
        jid, st, el, rem, name = parts
        if name != "wrap": continue
        cell = idx.get(jid)
        if cell:
            rows[cell] = (jid, st, el, rem)
    return rows


def train_status_str(disk, queue, ds, model, s, has_eval_data=False):
    qrow = queue.get((ds, model, s))
    if qrow:
        jid, st, el, rem = qrow
        if st == "R":
            return f"⏳ {jid} R {el} (≤{rem})"
        return f"⏳ {jid} {st}"
    ep = disk.get((ds, model, s), -1)
    if ep >= 1:
        return f"✓ ep={ep}"
    # If CSV has eval data but model is not on disk, model dir was likely
    # deleted/renamed. Mark "✓ (prior)" so the reader knows the cell is real.
    if has_eval_data:
        return "✓ (prior run)"
    return "–"


# -----------------------------------------------------------------------

def fmt_cell(cells_dict, csv_col):
    """Look up (mean, se, n, note) for a CSV column. Return formatted string."""
    if csv_col not in cells_dict:
        return "—"
    mean, se, n, note = cells_dict[csv_col]
    if mean == "" or n == "0":
        return "—"
    try:
        m = float(mean); s = float(se) if se else 0.0
    except Exception:
        return "—"
    return f"{m:.2f} ± {s:.2f}"


def load_table_cells(path: Path):
    """Return {method_num: {column: (mean, se, n, note)}}"""
    out = defaultdict(dict)
    if not path.is_file():
        return out
    with path.open() as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                mn = int(row["method_num"])
            except Exception:
                continue
            out[mn][row["column"]] = (row["mean"], row["se"], row["n"], row["note"])
    return out


def build_metric_doc(metric: str, disk: dict, queue: dict) -> str:
    full_name = METRIC_NAMES[metric]
    today = datetime.now(timezone.utc).strftime("%FT%TZ")
    out = []
    out.append(f"# Morning Results — {full_name} — {today}")
    out.append("")
    out.append(f"All cells: **{full_name} × 100 ± SE** (mean ± SE across the eval-task split for that section).")
    out.append("")
    out.append("Section header convention: `<model> × <eval-set label>`. The model is")
    out.append("the (LoRA-finetuned base) generator under test; the eval-set label says")
    out.append("which held-out task slice the cells were averaged over. For example,")
    out.append("`gemma-2-9b-it × membership (eval = rosch, all 10 tasks)` means: gemma-2-9b-it")
    out.append("trained on `membership-sans-rosch-v0` and evaluated on the 10 held-out Rosch")
    out.append("cross-categorization tasks (bird, carpenters-tool, clothing, fruit, furniture,")
    out.append("sport, toy, vegetable, vehicle, weapon). `persona ID` / `persona OOD` are the")
    out.append("3+3 splits of `persona-v1` (see headers for the per-persona task names).")
    out.append("")
    out.append("Train column: ✓ ep=N done · ⏳ jobid R elapsed (≤remaining) in-flight · – not started.")
    out.append("Empty cells (—): no eval CSV with that prefix yet.")
    out.append("")
    out.append("> **\u26a0\ufe0f IMPORTANT PROVENANCE CAVEAT — IFEVAL SECTIONS ARE V6, NOT V7.**")
    out.append(">")
    out.append("> All other sections (rosch, persona, humaneval) source from **v7 (fix1)** score")
    out.append("> files. The two ifeval sections below are built by the legacy")
    out.append("> `_build_ifeval_ood_table.py`, which hardcodes a v6 model prefix")
    out.append("> (`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). Their cells")
    out.append("> therefore reflect a **pre-fix1 v6 9b-it run**, not the v7 models we have on")
    out.append("> disk now. The v7 ifeval × s13 trains are still in flight; no v7 ifeval evals")
    out.append("> have been launched for s1\u2013s9 yet. ifeval section headers are tagged")
    out.append("> **`[v6 LEGACY]`** and each ifeval section repeats this warning right above")
    out.append("> the table.")
    out.append("")
    out.append("**Method-row coverage.** Tables include s1–s9 and s11–s13 (s10 was never run).")
    out.append("If a row shows all `—`, the eval CSV reports `n=0/missing` for that method;")
    out.append("ifeval and humaneval CSVs do not have an s13 row at all (the v6 builders")
    out.append("predate s13).")
    out.append("")

    # ifeval gen_roc CSV filename has no metric infix; other metrics do.
    if metric == "gen_roc":
        ifeval_id_name = "ifeval_id_table_cells.csv"
        ifeval_ood_name = "ifeval_ood_table_cells.csv"
    else:
        ifeval_id_name = f"ifeval_id_{metric}_table_cells.csv"
        ifeval_ood_name = f"ifeval_ood_{metric}_table_cells.csv"
    fmt_ctx = {"m": metric, "ifevalprefix_id": ifeval_id_name, "ifevalprefix_ood": ifeval_ood_name}

    for model, ds_label, tmpl, ds_key in SOURCES:
        path = METRICS_DIR / tmpl.format(**fmt_ctx)
        out.append(f"## {model} × {ds_label}")
        out.append("")
        if ds_key == "ifeval":
            out.append("> **\u26a0\ufe0f V6 LEGACY DATA — NOT v7.** All cells in this section come from")
            out.append("> a pre-fix1 v6 training run "
                       "(`v6-google_gemma-2-9b-it-delta0.15-epoch2_ifeval-concat-all`). "
                       "No v7 ifeval evaluations have been run yet "
                       "(v7 × s13 trains in flight; v7 × s1\u2013s9 not started). "
                       "Treat these numbers as legacy reference, not as v7 results.")
            out.append("")
        out.append(f"Source: [{path.name}](metrics-from-scores/{path.name})")
        out.append("")
        cells = load_table_cells(path)
        if not cells:
            out.append("(no data)")
            out.append("")
            continue
        # Header
        head = ["Setting", "Train"] + [c[0] for c in COL_MAP]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join(["---"] * len(head)) + "|")
        for mn, label, sN in SETTING_LABELS:
            cd = cells.get(mn, {})
            has_data = any(
                csv_col in cd and cd[csv_col][0] not in ("", "0") and cd[csv_col][2] != "0"
                for _, csv_col in COL_MAP
            )
            if mn == 0:
                train_s = "(base model)"
            else:
                train_s = train_status_str(disk, queue, ds_key, model, sN, has_eval_data=has_data)
            row_cells = []
            for header_col, csv_col in COL_MAP:
                row_cells.append(fmt_cell(cd, csv_col))
            row_label = f"{mn} {label}" if mn != 0 else "0 Base"
            out.append("| " + row_label + " | " + train_s + " | " + " | ".join(row_cells) + " |")
        out.append("")
    return "\n".join(out)


def main():
    disk = disk_coverage()
    queue = queue_status()
    for metric in METRICS:
        doc = build_metric_doc(metric, disk, queue)
        outpath = DOC_OUT_DIR / f"morning-results-{metric.replace('_', '')}-20260525.md"
        outpath.write_text(doc)
        print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
