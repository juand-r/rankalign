#!/usr/bin/env python3
"""End-to-end audit of docs/morning-results-*-20260525.md against source CSVs.

Checks:
 1. n-claim per section header (numbers in the label) vs CSV's max non-zero n.
 2. Cell values in markdown match CSV mean/se rounded to .2f.
 3. SETTING_LABELS coverage: are there method_nums in CSV not shown in doc?
 4. v6 vs v7 provenance per source: does the long.csv contain non-v7 refs
    other than the base row?
 5. Train column "ep=N" vs disk dir; "(prior run)" vs missing dir.
"""
import re, csv
from pathlib import Path
from collections import defaultdict

REPO = Path("/datastor1/jdr/gv-gap/rankalign")
DOCS = sorted((REPO / "docs").glob("morning-results-*-20260525.md"))
METRICS = REPO / "metrics-from-scores"
MODELS_DIR = Path("/datastor2/jdr/rankalign/models2")

# Same SETTING_LABELS as the renderer.
RENDER_METHOD_NUMS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13}

CLAIMED_N = {  # from human-readable labels in section headers
    "membership (eval = rosch, all 10 tasks)": 10,
    "persona (all 6 personas)": 6,
    "persona ID (3 in-domain": 3,
    "persona OOD (3 held-out": 3,
    "ifeval ID (held-out 50% of completions": "?",
    "ifeval OOD (20 fully held-out prompts": 20,
    "humaneval": "?",
}

def load_cells(path):
    out = {}  # (method_num, column) -> {mean, se, n, note}
    if not path.is_file(): return out
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                mn = int(r["method_num"])
            except Exception:
                continue
            out[(mn, r["column"])] = r
    return out

def load_long_versions(path):
    if not path.is_file(): return {}
    versions = defaultdict(int)
    with path.open() as f:
        reader = csv.DictReader(f)
        last_col = "file"
        for r in reader:
            fname = r.get("file") or r.get(last_col, "")
            m = re.search(r"(v[0-9]+)-google", fname)
            if m: versions[m.group(1)] += 1
    return dict(versions)

issues = []

def section_iter(md_text):
    blocks = re.split(r"^(## .+)$", md_text, flags=re.MULTILINE)
    # blocks alternates: prefix, header1, body1, header2, body2, ...
    for i in range(1, len(blocks), 2):
        yield blocks[i].strip(), blocks[i+1] if i+1 < len(blocks) else ""

def parse_md_cells(body):
    rows = []
    for line in body.splitlines():
        if not line.startswith("| "): continue
        if "Setting" in line and "Train" in line: continue
        if line.startswith("|---"): continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7: continue
        m = re.match(r"^(\d+)\s+(.+?)$", cells[0])
        if not m: continue
        rows.append({
            "method_num": int(m.group(1)),
            "method_name": m.group(2),
            "train": cells[1],
            "Raw": cells[2],
            "PMI base": cells[3],
            "PMI self": cells[4],
            "Neg base": cells[5],
            "Neg self": cells[6],
        })
    return rows

def parse_md_csvname(body):
    m = re.search(r"\[([^\]]+\.csv)\]\(metrics-from-scores/[^)]+\)", body)
    return m.group(1) if m else None

def fmt(mean, se):
    if mean == "" or mean is None: return "—"
    try:
        return f"{float(mean):.2f} ± {float(se or 0.0):.2f}"
    except Exception:
        return "—"

# Build disk coverage similarly to the renderer
disk_eps = {}  # (ds, model, sN) -> max epoch
DATASETS_FULL = {
    "membership": "membership-sans-rosch-v0",
    "persona":    "persona-v1",
    "ifeval":     "ifeval-concat",
    "humaneval":  "humaneval-v2.1correct-upper",
}
def classify_dir(d):
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

for ent in MODELS_DIR.iterdir():
    cls = classify_dir(ent.name)
    if not cls: continue
    model, ds, s, ep = cls
    key = (ds, model, s)
    disk_eps[key] = max(disk_eps.get(key, -1), ep)

# Iterate all 5 metric docs
for md_path in DOCS:
    metric = re.search(r"morning-results-(\w+)-", md_path.name).group(1)
    md_text = md_path.read_text()
    for header, body in section_iter(md_text):
        # extract model + label
        if " × " not in header: continue
        model_part, label_part = header.split(" × ", 1)
        model = model_part.replace("## ", "").strip()
        # find claimed n
        claimed_n = None
        for prefix, n in CLAIMED_N.items():
            if prefix in label_part: claimed_n = n; break
        # source CSV
        csv_name = parse_md_csvname(body)
        if not csv_name:
            issues.append(("missing_csv_link", md_path.name, header))
            continue
        cells = load_cells(METRICS / csv_name)
        if not cells:
            issues.append(("empty_csv", md_path.name, header, csv_name))
            continue
        # Check 1: claimed n vs actual max n
        actual_n = set()
        for (mn, col), r in cells.items():
            try:
                ni = int(r.get("n", "0") or "0")
                if ni > 0: actual_n.add(ni)
            except Exception: pass
        if isinstance(claimed_n, int) and actual_n and max(actual_n) != claimed_n:
            issues.append(("n_mismatch", md_path.name, header,
                           f"claimed={claimed_n} actual_max={max(actual_n)} actual_set={sorted(actual_n)}"))
        # Check 2: SETTING_LABELS coverage
        csv_method_nums = sorted({mn for mn, _ in cells.keys()})
        missing = [mn for mn in csv_method_nums if mn not in RENDER_METHOD_NUMS]
        if missing:
            issues.append(("methods_omitted", md_path.name, header,
                           f"CSV has method_nums={csv_method_nums}, doc renders only {sorted(RENDER_METHOD_NUMS)}, missing={missing}"))
        # Check 3: cell values match
        rendered = parse_md_cells(body)
        for row in rendered:
            for col_md, col_csv in [("Raw","Raw"),("PMI base","PMI base"),
                                    ("PMI self","PMI self"),("Neg base","Neg base"),
                                    ("Neg self","Neg self")]:
                csv_row = cells.get((row["method_num"], col_csv))
                if not csv_row:
                    if row[col_md] != "—":
                        issues.append(("missing_csv_cell_but_md_filled",
                                       md_path.name, header,
                                       f"mn={row['method_num']} col={col_csv} md={row[col_md]}"))
                    continue
                expected = fmt(csv_row.get("mean"), csv_row.get("se"))
                actual = row[col_md]
                # Treat both as either "—" or "X.XX ± Y.YY"
                if expected != actual:
                    issues.append(("cell_value_mismatch",
                                   md_path.name, header,
                                   f"mn={row['method_num']} col={col_csv} expected='{expected}' got='{actual}'"))
        # Check 4: long-csv version distribution
        long_path = METRICS / csv_name.replace("_table_cells.csv", "_table_long.csv")
        versions = load_long_versions(long_path)
        non_v7 = {k: v for k, v in versions.items() if k not in ("v7",)}
        # OK exception: rosch/persona base rows are v6 (small count). But if v6 dominates, that's bad.
        if versions and "v7" in versions and "v6" in versions and versions["v6"] > versions["v7"]:
            issues.append(("v6_majority", md_path.name, header,
                           f"versions={versions}"))
        if versions and "v7" not in versions and "v6" in versions:
            issues.append(("v6_only", md_path.name, header,
                           f"versions={versions}  (no v7 data)"))
        # Check 5: train column consistency
        for row in rendered:
            mn = row["method_num"]
            sN = f"s{mn}"
            tr = row["train"]
            ds_key = "membership" if "membership" in label_part \
                     else "persona" if "persona" in label_part \
                     else "ifeval" if "ifeval" in label_part \
                     else "humaneval" if "humaneval" in label_part \
                     else None
            if mn == 0:
                if tr != "(base model)":
                    issues.append(("base_train_label", md_path.name, header, f"got='{tr}'"))
                continue
            if "ep=" in tr:
                m = re.search(r"ep=(\d+)", tr)
                ep = int(m.group(1)) if m else None
                disk_ep = disk_eps.get((ds_key, model, sN))
                if disk_ep is None:
                    issues.append(("md_says_ep_but_no_dir",
                                   md_path.name, header,
                                   f"mn={mn} sN={sN} model={model} ds={ds_key} train='{tr}'"))
                elif disk_ep != ep:
                    issues.append(("epoch_mismatch",
                                   md_path.name, header,
                                   f"mn={mn} sN={sN} disk={disk_ep} md={ep}"))
            elif tr == "(base model)":
                pass
            elif tr.startswith("✓ (prior run)"):
                # Should mean: no model dir AND has eval data
                disk_ep = disk_eps.get((ds_key, model, sN))
                has_data = False
                for col in ("Raw","PMI base","PMI self","Neg base","Neg self"):
                    if row[col] != "—": has_data = True; break
                if disk_ep is not None:
                    issues.append(("prior_run_but_dir_exists",
                                   md_path.name, header,
                                   f"mn={mn} sN={sN} disk_ep={disk_ep}"))
                if not has_data:
                    issues.append(("prior_run_but_no_data",
                                   md_path.name, header, f"mn={mn} sN={sN}"))
            elif tr == "–":
                pass
            elif tr.startswith("⏳"):
                pass

# Print issues grouped
print(f"Total issues: {len(issues)}")
buckets = defaultdict(list)
for it in issues:
    buckets[it[0]].append(it[1:])
for cat in sorted(buckets):
    print(f"\n=== {cat}  (n={len(buckets[cat])}) ===")
    for x in buckets[cat][:30]:  # cap per category
        print("  -", " | ".join(map(str, x)))
    if len(buckets[cat]) > 30:
        print(f"  ...{len(buckets[cat])-30} more")
