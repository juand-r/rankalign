#!/usr/bin/env python3
"""Comprehensive inventory of score CSV files in outputs/.

Tracks ALL files, categorizes them, and reports completeness for every
expected (model, eval_mode, domain) configuration. Outputs both to
terminal and to a LaTeX PDF.

Eval TC types:
  neg:   prefix "neg-",  suffix "_tc"    (eval_by_claude.py --neg-typicality)
  self:  prefix "self-", suffix "_tc"    (eval_by_claude.py --self-typicality, current)
         prefix "self-", suffix "_evaltc" (eval_by_claude.py --self-typicality, older version)
  gpt2:  no prefix,      suffix "_evaltc" (eval.py --typicality-correction)

Filename format (eval_by_claude.py):
  scores_{prefix}{model_short}_{task}_{split}{v2_suf}{metric_suf}{tc_suf}{lenorm_suf}_{timestamp}.csv
"""

import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
LATEX_DIR = Path(__file__).resolve().parent.parent / "output-metrics"

# ---------------------------------------------------------------------------
# Task lists
# ---------------------------------------------------------------------------

TASKS_PQA = [
    "plausibleqa-nq_1114", "plausibleqa-nq_1324", "plausibleqa-nq_1328",
    "plausibleqa-nq_1369", "plausibleqa-nq_1394", "plausibleqa-nq_1438",
    "plausibleqa-nq_1663", "plausibleqa-nq_2031", "plausibleqa-nq_207",
    "plausibleqa-nq_2174", "plausibleqa-nq_2281", "plausibleqa-nq_2421",
    "plausibleqa-nq_2436", "plausibleqa-nq_2535", "plausibleqa-nq_2622",
    "plausibleqa-nq_2637", "plausibleqa-nq_2759", "plausibleqa-nq_2824",
    "plausibleqa-nq_2856", "plausibleqa-nq_2867", "plausibleqa-nq_2876",
    "plausibleqa-nq_3004", "plausibleqa-nq_3015", "plausibleqa-nq_3068",
    "plausibleqa-nq_3099", "plausibleqa-nq_3127", "plausibleqa-nq_3137",
    "plausibleqa-nq_316", "plausibleqa-nq_3276", "plausibleqa-nq_54",
    "plausibleqa-nq_562", "plausibleqa-nq_709", "plausibleqa-nq_958",
    "plausibleqa-trivia_1655", "plausibleqa-trivia_2984", "plausibleqa-trivia_3035",
    "plausibleqa-trivia_3043", "plausibleqa-trivia_3180", "plausibleqa-trivia_3245",
    "plausibleqa-trivia_3433", "plausibleqa-trivia_3492", "plausibleqa-trivia_3599",
    "plausibleqa-trivia_4009", "plausibleqa-trivia_4234", "plausibleqa-trivia_4489",
    "plausibleqa-trivia_4697", "plausibleqa-trivia_5003", "plausibleqa-trivia_560",
    "plausibleqa-trivia_5675", "plausibleqa-trivia_6317", "plausibleqa-trivia_6777",
    "plausibleqa-trivia_7272", "plausibleqa-trivia_7579", "plausibleqa-trivia_9589",
    "plausibleqa-webq_1000", "plausibleqa-webq_1046", "plausibleqa-webq_1086",
    "plausibleqa-webq_1097", "plausibleqa-webq_1163", "plausibleqa-webq_1187",
    "plausibleqa-webq_1278", "plausibleqa-webq_1307", "plausibleqa-webq_1310",
    "plausibleqa-webq_1338", "plausibleqa-webq_134", "plausibleqa-webq_1383",
    "plausibleqa-webq_141", "plausibleqa-webq_1421", "plausibleqa-webq_1442",
    "plausibleqa-webq_1476", "plausibleqa-webq_1498", "plausibleqa-webq_15",
    "plausibleqa-webq_1584", "plausibleqa-webq_1613", "plausibleqa-webq_1668",
    "plausibleqa-webq_1714", "plausibleqa-webq_1723", "plausibleqa-webq_1836",
    "plausibleqa-webq_1972", "plausibleqa-webq_212", "plausibleqa-webq_299",
    "plausibleqa-webq_342", "plausibleqa-webq_373", "plausibleqa-webq_428",
    "plausibleqa-webq_435", "plausibleqa-webq_520", "plausibleqa-webq_611",
    "plausibleqa-webq_650", "plausibleqa-webq_669", "plausibleqa-webq_672",
    "plausibleqa-webq_713", "plausibleqa-webq_744", "plausibleqa-webq_749",
    "plausibleqa-webq_760", "plausibleqa-webq_77", "plausibleqa-webq_803",
    "plausibleqa-webq_84", "plausibleqa-webq_88", "plausibleqa-webq_882",
    "plausibleqa-webq_898",
]

TASKS_AQA = [
    "ambigqa-american", "ambigqa-danube", "ambigqa-executed", "ambigqa-gives",
    "ambigqa-harry", "ambigqa-involved", "ambigqa-jack", "ambigqa-plays",
    "ambigqa-received", "ambigqa-sang", "ambigqa-soccer", "ambigqa-used",
    "ambigqa-voice", "ambigqa-winter", "ambigqa-won", "ambigqa-world",
    "ambigqa-year",
]

TASKS_HYP = [
    "hypernym-bananas", "hypernym-bazookas", "hypernym-cabinets", "hypernym-cars",
    "hypernym-chairs", "hypernym-crows", "hypernym-diapers", "hypernym-dogs",
    "hypernym-dolls", "hypernym-ducklings", "hypernym-elephants", "hypernym-guns",
    "hypernym-hammers", "hypernym-helmets", "hypernym-jackets", "hypernym-kayaks",
    "hypernym-kites", "hypernym-mirrors",
]

TASKS_IFE = [
    f"ifeval-prompt_{i}" for i in [
        1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
        29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
        54,56,57,58,59,61,63,64,65,66,67,68,70,72,73,74,75,76,77,78,79,80,82,83,
        84,85,87,88,89,90,91,92,93,94,95,96,97,98,99,100,102,103,104,105,106,107,
        108,109,
    ]
]

DOMAIN_TASKS = {
    "plausibleqa": TASKS_PQA,
    "ambigqa": TASKS_AQA,
    "hypernym": TASKS_HYP,
    "ifeval": TASKS_IFE,
}

# ---------------------------------------------------------------------------
# Eval mode definitions
# ---------------------------------------------------------------------------

EVAL_MODES = {
    # Three kinds of typicality correction used at eval time:
    #
    # neg:  eval_by_claude.py --neg-typicality   -> prefix "neg-",  suffix "_tc"
    # self: eval_by_claude.py --self-typicality  -> prefix "self-", suffix "_tc" (current)
    #                                               prefix "self-", suffix "_evaltc" (older eval_by_claude.py)
    # gpt2: eval.py --typicality-correction      -> no prefix,      suffix "_evaltc"
    "neg": {
        "label": "neg",
        "patterns": [("neg-", "_tc")],
    },
    "self": {
        "label": "self",
        "patterns": [("self-", "_tc"), ("self-", "_evaltc")],
    },
    "gpt2": {
        "label": "gpt2",
        "patterns": [("", "_evaltc")],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_dir_to_short(model_dir):
    """Convert model directory path to the short name used in filenames."""
    if '/' in model_dir and not model_dir.startswith('.'):
        return 'v6-' + model_dir.replace('/', '_')
    else:
        return model_dir.split('/')[-1].replace('--', '_')


def build_prefix(eval_prefix, model_short, task, split, metric, tc_suffix, lenorm):
    """Build the filename prefix that run_eval_semi.sh uses for skip logic."""
    v2 = "_v2" if task.startswith("hypernym-") else ""
    return f"scores_{eval_prefix}{model_short}_{task}_{split}{v2}{metric}{tc_suffix}{lenorm}_"


def find_matching(all_files, prefix):
    return [f for f in all_files if f.startswith(prefix)]


# ---------------------------------------------------------------------------
# Model / eval configuration registry
# ---------------------------------------------------------------------------

def _finetuned_variants(base_model, task_key, tc_suffix_train, msuf):
    """Return the 6 (or 5 for ifeval) training variants for a given domain."""
    tc = tc_suffix_train
    prefix_map = {
        "plausibleqa": "plausibleqa-all",
        "ambigqa": "ambigqa-all",
        "hypernym": "hypernym-concat-bananas-to-dogs-double-all",
        "ifeval": "ifeval-concat-all",
    }
    task_str = prefix_map[task_key]
    mp = f"v6-google--{base_model}-delta0.15-epoch2--{task_str}--d2g--random--alpha1.0{tc}"

    variants = [
        ("pref-only lo",      f"{mp}--full-completion--force-same-x--labelonly0.1{msuf}"),
        ("pref-only vlo lo",  f"{mp}--full-completion--force-same-x--vallogodds--labelonly0.1{msuf}"),
        ("pref-only semi",    f"{mp}--full-completion--force-same-x--semi0.1{msuf}"),
        ("comb lo",           f"{mp}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1{msuf}"),
        ("comb semi",         f"{mp}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1{msuf}"),
        ("sft semi",          f"{mp}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1{msuf}"),
        ("sft lo",            f"{mp}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--labelonly0.1{msuf}"),
    ]
    if task_key == "ifeval":
        variants = [v for v in variants if v[0] != "pref-only semi"]
    return variants


def _v2g_baseline(base_model, task_key, msuf):
    """Return the V2G baseline (paper RankAlign) model dir."""
    prefix_map = {
        "plausibleqa": "plausibleqa-all",
        "ambigqa": "ambigqa-all",
        "hypernym": "hypernym-concat-bananas-to-dogs-double-all",
        "ifeval": "ifeval-concat-all",
    }
    task_str = prefix_map[task_key]
    return f"v6-google--{base_model}-delta0.15-epoch2--{task_str}--d2g--random--alpha1.0--full-completion{msuf}"


def define_all_expected_evals():
    """Build the full registry of expected evaluations.

    Returns list of dicts with keys:
      base_model, domain, train_tc, variant, model_dir, eval_mode, metric, lenorm, split
    """
    evals = []

    for base_model in ["gemma-2-9b-it", "gemma-2-2b", "gemma-2-2b-it"]:
        msuf = "_merged" if "9b" in base_model else ""

        if base_model == "gemma-2-2b":
            domains = ["plausibleqa", "ambigqa", "hypernym"]
        elif base_model == "gemma-2-2b-it":
            domains = ["plausibleqa", "ambigqa", "hypernym"]
        else:
            domains = ["plausibleqa", "ambigqa", "hypernym", "ifeval"]

        tc_train_options = [("", "plain"), ("--tc-neg", "tc-neg"), ("--tc-self", "tc-self")]

        for eval_mode_key, eval_mode in EVAL_MODES.items():
            # --- Base model (no finetuning) ---
            for domain in domains:
                evals.append({
                    "base_model": base_model,
                    "domain": domain,
                    "train_tc": "base",
                    "variant": "base",
                    "model_dir": f"google/{base_model}",
                    "eval_mode": eval_mode_key,
                    "filename_patterns": eval_mode["patterns"],
                    "metric": "_log-odds",
                    "lenorm": "",
                    "split": "test",
                })

            # --- Finetuned: plain + tc-neg + tc-self trained ---
            for tc_train_suffix, tc_label in tc_train_options:
                for domain in domains:
                    for variant_name, model_dir in _finetuned_variants(
                        base_model, domain, tc_train_suffix, msuf
                    ):
                        evals.append({
                            "base_model": base_model,
                            "domain": domain,
                            "train_tc": tc_label,
                            "variant": variant_name,
                            "model_dir": model_dir,
                            "eval_mode": eval_mode_key,
                            "filename_patterns": eval_mode["patterns"],
                            "metric": "_log-odds",
                            "lenorm": "",
                            "split": "test",
                        })

            # --- tc-neg lenorm trained (2b only) ---
            if base_model == "gemma-2-2b":
                for domain in domains:
                    for variant_name, model_dir in _finetuned_variants(
                        base_model, domain, "--tc-neg--lenorm", msuf
                    ):
                        evals.append({
                            "base_model": base_model,
                            "domain": domain,
                            "train_tc": "tc-neg-lenorm",
                            "variant": variant_name,
                            "model_dir": model_dir,
                            "eval_mode": eval_mode_key,
                            "filename_patterns": eval_mode["patterns"],
                            "metric": "_log-odds",
                            "lenorm": "",
                            "split": "test",
                        })

            # --- V2G baselines (not for 2b-it) ---
            if base_model != "gemma-2-2b-it":
                for domain in domains:
                    evals.append({
                        "base_model": base_model,
                        "domain": domain,
                        "train_tc": "v2g",
                        "variant": "V2G baseline",
                        "model_dir": _v2g_baseline(base_model, domain, msuf),
                        "eval_mode": eval_mode_key,
                        "filename_patterns": eval_mode["patterns"],
                        "metric": "_log-odds",
                        "lenorm": "",
                        "split": "test",
                    })

    return evals


# ---------------------------------------------------------------------------
# Inventory engine
# ---------------------------------------------------------------------------

def run_inventory(all_files, evals):
    """Check each expected eval config against actual files.

    Returns (results, claimed_files).
    """
    results = []
    claimed = set()

    for ev in evals:
        model_short = model_dir_to_short(ev["model_dir"])
        tasks = DOMAIN_TASKS[ev["domain"]]
        found = 0
        missing_tasks = []
        dupes = []

        for task in tasks:
            matches = []
            for eval_prefix, tc_suffix in ev["filename_patterns"]:
                pfx = build_prefix(
                    eval_prefix, model_short, task, ev["split"],
                    ev["metric"], tc_suffix, ev["lenorm"],
                )
                matches.extend(find_matching(all_files, pfx))
            if matches:
                found += 1
                claimed.update(set(matches))
                if len(matches) > 1:
                    dupes.append((task, len(set(matches))))
            else:
                missing_tasks.append(task)

        total = len(tasks)
        results.append({
            **ev,
            "found": found,
            "total": total,
            "status": "COMPLETE" if found == total else f"MISSING {total - found}",
            "missing_tasks": missing_tasks,
            "dupes": dupes,
        })

    return results, claimed


# ---------------------------------------------------------------------------
# Classify unclaimed files
# ---------------------------------------------------------------------------

def classify_unclaimed(unclaimed):
    """Group unclaimed files into recognizable categories."""
    categories = defaultdict(list)
    for f in unclaimed:
        if "gemma-2-2b-it" in f:
            categories["gemma-2-2b-it (not tracked)"].append(f)
        elif "_evallenorm_" in f:
            categories["lenorm evals"].append(f)
        elif re.search(r'hypernym-(bananas|bazookas|cabinets|cars|chairs|crows|diapers|dogs|dolls|ducklings|elephants|guns|hammers|helmets|jackets|kayaks|kites|mirrors)-all_', f):
            categories["per-category hypernym (old)"].append(f)
        elif "v5-" in f:
            categories["v5 models (legacy)"].append(f)
        elif "_train_" in f:
            categories["train split evals"].append(f)
        elif "_log-probs_" in f:
            categories["log-probs metric"].append(f)
        elif "lenorm_full-completion" in f or ("tc-" in f and "lenorm" in f.split("full-completion")[0] if "full-completion" in f else False):
            categories["tc+lenorm trained models"].append(f)
        elif "epoch1" in f or "epoch0" in f:
            categories["intermediate epoch (0 or 1)"].append(f)
        elif "gemma-2-2b" in f and "gemma-2-2b-it" not in f and "ifeval" in f:
            categories["2b ifeval (excluded)"].append(f)
        else:
            categories["other"].append(f)
    return dict(categories)


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_results(results, claimed, all_files):
    total_files = len(all_files)
    claimed_count = len(claimed)
    unclaimed = [f for f in all_files if f not in claimed]

    print(f"\nTotal score CSV files: {total_files}")
    print(f"Claimed by expected configs: {claimed_count}")
    print(f"Unclaimed: {len(unclaimed)}")

    for base_model in ["gemma-2-2b", "gemma-2-9b-it"]:
        bm_results = [r for r in results if r["base_model"] == base_model]
        if not bm_results:
            continue

        print(f"\n{'='*100}")
        print(f"BASE MODEL: {base_model}")
        print(f"{'='*100}")

        for domain in ["plausibleqa", "ambigqa", "hypernym", "ifeval"]:
            dr = [r for r in bm_results if r["domain"] == domain]
            if not dr:
                continue

            n_tasks = len(DOMAIN_TASKS[domain])
            print(f"\n  --- {domain.upper()} ({n_tasks} tasks) ---")
            print(f"  {'Train':<8} {'Variant':<18} ", end="")
            for em in EVAL_MODES:
                print(f"  {em:<14}", end="")
            print()
            print(f"  {'-'*8} {'-'*18} ", end="")
            for _ in EVAL_MODES:
                print(f"  {'-'*14}", end="")
            print()

            seen_variants = []
            for r in dr:
                key = (r["train_tc"], r["variant"])
                if key not in seen_variants:
                    seen_variants.append(key)

            for train_tc, variant in sorted(set(seen_variants)):
                print(f"  {train_tc:<8} {variant:<18} ", end="")
                for em in EVAL_MODES:
                    matching = [r for r in dr if r["train_tc"] == train_tc
                                and r["variant"] == variant and r["eval_mode"] == em]
                    if matching:
                        r = matching[0]
                        cell = f"{r['found']}/{r['total']}"
                        if r["status"] == "COMPLETE":
                            cell += " ok"
                        print(f"  {cell:<14}", end="")
                    else:
                        print(f"  {'--':<14}", end="")
                print()

    # Unclaimed summary
    if unclaimed:
        cats = classify_unclaimed(unclaimed)
        print(f"\n{'='*100}")
        print(f"UNCLAIMED FILES: {len(unclaimed)} total")
        print(f"{'='*100}")
        for cat, files in sorted(cats.items(), key=lambda x: -len(x[1])):
            print(f"  {cat}: {len(files)}")
            for f in sorted(files)[:3]:
                print(f"    {f}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")

    # Summary
    complete = sum(1 for r in results if r["status"] == "COMPLETE")
    total_configs = len(results)
    total_missing = sum(r["total"] - r["found"] for r in results)
    nonzero = sum(1 for r in results if r["found"] > 0)

    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"  Configs checked:   {total_configs}")
    print(f"  Complete:          {complete}")
    print(f"  Partially done:    {nonzero - complete}")
    print(f"  Not started:       {total_configs - nonzero}")
    print(f"  Total missing:     {total_missing} task files")
    print(f"  Files claimed:     {claimed_count}/{total_files}")
    print(f"  Files unclaimed:   {len(unclaimed)}/{total_files}")


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def _esc(s):
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_latex(results, claimed, all_files, outpath):
    unclaimed = [f for f in all_files if f not in claimed]
    complete = sum(1 for r in results if r["status"] == "COMPLETE")

    lines = []
    lines.append(r"\documentclass[10pt,landscape]{article}")
    lines.append(r"\usepackage[margin=0.5in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{colortbl}")
    lines.append(r"\definecolor{done}{HTML}{C8E6C9}")
    lines.append(r"\definecolor{partial}{HTML}{FFF9C4}")
    lines.append(r"\definecolor{missing}{HTML}{FFCDD2}")
    lines.append(r"\begin{document}")
    lines.append(r"\section*{Score File Inventory}")
    lines.append(f"Total files: {len(all_files)}, "
                 f"claimed: {len(claimed)}, "
                 f"unclaimed: {len(unclaimed)}, "
                 f"configs complete: {complete}/{len(results)}")
    lines.append("")

    for base_model in ["gemma-2-2b", "gemma-2-9b-it"]:
        bm_results = [r for r in results if r["base_model"] == base_model]
        if not bm_results:
            continue

        lines.append(f"\\subsection*{{{_esc(base_model)}}}")

        for domain in ["plausibleqa", "ambigqa", "hypernym", "ifeval"]:
            dr = [r for r in bm_results if r["domain"] == domain]
            if not dr:
                continue

            n_tasks = len(DOMAIN_TASKS[domain])
            ncols = 2 + len(EVAL_MODES)
            col_spec = "ll" + "c" * len(EVAL_MODES)

            lines.append(f"\\paragraph{{{_esc(domain)} ({n_tasks} tasks)}}")
            lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            lines.append(r"\toprule")
            header = "Train & Variant"
            for em in EVAL_MODES:
                header += f" & {_esc(em)}"
            header += r" \\"
            lines.append(header)
            lines.append(r"\midrule")

            seen = []
            for r in dr:
                key = (r["train_tc"], r["variant"])
                if key not in seen:
                    seen.append(key)

            for train_tc, variant in sorted(set(seen)):
                row = f"{_esc(train_tc)} & {_esc(variant)}"
                for em in EVAL_MODES:
                    matching = [r for r in dr if r["train_tc"] == train_tc
                                and r["variant"] == variant and r["eval_mode"] == em]
                    if matching:
                        r = matching[0]
                        frac = f"{r['found']}/{r['total']}"
                        if r["found"] == r["total"]:
                            row += f" & \\cellcolor{{done}}{frac}"
                        elif r["found"] > 0:
                            row += f" & \\cellcolor{{partial}}{frac}"
                        else:
                            row += f" & \\cellcolor{{missing}}{frac}"
                    else:
                        row += " & --"
                row += r" \\"
                lines.append(row)

            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\vspace{1em}")
            lines.append("")

    # Unclaimed summary
    if unclaimed:
        cats = classify_unclaimed(unclaimed)
        lines.append(r"\subsection*{Unclaimed files}")
        lines.append(r"\begin{tabular}{lr}")
        lines.append(r"\toprule")
        lines.append(r"Category & Count \\")
        lines.append(r"\midrule")
        for cat, files in sorted(cats.items(), key=lambda x: -len(x[1])):
            lines.append(f"{_esc(cat)} & {len(files)} \\\\")
        lines.append(f"\\midrule")
        lines.append(f"Total & {len(unclaimed)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{document}")

    outpath.write_text("\n".join(lines))
    return outpath


def compile_latex(tex_path):
    pdf_dir = tex_path.parent
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(pdf_dir), str(tex_path)],
            capture_output=True, text=True, timeout=30,
        )
        pdf_path = tex_path.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"  PDF written to: {pdf_path}")
        else:
            print(f"  pdflatex ran but no PDF produced. Check {tex_path.with_suffix('.log')}")
    except FileNotFoundError:
        print("  pdflatex not found -- skipping PDF compilation. LaTeX source saved.")
    except subprocess.TimeoutExpired:
        print("  pdflatex timed out.")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(all_files):
    print("Running sanity checks...\n")

    assert model_dir_to_short("google/gemma-2-9b-it") == "v6-google_gemma-2-9b-it"
    assert model_dir_to_short("google/gemma-2-2b") == "v6-google_gemma-2-2b"

    ft = "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--tc-neg--full-completion--force-same-x--labelonly0.1_merged"
    expected = "v6-google_gemma-2-9b-it-delta0.15-epoch2_ambigqa-all_d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_labelonly0.1_merged"
    assert model_dir_to_short(ft) == expected, f"Got: {model_dir_to_short(ft)}"

    for prefix, task, tc_suf in [
        ("neg-", "ambigqa-american", "_tc"),
        ("self-", "hypernym-dogs", "_tc"),
        ("", "hypernym-dogs", "_evaltc"),
    ]:
        pfx = build_prefix(prefix, "v6-google_gemma-2-9b-it", task, "test", "_log-odds", tc_suf, "")
        matches = find_matching(all_files, pfx)
        status = f"found {len(matches)}" if matches else "NONE"
        print(f"  [{status:>10}] {pfx}*")

    print(f"\n  Sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive score file inventory")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to base model (e.g. gemma-2-9b-it)")
    parser.add_argument("--eval-mode", type=str, default=None,
                        choices=list(EVAL_MODES.keys()),
                        help="Filter to eval mode")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--no-latex", action="store_true", help="Skip LaTeX output")
    args = parser.parse_args()

    all_files = sorted(f for f in os.listdir(OUTPUTS_DIR)
                       if f.startswith("scores_") and f.endswith(".csv"))

    run_sanity_checks(all_files)
    if args.sanity_only:
        return

    evals = define_all_expected_evals()
    if args.model:
        evals = [e for e in evals if e["base_model"] == args.model]
    if args.eval_mode:
        evals = [e for e in evals if e["eval_mode"] == args.eval_mode]

    results, claimed = run_inventory(all_files, evals)
    print_results(results, claimed, all_files)

    if not args.no_latex:
        tex_path = LATEX_DIR / "inventory.tex"
        write_latex(results, claimed, all_files, tex_path)
        print(f"\n  LaTeX written to: {tex_path}")
        compile_latex(tex_path)


if __name__ == "__main__":
    main()
