#!/usr/bin/env python3
"""Inventory score CSV files in outputs/ and report completeness.

Parses filenames using the exact same logic as eval_by_claude.py and
run_eval_semi.sh to avoid any discrepancies.

Filename format (from eval_by_claude.py):
  scores_{self_prefix}{model_short}_{task}_{split}{v2_suffix}{metric_suffix}{eval_tc_suffix}{eval_lenorm_suffix}_{timestamp}.csv

Where:
  self_prefix:  "neg-" | "self-" | ""
  model_short:  for base models:     "v6-google_gemma-2-9b-it"
                for finetuned:       last path component with '--' -> '_'
                                     e.g. "v6-google_gemma-2-9b-it-delta0.15-epoch2_ambigqa-all_..."
  task:         the eval task name   e.g. "ambigqa-american", "hypernym-dogs", "ifeval-prompt_42"
  split:        "test" | "train"
  v2_suffix:    "_v2" for hypernym tasks, "" otherwise
  metric_suffix: "_log-odds" | "_log-probs"
  eval_tc_suffix: "_tc" if any typicality correction is used
  eval_lenorm_suffix: "_evallenorm" if length normalization is used
  timestamp:    "YYYYMMDD" (new) or "YYYYMMDD_HHMMSS" (old)
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

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


def build_expected_filename_prefix(self_prefix, model_short, task, split, metric_suffix, tc_suffix, lenorm_suffix):
    """Build the glob pattern that run_eval_semi.sh uses to check for existing files.

    This mirrors line 81 of run_eval_semi.sh:
      PATTERN="../outputs/scores_${SELF_PFX}${MODEL_SHORT}_${TASK}_test${V2_SUF}${METRIC_SUF}${TC_SUF}${LENORM_SUF}_*.csv"
    """
    v2_suffix = "_v2" if task.startswith("hypernym-") else ""
    return f"scores_{self_prefix}{model_short}_{task}_{split}{v2_suffix}{metric_suffix}{tc_suffix}{lenorm_suffix}_"


def model_dir_to_short(model_dir):
    """Convert a model directory path to the short name used in filenames.

    Mirrors eval_by_claude.py logic:
      - Base model "google/gemma-2-9b-it" -> "v6-google_gemma-2-9b-it"
      - Finetuned "../models/v6-google--gemma-2-9b-it-delta..." -> last component with '--' -> '_'
    """
    if '/' in model_dir and not model_dir.startswith('.'):
        return 'v6-' + model_dir.replace('/', '_')
    else:
        return model_dir.split('/')[-1].replace('--', '_')


def find_matching_files(all_files, prefix):
    """Find all files that match a given prefix (before the timestamp)."""
    return [f for f in all_files if f.startswith(prefix)]


def define_expected_evals():
    """Define all expected evaluation configurations from semi_supervised_eval_runs_neg.sh.

    Returns list of dicts, each describing one (model, eval_settings, domain) combination.
    """
    evals = []

    for base_model in ["gemma-2-9b-it", "gemma-2-2b"]:
        if "9b" in base_model:
            msuf = "_merged"
        else:
            msuf = ""

        for tc_suffix_train in ["--tc-neg", ""]:
            tc_label = "tc-neg" if tc_suffix_train == "--tc-neg" else "plain"

            # --- PLAUSIBLEQA: 6 variants ---
            mp = f"v6-google--{base_model}-delta0.15-epoch2--plausibleqa-all--d2g--random--alpha1.0{tc_suffix_train}"
            pqa_variants = [
                ("pref-only labelonly",           f"{mp}--full-completion--force-same-x--labelonly0.1{msuf}"),
                ("pref-only vallogodds labelonly", f"{mp}--full-completion--force-same-x--vallogodds--labelonly0.1{msuf}"),
                ("pref-only semi",                f"{mp}--full-completion--force-same-x--semi0.1{msuf}"),
                ("comb labelonly",                f"{mp}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1{msuf}"),
                ("comb semi",                     f"{mp}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1{msuf}"),
                ("sft semi",                      f"{mp}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1{msuf}"),
            ]
            for variant_name, model_dir in pqa_variants:
                evals.append({
                    "base_model": base_model,
                    "domain": "plausibleqa",
                    "train_tc": tc_label,
                    "variant": variant_name,
                    "model_dir": model_dir,
                    "eval_prefix": "neg-",
                    "metric": "_log-odds",
                    "eval_tc": "_tc",
                    "eval_lenorm": "",
                    "split": "test",
                })

            # --- AMBIGQA: 6 variants ---
            ma = f"v6-google--{base_model}-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0{tc_suffix_train}"
            aqa_variants = [
                ("pref-only labelonly",           f"{ma}--full-completion--force-same-x--labelonly0.1{msuf}"),
                ("pref-only vallogodds labelonly", f"{ma}--full-completion--force-same-x--vallogodds--labelonly0.1{msuf}"),
                ("pref-only semi",                f"{ma}--full-completion--force-same-x--semi0.1{msuf}"),
                ("comb labelonly",                f"{ma}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1{msuf}"),
                ("comb semi",                     f"{ma}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1{msuf}"),
                ("sft semi",                      f"{ma}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1{msuf}"),
            ]
            for variant_name, model_dir in aqa_variants:
                evals.append({
                    "base_model": base_model,
                    "domain": "ambigqa",
                    "train_tc": tc_label,
                    "variant": variant_name,
                    "model_dir": model_dir,
                    "eval_prefix": "neg-",
                    "metric": "_log-odds",
                    "eval_tc": "_tc",
                    "eval_lenorm": "",
                    "split": "test",
                })

            # --- HYPERNYM: 6 variants ---
            mh = f"v6-google--{base_model}-delta0.15-epoch2--hypernym-concat-bananas-to-dogs-double-all--d2g--random--alpha1.0{tc_suffix_train}"
            hyp_variants = [
                ("pref-only labelonly",           f"{mh}--full-completion--force-same-x--labelonly0.1{msuf}"),
                ("pref-only vallogodds labelonly", f"{mh}--full-completion--force-same-x--vallogodds--labelonly0.1{msuf}"),
                ("pref-only semi",                f"{mh}--full-completion--force-same-x--semi0.1{msuf}"),
                ("comb labelonly",                f"{mh}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1{msuf}"),
                ("comb semi",                     f"{mh}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1{msuf}"),
                ("sft semi",                      f"{mh}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1{msuf}"),
            ]
            for variant_name, model_dir in hyp_variants:
                evals.append({
                    "base_model": base_model,
                    "domain": "hypernym",
                    "train_tc": tc_label,
                    "variant": variant_name,
                    "model_dir": model_dir,
                    "eval_prefix": "neg-",
                    "metric": "_log-odds",
                    "eval_tc": "_tc",
                    "eval_lenorm": "",
                    "split": "test",
                })

            # --- IFEVAL: 5 variants (no pref-only semi) ---
            mi = f"v6-google--{base_model}-delta0.15-epoch2--ifeval-concat-all--d2g--random--alpha1.0{tc_suffix_train}"
            ife_variants = [
                ("pref-only labelonly",           f"{mi}--full-completion--force-same-x--labelonly0.1{msuf}"),
                ("pref-only vallogodds labelonly", f"{mi}--full-completion--force-same-x--vallogodds--labelonly0.1{msuf}"),
                ("comb labelonly",                f"{mi}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--labelonly0.1{msuf}"),
                ("comb semi",                     f"{mi}--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1{msuf}"),
                ("sft semi",                      f"{mi}--full-completion--pref0.0--nllv1.0--nllg1.0--force-same-x--semi0.1{msuf}"),
            ]
            for variant_name, model_dir in ife_variants:
                evals.append({
                    "base_model": base_model,
                    "domain": "ifeval",
                    "train_tc": tc_label,
                    "variant": variant_name,
                    "model_dir": model_dir,
                    "eval_prefix": "neg-",
                    "metric": "_log-odds",
                    "eval_tc": "_tc",
                    "eval_lenorm": "",
                    "split": "test",
                })

        # --- BASE MODEL (no finetuning) ---
        base_model_dir = f"google/{base_model}"
        for domain in ["plausibleqa", "ambigqa", "hypernym", "ifeval"]:
            evals.append({
                "base_model": base_model,
                "domain": domain,
                "train_tc": "base",
                "variant": "base (no finetuning)",
                "model_dir": base_model_dir,
                "eval_prefix": "neg-",
                "metric": "_log-odds",
                "eval_tc": "_tc",
                "eval_lenorm": "",
                "split": "test",
            })

    return evals


def run_inventory(filter_base_model=None, filter_eval_prefix=None):
    all_files = [f for f in os.listdir(OUTPUTS_DIR) if f.startswith("scores_") and f.endswith(".csv")]
    print(f"Total score CSV files in outputs/: {len(all_files)}")

    # Sanity check: make sure we can find some known files
    neg_files = [f for f in all_files if f.startswith("scores_neg-")]
    self_files = [f for f in all_files if f.startswith("scores_self-")]
    plain_files = [f for f in all_files if f.startswith("scores_v6-")]
    print(f"  neg- prefix: {len(neg_files)}")
    print(f"  self- prefix: {len(self_files)}")
    print(f"  plain (no prefix): {len(plain_files)}")
    assert len(neg_files) + len(self_files) + len(plain_files) == len(all_files), \
        f"File count mismatch: {len(neg_files)} + {len(self_files)} + {len(plain_files)} != {len(all_files)}. " \
        f"Some files don't match expected prefixes."

    evals = define_expected_evals()

    if filter_base_model:
        evals = [e for e in evals if e["base_model"] == filter_base_model]
    if filter_eval_prefix:
        evals = [e for e in evals if e["eval_prefix"] == filter_eval_prefix]

    print(f"\nChecking {len(evals)} expected evaluation configurations...\n")

    results = []
    claimed_files = set()

    for ev in evals:
        model_short = model_dir_to_short(ev["model_dir"])
        tasks = DOMAIN_TASKS[ev["domain"]]
        found = 0
        missing_tasks = []
        duplicate_tasks = []

        for task in tasks:
            prefix = build_expected_filename_prefix(
                ev["eval_prefix"], model_short, task, ev["split"],
                ev["metric"], ev["eval_tc"], ev["eval_lenorm"],
            )
            matches = find_matching_files(all_files, prefix)
            if len(matches) == 0:
                missing_tasks.append(task)
            else:
                found += 1
                claimed_files.update(matches)
                if len(matches) > 1:
                    duplicate_tasks.append((task, len(matches)))

        total = len(tasks)
        status = "COMPLETE" if found == total else f"MISSING {total - found}"
        results.append({
            **ev,
            "found": found,
            "total": total,
            "status": status,
            "missing_tasks": missing_tasks,
            "duplicate_tasks": duplicate_tasks,
        })

    # Print results grouped by base_model, then domain
    for base_model in sorted(set(r["base_model"] for r in results)):
        print(f"\n{'='*80}")
        print(f"BASE MODEL: {base_model}")
        print(f"{'='*80}")

        for domain in ["plausibleqa", "ambigqa", "hypernym", "ifeval"]:
            domain_results = [r for r in results if r["base_model"] == base_model and r["domain"] == domain]
            if not domain_results:
                continue

            total_tasks = len(DOMAIN_TASKS[domain])
            print(f"\n  --- {domain.upper()} ({total_tasks} tasks per variant) ---")
            print(f"  {'Train TC':<10} {'Variant':<35} {'Done':>5} {'Status':<15} {'Dupes'}")
            print(f"  {'-'*10} {'-'*35} {'-'*5} {'-'*15} {'-'*5}")

            for r in sorted(domain_results, key=lambda x: (x["train_tc"], x["variant"])):
                dupe_str = ""
                if r["duplicate_tasks"]:
                    dupe_str = f"{len(r['duplicate_tasks'])} dupes"
                status_str = r["status"]
                done_str = f"{r['found']}/{r['total']}"
                print(f"  {r['train_tc']:<10} {r['variant']:<35} {done_str:>5} {status_str:<15} {dupe_str}")

    # Check for unclaimed files (files we didn't expect)
    unclaimed = [f for f in all_files if f not in claimed_files]
    if filter_eval_prefix:
        unclaimed = [f for f in unclaimed if f.startswith(f"scores_{filter_eval_prefix}")]
    if filter_base_model:
        model_str = filter_base_model.replace("-", "_").replace("/", "_")
        unclaimed = [f for f in unclaimed if model_str in f]

    if unclaimed:
        print(f"\n{'='*80}")
        print(f"UNCLAIMED FILES ({len(unclaimed)} files not matched to any expected eval):")
        print(f"{'='*80}")
        for f in sorted(unclaimed)[:30]:
            print(f"  {f}")
        if len(unclaimed) > 30:
            print(f"  ... and {len(unclaimed) - 30} more")

    # Summary
    complete = sum(1 for r in results if r["status"] == "COMPLETE")
    incomplete = sum(1 for r in results if r["status"] != "COMPLETE")
    total_missing = sum(r["total"] - r["found"] for r in results)
    total_dupes = sum(len(r["duplicate_tasks"]) for r in results)

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Complete configurations: {complete}/{len(results)}")
    print(f"  Incomplete configurations: {incomplete}/{len(results)}")
    print(f"  Total missing task files: {total_missing}")
    print(f"  Total tasks with duplicates: {total_dupes}")

    return results


def run_sanity_checks():
    """Verify our filename logic matches actual files."""
    print("Running sanity checks...\n")
    all_files = [f for f in os.listdir(OUTPUTS_DIR) if f.startswith("scores_") and f.endswith(".csv")]

    # Check 1: model_dir_to_short works for known cases
    assert model_dir_to_short("google/gemma-2-9b-it") == "v6-google_gemma-2-9b-it", \
        f"Base model short name wrong: {model_dir_to_short('google/gemma-2-9b-it')}"
    assert model_dir_to_short("google/gemma-2-2b") == "v6-google_gemma-2-2b", \
        f"Base model short name wrong: {model_dir_to_short('google/gemma-2-2b')}"

    finetuned = "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--tc-neg--full-completion--force-same-x--labelonly0.1_merged"
    expected_short = "v6-google_gemma-2-9b-it-delta0.15-epoch2_ambigqa-all_d2g_random_alpha1.0_tc-neg_full-completion_force-same-x_labelonly0.1_merged"
    assert model_dir_to_short(finetuned) == expected_short, \
        f"Finetuned short name wrong:\n  got:      {model_dir_to_short(finetuned)}\n  expected: {expected_short}"

    # Check 2: can we find known base model files?
    prefix_base_aqa = build_expected_filename_prefix(
        "neg-", "v6-google_gemma-2-9b-it", "ambigqa-american", "test",
        "_log-odds", "_tc", "",
    )
    matches = find_matching_files(all_files, prefix_base_aqa)
    assert len(matches) > 0, f"Sanity check FAIL: no base 9b-it ambigqa-american files found with prefix '{prefix_base_aqa}'"
    print(f"  [OK] Base 9b-it ambigqa-american: found {len(matches)} file(s)")
    print(f"       Example: {matches[0]}")

    # Check 3: can we find known finetuned tc-neg files?
    prefix_ft = build_expected_filename_prefix(
        "neg-", expected_short, "ambigqa-american", "test",
        "_log-odds", "_tc", "",
    )
    matches_ft = find_matching_files(all_files, prefix_ft)
    assert len(matches_ft) > 0, f"Sanity check FAIL: no finetuned tc-neg ambigqa-american files found with prefix '{prefix_ft}'"
    print(f"  [OK] Finetuned tc-neg ambigqa-american: found {len(matches_ft)} file(s)")
    print(f"       Example: {matches_ft[0]}")

    # Check 4: verify hypernym files have _v2 suffix
    prefix_hyp = build_expected_filename_prefix(
        "neg-", "v6-google_gemma-2-9b-it", "hypernym-dogs", "test",
        "_log-odds", "_tc", "",
    )
    assert "_v2_" in prefix_hyp, f"Hypernym prefix missing _v2: {prefix_hyp}"
    matches_hyp = find_matching_files(all_files, prefix_hyp)
    assert len(matches_hyp) > 0, f"Sanity check FAIL: no base 9b-it hypernym-dogs files found with prefix '{prefix_hyp}'"
    print(f"  [OK] Base 9b-it hypernym-dogs (v2): found {len(matches_hyp)} file(s)")
    print(f"       Example: {matches_hyp[0]}")

    # Check 5: verify ifeval has NO _v2 suffix
    prefix_ife = build_expected_filename_prefix(
        "neg-", "v6-google_gemma-2-9b-it", "ifeval-prompt_1", "test",
        "_log-odds", "_tc", "",
    )
    assert "_v2" not in prefix_ife, f"IFEval prefix has unexpected _v2: {prefix_ife}"
    print(f"  [OK] IFEval prefix has no _v2 suffix")

    # Check 6: verify the prefix pattern matches expected format
    assert prefix_base_aqa == "scores_neg-v6-google_gemma-2-9b-it_ambigqa-american_test_log-odds_tc_", \
        f"Prefix format wrong: {prefix_base_aqa}"
    print(f"  [OK] Prefix format verified: {prefix_base_aqa}")

    # Check 7: spot-check a plain-trained model
    plain_model = "v6-google--gemma-2-9b-it-delta0.15-epoch2--ambigqa-all--d2g--random--alpha1.0--full-completion--force-same-x--labelonly0.1_merged"
    plain_short = model_dir_to_short(plain_model)
    prefix_plain = build_expected_filename_prefix(
        "neg-", plain_short, "ambigqa-american", "test", "_log-odds", "_tc", "",
    )
    matches_plain = find_matching_files(all_files, prefix_plain)
    if matches_plain:
        print(f"  [OK] Plain-trained ambigqa-american: found {len(matches_plain)} file(s)")
        print(f"       Example: {matches_plain[0]}")
    else:
        print(f"  [INFO] Plain-trained ambigqa-american: 0 files (model may not exist)")

    print(f"\n  All sanity checks passed.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inventory score CSV files")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to a specific base model (e.g. gemma-2-9b-it)")
    parser.add_argument("--eval-prefix", type=str, default=None,
                        help="Filter to a specific eval prefix (e.g. neg-)")
    parser.add_argument("--sanity-only", action="store_true",
                        help="Only run sanity checks, don't do full inventory")
    args = parser.parse_args()

    run_sanity_checks()

    if not args.sanity_only:
        run_inventory(filter_base_model=args.model, filter_eval_prefix=args.eval_prefix)
