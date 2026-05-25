#!/usr/bin/env python3
"""Audit v7 model + scores coverage for s1-s13 across all 3 models and 4 datasets.

Setting signatures (substring tests, looking for the EARLIEST adapter epoch on disk):

  s1  SFT labelonly 10%        : --pref0.0--nllv1.0--nllg1.0--labelonly0.1--fix1  (NOT cft)
  s2  RankAlign                : --full-completion--semi0.1--fix1, NO --tc-, NO --fsx, NO --vallogodds, NO --nllv/--nllg
  s3  New + fsx [-TC]          : --force-same-x AND --ppd AND --nllv1.0 AND --vallogodds AND NOT --tc-
  s4  New + PMI + fsx          : --tc-self AND --force-same-x AND --ppd AND --nllv1.0 AND --vallogodds
  s5  RA + PMI + fsx [-NLL]    : --tc-self AND --force-same-x AND --ppd AND NOT --nllv1.0 AND NOT --vallogodds
  s6  RA + PMI [+TC]           : --tc-self AND NOT --force-same-x AND NOT --nllv1.0 AND NOT --vallogodds
  s7  New + NegTC + fsx        : --tc-neg AND --force-same-x AND --ppd AND --nllv1.0 AND --vallogodds
  s11 New + PMI [-fsx]         : --tc-self AND NOT --force-same-x AND --nllv1.0 AND --vallogodds
  s12 New + NegTC [-fsx]       : --tc-neg AND NOT --force-same-x AND --nllv1.0 AND --vallogodds
  s13 SFT + CFT                : --cft

Eval coverage per cell (for tables):
  - TC-self trained (s4/s5/s6/s11): scores_basetyp-* and scores_self-* (PMI base + PMI self)
  - TC-neg trained  (s7/s12)      : scores_basetypneg-* and scores_neg-*  (Neg base + Neg self)
  - No-TC trained   (s1/s2/s3/s13): all four prefixes (PMI base, PMI self, Neg base, Neg self)
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

MODELS_DIR = Path("/datastor2/jdr/rankalign/models2")
SCORES_DIR = Path("/datastor1/jdr/gv-gap/rankalign/outputs")

MODELS = ["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it", "gemma-4-31B-it"]
DATASETS = {
    "membership-sans-rosch-v0": "membership",
    "persona-v1": "persona",
    "ifeval-concat": "ifeval",
    "humaneval-v2.1correct-upper": "humaneval",
}

SETTINGS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s11", "s12", "s13"]


def classify_setting(d: str) -> str | None:
    """Return setting tag from a model dirname signature, or None if unknown."""
    if "--cft--" in d:
        return "s13"
    if "--labelonly0.1--" in d and "--pref0.0--" in d and "--cft--" not in d:
        return "s1"
    has_tc_self = "--tc-self--" in d
    has_tc_neg = "--tc-neg--" in d
    has_fsx = "--force-same-x--" in d
    has_ppd = "--ppd--" in d
    has_nllv = "--nllv1.0--" in d
    has_vlo = "--vallogodds--" in d
    if not has_tc_self and not has_tc_neg:
        # no TC variant
        if not has_fsx and not has_nllv and not has_vlo:
            return "s2"
        if has_fsx and has_ppd and has_nllv and has_vlo:
            return "s3"
        return None
    if has_tc_self:
        if has_fsx and has_ppd and has_nllv and has_vlo:
            return "s4"
        if has_fsx and has_ppd and not has_nllv and not has_vlo:
            return "s5"
        if not has_fsx and not has_nllv and not has_vlo:
            return "s6"
        if not has_fsx and has_nllv and has_vlo:
            return "s11"
    if has_tc_neg:
        if has_fsx and has_ppd and has_nllv and has_vlo:
            return "s7"
        if not has_fsx and has_nllv and has_vlo:
            return "s12"
    return None


def model_from_dirname(d: str):
    for m in MODELS:
        if f"v7-google--{m}-delta" in d:
            return m
    return None


def dataset_from_dirname(d: str):
    for full, short in DATASETS.items():
        if f"--{full}-all--" in d:
            return full, short
    return None, None


def main():
    # Map (model, dataset_full, setting) -> set of epochs found on disk
    have_train = defaultdict(set)
    unknowns = []
    if MODELS_DIR.exists():
        for d in os.listdir(MODELS_DIR):
            if not d.startswith("v7-google--"):
                continue
            if d.endswith("_merged"):
                continue
            model = model_from_dirname(d)
            ds_full, ds_short = dataset_from_dirname(d)
            setting = classify_setting(d)
            m_ep = re.search(r"-epoch(\d+)--", d)
            ep = int(m_ep.group(1)) if m_ep else -1
            if model is None or ds_full is None or setting is None:
                unknowns.append((d, model, ds_full, setting))
                continue
            if ep >= 1:  # epoch1+ adapters exist
                have_train[(model, ds_short, setting)].add(ep)

    # Print train coverage matrix
    print("# v7 train coverage matrix (epoch>=1 saved)\n")
    print("Cells: epochs found, e.g. {1,2}=both saved, {1}=only epoch1, ∅=missing.\n")
    for ds_full, ds_short in DATASETS.items():
        print(f"## {ds_full} ({ds_short})\n")
        if ds_short == "humaneval":
            row_models = ["gemma-4-31B-it"]
        else:
            row_models = ["gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b-it"]
        print("| setting | " + " | ".join(row_models) + " |")
        print("|---|" + "|".join(["---"] * len(row_models)) + "|")
        for s in SETTINGS:
            row = [s]
            for m in row_models:
                eps = have_train.get((m, ds_short, s), set())
                row.append("{" + ",".join(str(e) for e in sorted(eps)) + "}" if eps else "∅")
            print("| " + " | ".join(row) + " |")
        print()

    if unknowns:
        print("## Unclassified model dirs (manual review):\n")
        for d, model, ds_full, setting in unknowns[:20]:
            print(f"  {d}  (model={model}, ds={ds_full}, setting={setting})")
        if len(unknowns) > 20:
            print(f"  ... and {len(unknowns)-20} more")


if __name__ == "__main__":
    main()
