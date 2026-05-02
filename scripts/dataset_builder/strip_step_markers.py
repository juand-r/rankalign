#!/usr/bin/env python3
"""
Strip PRM step markers (ки, Cyrillic U+043A U+0438) from GSM8K solution JSONL files.

These markers are artifacts from the RLHFlow PRM training pipeline and are not
meaningful content. This script reads a solutions JSONL, strips the markers from
the 'response' field, and writes to a new file.

Usage:
    python scripts/dataset_builder/strip_step_markers.py \
        data/gsm8k/rlhflow_mistral_solutions.jsonl \
        data/gsm8k/rlhflow_mistral_solutions_clean.jsonl

    # Preview what changes look like (dry run, 5 examples)
    python scripts/dataset_builder/strip_step_markers.py \
        data/gsm8k/rlhflow_mistral_solutions.jsonl --preview
"""

import argparse
import json
import re
import sys


# ки (U+043A U+0438) optionally preceded/followed by whitespace
STEP_MARKER_RE = re.compile(r'\s*\u043a\u0438\s*')


def strip_markers(text: str) -> str:
    """Remove ки step markers, collapsing surrounding whitespace to a single space."""
    # Replace " ки\n" with "\n" (preserve line breaks)
    cleaned = re.sub(r' ?\u043a\u0438\n', '\n', text)
    # Replace remaining " ки" (e.g., at end of string) with empty
    cleaned = re.sub(r' ?\u043a\u0438', '', cleaned)
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser(description="Strip ки step markers from GSM8K solutions")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", nargs="?", help="Output JSONL file (omit for --preview)")
    parser.add_argument("--preview", action="store_true",
                        help="Show before/after for a few examples, don't write")
    parser.add_argument("--n-preview", type=int, default=5)
    args = parser.parse_args()

    if not args.preview and not args.output:
        parser.error("Provide an output path, or use --preview")

    if args.preview:
        with open(args.input) as f:
            shown = 0
            for line in f:
                r = json.loads(line)
                resp = r['response']
                cleaned = strip_markers(resp)
                if resp != cleaned:
                    print(f"--- [{r['question_id']}] ---")
                    print(f"BEFORE (last 120): ...{resp[-120:]}")
                    print(f"AFTER  (last 120): ...{cleaned[-120:]}")
                    print()
                    shown += 1
                    if shown >= args.n_preview:
                        break
        return

    count = 0
    changed = 0
    with open(args.input) as fin, open(args.output, 'w') as fout:
        for line in fin:
            r = json.loads(line)
            original = r['response']
            r['response'] = strip_markers(original)
            if 'raw_response' in r:
                r['raw_response'] = strip_markers(r['raw_response'])
            if r['response'] != original:
                changed += 1
            fout.write(json.dumps(r) + '\n')
            count += 1

    print(f"Processed {count} solutions, {changed} had markers stripped")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
