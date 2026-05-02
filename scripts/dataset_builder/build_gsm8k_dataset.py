#!/usr/bin/env python3
"""
Build GSM8K rankalign dataset from solution JSONL files.

Creates two versions:
  1. full_response: complete model response including final answer
  2. truncated_response: response cut off before the final answer line

Reads from:
  - data/gsm8k/rlhflow_mistral_solutions.jsonl (downloaded from HF)
  - data/gsm8k/solutions.jsonl (our generated solutions, if any)

Outputs balanced CSVs to data/gsm8k/with_solutions/.

Usage:
    python scripts/build_gsm8k_dataset.py [--min-positive 10] [--min-negative 10] [--balance]
"""

import argparse
import csv
import json
import os
import re
import random
from collections import defaultdict


def truncate_before_answer(response: str) -> str:
    """Remove the final answer from a response, keeping only the reasoning.

    Tries to cut at various "final answer" patterns:
    1. "The answer is X" -> "The answer is"
    2. "#### X" -> everything before ####
    3. "\\boxed{X}" -> everything before \\boxed
    4. Last sentence containing a bare number -> remove it

    Returns the truncated response, or the original if no pattern found.
    """
    # Try "The answer is <number>" — cut after "The answer is"
    # Match various forms: "the answer is 42", "The answer is: 18 ки", etc.
    # Note: ки (U+043A U+0438) is a Cyrillic step marker from PRM datasets
    m = re.search(
        r'([Tt]he\s+(?:final\s+)?answer\s+is)\s*:?\s*\$?-?[\d,]+\.?\d*\$?\.?\s*(?:\u043a\u0438)?\s*$',
        response,
        re.MULTILINE
    )
    if m:
        return response[:m.end(1)].rstrip()

    # Try "#### <number>" — cut before ####
    m = re.search(r'\n?####\s*-?[\d,]+\.?\d*\s*$', response, re.MULTILINE)
    if m:
        return response[:m.start()].rstrip()

    # Try \boxed{X} at end
    m = re.search(r'\\boxed\{-?[\d,]+\.?\d*\}\s*\.?\s*$', response, re.MULTILINE)
    if m:
        return response[:m.start()].rstrip()

    # Fallback: remove the last line if it's just a number
    lines = response.rstrip().split('\n')
    if lines and re.match(r'^[\s$]*-?[\d,]+\.?\d*[\s.$]*$', lines[-1]):
        return '\n'.join(lines[:-1]).rstrip()

    # Can't find where to truncate — return original
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+",
                        default=["data/gsm8k/rlhflow_mistral_solutions.jsonl",
                                 "data/gsm8k/solutions.jsonl"],
                        help="Input JSONL files with solutions")
    parser.add_argument("--output-dir", type=str, default="data/gsm8k/with_solutions")
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--min-negative", type=int, default=10)
    parser.add_argument("--balance", action="store_true",
                        help="Downsample majority class per problem for 50/50")
    parser.add_argument("--max-per-side", type=int, default=50,
                        help="Max solutions per side per problem (to keep dataset manageable)")
    parser.add_argument("--train-count", type=int, default=None,
                        help="Number of train problems (default: auto from question_id)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all solutions
    by_question = defaultdict(list)
    for input_path in args.input:
        if not os.path.exists(input_path):
            print(f"Skipping {input_path} (not found)")
            continue
        print(f"Loading {input_path}...")
        with open(input_path) as f:
            for line in f:
                r = json.loads(line)
                by_question[r['question_id']].append(r)

    print(f"Loaded solutions for {len(by_question)} questions")

    # Classify train vs test by question_id
    train_questions = {qid: sols for qid, sols in by_question.items() if 'train' in qid}
    test_questions = {qid: sols for qid, sols in by_question.items() if 'test' in qid}
    print(f"Train questions: {len(train_questions)}, Test questions: {len(test_questions)}")

    # Filter for qualification
    def qualifies(sols):
        n_pass = sum(1 for s in sols if s['correct'])
        n_fail = sum(1 for s in sols if not s['correct'])
        return n_pass >= args.min_positive and n_fail >= args.min_negative

    train_qualified = {qid: sols for qid, sols in train_questions.items() if qualifies(sols)}
    test_qualified = {qid: sols for qid, sols in test_questions.items() if qualifies(sols)}
    print(f"Qualified train: {len(train_qualified)}/{len(train_questions)}")
    print(f"Qualified test: {len(test_qualified)}/{len(test_questions)}")

    # Build CSVs for both versions
    for version in ['full_response', 'truncated_response']:
        version_dir = os.path.join(args.output_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        def select_solutions(sols):
            pos = [s for s in sols if s['correct']]
            neg = [s for s in sols if not s['correct']]
            random.shuffle(pos)
            random.shuffle(neg)
            if args.balance:
                n = min(len(pos), len(neg), args.max_per_side)
                return pos[:n] + neg[:n]
            else:
                return pos[:args.max_per_side] + neg[:args.max_per_side]

        def get_response(sol):
            resp = sol['response']
            if version == 'truncated_response':
                return truncate_before_answer(resp)
            return resp

        # Write train.csv
        train_csv = os.path.join(version_dir, 'train.csv')
        train_rows = 0
        with open(train_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['question', 'answer', 'correct', 'strategy'])
            writer.writeheader()
            for qid in sorted(train_qualified):
                sols = select_solutions(train_qualified[qid])
                for sol in sols:
                    writer.writerow({
                        'question': sol['question'],
                        'answer': get_response(sol),
                        'correct': 'Yes' if sol['correct'] else 'No',
                        'strategy': 'gsm8k',
                    })
                    train_rows += 1
        print(f"[{version}] Wrote {train_rows} train rows")

        # Write per-problem test CSVs
        test_rows = 0
        for qid in sorted(test_qualified):
            slug = qid.replace('/', '_')
            test_csv = os.path.join(version_dir, f'{slug}.csv')
            sols = select_solutions(test_qualified[qid])
            with open(test_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['question', 'answer', 'correct', 'strategy'])
                writer.writeheader()
                for sol in sols:
                    writer.writerow({
                        'question': sol['question'],
                        'answer': get_response(sol),
                        'correct': 'Yes' if sol['correct'] else 'No',
                        'strategy': 'gsm8k',
                    })
                    test_rows += 1
        print(f"[{version}] Wrote {test_rows} test rows across {len(test_qualified)} problems")

    # Summary
    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}")
    print(f"Train: {len(train_qualified)} problems")
    print(f"Test:  {len(test_qualified)} problems")
    print(f"Versions: full_response, truncated_response")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
