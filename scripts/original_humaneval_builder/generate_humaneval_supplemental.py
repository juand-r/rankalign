"""
Supplemental generation for HumanEval problems that need more positive or negative solutions.

Strategy (per user guidance):
- Problems with <10 negatives: GPT-4o-mini at high temperature (weaker model makes more mistakes)
- Problems with <10 positives: GPT-4o at low temperature (stronger model, deterministic)

Appends to the existing solutions.jsonl file.

Usage:
    python scripts/generate_humaneval_supplemental.py [--output data/humaneval/solutions.jsonl]
"""

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from human_eval.data import read_problems
from openai import OpenAI

# Reuse core functions from generate_humaneval_solutions.py
sys.path.insert(0, str(Path(__file__).parent))
from generate_humaneval_solutions import generate_solution, clean_solution, test_solution


THRESHOLD = 10  # Need at least this many pass AND fail per problem


def load_existing_counts(path):
    """Load existing solutions and count pass/fail per task."""
    per_task = defaultdict(lambda: {'pass': 0, 'fail': 0})
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                r = json.loads(line)
                if r['passed']:
                    per_task[r['task_id']]['pass'] += 1
                else:
                    per_task[r['task_id']]['fail'] += 1
    return per_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/humaneval/solutions.jsonl")
    parser.add_argument("--neg-samples", type=int, default=40,
                        help="Extra samples for problems needing negatives")
    parser.add_argument("--pos-samples", type=int, default=20,
                        help="Extra samples for problems needing positives")
    parser.add_argument("--neg-model", type=str, default="gpt-4o-mini",
                        help="Weaker model to generate more failures")
    parser.add_argument("--pos-model", type=str, default="gpt-4o",
                        help="Strong model for generating passes")
    args = parser.parse_args()

    client = OpenAI()
    problems = read_problems()
    per_task = load_existing_counts(args.output)

    need_neg = sorted([tid for tid in problems if per_task[tid]['fail'] < THRESHOLD])
    need_pos = sorted([tid for tid in problems if per_task[tid]['pass'] < THRESHOLD])

    print(f"Problems needing more negatives (<{THRESHOLD} fail): {len(need_neg)}")
    print(f"Problems needing more positives (<{THRESHOLD} pass): {len(need_pos)}")

    # High-temp schedule for generating failures
    neg_temps = [1.2] * 15 + [1.4] * 15 + [1.6] * 10
    # Low-temp schedule for generating passes
    pos_temps = [0.0] * 10 + [0.2] * 10

    total_generated = 0
    total_new_neg = 0
    total_new_pos = 0

    with open(args.output, 'a') as fout:
        # Phase 1: Generate failures with weaker model
        print(f"\n=== Phase 1: Generating negatives with {args.neg_model} ===")
        for i, task_id in enumerate(need_neg):
            problem = problems[task_id]
            current_fail = per_task[task_id]['fail']
            n_samples = args.neg_samples

            print(f"[{i+1}/{len(need_neg)}] {task_id} (have {current_fail} fail, {per_task[task_id]['pass']} pass)")

            n_pass = 0
            n_fail = 0
            for j in range(n_samples):
                temp = neg_temps[j % len(neg_temps)]
                try:
                    raw = generate_solution(client, problem['prompt'],
                                          model=args.neg_model, temperature=temp)
                    solution = clean_solution(raw, problem['prompt'])
                    passed, error = test_solution(
                        problem['prompt'], solution,
                        problem['test'], problem['entry_point']
                    )

                    record = {
                        'task_id': task_id,
                        'prompt': problem['prompt'],
                        'solution': solution,
                        'raw_solution': raw,
                        'entry_point': problem['entry_point'],
                        'passed': passed,
                        'error': error,
                        'temperature': temp,
                        'model': args.neg_model,
                    }

                    fout.write(json.dumps(record) + "\n")
                    fout.flush()

                    if passed:
                        n_pass += 1
                    else:
                        n_fail += 1

                except Exception as e:
                    print(f"  Error on sample {j}: {e}")
                    traceback.print_exc()

            print(f"  -> {n_pass} pass, {n_fail} fail (new)")
            total_generated += n_pass + n_fail
            total_new_neg += n_fail
            total_new_pos += n_pass

        # Phase 2: Generate passes with strong model at low temp
        print(f"\n=== Phase 2: Generating positives with {args.pos_model} ===")
        for i, task_id in enumerate(need_pos):
            problem = problems[task_id]
            current_pass = per_task[task_id]['pass']
            n_samples = args.pos_samples

            print(f"[{i+1}/{len(need_pos)}] {task_id} (have {current_pass} pass, {per_task[task_id]['fail']} fail)")

            n_pass = 0
            n_fail = 0
            for j in range(n_samples):
                temp = pos_temps[j % len(pos_temps)]
                try:
                    raw = generate_solution(client, problem['prompt'],
                                          model=args.pos_model, temperature=temp)
                    solution = clean_solution(raw, problem['prompt'])
                    passed, error = test_solution(
                        problem['prompt'], solution,
                        problem['test'], problem['entry_point']
                    )

                    record = {
                        'task_id': task_id,
                        'prompt': problem['prompt'],
                        'solution': solution,
                        'raw_solution': raw,
                        'entry_point': problem['entry_point'],
                        'passed': passed,
                        'error': error,
                        'temperature': temp,
                        'model': args.pos_model,
                    }

                    fout.write(json.dumps(record) + "\n")
                    fout.flush()

                    if passed:
                        n_pass += 1
                    else:
                        n_fail += 1

                except Exception as e:
                    print(f"  Error on sample {j}: {e}")
                    traceback.print_exc()

            print(f"  -> {n_pass} pass, {n_fail} fail (new)")
            total_generated += n_pass + n_fail
            total_new_neg += n_fail
            total_new_pos += n_pass

    # Final summary
    print(f"\n=== Supplemental Generation Complete ===")
    print(f"Total new solutions: {total_generated}")
    print(f"New passes: {total_new_pos}")
    print(f"New failures: {total_new_neg}")

    # Recount and show qualification status
    final_counts = load_existing_counts(args.output)
    qualified = sum(1 for s in final_counts.values() if s['pass'] >= THRESHOLD and s['fail'] >= THRESHOLD)
    still_low_neg = sum(1 for s in final_counts.values() if s['fail'] < THRESHOLD)
    still_low_pos = sum(1 for s in final_counts.values() if s['pass'] < THRESHOLD)
    total_solutions = sum(s['pass'] + s['fail'] for s in final_counts.values())

    print(f"\nTotal solutions now: {total_solutions}")
    print(f"Qualified problems (>={THRESHOLD}/{THRESHOLD}): {qualified}/164")
    print(f"Still need negatives: {still_low_neg}")
    print(f"Still need positives: {still_low_pos}")


if __name__ == "__main__":
    main()
