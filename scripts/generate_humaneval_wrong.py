"""
Generate plausible-but-wrong solutions for HumanEval problems that have too few negatives.

Uses a "write a subtly wrong implementation" prompt to elicit failures.
Only targets problems that still have <10 failing solutions.

Appends to the existing solutions.jsonl file.
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

sys.path.insert(0, str(Path(__file__).parent))
from generate_humaneval_solutions import clean_solution, test_solution


THRESHOLD = 10


def generate_wrong_solution(client, prompt, model="gpt-4o-mini", temperature=0.8):
    """Generate a plausible-but-wrong solution."""
    system_msg = (
        "You are a Python programmer who makes subtle mistakes. "
        "Complete the given function, but introduce a subtle bug that would cause it to fail on some inputs. "
        "The bug should be plausible — the kind of mistake a real programmer might make "
        "(off-by-one errors, wrong comparison operators, missing edge cases, wrong variable, etc.). "
        "Return ONLY the function body. Do NOT include the function signature, imports, or any explanation. "
        "Do NOT wrap in markdown code blocks. Do NOT comment about the bug."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Complete this function (with a subtle bug):\n\n{prompt}"},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def load_existing_counts(path):
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
    parser.add_argument("--samples", type=int, default=20,
                        help="Wrong solutions to generate per problem")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    client = OpenAI()
    problems = read_problems()
    per_task = load_existing_counts(args.output)

    need_neg = sorted([tid for tid in problems if per_task[tid]['fail'] < THRESHOLD])
    print(f"Problems needing negatives: {len(need_neg)}")

    temps = [0.6, 0.8, 1.0, 1.2]
    total_pass = 0
    total_fail = 0

    with open(args.output, 'a') as fout:
        for i, task_id in enumerate(need_neg):
            problem = problems[task_id]
            current = per_task[task_id]
            print(f"[{i+1}/{len(need_neg)}] {task_id} (have {current['fail']} fail, {current['pass']} pass)")

            n_pass = 0
            n_fail = 0
            for j in range(args.samples):
                temp = temps[j % len(temps)]
                try:
                    raw = generate_wrong_solution(client, problem['prompt'],
                                                  model=args.model, temperature=temp)
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
                        'model': args.model,
                        'strategy': 'intentional_bug',
                    }

                    fout.write(json.dumps(record) + "\n")
                    fout.flush()

                    if passed:
                        n_pass += 1
                    else:
                        n_fail += 1

                except Exception as e:
                    print(f"  Error: {e}")
                    traceback.print_exc()

            print(f"  -> {n_pass} pass, {n_fail} fail")
            total_pass += n_pass
            total_fail += n_fail

    print(f"\n=== Done ===")
    print(f"Generated: {total_pass + total_fail} ({total_pass} pass, {total_fail} fail)")

    final = load_existing_counts(args.output)
    qualified = sum(1 for s in final.values() if s['pass'] >= THRESHOLD and s['fail'] >= THRESHOLD)
    still_low = sum(1 for s in final.values() if s['fail'] < THRESHOLD)
    total = sum(s['pass'] + s['fail'] for s in final.values())
    print(f"Total solutions: {total}")
    print(f"Qualified: {qualified}/164")
    print(f"Still need negatives: {still_low}")


if __name__ == "__main__":
    main()
