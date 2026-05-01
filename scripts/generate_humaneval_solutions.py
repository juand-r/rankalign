"""
Generate solutions for HumanEval problems using GPT-4o and classify as pass/fail.

Samples multiple solutions per problem at varying temperatures, runs HumanEval
unit tests to classify, and saves results to a JSONL file.

Usage:
    python scripts/generate_humaneval_solutions.py [--samples-per-problem 40] [--output data/humaneval/solutions.jsonl]
"""

import argparse
import json
import os
import sys
import signal
import traceback
import tempfile
import subprocess
from pathlib import Path

from human_eval.data import read_problems
from openai import OpenAI


def generate_solution(client, prompt, model="gpt-4o", temperature=1.0):
    """Generate a single solution completion using OpenAI API."""
    system_msg = (
        "You are an expert Python programmer. Complete the given function. "
        "Return ONLY the function body (the indented code that goes after the function signature). "
        "Do NOT include the function signature, imports, or any explanation. "
        "Do NOT wrap in markdown code blocks."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Complete this function:\n\n{prompt}"},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def clean_solution(raw_solution, prompt):
    """Clean up a raw LLM solution to extract just the function body."""
    sol = raw_solution

    # Strip markdown code blocks if present
    if sol.startswith("```"):
        lines = sol.split("\n")
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last ``` if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sol = "\n".join(lines)

    # If the solution contains the full function signature, extract body only
    # Look for 'def ' at the start of a line
    lines = sol.split("\n")
    body_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            # Find the end of the signature (the colon)
            for j in range(i, len(lines)):
                if lines[j].rstrip().endswith(":"):
                    body_start = j + 1
                    break
            break

    if body_start is not None and body_start < len(lines):
        sol = "\n".join(lines[body_start:])

    # Ensure proper indentation (4 spaces)
    result_lines = []
    for line in sol.split("\n"):
        if line.strip() == "":
            result_lines.append("")
        elif not line.startswith("    ") and line.strip():
            result_lines.append("    " + line)
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def test_solution(prompt, solution, test_code, entry_point, timeout=10):
    """Test a solution by executing it with the HumanEval test harness.

    Returns (passed: bool, error_msg: str or None)
    """
    full_code = prompt + solution + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        passed = result.returncode == 0
        error_msg = result.stderr.strip() if not passed else None
        return passed, error_msg
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-problem", type=int, default=40,
                        help="Number of solutions to sample per problem")
    parser.add_argument("--output", type=str, default="data/humaneval/solutions.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    client = OpenAI()
    problems = read_problems()
    task_ids = sorted(problems.keys())
    print(f"Loaded {len(task_ids)} HumanEval problems")

    # Load existing results if resuming
    done_tasks = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            for line in f:
                rec = json.loads(line)
                tid = rec['task_id']
                if tid not in done_tasks:
                    done_tasks[tid] = 0
                done_tasks[tid] += 1
        print(f"Resuming: {len(done_tasks)} tasks have some solutions")

    # Temperature schedule: mix of temps for diversity
    # Lower temps for reliable correct solutions, higher for plausible failures
    temp_schedule = (
        [0.2] * 5 +   # 5 low-temp (likely correct)
        [0.6] * 5 +   # 5 medium-temp
        [1.0] * 15 +  # 15 high-temp (more failures)
        [1.2] * 10 +  # 10 very high temp
        [1.4] * 5     # 5 extreme temp
    )

    with open(args.output, 'a') as fout:
        for i, task_id in enumerate(task_ids):
            problem = problems[task_id]
            existing = done_tasks.get(task_id, 0)
            remaining = args.samples_per_problem - existing
            if remaining <= 0:
                print(f"[{i+1}/{len(task_ids)}] {task_id}: already have {existing} solutions, skipping")
                continue

            print(f"[{i+1}/{len(task_ids)}] {task_id}: generating {remaining} solutions (have {existing})...")

            n_pass = 0
            n_fail = 0

            for j in range(remaining):
                temp = temp_schedule[j % len(temp_schedule)]
                try:
                    raw = generate_solution(client, problem['prompt'], model=args.model, temperature=temp)
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

            print(f"  -> {n_pass} pass, {n_fail} fail")

    # Print summary
    print("\n=== Summary ===")
    all_records = []
    with open(args.output, 'r') as f:
        for line in f:
            all_records.append(json.loads(line))

    n_total = len(all_records)
    n_pass = sum(1 for r in all_records if r['passed'])
    n_fail = n_total - n_pass
    print(f"Total solutions: {n_total}")
    print(f"Passed: {n_pass} ({100*n_pass/n_total:.1f}%)")
    print(f"Failed: {n_fail} ({100*n_fail/n_total:.1f}%)")

    # Per-problem stats
    from collections import Counter, defaultdict
    per_task = defaultdict(lambda: {'pass': 0, 'fail': 0})
    for r in all_records:
        if r['passed']:
            per_task[r['task_id']]['pass'] += 1
        else:
            per_task[r['task_id']]['fail'] += 1

    low_pass = [(tid, s) for tid, s in per_task.items() if s['pass'] < 10]
    low_fail = [(tid, s) for tid, s in per_task.items() if s['fail'] < 10]
    print(f"\nProblems with <10 passing: {len(low_pass)}")
    print(f"Problems with <10 failing: {len(low_fail)}")
    if low_fail:
        print("  (these need more sampling at higher temp or a weaker model)")
        for tid, s in sorted(low_fail, key=lambda x: x[1]['fail']):
            print(f"    {tid}: {s['pass']} pass, {s['fail']} fail")


if __name__ == "__main__":
    main()
