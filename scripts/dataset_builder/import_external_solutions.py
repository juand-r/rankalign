#!/usr/bin/env python3
"""
Import HumanEval solutions from external repos into our solutions.jsonl format.

Sources:
1. jamesmurdza/humaneval-results — CodeLlama-34b, GPT-3.5-turbo, GPT-4 (10 runs each, markdown)
2. breath24/FailureBench — Claude Sonnet-4, DeepSeek-V3, GPT-4o, Llama-3.3-70B, Mistral-3.2-24B, Qwen3-Coder (1 solution each, JSON)

Usage:
    python scripts/dataset_builder/import_external_solutions.py \
        --humaneval-results /tmp/humaneval-results \
        --failurebench /tmp/FailureBench \
        --problems data/humaneval/problems.jsonl \
        --output data/humaneval/solutions.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def load_problems(path):
    """Load problems.jsonl for validation."""
    problems = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            problems[p['task_id']] = p
    return problems


def validate_solution(prompt, solution, test_code, entry_point):
    """Run solution against HumanEval unit tests. Returns (passed, error)."""
    full_code = prompt + solution + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        passed = result.returncode == 0
        error = result.stderr.strip() if not passed else None
        return passed, error
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


def clean_solution_body(raw_code, prompt):
    """Extract just the function body from a full solution, matching our format."""
    # If the raw code contains the full function (signature + body), extract body
    lines = raw_code.split('\n')
    body_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            for j in range(i, len(lines)):
                if lines[j].rstrip().endswith(':'):
                    body_start = j + 1
                    break
            break

    if body_start is not None and body_start < len(lines):
        sol = '\n'.join(lines[body_start:])
    else:
        sol = raw_code

    # Strip markdown code blocks
    if sol.startswith('```'):
        sol_lines = sol.split('\n')
        sol_lines = sol_lines[1:]
        if sol_lines and sol_lines[-1].strip() == '```':
            sol_lines = sol_lines[:-1]
        sol = '\n'.join(sol_lines)

    # Ensure proper indentation
    result_lines = []
    for line in sol.split('\n'):
        if line.strip() == '':
            result_lines.append('')
        elif not line.startswith('    ') and line.strip():
            result_lines.append('    ' + line)
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def parse_humaneval_results(repo_dir, problems):
    """Parse jamesmurdza/humaneval-results markdown files."""
    records = []
    model_dirs = {
        'codellama-34b-instruct': 'CodeLlama-34b-Instruct',
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'gpt-4': 'gpt-4',
    }

    for dir_name, model_name in model_dirs.items():
        model_dir = os.path.join(repo_dir, dir_name)
        if not os.path.exists(model_dir):
            print(f"  Skipping {model_name}: {model_dir} not found")
            continue

        n_pass = 0
        n_fail = 0
        n_error = 0

        for filename in sorted(os.listdir(model_dir)):
            if not filename.endswith('.md'):
                continue
            problem_num = filename[:-3]  # strip .md
            task_id = f"HumanEval/{problem_num}"

            if task_id not in problems:
                continue

            problem = problems[task_id]
            filepath = os.path.join(model_dir, filename)

            with open(filepath) as f:
                content = f.read()

            # Extract individual runs
            # Pattern: ### ✅ Run N or ### ❌ Run N followed by code block
            run_pattern = re.compile(
                r'### ([✅❌]) Run (\d+).*?```python\n(.*?)```',
                re.DOTALL
            )

            for match in run_pattern.finditer(content):
                status_emoji = match.group(1)
                run_num = int(match.group(2))
                raw_code = match.group(3).strip()

                # Clean to function body
                solution = clean_solution_body(raw_code, problem['prompt'])

                # Validate ourselves
                passed, error = validate_solution(
                    problem['prompt'], solution, problem['test'], problem['entry_point']
                )

                if passed:
                    n_pass += 1
                else:
                    n_fail += 1

                records.append({
                    'task_id': task_id,
                    'prompt': problem['prompt'],
                    'solution': solution,
                    'raw_solution': raw_code,
                    'entry_point': problem['entry_point'],
                    'passed': passed,
                    'error': error,
                    'temperature': 0.2,  # from the markdown headers
                    'model': model_name,
                    'strategy': 'normal',
                    'source': 'jamesmurdza/humaneval-results',
                })

        print(f"  {model_name}: {n_pass} pass, {n_fail} fail, {n_error} errors")

    return records


def parse_failurebench(repo_dir, problems):
    """Parse breath24/FailureBench JSON files."""
    records = []
    he_dir = os.path.join(repo_dir, 'evaluation-results', 'llm-generated-code', 'HumanEval')

    if not os.path.exists(he_dir):
        print(f"  FailureBench HumanEval dir not found: {he_dir}")
        return records

    for model_dir_name in sorted(os.listdir(he_dir)):
        raw_dir = os.path.join(he_dir, model_dir_name, 'raw')
        if not os.path.isdir(raw_dir):
            continue

        model_name = model_dir_name.strip()
        n_pass = 0
        n_fail = 0

        for filename in sorted(os.listdir(raw_dir)):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(raw_dir, filename)
            with open(filepath) as f:
                data = json.load(f)

            task_id = data['task_id']
            if task_id not in problems:
                continue

            problem = problems[task_id]
            raw_code = data['llm_response']

            # Clean to function body
            solution = clean_solution_body(raw_code, problem['prompt'])

            # Validate ourselves
            passed, error = validate_solution(
                problem['prompt'], solution, problem['test'], problem['entry_point']
            )

            if passed:
                n_pass += 1
            else:
                n_fail += 1

            records.append({
                'task_id': task_id,
                'prompt': problem['prompt'],
                'solution': solution,
                'raw_solution': raw_code,
                'entry_point': problem['entry_point'],
                'passed': passed,
                'error': error,
                'temperature': None,
                'model': model_name,
                'strategy': 'normal',
                'source': 'breath24/FailureBench',
            })

        print(f"  {model_name}: {n_pass} pass, {n_fail} fail")

    return records


def main():
    parser = argparse.ArgumentParser(description="Import external HumanEval solutions")
    parser.add_argument("--humaneval-results", type=str, help="Path to jamesmurdza/humaneval-results clone")
    parser.add_argument("--failurebench", type=str, help="Path to breath24/FailureBench clone")
    parser.add_argument("--problems", type=str, required=True, help="Path to problems.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL (appends)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report stats, don't write")
    args = parser.parse_args()

    problems = load_problems(args.problems)
    print(f"Loaded {len(problems)} problems")

    all_records = []

    if args.humaneval_results:
        print(f"\nParsing humaneval-results from {args.humaneval_results}...")
        records = parse_humaneval_results(args.humaneval_results, problems)
        all_records.extend(records)
        print(f"  Total: {len(records)} solutions")

    if args.failurebench:
        print(f"\nParsing FailureBench from {args.failurebench}...")
        records = parse_failurebench(args.failurebench, problems)
        all_records.extend(records)
        print(f"  Total: {len(records)} solutions")

    print(f"\nGrand total: {len(all_records)} new solutions")
    pass_count = sum(1 for r in all_records if r['passed'])
    fail_count = len(all_records) - pass_count
    print(f"  Pass: {pass_count}, Fail: {fail_count}")

    # Per-model summary
    from collections import Counter
    model_counts = Counter()
    for r in all_records:
        model_counts[r['model']] += 1
    print("\nPer model:")
    for m, c in model_counts.most_common():
        p = sum(1 for r in all_records if r['model'] == m and r['passed'])
        print(f"  {m}: {c} solutions ({p} pass, {c-p} fail)")

    if not args.dry_run:
        with open(args.output, 'a') as f:
            for r in all_records:
                f.write(json.dumps(r) + '\n')
        print(f"\nAppended {len(all_records)} records to {args.output}")


if __name__ == "__main__":
    main()
