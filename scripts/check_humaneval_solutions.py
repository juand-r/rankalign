"""
Verify HumanEval solutions against the original test cases.

Loads our dataset CSVs, reconstructs full programs by prepending the
HumanEval function signature, appends the test harness, and executes.
Reports pass/fail for each solution and flags any mismatches with our
'correct' labels.

Usage:
    python scripts/check_humaneval_solutions.py                     # all problems
    python scripts/check_humaneval_solutions.py --problems 19 40    # specific ones
    python scripts/check_humaneval_solutions.py --only-mismatches   # show only label bugs
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile

from human_eval.data import read_problems


def run_solution(prompt, solution, test_code, entry_point, timeout=10):
    """Execute solution against HumanEval tests. Returns (passed, error_msg)."""
    full_code = prompt + solution + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        passed = result.returncode == 0
        error_msg = None
        if not passed:
            lines = result.stderr.strip().split('\n')
            error_msg = lines[-1] if lines else 'unknown error'
        return passed, error_msg
    except subprocess.TimeoutExpired:
        return False, 'TIMEOUT'
    finally:
        os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problems', nargs='*', type=str, default=None,
                        help='Problem numbers to check (e.g. 19 40). Default: all.')
    parser.add_argument('--only-mismatches', action='store_true',
                        help='Only print solutions where test result disagrees with label.')
    parser.add_argument('--data-dir', type=str,
                        default='data/humaneval/with_solutions',
                        help='Directory containing per-problem CSVs.')
    parser.add_argument('--timeout', type=int, default=10)
    args = parser.parse_args()

    he_problems = read_problems()

    csvs = sorted(glob.glob(os.path.join(args.data_dir, 'humaneval_*.csv')))
    if args.problems:
        keep = {f'humaneval_{p}' for p in args.problems}
        csvs = [c for c in csvs if os.path.basename(c)[:-4] in keep]

    total = 0
    mismatches = 0
    pass_count = 0
    fail_count = 0

    for csv_path in csvs:
        slug = os.path.basename(csv_path)[:-4]  # humaneval_19
        num = slug.split('_')[1]
        he_id = f'HumanEval/{num}'

        if he_id not in he_problems:
            print(f'WARNING: {he_id} not found in human_eval dataset, skipping')
            continue

        he = he_problems[he_id]

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f'{"="*70}')
        print(f'{he_id}  |  {he["entry_point"]}()  |  {len(rows)} solutions')
        print(f'{"="*70}')
        print()

        # Show test cases
        if not args.only_mismatches:
            for line in he['test'].split('\n'):
                if 'assert' in line:
                    print(f'  {line.strip()}')
            print()

        for i, row in enumerate(rows):
            label_correct = row['correct'].strip().lower() == 'yes'
            answer = row['answer']

            passed, error = run_solution(
                he['prompt'], answer, he['test'], he['entry_point'],
                timeout=args.timeout,
            )

            total += 1
            if passed:
                pass_count += 1
            else:
                fail_count += 1

            mismatch = passed != label_correct
            if mismatch:
                mismatches += 1

            if args.only_mismatches and not mismatch:
                continue

            tag = 'MISMATCH' if mismatch else 'ok'
            label_str = 'correct' if label_correct else 'incorrect'
            test_str = 'PASSES' if passed else f'FAILS'

            preview = answer.replace('\n', '\\n')[:100]
            print(f'  [{tag:>8s}] label={label_str:<10s} test={test_str:<7s} | {preview}')
            if error and not args.only_mismatches:
                print(f'           error: {error}')

        print()

    print(f'{"="*70}')
    print(f'SUMMARY: {total} solutions checked')
    print(f'  Passed tests: {pass_count}')
    print(f'  Failed tests: {fail_count}')
    print(f'  Label mismatches: {mismatches}')
    if mismatches:
        print(f'  WARNING: {mismatches} solutions have labels that disagree with test results!')
    else:
        print(f'  All labels are consistent with test results.')


if __name__ == '__main__':
    main()
