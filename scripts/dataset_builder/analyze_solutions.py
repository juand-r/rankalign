#!/usr/bin/env python3
"""
Comprehensive analysis of the solution pool for HumanEval v1.

Checks:
  - Coverage: model × strategy × problem completeness
  - Balance: pass/fail ratio by model, strategy, problem
  - Quality: error types, solution lengths, trailing-markdown contamination
  - Diversity: solution length distributions, model-difficulty confounds

Usage:
    python scripts/dataset_builder/analyze_solutions.py --input data/humaneval/solutions.jsonl
"""

import argparse
import json
import random
from collections import defaultdict, Counter


EXCLUDED = {'HumanEval/53', 'HumanEval/145'}


def load_solutions(path, exclude_intentional_bug=True):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r['task_id'] in EXCLUDED:
                continue
            if exclude_intentional_bug and r.get('strategy') == 'intentional_bug':
                continue
            rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="solutions.jsonl path")
    args = parser.parse_args()

    rows = load_solutions(args.input)
    print(f"Total solutions (excl intentional_bug, excl 53/145): {len(rows)}")

    models = sorted(set(r['model'] for r in rows))
    strategies = sorted(set(r.get('strategy', 'normal') for r in rows))
    problems = sorted(set(r['task_id'] for r in rows), key=lambda x: int(x.split('/')[1]))

    print(f"\nModels ({len(models)}): {models}")
    print(f"Strategies ({len(strategies)}): {strategies}")
    print(f"Problems: {len(problems)}")

    # --- 1. Model × Strategy counts ---
    print("\n" + "=" * 70)
    print("1. MODEL × STRATEGY SOLUTION COUNTS")
    print("=" * 70)
    ms_counts = defaultdict(int)
    ms_pass = defaultdict(int)
    for r in rows:
        key = (r['model'], r.get('strategy', 'normal'))
        ms_counts[key] += 1
        if r['passed']:
            ms_pass[key] += 1

    # Header
    header = f"{'Model':<45}"
    for s in strategies:
        header += f" {s[:8]:>10}"
    header += f" {'Total':>10}"
    print(header)
    print("-" * len(header))
    for m in models:
        line = f"{m:<45}"
        total = 0
        for s in strategies:
            c = ms_counts.get((m, s), 0)
            total += c
            line += f" {c:>10}"
        line += f" {total:>10}"
        print(line)

    # --- 2. Pass/Fail by model ---
    print("\n" + "=" * 70)
    print("2. PASS/FAIL BY MODEL")
    print("=" * 70)
    model_pass = defaultdict(int)
    model_fail = defaultdict(int)
    for r in rows:
        if r['passed']:
            model_pass[r['model']] += 1
        else:
            model_fail[r['model']] += 1
    print(f"{'Model':<45} {'Pass':>8} {'Fail':>8} {'Total':>8} {'Fail%':>8}")
    print("-" * 80)
    for m in models:
        p, f = model_pass[m], model_fail[m]
        t = p + f
        print(f"{m:<45} {p:>8} {f:>8} {t:>8} {100*f/t:>7.1f}%")

    # --- 3. Pass/Fail by strategy ---
    print("\n" + "=" * 70)
    print("3. PASS/FAIL BY STRATEGY")
    print("=" * 70)
    strat_pass = defaultdict(int)
    strat_fail = defaultdict(int)
    for r in rows:
        s = r.get('strategy', 'normal')
        if r['passed']:
            strat_pass[s] += 1
        else:
            strat_fail[s] += 1
    print(f"{'Strategy':<20} {'Pass':>8} {'Fail':>8} {'Total':>8} {'Fail%':>8}")
    print("-" * 50)
    for s in strategies:
        p, f = strat_pass[s], strat_fail[s]
        t = p + f
        print(f"{s:<20} {p:>8} {f:>8} {t:>8} {100*f/t:>7.1f}%")

    # --- 4. Per-problem qualification ---
    print("\n" + "=" * 70)
    print("4. PER-PROBLEM QUALIFICATION (>=10 pass AND >=10 fail)")
    print("=" * 70)
    prob_pass = defaultdict(int)
    prob_fail = defaultdict(int)
    for r in rows:
        if r['passed']:
            prob_pass[r['task_id']] += 1
        else:
            prob_fail[r['task_id']] += 1

    qualified = []
    too_few_pass = []
    too_few_fail = []
    for p in problems:
        pp, pf = prob_pass[p], prob_fail[p]
        if pp >= 10 and pf >= 10:
            qualified.append(p)
        elif pp < 10:
            too_few_pass.append((p, pp, pf))
        else:
            too_few_fail.append((p, pp, pf))

    print(f"Qualified: {len(qualified)}/162")
    if too_few_pass:
        print(f"\nToo few passes ({len(too_few_pass)}):")
        for p, pp, pf in too_few_pass[:20]:
            print(f"  {p}: {pp}p/{pf}f")
    if too_few_fail:
        print(f"\nToo few failures ({len(too_few_fail)}):")
        for p, pp, pf in too_few_fail[:20]:
            print(f"  {p}: {pp}p/{pf}f")

    # --- 5. Solution length analysis ---
    print("\n" + "=" * 70)
    print("5. SOLUTION LENGTH (chars) BY PASS/FAIL")
    print("=" * 70)
    pass_lens = [len(r['solution']) for r in rows if r['passed']]
    fail_lens = [len(r['solution']) for r in rows if not r['passed']]
    if pass_lens:
        pass_lens.sort()
        print(f"Passing:  n={len(pass_lens)}, median={pass_lens[len(pass_lens)//2]}, "
              f"mean={sum(pass_lens)/len(pass_lens):.0f}, "
              f"p10={pass_lens[len(pass_lens)//10]}, p90={pass_lens[9*len(pass_lens)//10]}")
    if fail_lens:
        fail_lens.sort()
        print(f"Failing:  n={len(fail_lens)}, median={fail_lens[len(fail_lens)//2]}, "
              f"mean={sum(fail_lens)/len(fail_lens):.0f}, "
              f"p10={fail_lens[len(fail_lens)//10]}, p90={fail_lens[9*len(fail_lens)//10]}")

    # --- 6. Error type analysis ---
    print("\n" + "=" * 70)
    print("6. ERROR TYPES IN FAILING SOLUTIONS")
    print("=" * 70)
    error_types = Counter()
    for r in rows:
        if not r['passed'] and r.get('error'):
            err = r['error'].split('\n')[-1]
            # Classify
            if 'SyntaxError' in err:
                error_types['SyntaxError'] += 1
            elif 'NameError' in err:
                error_types['NameError'] += 1
            elif 'TypeError' in err:
                error_types['TypeError'] += 1
            elif 'IndexError' in err:
                error_types['IndexError'] += 1
            elif 'AssertionError' in err or 'AssertionError' in err:
                error_types['AssertionError (test fail)'] += 1
            elif 'TIMEOUT' in err:
                error_types['TIMEOUT'] += 1
            elif 'ValueError' in err:
                error_types['ValueError'] += 1
            elif 'AttributeError' in err:
                error_types['AttributeError'] += 1
            elif 'ZeroDivisionError' in err:
                error_types['ZeroDivisionError'] += 1
            elif 'RecursionError' in err:
                error_types['RecursionError'] += 1
            else:
                error_types['Other'] += 1
    for et, c in error_types.most_common(15):
        print(f"  {et:<30} {c:>6}")

    # --- 7. Trailing markdown contamination check ---
    print("\n" + "=" * 70)
    print("7. TRAILING MARKDOWN/EXPLANATION CHECK")
    print("=" * 70)
    has_backticks = sum(1 for r in rows if '```' in r['solution'])
    has_explanation = sum(1 for r in rows
                        if any(line.strip() and not line.strip().startswith('#')
                               and not line.startswith(' ') and not line.startswith('\t')
                               and len(line.strip()) > 20
                               for line in r['solution'].split('\n')
                               if not line.strip().startswith('def ')
                               and not line.strip().startswith('return ')
                               and not line.strip().startswith('import ')))
    print(f"Solutions with ``` in body: {has_backticks} ({100*has_backticks/len(rows):.1f}%)")

    # --- 8. Sample outputs ---
    print("\n" + "=" * 70)
    print("8. SAMPLE OUTPUTS (random correct + incorrect per strategy)")
    print("=" * 70)
    random.seed(42)
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get('strategy', 'normal'), r['passed'])].append(r)

    for s in strategies:
        print(f"\n--- {s} ---")
        for passed in [True, False]:
            label = "CORRECT" if passed else "INCORRECT"
            pool = groups.get((s, passed), [])
            if not pool:
                print(f"  [{label}] No examples")
                continue
            ex = random.choice(pool)
            print(f"  [{label}] {ex['task_id']} model={ex['model']} temp={ex['temperature']}")
            lines = ex['solution'].split('\n')[:12]
            for line in lines:
                print(f"    {line}")
            if len(ex['solution'].split('\n')) > 12:
                print(f"    ... ({len(ex['solution'].split(chr(10)))} lines)")
            if not passed and ex.get('error'):
                err_last = ex['error'].split('\n')[-1][:100]
                print(f"  Error: {err_last}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
