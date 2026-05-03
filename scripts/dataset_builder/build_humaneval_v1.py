#!/usr/bin/env python3
"""
Build humaneval-v1 task dataset from the merged solution pool.

Stratified sampling strategy:
- Split 162 problems into ~80 train + ~82 test
- Per problem, sample N solutions balanced across:
  - pass/fail (50/50 target)
  - models (diverse mix)
  - strategies (diverse mix)
- Output: data/humaneval/with_solutions/train.csv + humaneval_<N>.csv files

Usage:
    .tools-venv/bin/python scripts/dataset_builder/build_humaneval_v1.py \
        --input data/humaneval/solutions_merged.jsonl \
        --output-dir data/humaneval/with_solutions \
        --samples-per-problem 30 \
        --seed 42
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


EXCLUDED = {'HumanEval/53', 'HumanEval/145'}


def load_solutions(path):
    """Load solutions, excluding intentional_bug, excluded problems, and empty solutions."""
    rows = []
    empty = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r['task_id'] in EXCLUDED:
                continue
            if r.get('strategy') == 'intentional_bug':
                continue
            if not r.get('solution', '').strip():
                empty += 1
                continue
            rows.append(r)
    if empty:
        print(f"  Skipped {empty} empty solutions")
    return rows


def stratified_sample(pool, n, rng):
    """Sample n items from pool, maximizing model and strategy diversity."""
    if len(pool) <= n:
        return pool[:]

    # Group by (model, strategy) to ensure diversity
    by_group = defaultdict(list)
    for r in pool:
        by_group[(r['model'], r.get('strategy', 'normal'))].append(r)

    selected = []
    # Round-robin from each group
    group_keys = list(by_group.keys())
    rng.shuffle(group_keys)

    group_iters = {}
    for k in group_keys:
        items = by_group[k][:]
        rng.shuffle(items)
        group_iters[k] = iter(items)

    # Round-robin until we have enough
    while len(selected) < n:
        added_this_round = False
        for k in group_keys:
            if len(selected) >= n:
                break
            try:
                selected.append(next(group_iters[k]))
                added_this_round = True
            except StopIteration:
                continue
        if not added_this_round:
            break

    return selected[:n]


def make_split(problems, rng, n_train=80):
    """Split problems into train/test, balancing difficulty."""
    # Sort by problem number for determinism, then shuffle with seed
    problems = sorted(problems, key=lambda x: int(x.split('/')[1]))
    rng.shuffle(problems)
    train_problems = set(problems[:n_train])
    test_problems = set(problems[n_train:])
    return train_problems, test_problems


def write_csv(rows, output_path):
    """Write rows as CSV in the rankalign humaneval format."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer', 'correct', 'strategy', 'model', 'temperature', 'task_id', 'error'])
        for r in rows:
            writer.writerow([
                r['prompt'],
                r['solution'],
                'Yes' if r['passed'] else 'No',
                r.get('strategy', 'normal'),
                r.get('model', ''),
                r.get('temperature', ''),
                r.get('task_id', ''),
                r.get('error', ''),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="solutions_merged.jsonl path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--samples-per-problem", type=int, default=30,
                        help="Target solutions per problem")
    parser.add_argument("--n-train", type=int, default=80, help="Number of train problems")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_solutions(args.input)
    print(f"Loaded {len(rows)} solutions")

    # Group by problem
    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r['task_id']].append(r)

    problems = sorted(by_problem.keys(), key=lambda x: int(x.split('/')[1]))
    print(f"Problems: {len(problems)}")

    # Split problems
    train_problems, test_problems = make_split(problems, rng, args.n_train)
    print(f"Train problems: {len(train_problems)}, Test problems: {len(test_problems)}")

    n = args.samples_per_problem
    half = n // 2

    # Sample per problem
    train_rows = []
    test_rows_by_problem = {}
    stats = {'train_pass': 0, 'train_fail': 0, 'test_pass': 0, 'test_fail': 0}

    for pid in problems:
        pool = by_problem[pid]
        pass_pool = [r for r in pool if r['passed']]
        fail_pool = [r for r in pool if not r['passed']]

        # Sample balanced pass/fail, up to half each
        n_pass = min(half, len(pass_pool))
        n_fail = min(half, len(fail_pool))
        # If one side is short, give the other more
        if n_pass < half and len(fail_pool) > half:
            n_fail = min(n - n_pass, len(fail_pool))
        elif n_fail < half and len(pass_pool) > half:
            n_pass = min(n - n_fail, len(pass_pool))

        sampled_pass = stratified_sample(pass_pool, n_pass, rng)
        sampled_fail = stratified_sample(fail_pool, n_fail, rng)
        sampled = sampled_pass + sampled_fail
        rng.shuffle(sampled)

        if pid in train_problems:
            train_rows.extend(sampled)
            stats['train_pass'] += len(sampled_pass)
            stats['train_fail'] += len(sampled_fail)
        else:
            test_rows_by_problem[pid] = sampled
            stats['test_pass'] += len(sampled_pass)
            stats['test_fail'] += len(sampled_fail)

    print(f"\nSampling results:")
    print(f"  Train: {len(train_rows)} rows ({stats['train_pass']}p/{stats['train_fail']}f)")
    total_test = sum(len(v) for v in test_rows_by_problem.values())
    print(f"  Test:  {total_test} rows across {len(test_rows_by_problem)} problems "
          f"({stats['test_pass']}p/{stats['test_fail']}f)")

    # Model and strategy diversity
    train_models = set(r['model'] for r in train_rows)
    train_strats = set(r.get('strategy', 'normal') for r in train_rows)
    print(f"  Train models: {len(train_models)}, strategies: {len(train_strats)}")

    if args.dry_run:
        print("\n[DRY RUN] Not writing files.")
        return

    # Write files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing train.csv if it exists
    train_path = output_dir / "train.csv"
    if train_path.exists():
        backup = output_dir / "train_old.csv"
        train_path.rename(backup)
        print(f"  Backed up existing train.csv to train_old.csv")

    # Write train.csv
    rng.shuffle(train_rows)
    write_csv(train_rows, train_path)
    print(f"  Wrote {len(train_rows)} rows to {train_path}")

    # Remove old test files
    for f in output_dir.glob("humaneval_*.csv"):
        f.unlink()

    # Write per-problem test CSVs
    for pid, test_rows in sorted(test_rows_by_problem.items()):
        num = pid.split('/')[1]
        test_path = output_dir / f"humaneval_{num}.csv"
        write_csv(test_rows, test_path)

    print(f"  Wrote {len(test_rows_by_problem)} test problem CSVs")

    # Summary
    print(f"\n=== HUMANEVAL-V1 DATASET ===")
    print(f"Train: {len(train_rows)} solutions across {len(train_problems)} problems")
    print(f"Test: {total_test} solutions across {len(test_rows_by_problem)} problems")
    print(f"Samples/problem target: {args.samples_per_problem}")
    print(f"Models: {len(train_models)}")
    print(f"Strategies: {sorted(train_strats)}")


if __name__ == "__main__":
    main()
