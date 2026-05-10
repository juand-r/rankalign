#!/usr/bin/env python3
"""
Build gsm8k-v1 task dataset from the merged solution pool. Mirrors
build_humaneval_v1.py but for GSM8K's CSV schema and task naming.

Stratified sampling strategy:
- Filter out intentional_bug, excluded problems, empty solutions.
- Split qualified problems into ~80 test + rest train (configurable).
- Per problem, sample N rows balanced 50/50 pass/fail across (model, strategy)
  groups via round-robin (so no single (model, strategy) dominates).
- Output:
    train.csv (all train-side rows shuffled)
    gsm8k_test_<NUM>.csv (one per held-out test problem)

Usage:
    .tools-venv/bin/python scripts/dataset_builder/build_gsm8k_v1.py \\
        --input data/gsm8k/v1/solutions_merged.jsonl \\
        --output-dir data/gsm8k/v1 \\
        --samples-per-problem 30 \\
        --n-test 100 --seed 42

Notes
-----
- The output dir is `data/gsm8k/v1/` (NOT `data/gsm8k/with_solutions/`) so the
  existing v0 build at `data/gsm8k/with_solutions/full_response/...` is not
  touched in any way.
- Per-problem CSVs are named `gsm8k_test_<NUM>.csv` to match the v0 layout —
  task_id `gsm8k_test_42` becomes `gsm8k_test_42.csv`.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


# Hard-coded exclusions live here. Empty for v1 day-1; can be extended after the
# pool is built and we discover unsolvable_easy / unsolvable_hard problems.
EXCLUDED = set()

# Models excluded from the v1 build. gpt-4o passes ~96% of the test split, so
# its rows are almost all "easy correct" with little discriminator/generator
# signal — we still keep them in the pool but skip at sampling time. (Mirrors
# how humaneval-v1 keeps `intentional_bug` rows in the pool and filters at
# build time.)
EXCLUDED_MODELS = {"gpt-4o"}

# When sampling the per-(model, strategy) test set, only consider rows from
# this 4-model × 3-strategy "cycle" — the same matrix used to determine which
# 111 problems qualify with ≥2p ∧ ≥2f everywhere. Train is unrestricted (uses
# all available models/strategies for breadth) but test is locked to this
# 12-cell grid for clean per-cell analysis downstream.
CYCLE_TEST_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
]
CYCLE_TEST_STRATEGIES = ["normal", "beginner", "unusual"]


def load_solutions(path):
    """Load merged pool, dropping intentional_bug, excluded, and empty rows."""
    rows = []
    empty = 0
    excluded_model = 0
    malformed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if r.get('task_id') in EXCLUDED:
                continue
            if r.get('strategy') == 'intentional_bug':
                continue
            if r.get('model') in EXCLUDED_MODELS:
                excluded_model += 1
                continue
            if not (r.get('solution') or '').strip():
                empty += 1
                continue
            rows.append(r)
    if empty:
        print(f"  Skipped {empty} empty solutions")
    if excluded_model:
        print(f"  Skipped {excluded_model} rows from EXCLUDED_MODELS={sorted(EXCLUDED_MODELS)}")
    if malformed:
        print(f"  Skipped {malformed} malformed JSON lines (likely concurrent-write corruption)")
    return rows


def stratified_sample(pool, n, rng):
    """Sample n items maximizing (model, strategy) diversity via round-robin."""
    if len(pool) <= n:
        return pool[:]

    by_group = defaultdict(list)
    for r in pool:
        by_group[(r.get('model', ''), r.get('strategy', 'normal'))].append(r)

    selected = []
    group_keys = list(by_group.keys())
    rng.shuffle(group_keys)

    group_iters = {}
    for k in group_keys:
        items = by_group[k][:]
        rng.shuffle(items)
        group_iters[k] = iter(items)

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


def make_split(qualified_problems, rng, n_test):
    """Split qualified problems into train / test by deterministic shuffle.

    Sorts by integer suffix in task_id (gsm8k_test_<N>) for determinism, then
    shuffles with the seeded RNG, holds out the last n_test as the test set.
    """
    def numeric_key(qid):
        try:
            return int(qid.rsplit('_', 1)[-1])
        except ValueError:
            return 0
    problems = sorted(qualified_problems, key=numeric_key)
    rng.shuffle(problems)
    test_problems = set(problems[:n_test])
    train_problems = set(problems[n_test:])
    return train_problems, test_problems


def write_csv(rows, output_path):
    """Write rows in the canonical gsm8k v1 CSV schema (mirrors humaneval-v1)."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer', 'correct', 'strategy',
                         'model', 'temperature', 'task_id', 'error'])
        for r in rows:
            writer.writerow([
                r.get('question', ''),
                r.get('solution', ''),
                'Yes' if r.get('passed') else 'No',
                r.get('strategy', 'normal'),
                r.get('model', ''),
                r.get('temperature', ''),
                r.get('task_id', ''),
                r.get('error', '') or '',
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="solutions_merged.jsonl path")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (will be created); separate from data/gsm8k/with_solutions/")
    parser.add_argument("--samples-per-problem", type=int, default=30,
                        help="Target solutions per problem (used only in legacy random-split mode)")
    parser.add_argument("--n-test", type=int, default=100,
                        help="Number of test problems (legacy random-split mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pos", type=int, default=10,
                        help="Minimum pass count for problem to qualify (legacy random-split mode)")
    parser.add_argument("--min-neg", type=int, default=10,
                        help="Minimum fail count for problem to qualify (legacy random-split mode)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")

    # === explicit-list mode (added 2026-05-09) ===
    parser.add_argument("--test-ids-file", default=None,
                        help="Path to a file with one question_id per line. If given, this "
                             "replaces the random-split logic — the listed problems become the "
                             "test set, all OTHER pool problems become train.")
    parser.add_argument("--test-per-cell-pass", type=int, default=None,
                        help="Sample exactly this many pass rows per (model, strategy) cell "
                             "for each test problem. Required when --test-ids-file is set.")
    parser.add_argument("--test-per-cell-fail", type=int, default=None,
                        help="Sample exactly this many fail rows per (model, strategy) cell "
                             "for each test problem. Required when --test-ids-file is set.")
    parser.add_argument("--train-keep-all-balanced", action="store_true",
                        help="Train mode: keep all rows per problem but per-problem subsample "
                             "the majority class so the pass-ratio is within --train-balance-tol "
                             "of 0.5. Single-class problems (0 pass or 0 fail) keep all rows.")
    parser.add_argument("--train-balance-tol", type=float, default=0.05,
                        help="Pass-ratio tolerance around 0.5; default 0.05 → range [0.45, 0.55].")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_solutions(args.input)
    print(f"Loaded {len(rows)} solutions")

    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r['task_id']].append(r)

    train_rows = []
    test_rows_by_problem = {}
    stats = {'train_pass': 0, 'train_fail': 0, 'test_pass': 0, 'test_fail': 0}

    if args.test_ids_file is not None:
        # === Explicit-list mode (per-cell N pass + N fail for test, balanced train) ===
        if args.test_per_cell_pass is None or args.test_per_cell_fail is None:
            raise SystemExit("--test-per-cell-pass and --test-per-cell-fail are required with --test-ids-file")

        with open(args.test_ids_file) as f:
            test_problems = set(line.strip() for line in f if line.strip())
        print(f"\nLoaded {len(test_problems)} test problem ids from {args.test_ids_file}")

        train_problems = set(by_problem.keys()) - test_problems
        print(f"Train problems (rest of pool): {len(train_problems)}")

        # === Test sampling: per (model, strategy) cell, sample N_pass + N_fail.
        # Locked to the CYCLE_TEST_MODELS × CYCLE_TEST_STRATEGIES grid (12 cells). ===
        n_pass = args.test_per_cell_pass
        n_fail = args.test_per_cell_fail
        cycle_models_set = set(CYCLE_TEST_MODELS)
        cycle_strats_set = set(CYCLE_TEST_STRATEGIES)
        cycle_cells_n = len(CYCLE_TEST_MODELS) * len(CYCLE_TEST_STRATEGIES)
        underfilled_cells = 0
        missing_cells = 0
        for qid in sorted(test_problems):
            pool = by_problem.get(qid, [])
            if not pool:
                print(f"  WARNING: test qid {qid} has no rows in pool; skipping")
                continue
            # Filter pool to cycle models × cycle strategies only.
            cycle_pool = [r for r in pool
                          if r.get('model', '') in cycle_models_set
                          and r.get('strategy', 'normal') in cycle_strats_set]
            by_cell = defaultdict(lambda: {'pass': [], 'fail': []})
            for r in cycle_pool:
                key = (r.get('model', ''), r.get('strategy', 'normal'))
                by_cell[key]['pass' if r.get('passed') else 'fail'].append(r)

            # Walk the 12 expected cells in fixed order so missing cells are visible.
            sampled = []
            for m in CYCLE_TEST_MODELS:
                for s in CYCLE_TEST_STRATEGIES:
                    key = (m, s)
                    lists = by_cell.get(key, {'pass': [], 'fail': []})
                    p_avail = len(lists['pass']); f_avail = len(lists['fail'])
                    if p_avail == 0 and f_avail == 0:
                        missing_cells += 1
                        continue
                    p_take = min(n_pass, p_avail); f_take = min(n_fail, f_avail)
                    if p_take < n_pass or f_take < n_fail:
                        underfilled_cells += 1
                    pp = lists['pass'][:]; ff = lists['fail'][:]
                    rng.shuffle(pp); rng.shuffle(ff)
                    sampled.extend(pp[:p_take]); sampled.extend(ff[:f_take])
            rng.shuffle(sampled)
            test_rows_by_problem[qid] = sampled
            stats['test_pass'] += sum(1 for r in sampled if r.get('passed'))
            stats['test_fail'] += sum(1 for r in sampled if not r.get('passed'))
        print(f"  Test cycle: {len(CYCLE_TEST_MODELS)} models × {len(CYCLE_TEST_STRATEGIES)} strategies = {cycle_cells_n} cells per problem")
        print(f"  Expected per problem: {cycle_cells_n} cells × {n_pass+n_fail} rows = {cycle_cells_n*(n_pass+n_fail)} rows")
        if missing_cells:
            print(f"  NOTE: {missing_cells} (test_problem, cell) pairs were MISSING (no rows at all)")
        if underfilled_cells:
            print(f"  NOTE: {underfilled_cells} (test_problem, cell) pairs were underfilled "
                  f"(fewer than {n_pass} pass or {n_fail} fail available)")

        # === Train sampling: keep all rows per problem, balance per-problem ===
        # Train is also locked to the same CYCLE grid as test (4 models × 3 strategies)
        # so train and test have matching distributions.
        if not args.train_keep_all_balanced:
            print("WARNING: --test-ids-file given but --train-keep-all-balanced not set. "
                  "Train rows will be raw (unbalanced). To balance, pass --train-keep-all-balanced.")
        print(f"  Train cycle filter: {len(CYCLE_TEST_MODELS)} models × {len(CYCLE_TEST_STRATEGIES)} strategies (matches test)")
        tol = args.train_balance_tol
        single_class = balanced = subsampled = 0
        for qid in sorted(train_problems):
            pool = [r for r in by_problem[qid]
                    if r.get('model', '') in cycle_models_set
                    and r.get('strategy', 'normal') in cycle_strats_set]
            pass_pool = [r for r in pool if r.get('passed')]
            fail_pool = [r for r in pool if not r.get('passed')]
            np_, nf_ = len(pass_pool), len(fail_pool)
            if np_ == 0 or nf_ == 0:
                kept = pool[:]
                single_class += 1
            elif args.train_keep_all_balanced:
                # Subsample majority class so pass-ratio in [0.5-tol, 0.5+tol].
                ratio = np_ / (np_ + nf_)
                if ratio > 0.5 + tol:
                    # Too many passes: cap pass count so ratio = 0.5 + tol.
                    # target_pass / (target_pass + nf_) = 0.5 + tol
                    # target_pass = (0.5+tol) * nf_ / (0.5 - tol)
                    target_pass = int(round((0.5 + tol) * nf_ / (0.5 - tol)))
                    target_pass = max(target_pass, 1)
                    pp = pass_pool[:]; rng.shuffle(pp)
                    kept = pp[:target_pass] + fail_pool[:]
                    subsampled += 1
                elif ratio < 0.5 - tol:
                    target_fail = int(round((0.5 + tol) * np_ / (0.5 - tol)))
                    target_fail = max(target_fail, 1)
                    ff = fail_pool[:]; rng.shuffle(ff)
                    kept = pass_pool[:] + ff[:target_fail]
                    subsampled += 1
                else:
                    kept = pool[:]
                    balanced += 1
            else:
                kept = pool[:]
            rng.shuffle(kept)
            train_rows.extend(kept)
            stats['train_pass'] += sum(1 for r in kept if r.get('passed'))
            stats['train_fail'] += sum(1 for r in kept if not r.get('passed'))
        if args.train_keep_all_balanced:
            print(f"  Train per-problem: balanced={balanced}, subsampled-majority={subsampled}, single-class={single_class}")

    else:
        # === Legacy random-split mode ===
        # Filter to qualified problems (>=min-pos pass AND >=min-neg fail)
        qualified = []
        short_pos = 0
        short_neg = 0
        short_both = 0
        for qid, pool in by_problem.items():
            npass = sum(1 for r in pool if r.get('passed'))
            nfail = len(pool) - npass
            if npass < args.min_pos and nfail < args.min_neg:
                short_both += 1
            elif npass < args.min_pos:
                short_pos += 1
            elif nfail < args.min_neg:
                short_neg += 1
            else:
                qualified.append(qid)

        print(f"\nProblems with at least one row: {len(by_problem)}")
        print(f"  Qualified (>={args.min_pos}p AND >={args.min_neg}f): {len(qualified)}")
        print(f"  Short on pos only: {short_pos}")
        print(f"  Short on neg only: {short_neg}")
        print(f"  Short on both:     {short_both}")

        if args.n_test >= len(qualified):
            raise SystemExit(f"n-test ({args.n_test}) >= qualified count ({len(qualified)}); aborting")

        train_problems, test_problems = make_split(qualified, rng, args.n_test)
        print(f"Train problems: {len(train_problems)}, Test problems: {len(test_problems)}")

        n = args.samples_per_problem
        half = n // 2

        for qid in qualified:
            pool = by_problem[qid]
            pass_pool = [r for r in pool if r.get('passed')]
            fail_pool = [r for r in pool if not r.get('passed')]

            n_pass = min(half, len(pass_pool))
            n_fail = min(half, len(fail_pool))
            if n_pass < half and len(fail_pool) > half:
                n_fail = min(n - n_pass, len(fail_pool))
            elif n_fail < half and len(pass_pool) > half:
                n_pass = min(n - n_fail, len(pass_pool))

            sampled_pass = stratified_sample(pass_pool, n_pass, rng)
            sampled_fail = stratified_sample(fail_pool, n_fail, rng)
            sampled = sampled_pass + sampled_fail
            rng.shuffle(sampled)

            if qid in train_problems:
                train_rows.extend(sampled)
                stats['train_pass'] += len(sampled_pass)
                stats['train_fail'] += len(sampled_fail)
            else:
                test_rows_by_problem[qid] = sampled
                stats['test_pass'] += len(sampled_pass)
                stats['test_fail'] += len(sampled_fail)

    print(f"\nSampling results:")
    print(f"  Train: {len(train_rows)} rows ({stats['train_pass']}p/{stats['train_fail']}f)")
    total_test = sum(len(v) for v in test_rows_by_problem.values())
    print(f"  Test:  {total_test} rows across {len(test_rows_by_problem)} problems "
          f"({stats['test_pass']}p/{stats['test_fail']}f)")

    train_models = set(r.get('model', '') for r in train_rows)
    train_strats = set(r.get('strategy', 'normal') for r in train_rows)
    print(f"  Train models: {len(train_models)}, strategies: {len(train_strats)}")

    if args.dry_run:
        print("\n[DRY RUN] Not writing files.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup existing train.csv if present
    train_path = output_dir / "train.csv"
    if train_path.exists():
        backup = output_dir / "train_old.csv"
        train_path.rename(backup)
        print(f"  Backed up existing train.csv to train_old.csv")

    rng.shuffle(train_rows)
    write_csv(train_rows, train_path)
    print(f"  Wrote {len(train_rows)} rows to {train_path}")

    # Remove old per-problem test CSVs
    for f in output_dir.glob("gsm8k_test_*.csv"):
        f.unlink()

    for qid, test_rows in sorted(test_rows_by_problem.items()):
        # qid like 'gsm8k_test_42' → file 'gsm8k_test_42.csv'
        test_path = output_dir / f"{qid}.csv"
        write_csv(test_rows, test_path)

    print(f"  Wrote {len(test_rows_by_problem)} test problem CSVs")

    print(f"\n=== GSM8K-V1 DATASET ===")
    print(f"Train: {len(train_rows)} solutions across {len(train_problems)} problems")
    print(f"Test:  {total_test} solutions across {len(test_rows_by_problem)} problems")
    print(f"Samples/problem target: {args.samples_per_problem}")
    print(f"Models in train: {sorted(train_models)}")
    print(f"Strategies in train: {sorted(train_strats)}")


if __name__ == "__main__":
    main()
