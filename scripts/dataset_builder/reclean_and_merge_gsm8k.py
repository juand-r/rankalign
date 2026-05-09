#!/usr/bin/env python3
"""
Re-clean and dedup the gsm8k-v1 solutions pool. Mirrors humaneval-v1's
reclean_and_merge.py but for the gsm8k canonical schema.

- Reads `data/gsm8k/v1/solutions.jsonl` (append-only pool produced by
  generate_solutions_parallel_gsm8k.py).
- For each row: re-runs `clean_solution()` on `raw_solution`. If the cleaned
  text differs, re-validates against the gold answer (cheap; pure-Python regex,
  no subprocess).
- Drops rows with empty `solution`.
- Drops `intentional_bug` strategy at the merge stage (defense in depth — v1
  is not supposed to have any, but we filter just in case).
- Dedup by (task_id, model, strategy, md5(solution)).
- Writes `data/gsm8k/v1/solutions_merged.jsonl`.

Usage:
    .tools-venv/bin/python scripts/dataset_builder/reclean_and_merge_gsm8k.py
"""

import json
import sys
import hashlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from solution_generator.gsm8k_config import GSM8KConfig


def reclean_file(path, task_config, problems_by_id):
    """Re-clean solutions from a file. Returns list of records, normalized to canonical schema."""
    records = []
    recleaned = 0
    revalidated = 0
    flipped = 0
    skipped_no_qid = 0

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            qid = r.get('task_id') or r.get('question_id', '')
            if not qid:
                skipped_no_qid += 1
                continue

            raw = r.get('raw_solution') or r.get('raw_response') or ''
            old_solution = r.get('solution') or r.get('response') or ''

            if raw:
                new_solution = task_config.clean_solution(raw)
                if new_solution != old_solution:
                    recleaned += 1
                    old_passed = r.get('passed', r.get('correct', False))
                    problem = problems_by_id.get(qid)
                    if problem:
                        passed, extracted, error = task_config.validate(problem, new_solution)
                        revalidated += 1
                        if passed != old_passed:
                            flipped += 1
                        r['solution'] = new_solution
                        r['passed'] = passed
                        r['extracted_answer'] = extracted
                        r['error'] = error
                    else:
                        r['solution'] = new_solution

            # Normalize to canonical fields used by build_gsm8k_v1.py
            if 'task_id' not in r and 'question_id' in r:
                r['task_id'] = r['question_id']
            if 'solution' not in r and 'response' in r:
                r['solution'] = r['response']
            if 'passed' not in r and 'correct' in r:
                r['passed'] = bool(r['correct'])

            records.append(r)

            if (i + 1) % 5000 == 0:
                print(f"  [{path.name}] Processed {i+1} rows, {recleaned} re-cleaned, {flipped} flipped")

    print(f"  [{path.name}] Done: {len(records)} records, {recleaned} re-cleaned, "
          f"{revalidated} re-validated, {flipped} flipped, {skipped_no_qid} skipped (no qid)")
    return records


def filter_empty_solutions(records):
    valid = [r for r in records if (r.get('solution') or '').strip()]
    removed = len(records) - len(valid)
    if removed:
        print(f"  Removed {removed} empty solutions")
    return valid


def filter_intentional_bug(records):
    valid = [r for r in records if r.get('strategy') != 'intentional_bug']
    removed = len(records) - len(valid)
    if removed:
        print(f"  Removed {removed} intentional_bug rows (excluded from v1)")
    return valid


def dedup_records(records):
    seen = set()
    deduped = []
    dupes = 0
    for r in records:
        key = (
            r.get('task_id', ''),
            r.get('model', ''),
            r.get('strategy', 'normal'),
            hashlib.md5(r.get('solution', '').encode()).hexdigest(),
        )
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        deduped.append(r)
    print(f"  Dedup: {len(records)} -> {len(deduped)} ({dupes} duplicates removed)")
    return deduped


def main():
    data_dir = Path("data/gsm8k")
    v1_dir = data_dir / "v1"
    pool_path = v1_dir / "solutions.jsonl"
    output_path = v1_dir / "solutions_merged.jsonl"
    problems_path = data_dir / "gsm8k_test_problems.jsonl"

    if not pool_path.exists():
        print(f"ERROR: pool not found at {pool_path}")
        sys.exit(1)
    if not problems_path.exists():
        print(f"ERROR: problems file not found at {problems_path}")
        sys.exit(1)

    problems_by_id = {}
    with open(problems_path) as f:
        for line in f:
            p = json.loads(line)
            qid = p.get('question_id') or p.get('task_id', '')
            problems_by_id[qid] = p
    print(f"Loaded {len(problems_by_id)} problems")

    task_config = GSM8KConfig()

    print(f"\nRe-cleaning pool ({pool_path})...")
    records = reclean_file(pool_path, task_config, problems_by_id)

    records = filter_intentional_bug(records)
    records = filter_empty_solutions(records)
    records = dedup_records(records)

    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f"\nWrote {len(records)} records to {output_path}")

    per_problem = defaultdict(lambda: {'pass': 0, 'fail': 0, 'models': set(), 'strategies': set()})
    for r in records:
        qid = r.get('task_id', '')
        per_problem[qid]['pass' if r.get('passed') else 'fail'] += 1
        per_problem[qid]['models'].add(r.get('model', ''))
        per_problem[qid]['strategies'].add(r.get('strategy', ''))

    qualified = sum(1 for v in per_problem.values() if v['pass'] >= 10 and v['fail'] >= 10)
    print(f"\nProblems with at least one row: {len(per_problem)}")
    print(f"Problems qualified (>=10p AND >=10f): {qualified}")
    pass_counts = sorted(v['pass'] for v in per_problem.values())
    fail_counts = sorted(v['fail'] for v in per_problem.values())
    if pass_counts:
        print(f"Pass per problem: min={pass_counts[0]} median={pass_counts[len(pass_counts)//2]} max={pass_counts[-1]}")
    if fail_counts:
        print(f"Fail per problem: min={fail_counts[0]} median={fail_counts[len(fail_counts)//2]} max={fail_counts[-1]}")


if __name__ == "__main__":
    main()
