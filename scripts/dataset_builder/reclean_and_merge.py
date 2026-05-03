#!/usr/bin/env python3
"""
Re-clean all solutions using updated clean_solution, re-validate changed ones,
merge spark + local, and deduplicate.

Usage:
    .tools-venv/bin/python scripts/dataset_builder/reclean_and_merge.py
"""

import json
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from solution_generator.humaneval_config import HumanEvalConfig

EXCLUDED = {'HumanEval/53', 'HumanEval/145'}


def reclean_file(path, task_config, problems_by_id):
    """Re-clean solutions from a file. Returns list of records."""
    records = []
    recleaned = 0
    revalidated = 0
    flipped = 0

    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r['task_id'] in EXCLUDED:
                continue

            raw = r.get('raw_solution', '')
            if not raw:
                records.append(r)
                continue

            new_solution = task_config.clean_solution(raw)
            if new_solution != r['solution']:
                recleaned += 1
                old_passed = r['passed']
                # Re-validate
                problem = problems_by_id.get(r['task_id'])
                if problem:
                    passed, _, error = task_config.validate(problem, new_solution)
                    revalidated += 1
                    if passed != old_passed:
                        flipped += 1
                    r['solution'] = new_solution
                    r['passed'] = passed
                    r['error'] = error
                else:
                    r['solution'] = new_solution

            records.append(r)

            if (i + 1) % 5000 == 0:
                print(f"  [{path.name}] Processed {i+1} rows, {recleaned} re-cleaned, {flipped} flipped")

    print(f"  [{path.name}] Done: {len(records)} records, {recleaned} re-cleaned, "
          f"{revalidated} re-validated, {flipped} flipped")
    return records


def filter_empty_solutions(records):
    """Remove records with empty solutions (API failures, empty model responses)."""
    valid = [r for r in records if r.get('solution', '').strip()]
    removed = len(records) - len(valid)
    if removed:
        print(f"  Removed {removed} empty solutions")
    return valid


def dedup_records(records):
    """Deduplicate by (task_id, model, strategy, solution hash)."""
    seen = set()
    deduped = []
    dupes = 0
    for r in records:
        key = (
            r['task_id'],
            r['model'],
            r.get('strategy', 'normal'),
            hashlib.md5(r['solution'].encode()).hexdigest(),
        )
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        deduped.append(r)
    print(f"  Dedup: {len(records)} -> {len(deduped)} ({dupes} duplicates removed)")
    return deduped


def main():
    data_dir = Path("data/humaneval")
    local_path = data_dir / "solutions.jsonl"
    spark_path = data_dir / "solutions_spark_raw.jsonl"
    output_path = data_dir / "solutions_merged.jsonl"

    # Load problems
    problems_by_id = {}
    with open(data_dir / "problems.jsonl") as f:
        for line in f:
            p = json.loads(line)
            problems_by_id[p['task_id']] = p
    print(f"Loaded {len(problems_by_id)} problems")

    task_config = HumanEvalConfig()

    # Re-clean both files
    print(f"\nRe-cleaning local ({local_path})...")
    local_records = reclean_file(local_path, task_config, problems_by_id)

    print(f"\nRe-cleaning spark ({spark_path})...")
    spark_records = reclean_file(spark_path, task_config, problems_by_id)

    # Merge
    all_records = local_records + spark_records
    print(f"\nMerged: {len(local_records)} local + {len(spark_records)} spark = {len(all_records)} total")

    # Remove empty solutions
    all_records = filter_empty_solutions(all_records)

    # Deduplicate
    all_records = dedup_records(all_records)

    # Write output
    with open(output_path, 'w') as f:
        for r in all_records:
            f.write(json.dumps(r) + '\n')
    print(f"\nWrote {len(all_records)} records to {output_path}")


if __name__ == "__main__":
    main()
