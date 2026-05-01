"""
Preprocess the CodeContests dataset (deepmind/code_contests) into local files
for the task registry.

Output layout:
    data/codecontests/
        descriptions.json       - {problem_name: {description, difficulty}} for all splits
        train.jsonl             - compact train items (no description; joined at load time)
        split_manifest.json     - which eval slugs are "test" vs "valid" (see below)
        problems_train.jsonl    - per-problem metadata for train
        problems_test.jsonl     - per-problem metadata for test
        problems_valid.jsonl    - per-problem metadata for valid
        test/<slug>.jsonl       - self-contained per-problem test/valid items

split_manifest.json maps each slug to its origin split so that downstream code
(src/tasks/codecontests.py, eval scripts) can distinguish TEST from VALID eval
tasks.  Format: {"test": ["1575a", ...], "valid": ["1548c", ...]}

Train items (compact, ~2MB):
    {"problem_name": "...", "solution": "...", "correct": "Yes"/"No", "language": 2}

Test items (self-contained, small files):
    {"problem_name": "...", "description": "...", "solution": "...", "correct": "Yes"/"No", "language": 2}

Language codes (protobuf enum):
    0 = UNKNOWN_LANGUAGE, 1 = PYTHON2, 2 = CPP, 3 = PYTHON3, 4 = JAVA
"""

import json
import os
import re
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

LANGUAGE_NAMES = {0: "unknown", 1: "python2", 2: "cpp", 3: "python3", 4: "java"}

MAX_SOLUTION_CHARS = 3000
MAX_DESCRIPTION_CHARS = 6000
TRAIN_MAX_CORRECT = 20
TRAIN_MAX_INCORRECT = 20
TEST_MAX_CORRECT = 20
TEST_MAX_INCORRECT = 20
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "codecontests"
TEST_DIR = OUTPUT_DIR / "test"


def problem_name_to_slug(name: str) -> str:
    """Convert '1575_A. Another Sorting Problem' -> '1575a'."""
    m = re.match(r'(\d+)_([A-Z]\d?)', name)
    if m:
        return m.group(1) + m.group(2).lower()
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def filter_solutions(codes, langs, max_count, rng):
    """Filter solutions by length, shuffle, and cap."""
    candidates = []
    for code, lang in zip(codes, langs):
        code = code.strip()
        if code and len(code) <= MAX_SOLUTION_CHARS:
            candidates.append((code, lang))
    rng.shuffle(candidates)
    return candidates[:max_count]


def main():
    print("Loading CodeContests from HuggingFace...")
    ds = load_dataset("deepmind/code_contests")
    rng = random.Random(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Collect all descriptions across splits ----
    all_descriptions = {}

    for split_name in ['train', 'test', 'valid']:
        for ex in ds[split_name]:
            desc = ex['description'].strip()
            if desc and len(desc) <= MAX_DESCRIPTION_CHARS:
                all_descriptions[ex['name']] = {
                    'description': desc,
                    'difficulty': ex.get('difficulty', -1),
                }

    desc_path = OUTPUT_DIR / "descriptions.json"
    with open(desc_path, 'w') as f:
        json.dump(all_descriptions, f, ensure_ascii=False)
    print(f"Descriptions: {len(all_descriptions)} problems -> {desc_path} ({desc_path.stat().st_size / 1e6:.1f} MB)")

    # ---- TRAIN SPLIT (compact: no description in items) ----
    print("\n=== Processing TRAIN split ===")
    train_items = []
    train_problems_meta = []
    skipped = 0

    for ex in ds['train']:
        name = ex['name']
        if name not in all_descriptions:
            skipped += 1
            continue

        sols = ex.get('solutions', {})
        isols = ex.get('incorrect_solutions', {})

        correct = filter_solutions(
            sols.get('solution', []), sols.get('language', []),
            TRAIN_MAX_CORRECT, rng
        )
        incorrect = filter_solutions(
            isols.get('solution', []), isols.get('language', []),
            TRAIN_MAX_INCORRECT, rng
        )

        if not correct and not incorrect:
            skipped += 1
            continue

        for code, lang in correct:
            train_items.append({
                'problem_name': name,
                'solution': code,
                'correct': 'Yes',
                'language': lang,
            })
        for code, lang in incorrect:
            train_items.append({
                'problem_name': name,
                'solution': code,
                'correct': 'No',
                'language': lang,
            })

        train_problems_meta.append({
            'problem_name': name,
            'slug': problem_name_to_slug(name),
            'n_correct': len(correct),
            'n_incorrect': len(incorrect),
            'difficulty': all_descriptions[name]['difficulty'],
        })

    train_path = OUTPUT_DIR / "train.jsonl"
    with open(train_path, 'w') as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(OUTPUT_DIR / "problems_train.jsonl", 'w') as f:
        for m in train_problems_meta:
            f.write(json.dumps(m) + '\n')

    n_correct = sum(1 for it in train_items if it['correct'] == 'Yes')
    n_incorrect = sum(1 for it in train_items if it['correct'] == 'No')
    n_both = sum(1 for m in train_problems_meta if m['n_correct'] > 0 and m['n_incorrect'] > 0)
    print(f"  Problems: {len(train_problems_meta)}/{len(ds['train'])} (skipped {skipped})")
    print(f"  Problems with BOTH correct+incorrect: {n_both}")
    print(f"  Items: {len(train_items)} ({n_correct} correct, {n_incorrect} incorrect)")
    print(f"  File: {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)")

    lang_counts = Counter(it['language'] for it in train_items)
    print(f"  Languages: {', '.join(f'{LANGUAGE_NAMES.get(k,k)}={v}' for k,v in lang_counts.most_common())}")

    sol_lens = sorted(len(it['solution']) for it in train_items)
    print(f"  Solution chars: p50={sol_lens[len(sol_lens)//2]}, "
          f"p90={sol_lens[int(len(sol_lens)*0.9)]}, max={sol_lens[-1]}")

    # ---- TEST + VALID SPLITS (self-contained items with description) ----
    slug_counts = Counter()

    for split_name, meta_filename, max_c, max_i in [
        ('test', 'problems_test.jsonl', TEST_MAX_CORRECT, TEST_MAX_INCORRECT),
        ('valid', 'problems_valid.jsonl', TEST_MAX_CORRECT, TEST_MAX_INCORRECT),
    ]:
        print(f"\n=== Processing {split_name.upper()} split ===")
        problems_meta = []
        total_items = 0

        for ex in ds[split_name]:
            name = ex['name']
            if name not in all_descriptions:
                print(f"  Skipping {name} (no description)")
                continue

            sols = ex.get('solutions', {})
            isols = ex.get('incorrect_solutions', {})

            correct = filter_solutions(
                sols.get('solution', []), sols.get('language', []), max_c, rng)
            incorrect = filter_solutions(
                isols.get('solution', []), isols.get('language', []), max_i, rng)

            if not correct and not incorrect:
                print(f"  Skipping {name} (no usable solutions)")
                continue

            slug = problem_name_to_slug(name)
            if slug in slug_counts:
                slug_counts[slug] += 1
                slug = f"{slug}-{slug_counts[slug]}"
            else:
                slug_counts[slug] = 1

            items = []
            desc = all_descriptions[name]['description']
            for code, lang in correct:
                items.append({
                    'problem_name': name,
                    'description': desc,
                    'solution': code,
                    'correct': 'Yes',
                    'language': lang,
                })
            for code, lang in incorrect:
                items.append({
                    'problem_name': name,
                    'description': desc,
                    'solution': code,
                    'correct': 'No',
                    'language': lang,
                })

            out_path = TEST_DIR / f"{slug}.jsonl"
            with open(out_path, 'w') as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            problems_meta.append({
                'problem_name': name,
                'slug': slug,
                'n_correct': len(correct),
                'n_incorrect': len(incorrect),
                'n_items': len(items),
                'difficulty': all_descriptions[name]['difficulty'],
            })
            total_items += len(items)

        with open(OUTPUT_DIR / meta_filename, 'w') as f:
            for m in problems_meta:
                f.write(json.dumps(m) + '\n')

        items_per = [m['n_items'] for m in problems_meta]
        print(f"  Problems: {len(problems_meta)}/{len(ds[split_name])}")
        print(f"  Items: {total_items} (min={min(items_per)}, mean={sum(items_per)/len(items_per):.1f}, max={max(items_per)} per problem)")

    # ---- SUMMARY ----
    total_test_files = len(list(TEST_DIR.glob("*.jsonl")))
    total_size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\n=== SUMMARY ===")
    print(f"  {desc_path.name}: {desc_path.stat().st_size / 1e6:.1f} MB")
    print(f"  {train_path.name}: {train_path.stat().st_size / 1e6:.1f} MB ({len(train_items)} items)")
    print(f"  test/ : {total_test_files} files")
    print(f"  Total: {total_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
