"""Explore the CodeContests dataset from HuggingFace.

Related files:
    scripts/preprocess_codecontests.py  - generates data/codecontests/ from HF
    src/tasks/codecontests.py           - task module (registers codecontests-*)
    data/codecontests/split_manifest.json - maps eval slugs to "test" vs "valid"
"""

import sys
from collections import Counter, defaultdict
from datasets import load_dataset

print("Loading CodeContests dataset (this may take a while on first download)...")
ds = load_dataset("deepmind/code_contests")

print(f"\nSplits: {list(ds.keys())}")
for split_name, split_data in ds.items():
    print(f"  {split_name}: {len(split_data)} problems")

print("\n--- Column names ---")
print(ds['train'].column_names)

print("\n--- First train example (keys + types) ---")
ex = ds['train'][0]
for k, v in ex.items():
    if isinstance(v, (list, dict)):
        print(f"  {k}: {type(v).__name__}, len={len(v)}")
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, list):
                    print(f"    {kk}: list, len={len(vv)}")
                else:
                    print(f"    {kk}: {type(vv).__name__}")
        elif isinstance(v, list) and len(v) > 0:
            print(f"    first element type: {type(v[0]).__name__}")
            if isinstance(v[0], str) and len(v[0]) < 200:
                print(f"    first element: {v[0][:200]}")
    elif isinstance(v, str):
        print(f"  {k}: str, len={len(v)}, preview={v[:120]}...")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

print("\n--- Solutions structure ---")
sols = ex.get('solutions', {})
print(f"  solutions type: {type(sols)}")
if isinstance(sols, dict):
    for k, v in sols.items():
        print(f"    {k}: {type(v).__name__}, len={len(v) if isinstance(v, list) else 'N/A'}")
        if isinstance(v, list) and len(v) > 0:
            print(f"      first: {type(v[0]).__name__}", end="")
            if isinstance(v[0], str):
                print(f", len={len(v[0])}")
            elif isinstance(v[0], (int, float)):
                print(f", val={v[0]}")
            else:
                print()

print("\n--- Incorrect solutions structure ---")
isols = ex.get('incorrect_solutions', {})
print(f"  incorrect_solutions type: {type(isols)}")
if isinstance(isols, dict):
    for k, v in isols.items():
        print(f"    {k}: {type(v).__name__}, len={len(v) if isinstance(v, list) else 'N/A'}")

print("\n\n=== STATS ACROSS SPLITS ===")
for split_name in ds.keys():
    split_data = ds[split_name]
    n_correct = []
    n_incorrect = []
    desc_lens = []
    languages_counter = Counter()
    difficulty_counter = Counter()
    problems_with_incorrect = 0

    for i, ex in enumerate(split_data):
        sols = ex.get('solutions', {})
        isols = ex.get('incorrect_solutions', {})

        sol_list = sols.get('solution', []) if isinstance(sols, dict) else []
        isol_list = isols.get('solution', []) if isinstance(isols, dict) else []
        n_correct.append(len(sol_list))
        n_incorrect.append(len(isol_list))
        if len(isol_list) > 0:
            problems_with_incorrect += 1

        desc = ex.get('description', '')
        desc_lens.append(len(desc))

        if isinstance(sols, dict) and 'language' in sols:
            for lang in sols['language']:
                languages_counter[lang] += 1

        diff = ex.get('difficulty', None)
        if diff is not None:
            difficulty_counter[diff] += 1

    total = len(split_data)
    print(f"\n--- {split_name} ({total} problems) ---")
    print(f"  Correct solutions: min={min(n_correct)}, max={max(n_correct)}, "
          f"mean={sum(n_correct)/total:.1f}, median={sorted(n_correct)[total//2]}")
    print(f"  Incorrect solutions: min={min(n_incorrect)}, max={max(n_incorrect)}, "
          f"mean={sum(n_incorrect)/total:.1f}, median={sorted(n_incorrect)[total//2]}")
    print(f"  Problems with >=1 incorrect: {problems_with_incorrect}/{total} ({100*problems_with_incorrect/total:.1f}%)")
    print(f"  Description lengths (chars): min={min(desc_lens)}, max={max(desc_lens)}, "
          f"mean={sum(desc_lens)/total:.1f}")
    print(f"  Top languages: {languages_counter.most_common(6)}")
    print(f"  Difficulty distribution: {difficulty_counter.most_common(10)}")

print("\n\n=== EXAMPLE PROBLEM (test split, first) ===")
test_ex = ds['test'][0]
print(f"Name: {test_ex.get('name', 'N/A')}")
print(f"Difficulty: {test_ex.get('difficulty', 'N/A')}")
print(f"Source: {test_ex.get('source', 'N/A')}")
desc = test_ex.get('description', '')
print(f"Description ({len(desc)} chars):\n{desc[:800]}")
sols = test_ex.get('solutions', {})
sol_list = sols.get('solution', []) if isinstance(sols, dict) else []
sol_langs = sols.get('language', []) if isinstance(sols, dict) else []
print(f"\nCorrect solutions: {len(sol_list)}")
if sol_list:
    print(f"  First solution language: {sol_langs[0] if sol_langs else '?'}")
    print(f"  First solution ({len(sol_list[0])} chars):\n{sol_list[0][:500]}")

isols = test_ex.get('incorrect_solutions', {})
isol_list = isols.get('solution', []) if isinstance(isols, dict) else []
print(f"\nIncorrect solutions: {len(isol_list)}")
if isol_list:
    print(f"  First incorrect ({len(isol_list[0])} chars):\n{isol_list[0][:500]}")

# Show a few problem names for slug generation ideas
print("\n\n=== PROBLEM NAMES (test split) ===")
for i, ex in enumerate(ds['test'][:20]):
    name = ex.get('name', f'problem_{i}')
    n_sol = len(ex.get('solutions', {}).get('solution', []))
    n_isol = len(ex.get('incorrect_solutions', {}).get('solution', []))
    print(f"  {i}: {name} (correct={n_sol}, incorrect={n_isol})")
