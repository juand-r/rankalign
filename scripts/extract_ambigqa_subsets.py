#!/usr/bin/env python3
"""
Extract subsets of AmbigQA where all annotators agree the question is ambiguous.

Creates two JSONL files:
- train: questions with >= MIN_ANSWERS_TRAIN unique answers
- test: questions with >= MIN_ANSWERS_TEST unique answers

Each line is: {"question": "...", "answers": ["...", ...]}
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MIN_ANSWERS_TRAIN = 10
MIN_ANSWERS_TEST = 20


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def flatten_answers(answer_field):
    """Recursively flatten nested answer lists."""
    if answer_field is None:
        return []
    if isinstance(answer_field, str):
        return [answer_field]
    if isinstance(answer_field, list):
        out = []
        for a in answer_field:
            out.extend(flatten_answers(a))
        return out
    return []


def get_unique_answers(row):
    """Extract unique answers (case-folded for dedup, but keep original casing)."""
    qa_pairs = (row.get("annotations") or {}).get("qaPairs") or []
    answers = []
    for qa in (qa_pairs if isinstance(qa_pairs, list) else []):
        if isinstance(qa, dict):
            answers.extend(flatten_answers(qa.get("answer") or qa.get("answers")))
    if not answers:
        answers = flatten_answers(row.get("nq_answer"))
    
    # Deduplicate by lowercase, keep first occurrence's casing
    seen = {}
    for a in answers:
        if isinstance(a, str) and a.strip():
            key = a.strip().lower()
            if key not in seen:
                seen[key] = a.strip()
    return list(seen.values())


def is_agreed_ambiguous(row):
    """Return True if all annotators agree this is multipleQAs (ambiguous)."""
    ann = row.get("annotations") or {}
    types = ann.get("type") or []
    if not isinstance(types, list):
        types = [types]
    return len(types) > 0 and all(t == "multipleQAs" for t in types)


def extract_subset(dataset, min_answers):
    """Extract questions with >= min_answers unique answers."""
    results = []
    for row in dataset:
        if not is_agreed_ambiguous(row):
            continue
        answers = get_unique_answers(row)
        if len(answers) >= min_answers:
            results.append({
                "question": row["question"],
                "answers": answers,
            })
    return results


def save_jsonl(data, path):
    """Save list of dicts to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples to {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Extract AmbigQA subsets")
    parser.add_argument("--dataset", default="sewon/ambig_qa")
    parser.add_argument("--config", default="full")
    parser.add_argument("--output-dir", default="data/ambigqa")
    parser.add_argument("--min-train", type=int, default=MIN_ANSWERS_TRAIN,
                        help=f"Min unique answers for train (default: {MIN_ANSWERS_TRAIN})")
    parser.add_argument("--min-test", type=int, default=MIN_ANSWERS_TEST,
                        help=f"Min unique answers for test (default: {MIN_ANSWERS_TEST})")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.config})...")
    ds = load_dataset(args.dataset, args.config)

    # Extract from train split
    print(f"\nExtracting train examples (>= {args.min_train} answers)...")
    train_data = extract_subset(ds["train"], args.min_train)

    # Extract from validation split (used as test)
    print(f"Extracting test examples (>= {args.min_test} answers)...")
    test_data = extract_subset(ds["validation"], args.min_test)

    # Save
    output_dir = Path(args.output_dir)
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")

    print(f"\nDone! Train: {len(train_data)}, Test: {len(test_data)}")


if __name__ == "__main__":
    main()
