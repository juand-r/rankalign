#!/usr/bin/env python3
"""
Preprocess hypernym CSV files:
1. Lowercase all predicted_hypernym values
2. Remove duplicates within each file (keeping first occurrence)
3. For overlapping hypernyms between train and test: REMOVE from train, KEEP in test
   (to avoid train->test leakage)

Output files are saved to data/lowercase-deduped-hypernyms/
"""

import argparse
import csv
import os
import re
from pathlib import Path
from collections import defaultdict


def get_noun_from_filename(filename: str) -> str:
    """Extract noun from filename like hypernym_elephants_google-gemma-2-2b_train.csv"""
    match = re.match(r'hypernym_(.+)_google-gemma-2-2b_(train|test)\.csv', filename)
    if match:
        return match.group(1)
    return None


def get_split_from_filename(filename: str) -> str:
    """Extract split (train/test) from filename"""
    match = re.match(r'hypernym_(.+)_google-gemma-2-2b_(train|test)\.csv', filename)
    if match:
        return match.group(2)
    return None


def load_csv(filepath: str) -> tuple[list[str], list[dict]]:
    """Load CSV and return (fieldnames, rows)"""
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def save_csv(filepath: str, fieldnames: list[str], rows: list[dict]):
    """Save rows to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_noun(noun: str, data_dir: str, output_dir: str, dry_run: bool = False):
    """Process train and test files for a single noun"""
    train_file = os.path.join(data_dir, f'hypernym_{noun}_google-gemma-2-2b_train.csv')
    test_file = os.path.join(data_dir, f'hypernym_{noun}_google-gemma-2-2b_test.csv')
    
    if not os.path.exists(train_file):
        print(f"  WARNING: Train file not found: {train_file}")
        return None
    if not os.path.exists(test_file):
        print(f"  WARNING: Test file not found: {test_file}")
        return None
    
    # Load both files
    train_fieldnames, train_rows = load_csv(train_file)
    test_fieldnames, test_rows = load_csv(test_file)
    
    # Lowercase hypernyms and collect unique ones
    # For test: dedupe and lowercase
    test_seen = set()
    test_deduped = []
    for row in test_rows:
        hyp_lower = row['predicted_hypernym'].lower()
        if hyp_lower not in test_seen:
            test_seen.add(hyp_lower)
            row['predicted_hypernym'] = hyp_lower
            test_deduped.append(row)
    
    # For train: dedupe, lowercase, AND remove any that overlap with test
    train_seen = set()
    train_deduped = []
    overlap_removed = 0
    for row in train_rows:
        hyp_lower = row['predicted_hypernym'].lower()
        # Skip if this hypernym is in test set (to avoid leakage)
        if hyp_lower in test_seen:
            overlap_removed += 1
            continue
        # Skip if already seen in train
        if hyp_lower not in train_seen:
            train_seen.add(hyp_lower)
            row['predicted_hypernym'] = hyp_lower
            train_deduped.append(row)
    
    stats = {
        'noun': noun,
        'train_original': len(train_rows),
        'train_after_dedupe': len(train_deduped),
        'train_duplicates_removed': len(train_rows) - len(train_deduped) - overlap_removed,
        'train_overlap_removed': overlap_removed,
        'test_original': len(test_rows),
        'test_after_dedupe': len(test_deduped),
        'test_duplicates_removed': len(test_rows) - len(test_deduped),
    }
    
    if not dry_run:
        # Save processed files
        train_output = os.path.join(output_dir, f'hypernym_{noun}_google-gemma-2-2b_train.csv')
        test_output = os.path.join(output_dir, f'hypernym_{noun}_google-gemma-2-2b_test.csv')
        save_csv(train_output, train_fieldnames, train_deduped)
        save_csv(test_output, test_fieldnames, test_deduped)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Preprocess hypernym CSV files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without writing files')
    parser.add_argument('--data-dir', default='data', help='Input data directory')
    parser.add_argument('--output-dir', default='data/lowercase-deduped-hypernyms', help='Output directory')
    args = parser.parse_args()
    
    # Find all unique nouns
    nouns = set()
    for filename in os.listdir(args.data_dir):
        noun = get_noun_from_filename(filename)
        if noun:
            nouns.add(noun)
    
    nouns = sorted(nouns)
    print(f"Found {len(nouns)} nouns to process")
    print(f"Output directory: {args.output_dir}")
    if args.dry_run:
        print("DRY RUN - no files will be written")
    print()
    
    # Process each noun
    total_stats = {
        'train_original': 0,
        'train_after_dedupe': 0,
        'train_duplicates_removed': 0,
        'train_overlap_removed': 0,
        'test_original': 0,
        'test_after_dedupe': 0,
        'test_duplicates_removed': 0,
    }
    
    for noun in nouns:
        print(f"Processing: {noun}")
        stats = process_noun(noun, args.data_dir, args.output_dir, args.dry_run)
        if stats:
            print(f"  Train: {stats['train_original']} -> {stats['train_after_dedupe']} "
                  f"(removed {stats['train_duplicates_removed']} duplicates, "
                  f"{stats['train_overlap_removed']} overlaps with test)")
            print(f"  Test:  {stats['test_original']} -> {stats['test_after_dedupe']} "
                  f"(removed {stats['test_duplicates_removed']} duplicates)")
            
            for key in total_stats:
                total_stats[key] += stats[key]
    
    print()
    print("=" * 60)
    print("TOTALS:")
    print(f"  Train: {total_stats['train_original']} -> {total_stats['train_after_dedupe']} "
          f"(removed {total_stats['train_duplicates_removed']} duplicates, "
          f"{total_stats['train_overlap_removed']} overlaps with test)")
    print(f"  Test:  {total_stats['test_original']} -> {total_stats['test_after_dedupe']} "
          f"(removed {total_stats['test_duplicates_removed']} duplicates)")
    print("=" * 60)


if __name__ == '__main__':
    main()

