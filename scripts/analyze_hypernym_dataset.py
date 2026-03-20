#!/usr/bin/env python3
"""
Quick script to analyze the structure of the hypernym dataset.
Uses only standard library - no pandas required.
"""
import csv
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def analyze_csv_dataset(filepath: Path, name: str):
    """Analyze a CSV hypernym dataset."""
    print(f"\n{'='*60}")
    print(f"Analysis of: {name}")
    print(f"{'='*60}")
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\nTotal rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys()) if rows else 'N/A'}")
    
    # Hyponyms (noun1)
    hyponyms = [row['noun1'] for row in rows]
    unique_hyponyms = set(hyponyms)
    print(f"\n--- Hyponyms (noun1) ---")
    print(f"Total unique hyponyms: {len(unique_hyponyms)}")
    
    hyponym_counts = Counter(hyponyms)
    print(f"\nHyponym distribution:")
    for hyponym, count in hyponym_counts.most_common(20):
        print(f"  {hyponym}: {count} hypernyms")
    if len(hyponym_counts) > 20:
        print(f"  ... and {len(hyponym_counts) - 20} more")
    
    # Hypernyms (predicted_hypernym)
    hypernyms = [row['predicted_hypernym'] for row in rows]
    unique_hypernyms = set(hypernyms)
    print(f"\n--- Hypernyms (predicted_hypernym) ---")
    print(f"Total unique hypernyms: {len(unique_hypernyms)}")
    
    hypernym_counts = Counter(hypernyms)
    print(f"\nTop 20 most common hypernyms:")
    for hypernym, count in hypernym_counts.most_common(20):
        print(f"  '{hypernym}': appears {count} time(s)")
    
    # Strategy distribution
    if 'strategy' in rows[0]:
        strategies = [row['strategy'] for row in rows]
        strategy_counts = Counter(strategies)
        print(f"\n--- Strategy distribution ---")
        for strategy, count in strategy_counts.most_common():
            print(f"  {strategy}: {count}")
    
    # GPT-4 ground truth distribution
    if 'gpt4_ground_truth' in rows[0]:
        gts = [row['gpt4_ground_truth'] for row in rows]
        gt_counts = Counter(gts)
        print(f"\n--- GPT-4 Ground Truth distribution ---")
        for gt, count in gt_counts.most_common():
            print(f"  {gt}: {count}")


def analyze_json_dataset(filepath: Path, name: str):
    """Analyze a JSON hypernym dataset."""
    print(f"\n{'='*60}")
    print(f"Analysis of: {name}")
    print(f"{'='*60}")
    
    with open(filepath) as f:
        data = json.load(f)
    
    print(f"\nTotal rows: {len(data)}")
    if data:
        print(f"Fields per entry: {list(data[0].keys())}")
    
    # Hyponyms (noun1)
    hyponyms = [d['noun1'] for d in data]
    unique_hyponyms = set(hyponyms)
    print(f"\n--- Hyponyms (noun1) ---")
    print(f"Total unique hyponyms: {len(unique_hyponyms)}")
    
    hyponym_counts = Counter(hyponyms)
    print(f"\nTop 20 hyponyms by frequency:")
    for hyponym, count in hyponym_counts.most_common(20):
        print(f"  {hyponym}: {count} hypernym pairs")
    if len(hyponym_counts) > 20:
        print(f"  ... and {len(hyponym_counts) - 20} more")
    
    # Hypernyms (noun2)
    hypernyms = [d['noun2'] for d in data]
    unique_hypernyms = set(hypernyms)
    print(f"\n--- Hypernyms (noun2) ---")
    print(f"Total unique hypernyms: {len(unique_hypernyms)}")
    
    hypernym_counts = Counter(hypernyms)
    print(f"\nTop 20 most common hypernyms:")
    for hypernym, count in hypernym_counts.most_common(20):
        print(f"  '{hypernym}': appears with {count} different hyponyms")
    
    # Taxonomic relationship distribution
    if 'taxonomic' in data[0]:
        taxonomic_counts = Counter(d['taxonomic'] for d in data)
        print(f"\n--- Taxonomic relationship distribution ---")
        for tax, count in taxonomic_counts.items():
            print(f"  {tax}: {count}")
    
    # How many hypernyms does each hyponym appear with?
    print(f"\n--- Hyponym-Hypernym pairing analysis ---")
    hyponym_to_hypernyms = {}
    for d in data:
        hyponym = d['noun1']
        hypernym = d['noun2']
        if hyponym not in hyponym_to_hypernyms:
            hyponym_to_hypernyms[hyponym] = set()
        hyponym_to_hypernyms[hyponym].add(hypernym)
    
    hypernym_per_hyponym = [len(v) for v in hyponym_to_hypernyms.values()]
    print(f"Average hypernyms per hyponym: {sum(hypernym_per_hyponym)/len(hypernym_per_hyponym):.2f}")
    print(f"Min hypernyms per hyponym: {min(hypernym_per_hyponym)}")
    print(f"Max hypernyms per hyponym: {max(hypernym_per_hyponym)}")
    
    # Show some examples
    print(f"\nExamples of hyponyms with multiple hypernyms:")
    multi_hypernym = {k: v for k, v in hyponym_to_hypernyms.items() if len(v) > 1}
    for hyponym, hypernyms in list(multi_hypernym.items())[:5]:
        print(f"  {hyponym} -> {hypernyms}")
    
    # How many hyponyms does each hypernym appear with?
    hypernym_to_hyponyms = {}
    for d in data:
        hyponym = d['noun1']
        hypernym = d['noun2']
        if hypernym not in hypernym_to_hyponyms:
            hypernym_to_hyponyms[hypernym] = set()
        hypernym_to_hyponyms[hypernym].add(hyponym)
    
    print(f"\nExamples of hypernyms with multiple hyponyms:")
    multi_hyponym = {k: v for k, v in hypernym_to_hyponyms.items() if len(v) > 1}
    for hypernym, hyponyms in list(sorted(multi_hyponym.items(), key=lambda x: -len(x[1])))[:10]:
        print(f"  {hypernym} ({len(hyponyms)} hyponyms) -> {list(hyponyms)[:5]}{'...' if len(hyponyms) > 5 else ''}")


def main():
    print("="*60)
    print("HYPERNYM DATASET STRUCTURE ANALYSIS")
    print("="*60)
    
    # Analyze CSV datasets
    csv_files = [
        (DATA_DIR / "hypernym_car_test.csv", "hypernym_car_test.csv"),
        (DATA_DIR / "hypernym_car_train.csv", "hypernym_car_train.csv"),
    ]
    
    for filepath, name in csv_files:
        if filepath.exists():
            analyze_csv_dataset(filepath, name)
    
    # Analyze JSON datasets
    json_files = [
        (DATA_DIR / "hypernymy-train.json", "hypernymy-train.json"),
        (DATA_DIR / "hypernym-train-gemma-2-2b.json", "hypernym-train-gemma-2-2b.json"),
    ]
    
    for filepath, name in json_files:
        if filepath.exists():
            analyze_json_dataset(filepath, name)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
