#!/usr/bin/env python3
"""
Combine mid-range log_prob data with existing combined data.

Takes:
- cars_midrange_logprob_with_gpt4_gt_and_typicality.csv (new mid-range points)
- cars_all_strategies_with_gpt4_gt_and_typicality.csv (existing combined data)

Produces:
- cars_all_strategies_plus_midrange_final.csv (merged, deduplicated)

Usage:
    python combine_midrange_with_existing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("\n" + "="*60)
    print("COMBINING MID-RANGE DATA WITH EXISTING DATA")
    print("="*60)
    
    # File paths
    midrange_file = Path('cars_midrange_logprob_with_gpt4_gt_and_typicality.csv')
    existing_file = Path('cars_all_strategies_with_gpt4_gt_and_typicality.csv')
    output_file = Path('cars_all_strategies_plus_midrange_final.csv')
    
    # Load files
    print(f"\nLoading {midrange_file}...")
    df_midrange = pd.read_csv(midrange_file)
    print(f"  Rows: {len(df_midrange)}")
    
    print(f"\nLoading {existing_file}...")
    df_existing = pd.read_csv(existing_file)
    print(f"  Rows: {len(df_existing)}")
    
    # Check for overlapping points (same noun1 + predicted_hypernym)
    print("\nChecking for duplicates...")
    df_midrange['key'] = df_midrange['noun1'] + '|' + df_midrange['predicted_hypernym']
    df_existing['key'] = df_existing['noun1'] + '|' + df_existing['predicted_hypernym']
    
    overlap = set(df_midrange['key']) & set(df_existing['key'])
    print(f"  Overlapping entries: {len(overlap)}")
    
    # Remove duplicates from midrange (keep existing version)
    df_midrange_unique = df_midrange[~df_midrange['key'].isin(overlap)]
    print(f"  Unique new entries from midrange: {len(df_midrange_unique)}")
    
    # Drop the key column before combining
    df_midrange_unique = df_midrange_unique.drop(columns=['key'])
    df_existing = df_existing.drop(columns=['key'])
    
    # Ensure columns match (add missing columns with NaN)
    all_cols = set(df_existing.columns) | set(df_midrange_unique.columns)
    for col in all_cols:
        if col not in df_existing.columns:
            df_existing[col] = np.nan
        if col not in df_midrange_unique.columns:
            df_midrange_unique[col] = np.nan
    
    # Reorder columns to match
    df_midrange_unique = df_midrange_unique[df_existing.columns]
    
    # Combine
    df_combined = pd.concat([df_existing, df_midrange_unique], ignore_index=True)
    print(f"\nCombined total: {len(df_combined)} rows")
    
    # Save
    df_combined.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nStrategy breakdown:")
    print(df_combined['strategy'].value_counts())
    
    print(f"\nGPT-4 ground truth:")
    print(df_combined['gpt4_ground_truth'].value_counts())
    
    print(f"\nlog_prob distribution:")
    bins = [-float('inf'), -15, -12, -9, -6, -3, 0]
    labels = ['<-15', '-15 to -12', '-12 to -9', '-9 to -6', '-6 to -3', '-3 to 0']
    df_combined['bin'] = pd.cut(df_combined['log_prob'], bins=bins, labels=labels)
    print(df_combined['bin'].value_counts().sort_index())
    
    print(f"\n✓ Done! Output: {output_file}")


if __name__ == '__main__':
    main()

