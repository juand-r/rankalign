"""
Merge GPT-2 typicality scores with eval.py debug output.

This creates a combined CSV with:
- noun pairs and ground truth
- generator and discriminator scores from target LLM
- typicality scores from GPT-2

Usage:
python merge_data.py \
    --typicality gpt2_typicality_scores.csv \
    --eval_output ../outputs/debug_values_hypernym_logodds.csv \
    --output merged_analysis_data.csv
"""

import argparse
import pandas as pd


def main(args):
    print("="*60)
    print("Merging Typicality Scores with Eval Output")
    print("="*60)
    
    # Load typicality scores
    print(f"\nLoading typicality scores from: {args.typicality}")
    df_typicality = pd.read_csv(args.typicality)
    print(f"  Loaded {len(df_typicality)} rows")
    print(f"  Columns: {list(df_typicality.columns)}")
    
    # Load eval output
    print(f"\nLoading eval output from: {args.eval_output}")
    df_eval = pd.read_csv(args.eval_output)
    print(f"  Loaded {len(df_eval)} rows")
    print(f"  Columns: {list(df_eval.columns)}")
    
    # Verify row counts match
    if len(df_typicality) != len(df_eval):
        print(f"\n❌ ERROR: Row counts don't match!")
        print(f"  Typicality file: {len(df_typicality)} rows")
        print(f"  Eval file: {len(df_eval)} rows")
        print(f"  Difference: {abs(len(df_typicality) - len(df_eval))} rows")
        raise ValueError("Merge verification failed: row counts don't match! Both files must have same number of rows.")
    
    # Verify indices match
    indices_match = (df_typicality['index'].values == df_eval['index'].values).all()
    if not indices_match:
        print(f"\n❌ ERROR: Index values don't match!")
        mismatched_indices = df_typicality['index'].values != df_eval['index'].values
        print(f"  Number of mismatched indices: {mismatched_indices.sum()}")
        print("\nFirst few mismatches:")
        for i in range(min(10, len(mismatched_indices))):
            if mismatched_indices[i]:
                print(f"  Row {i}: typicality index={df_typicality.iloc[i]['index']}, eval index={df_eval.iloc[i]['index']}")
        raise ValueError("Merge verification failed: index values don't match! Files may be using different data or ordering.")
    
    print("✓ Indices match")
    
    # Merge on index
    print("\nMerging dataframes on index...")
    df_merged = pd.merge(
        df_typicality,
        df_eval,
        on='index',
        how='inner',
        suffixes=('_typicality', '_eval')
    )
    
    print(f"  Merged {len(df_merged)} rows")
    
    # CRITICAL: Verify noun pairs match
    if 'noun1_typicality' in df_merged.columns and 'noun1_eval' in df_merged.columns:
        noun1_mismatches = (df_merged['noun1_typicality'] != df_merged['noun1_eval']).sum()
        noun2_mismatches = (df_merged['noun2_typicality'] != df_merged['noun2_eval']).sum()
        
        if noun1_mismatches > 0 or noun2_mismatches > 0:
            print(f"\n❌ ERROR: Noun pairs don't match!")
            print(f"  noun1 mismatches: {noun1_mismatches}")
            print(f"  noun2 mismatches: {noun2_mismatches}")
            print("\nFirst few mismatches:")
            mask = (df_merged['noun1_typicality'] != df_merged['noun1_eval']) | \
                   (df_merged['noun2_typicality'] != df_merged['noun2_eval'])
            print(df_merged[mask][['index', 'noun1_typicality', 'noun1_eval', 
                                     'noun2_typicality', 'noun2_eval']].head(10))
            raise ValueError("Merge verification failed: noun pairs don't match!")
        else:
            print("\n✓ Noun pairs match perfectly")
            # Keep only one set of noun columns
            df_merged['noun1'] = df_merged['noun1_typicality']
            df_merged['noun2'] = df_merged['noun2_typicality']
            df_merged = df_merged.drop(columns=['noun1_typicality', 'noun1_eval', 
                                                  'noun2_typicality', 'noun2_eval'])
    
    # Verify ground truth matches (if both have it)
    if 'ground_truth_typicality' in df_merged.columns and 'ground_truth_eval' in df_merged.columns:
        mismatches = (df_merged['ground_truth_typicality'] != df_merged['ground_truth_eval']).sum()
        if mismatches > 0:
            print(f"\n❌ ERROR: {mismatches} ground truth mismatches!")
            print("\nFirst few mismatches:")
            mask = df_merged['ground_truth_typicality'] != df_merged['ground_truth_eval']
            print(df_merged[mask][['index', 'noun1', 'noun2', 'ground_truth_typicality', 'ground_truth_eval']].head(10))
            raise ValueError("Merge verification failed: ground truth labels don't match!")
        else:
            print("✓ Ground truth labels match")
            # Keep only one ground_truth column
            df_merged['ground_truth'] = df_merged['ground_truth_typicality']
            df_merged = df_merged.drop(columns=['ground_truth_typicality', 'ground_truth_eval'])
    
    # Also verify taxonomic if present
    if 'taxonomic_typicality' in df_merged.columns and 'taxonomic_eval' in df_merged.columns:
        taxonomic_mismatches = (df_merged['taxonomic_typicality'] != df_merged['taxonomic_eval']).sum()
        if taxonomic_mismatches > 0:
            print(f"\n❌ ERROR: {taxonomic_mismatches} taxonomic label mismatches!")
            print("\nFirst few mismatches:")
            mask = df_merged['taxonomic_typicality'] != df_merged['taxonomic_eval']
            print(df_merged[mask][['index', 'noun1', 'noun2', 'taxonomic_typicality', 'taxonomic_eval']].head(10))
            raise ValueError("Merge verification failed: taxonomic labels don't match!")
        else:
            print("✓ Taxonomic labels match")
            df_merged['taxonomic'] = df_merged['taxonomic_typicality']
            df_merged = df_merged.drop(columns=['taxonomic_typicality', 'taxonomic_eval'])
    
    # Reorder columns for readability
    column_order = [
        'index',
        'noun1',
        'noun2',
        'taxonomic',
        'ground_truth',
        'gen_score',
        'disc_score',
        'log_prob_noun2',
        'log_prob_noun1',
        'log_prob_noun2_given_context'
    ]
    
    # Keep only columns that exist
    column_order = [col for col in column_order if col in df_merged.columns]
    # Add any remaining columns
    remaining_cols = [col for col in df_merged.columns if col not in column_order]
    df_merged = df_merged[column_order + remaining_cols]
    
    # Save merged data
    print(f"\nSaving merged data to: {args.output}")
    df_merged.to_csv(args.output, index=False)
    print(f"  Saved {len(df_merged)} rows with {len(df_merged.columns)} columns")
    
    # Print summary
    print("\n" + "="*60)
    print("Merged Data Summary")
    print("="*60)
    print(f"\nColumns in merged data:")
    for col in df_merged.columns:
        print(f"  - {col}")
    
    print(f"\nFirst few rows:")
    print(df_merged.head(3).to_string())
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge typicality scores with eval output")
    parser.add_argument("--typicality", type=str, required=True,
                        help="Path to GPT-2 typicality scores CSV")
    parser.add_argument("--eval_output", type=str, required=True,
                        help="Path to eval.py debug output CSV")
    parser.add_argument("--output", type=str, default="merged_analysis_data.csv",
                        help="Output CSV file path")
    
    args = parser.parse_args()
    main(args)

