"""
Subsample CSV by taking every Nth row.

Usage:
python subsample_csv.py --input cars_combined_tok1_tok2.csv --output cars_combined_tok1_tok2-subsampled.csv --step 20
"""

import argparse
import pandas as pd
from pathlib import Path

def main(args):
    print("="*60)
    print("Subsampling CSV")
    print("="*60)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"\nLoading CSV: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  ✓ Loaded {len(df)} rows")
    
    # Subsample by taking every Nth row
    df_subsampled = df.iloc[::args.step]
    
    print(f"\nSubsampling every {args.step}th row:")
    print(f"  Original rows: {len(df)}")
    print(f"  Subsampled rows: {len(df_subsampled)}")
    print(f"  Ratio: {len(df_subsampled)/len(df):.3f} ({len(df_subsampled)/len(df)*100:.1f}%)")
    
    # Save to output
    df_subsampled.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print("✅ DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample CSV by taking every Nth row")
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--step', type=int, default=20,
                        help='Take every Nth row (default: 20)')
    
    args = parser.parse_args()
    main(args)

