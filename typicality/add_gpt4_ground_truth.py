"""
Add GPT-4.1-mini ground truth answers to predictions CSV.

Queries GPT-4.1-mini with: "Is it true that {noun1} are a kind of {predicted_hypernym}? 
Answer only Yes or No. Answer: "

Adds a new column 'gpt4_ground_truth' with the LLM's answer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import os
from openai import OpenAI

# Configuration
MODEL_NAME = "gpt-4.1-mini"

def get_gpt4_answer(noun1, predicted_hypernym, client, max_retries=3):
    """
    Query GPT-4.1-mini for ground truth answer.
    
    Args:
        noun1: The hyponym (e.g., "cars")
        predicted_hypernym: The predicted hypernym (e.g., "vehicles")
        client: OpenAI client
        max_retries: Maximum number of retries on failure
    
    Returns:
        String: "Yes", "No", or "Error" if failed
    """
    prompt = f"Is it true that {noun1} are a kind of {predicted_hypernym}? Answer only Yes or No. Answer:"
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic
                max_tokens=10,    # Short answer
                n=1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Normalize answer to "Yes" or "No"
            answer_lower = answer.lower()
            if "yes" in answer_lower:
                return "Yes"
            elif "no" in answer_lower:
                return "No"
            else:
                # If not clear Yes/No, return the raw answer
                return answer
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n  ⚠ Error on attempt {attempt + 1}: {e}")
                print(f"  ↻ Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"\n  ✗ Failed after {max_retries} attempts: {e}")
                return "Error"
    
    return "Error"

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Add GPT-4.1-mini ground truth answers to predictions CSV"
    )
    parser.add_argument('--input', type=str, 
                        required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', type=str,
                        required=True,
                        help='Output CSV file path')
    parser.add_argument('--api_key', type=str,
                        default=None,
                        help='OpenAI API key (defaults to OPENAI_MY_API_KEY env var)')
    args = parser.parse_args()
    
    INPUT_CSV = Path(args.input)
    OUTPUT_CSV = Path(args.output)
    
    print("="*70)
    print("Adding GPT-4.1-mini Ground Truth Answers to Predictions CSV")
    print("="*70)
    
    # Initialize OpenAI client
    print(f"\nInitializing OpenAI client...")
    if args.api_key:
        api_key = args.api_key
    else:
        api_key = os.environ.get('OPENAI_MY_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_MY_API_KEY environment variable not set and no --api_key provided")
    
    client = OpenAI(api_key=api_key)
    print(f"  ✓ Using model: {MODEL_NAME}")
    
    # Load the CSV
    print(f"\nLoading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  ✓ Loaded {len(df)} predictions")
    print(f"  ✓ Existing columns: {list(df.columns)}")
    
    # Query GPT-4.1-mini for each prediction
    print(f"\nQuerying {MODEL_NAME} for ground truth answers...")
    print(f"  Prompt template: 'Is it true that {{noun1}} are a kind of {{hypernym}}? Answer only Yes or No. Answer:'")
    
    ground_truth_answers = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Querying GPT-4.1-mini"):
        noun1 = row['noun1']
        predicted_hypernym = row['predicted_hypernym']
        
        answer = get_gpt4_answer(noun1, predicted_hypernym, client)
        ground_truth_answers.append(answer)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    # Add to dataframe
    df['gpt4_ground_truth'] = ground_truth_answers
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Added 'gpt4_ground_truth' column")
    print(f"✓ Saved to: {OUTPUT_CSV}")
    print(f"{'='*70}")
    
    # Print statistics
    print(f"\nGround Truth Answer Distribution:")
    answer_counts = df['gpt4_ground_truth'].value_counts()
    for answer, count in answer_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {answer}: {count} ({percentage:.1f}%)")
    
    # Print all predictions with their ground truth
    print(f"\nAll Predictions with Ground Truth:")
    print(f"{'Rank':<6} {'Hypernym':<25} {'Gen LogP':<10} {'Val LogP':<10} {'GPT-4 GT':<10}")
    print("-" * 70)
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        gen_lp = row['log_prob']
        val_lp = row.get('validator_log_prob', np.nan)
        gt = row['gpt4_ground_truth']
        hypernym = row['predicted_hypernym']
        
        if pd.notna(val_lp):
            print(f"{rank:<6} {hypernym:<25} {gen_lp:<10.4f} {val_lp:<10.4f} {gt:<10}")
        else:
            print(f"{rank:<6} {hypernym:<25} {gen_lp:<10.4f} {'N/A':<10} {gt:<10}")
    
    print(f"\n✅ DONE!")

