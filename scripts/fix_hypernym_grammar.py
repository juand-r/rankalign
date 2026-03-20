#!/usr/bin/env python3
"""
Fix grammaticality issues in hypernym predictions using GPT-4.

For each entry in the CSV, prompts GPT-4 to check if the hypernym is grammatical
in the context of the hypernymy task prompts, and if not, whether it can be fixed.

Usage:
    python fix_hypernym_grammar.py <csv_file> [--dry-run] [--limit N]
    
Example:
    python fix_hypernym_grammar.py data/hypernym_elephants_google-gemma-2-2b_train.csv --dry-run --limit 5
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import openai
from openai import OpenAI

# Configuration
OUTPUT_DIR = Path("data/fixed-hypernyms")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RATE_LIMIT_DELAY = 0.5  # seconds between requests

# GPT-4 prompt templates - separate for generator and discriminator
GPT4_GENERATOR_TEMPLATE = '''Consider this sentence:

"{word} are a kind of {hypernym}."

Is this sentence grammatical? I don't care about casing or whether the sentence makes sense semantically. Just the grammar.

IMPORTANT RULES:
1. After "a kind of", the noun should be SINGULAR, not plural.
   - "elephants are a kind of mammals" → "elephants are a kind of mammal"
   - "elephants are a kind of large mammals" → "elephants are a kind of large mammal"
2. Make MINIMAL changes. Only fix the grammar, preserve all other words.
   - Do NOT remove adjectives or modifiers
   - Just singularize the final noun if needed

If the sentence is already grammatical, return it as-is.

Output your report in JSON format:
{{"grammatical": "Yes/No",
  "reason": "<reason if No, e.g., plural after 'a kind of', verb instead of noun, etc. If Yes, say 'N/A'>",
  "can_be_fixed": "Yes/No",
  "corrected_sentence": "<the full corrected sentence with minimal changes>"
}}

Only output the JSON, nothing else.'''


GPT4_DISCRIMINATOR_TEMPLATE = '''Consider this sentence:

"Do you think {word} are a {hypernym}?"

Is this sentence grammatical? I don't care about casing or whether the sentence makes sense semantically. Just the grammar.

If the sentence is already grammatical, return it as-is.
If it needs correction, provide the most natural grammatical form. For example:
- "Do you think elephants are a animal?" -> "Do you think elephants are animals?" (plural is more natural)
- "Do you think elephants are a mammal?" -> OK as is, or "Do you think elephants are mammals?"

Prefer the more natural plural form when the subject is plural.

Output your report in JSON format:
{{"grammatical": "Yes/No",
  "reason": "<reason if No, e.g., article mismatch, verb instead of noun, etc. If Yes, say 'N/A'>",
  "can_be_fixed": "Yes/No",
  "corrected_sentence": "<the full corrected sentence with actual words, e.g. 'Do you think elephants are mammals?' - not placeholders>"
}}

Only output the JSON, nothing else.'''


def create_generator_prompt(word: str, hypernym: str) -> str:
    """Create the generator prompt for GPT-4."""
    return GPT4_GENERATOR_TEMPLATE.format(word=word, hypernym=hypernym)


def create_discriminator_prompt(word: str, hypernym: str) -> str:
    """Create the discriminator prompt for GPT-4."""
    return GPT4_DISCRIMINATOR_TEMPLATE.format(word=word, hypernym=hypernym)


def extract_hypernym_from_generator(word: str, generator_sentence: str) -> str | None:
    """Extract the hypernym from a generator sentence of the form '$word are a kind of X.'"""
    prefix = f"{word} are a kind of "
    # Case-insensitive prefix check
    if generator_sentence.lower().startswith(prefix.lower()):
        # Extract everything after the prefix, removing trailing period
        hypernym = generator_sentence[len(prefix):].rstrip('.')
        return hypernym.strip()
    return None


def query_gpt4(client: OpenAI, prompt: str) -> dict:
    """Query GPT-4 and parse the response."""
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            # Handle potential markdown code blocks
            if content.startswith("```"):
                # Remove markdown code block
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            result = json.loads(content)
            return result
            
        except json.JSONDecodeError as e:
            print(f"  Warning: Failed to parse JSON response: {e}")
            print(f"  Raw response: {content}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"error": "JSON parse error", "raw": content}
                
        except openai.RateLimitError as e:
            print(f"  Rate limit hit, waiting {RETRY_DELAY * (attempt + 1)}s...")
            time.sleep(RETRY_DELAY * (attempt + 1))
            
        except openai.APIError as e:
            print(f"  API error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {"error": str(e)}
    
    return {"error": "Max retries exceeded"}


def process_csv(input_file: Path, dry_run: bool = False, limit: int = None):
    """Process the CSV file and check grammaticality of hypernyms."""
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine output filenames
    stem = input_file.stem
    fixed_file = OUTPUT_DIR / f"{stem}-fixed.csv"
    unfixable_file = OUTPUT_DIR / f"{stem}-unfixable.csv"
    
    print(f"Processing: {input_file}")
    print(f"Output (fixed): {fixed_file}")
    print(f"Output (unfixable): {unfixable_file}")
    if dry_run:
        print("*** DRY RUN - no files will be written ***")
    print()
    
    # Read input CSV
    with open(input_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames
        rows = list(reader)
    
    if limit:
        rows = rows[:limit]
        print(f"Limiting to first {limit} rows")
    
    # Lowercase hypernyms and remove duplicates
    seen_pairs = set()
    deduplicated_rows = []
    duplicate_rows = []  # Will be saved to unfixable file
    
    for row in rows:
        original_word = row['noun1']
        original_hypernym = row['predicted_hypernym']
        word = original_word.lower()
        hypernym = original_hypernym.lower()
        pair = (word, hypernym)
        
        if pair in seen_pairs:
            # Save duplicate with original casing info
            duplicate_rows.append({
                'noun1': word,
                'original_hypernym': original_hypernym,
                'reason': f"Duplicate after lowercasing (original: '{original_hypernym}')",
                **{k: v for k, v in row.items() if k not in ['noun1', 'predicted_hypernym']}
            })
            continue
        
        seen_pairs.add(pair)
        # Update the row with lowercased values
        row['noun1'] = word
        row['predicted_hypernym'] = hypernym
        deduplicated_rows.append(row)
    
    rows = deduplicated_rows
    if duplicate_rows:
        print(f"Found {len(duplicate_rows)} duplicate entries after lowercasing (will save to unfixable)")
    
    # New fieldnames for output
    output_fieldnames = ['noun1', 'fixed_hypernym_generator', 'original_hypernym', 
                         'corrected_generator', 'corrected_discriminator',
                         'discriminator_sentence', 'generator_sentence'] + \
                        [f for f in original_fieldnames if f not in ['noun1', 'predicted_hypernym']]
    
    fixed_rows = []
    unfixable_rows = []
    error_rows = []
    
    total = len(rows)
    
    for i, row in enumerate(rows):
        word = row['noun1']
        hypernym = row['predicted_hypernym']
        
        print(f"[{i+1}/{total}] Checking: '{word}' + '{hypernym}'")
        
        if dry_run:
            # In dry run, just show what would be done
            print(f"  Would query GPT-4 for generator and discriminator")
            continue
        
        # Query GPT-4 for GENERATOR
        gen_prompt = create_generator_prompt(word, hypernym)
        gen_result = query_gpt4(client, gen_prompt)
        
        if "error" in gen_result:
            print(f"  Error (generator): {gen_result['error']}")
            error_rows.append((row, gen_result))
            continue
        
        time.sleep(RATE_LIMIT_DELAY)
        
        # Query GPT-4 for DISCRIMINATOR
        disc_prompt = create_discriminator_prompt(word, hypernym)
        disc_result = query_gpt4(client, disc_prompt)
        
        if "error" in disc_result:
            print(f"  Error (discriminator): {disc_result['error']}")
            error_rows.append((row, disc_result))
            continue
        
        # Parse generator result
        gen_grammatical = gen_result.get("grammatical", "").lower() == "yes"
        gen_can_fix = gen_result.get("can_be_fixed", "").lower() == "yes"
        gen_reason = gen_result.get("reason", "")
        gen_sentence = gen_result.get("corrected_sentence", f"{word} are a kind of {hypernym}.")
        
        # Parse discriminator result
        disc_grammatical = disc_result.get("grammatical", "").lower() == "yes"
        disc_can_fix = disc_result.get("can_be_fixed", "").lower() == "yes"
        disc_reason = disc_result.get("reason", "")
        disc_sentence = disc_result.get("corrected_sentence", f"Do you think {word} are a {hypernym}?")
        
        # Extract hypernym from generator sentence
        extracted_hypernym = extract_hypernym_from_generator(word, gen_sentence)
        
        if extracted_hypernym is None:
            print(f"  ⚠ Could not extract hypernym from generator sentence: '{gen_sentence}'")
            error_rows.append((row, {"error": "Could not extract hypernym", "generator_sentence": gen_sentence}))
            continue
        
        # Determine if corrections were made
        gen_was_corrected = extracted_hypernym.lower() != hypernym.lower()
        orig_disc = f"Do you think {word} are a {hypernym}?"
        disc_was_corrected = disc_sentence.lower() != orig_disc.lower()
        
        # Build output row
        output_row = {
            'noun1': word,
            'original_hypernym': hypernym,
            'fixed_hypernym_generator': extracted_hypernym,
            'corrected_generator': 'Yes' if gen_was_corrected else 'No',
            'corrected_discriminator': 'Yes' if disc_was_corrected else 'No',
            'discriminator_sentence': disc_sentence,
            'generator_sentence': gen_sentence,
        }
        
        # Copy other fields
        for f in original_fieldnames:
            if f not in ['noun1', 'predicted_hypernym']:
                output_row[f] = row[f]
        
        # Determine overall fixability
        gen_ok = gen_grammatical or gen_can_fix
        disc_ok = disc_grammatical or disc_can_fix
        
        if gen_ok and disc_ok:
            fixed_rows.append(output_row)
            gen_status = "✓" if gen_grammatical else f"→ '{extracted_hypernym}'"
            disc_status = "✓" if disc_grammatical else "→ fixed"
            print(f"  Generator: {gen_status} | Discriminator: {disc_status}")
        else:
            unfixable_rows.append(output_row)
            reasons = []
            if not gen_ok:
                reasons.append(f"gen: {gen_reason}")
            if not disc_ok:
                reasons.append(f"disc: {disc_reason}")
            print(f"  ✗ Unfixable: {'; '.join(reasons)}")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    if dry_run:
        print(f"\nDry run complete. Would process {total} rows.")
        return
    
    # Write output files
    if fixed_rows:
        with open(fixed_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(fixed_rows)
        print(f"\nWrote {len(fixed_rows)} fixed rows to {fixed_file}")
    
    # Add duplicate rows to unfixable (format them properly)
    for dup_row in duplicate_rows:
        formatted_dup = {
            'noun1': dup_row['noun1'],
            'fixed_hypernym_generator': '',
            'original_hypernym': dup_row['original_hypernym'],
            'corrected_generator': 'N/A',
            'corrected_discriminator': 'N/A',
            'discriminator_sentence': f"DUPLICATE: {dup_row['reason']}",
            'generator_sentence': '',
        }
        # Copy other fields
        for f in original_fieldnames:
            if f not in ['noun1', 'predicted_hypernym'] and f in dup_row:
                formatted_dup[f] = dup_row[f]
        unfixable_rows.append(formatted_dup)
    
    if unfixable_rows:
        with open(unfixable_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(unfixable_rows)
        print(f"Wrote {len(unfixable_rows)} unfixable rows to {unfixable_file}")
        if duplicate_rows:
            print(f"  (includes {len(duplicate_rows)} duplicates from lowercasing)")
    
    if error_rows:
        print(f"\n{len(error_rows)} rows had errors and were skipped")
    
    print(f"\nSummary:")
    print(f"  Total processed: {total}")
    print(f"  Fixed/grammatical: {len(fixed_rows)}")
    print(f"  Unfixable: {len(unfixable_rows)} ({len(duplicate_rows)} duplicates)")
    print(f"  Errors: {len(error_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix grammaticality issues in hypernym predictions using GPT-4"
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the input CSV file (e.g., data/hypernym_elephants_google-gemma-2-2b_train.csv)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls or writing files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N rows (useful for testing)"
    )
    
    args = parser.parse_args()
    
    if not args.csv_file.exists():
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)
    
    process_csv(args.csv_file, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()

