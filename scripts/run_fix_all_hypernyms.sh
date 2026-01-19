#!/bin/bash

# Run fix_hypernym_grammar.py on all hypernym CSV files (train and test)
#
# This script:
# 1. First runs preprocess_hypernym_csvs.py to create lowercase, deduped files
#    (with train->test overlap removed from train)
# 2. Then runs fix_hypernym_grammar.py on each file
#
# Usage: ./run_fix_all_hypernyms.sh [--dry-run] [--limit N]
# Example: ./run_fix_all_hypernyms.sh
# Example: ./run_fix_all_hypernyms.sh --dry-run
# Example: ./run_fix_all_hypernyms.sh --limit 10

cd "$(dirname "$0")/.."

# Pass through all arguments to the Python script
ARGS="$@"

# Directory for preprocessed (lowercase, deduped) files
PREPROCESSED_DIR="data/lowercase-deduped-hypernyms"

# Step 1: Run preprocessing if needed
if [ ! -d "$PREPROCESSED_DIR" ] || [ -z "$(ls -A "$PREPROCESSED_DIR" 2>/dev/null)" ]; then
    echo "========================================"
    echo "Running preprocessing (lowercase + dedupe + remove train/test overlap)..."
    echo "========================================"
    python scripts/preprocess_hypernym_csvs.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed"
        exit 1
    fi
    echo ""
fi

# Find all hypernym train and test CSV files from preprocessed directory
# Use find to handle filenames with spaces properly
mapfile -t ALL_FILES < <(find "$PREPROCESSED_DIR" -maxdepth 1 -name "hypernym_*_google-gemma-2-2b_train.csv" -o -name "hypernym_*_google-gemma-2-2b_test.csv" | sort)

if [ ${#ALL_FILES[@]} -eq 0 ]; then
    echo "No hypernym CSV files found in $PREPROCESSED_DIR/"
    exit 1
fi

NUM_FILES=${#ALL_FILES[@]}
echo "Found $NUM_FILES hypernym CSV files"
echo "Arguments: $ARGS"
echo "========================================"

# Check which files are already processed (have output files)
SKIPPED=0
TO_PROCESS=()
for FILE in "${ALL_FILES[@]}"; do
    # Get the expected output filename
    BASENAME=$(basename "$FILE" .csv)
    OUTPUT_FILE="data/fixed-hypernyms/${BASENAME}-fixed.csv"
    
    if [ -f "$OUTPUT_FILE" ]; then
        SKIPPED=$((SKIPPED + 1))
    else
        TO_PROCESS+=("$FILE")
    fi
done

if [ $SKIPPED -gt 0 ]; then
    echo "Skipping $SKIPPED already processed files"
fi

TO_PROCESS_COUNT=${#TO_PROCESS[@]}
echo "Files to process: $TO_PROCESS_COUNT"
echo ""

if [ $TO_PROCESS_COUNT -eq 0 ]; then
    echo "All files already processed!"
    exit 0
fi

# Process each file
COUNT=0
for FILE in "${TO_PROCESS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "========================================"
    echo "[$COUNT/$TO_PROCESS_COUNT] Processing: $FILE"
    echo "========================================"
    
    python scripts/fix_hypernym_grammar.py "$FILE" $ARGS
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to process $FILE"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All done! Processed $TO_PROCESS_COUNT files."
echo "========================================"

