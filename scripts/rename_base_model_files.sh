#!/bin/bash
# Script to rename base model files to include v5- or v6- prefix based on date
# Usage: ./rename_base_model_files.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be renamed ==="
fi

OUTPUTS_DIR="/datastor1/jdr/gv-gap/rankalign/outputs"
cd "$OUTPUTS_DIR" || exit 1

# Find all base model files (scores_gemma-2-2b_*.csv)
# Exclude any that already have v5- or v6- prefix
files=$(ls -1 scores_gemma-2-2b_*.csv 2>/dev/null | grep -v "^scores_v[56]-")

if [ -z "$files" ]; then
    echo "No base model files found to rename."
    exit 0
fi

echo "Found $(echo "$files" | wc -l) base model files to process"
echo ""

renamed_count=0
skipped_count=0

for file in $files; do
    # Extract date from filename (format: _YYYYMMDD_)
    date_match=$(echo "$file" | grep -oE '_([0-9]{8})_' | head -1 | sed 's/_//g')
    
    if [ -z "$date_match" ]; then
        echo "⚠️  SKIP: $file (no date found in filename)"
        ((skipped_count++))
        continue
    fi
    
    # Determine version based on date
    # Files from 20260112-20260120: v5
    # Files from 20260127+: v6
    if [[ "$date_match" -le "20260120" ]]; then
        version="v5"
    elif [[ "$date_match" -ge "20260127" ]]; then
        version="v6"
    else
        echo "⚠️  SKIP: $file (date $date_match is between v5 and v6, unclear version)"
        ((skipped_count++))
        continue
    fi
    
    # Create new filename: scores_gemma-2-2b_... -> scores_v5-google_gemma-2-2b_...
    # Remove "scores_" prefix, add version and "google_" (with underscore, not hyphen)
    rest="${file#scores_}"
    new_file="scores_${version}-google_${rest}"
    
    if [ -f "$new_file" ]; then
        echo "⚠️  SKIP: $file -> $new_file (target already exists)"
        ((skipped_count++))
        continue
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "WOULD RENAME: $file -> $new_file (date: $date_match, version: $version)"
    else
        mv "$file" "$new_file"
        echo "✓ RENAMED: $file -> $new_file (date: $date_match, version: $version)"
    fi
    ((renamed_count++))
done

echo ""
echo "Summary:"
echo "  Renamed: $renamed_count"
echo "  Skipped: $skipped_count"
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "This was a DRY RUN. Run without --dry-run to actually rename files."
fi
