#!/bin/bash
# Script to fix base model file names: scores_v5-google-gemma-2-2b_... -> scores_v5-google_gemma-2-2b_...
# Usage: ./fix_google_underscore.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be renamed ==="
fi

OUTPUTS_DIR="/datastor1/jdr/gv-gap/rankalign/outputs"
cd "$OUTPUTS_DIR" || exit 1

# Find files that need fixing: scores_v5-google-gemma-2-2b_ or scores_v6-google-gemma-2-2b_ (with hyphen)
files=$(ls -1 scores_v[56]-google-gemma-2-2b_*.csv 2>/dev/null)

if [ -z "$files" ]; then
    echo "No files found to fix."
    exit 0
fi

echo "Found $(echo "$files" | wc -l) files to fix"
echo ""

renamed_count=0
skipped_count=0

for file in $files; do
    # Replace google-gemma with google_gemma
    new_file=$(echo "$file" | sed 's/\(v[56]-google\)-gemma/\1_gemma/')
    
    if [ "$file" == "$new_file" ]; then
        echo "⚠️  SKIP: $file (no change needed)"
        ((skipped_count++))
        continue
    fi
    
    if [ -f "$new_file" ]; then
        echo "⚠️  SKIP: $file -> $new_file (target already exists)"
        ((skipped_count++))
        continue
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "WOULD RENAME: $file -> $new_file"
    else
        mv "$file" "$new_file"
        echo "✓ RENAMED: $file -> $new_file"
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
