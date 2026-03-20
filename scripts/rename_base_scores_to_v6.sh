#!/bin/bash

# Rename base model scores files from old format to new v6 format
# Old: scores_gemma-2-2b_hypernym-...
# New: scores_v6-google_gemma-2-2b_hypernym-...

OUTPUTS_DIR="../outputs"

# Dry run by default, pass --execute to actually rename
DRY_RUN=true
if [ "$1" == "--execute" ]; then
    DRY_RUN=false
    echo "EXECUTING renames..."
else
    echo "DRY RUN - pass --execute to actually rename files"
fi

echo ""

# Count files
count=0

# Find and rename scores files only
for f in "$OUTPUTS_DIR"/scores_gemma-2-2b_*; do
    if [ -f "$f" ]; then
        basename=$(basename "$f")
        newname="${basename/scores_gemma-2-2b_/scores_v6-google_gemma-2-2b_}"
        
        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] $basename -> $newname"
        else
            mv "$f" "$OUTPUTS_DIR/$newname"
            echo "Renamed: $basename -> $newname"
        fi
        ((count++))
    fi
done

echo ""
echo "Total files to rename: $count"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Run with --execute to perform the renames"
fi
