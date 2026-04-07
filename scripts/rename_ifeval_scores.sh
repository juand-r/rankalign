#!/bin/bash
# Rename IFEval base-model score files to include v6-google_ prefix.
#
# Old format:  scores_{prefix}gemma-2-{X}_ifeval-...
# New format:  scores_{prefix}v6-google_gemma-2-{X}_ifeval-...
#
# where {prefix} is "" | "self-" | "neg-"
# and {X} is "2b" | "9b-it"
#
# Usage:
#   bash scripts/rename_ifeval_scores.sh          # dry-run (default)
#   bash scripts/rename_ifeval_scores.sh --apply   # actually rename

set -e

DIR="outputs"
APPLY=false

if [ "$1" = "--apply" ]; then
    APPLY=true
    echo "=== APPLY MODE: files will be renamed ==="
else
    echo "=== DRY RUN: no files will be changed (pass --apply to rename) ==="
fi

COUNT=0

for old in "$DIR"/scores_*gemma-2-*_ifeval-*.csv; do
    [ -f "$old" ] || continue
    base=$(basename "$old")

    # Skip files that already have the correct v6- format
    if echo "$base" | grep -qE '^scores_(self-|neg-)?v[56]-'; then
        continue
    fi

    # Determine the replacement:
    #   scores_gemma-2-        -> scores_v6-google_gemma-2-
    #   scores_self-gemma-2-   -> scores_self-v6-google_gemma-2-
    #   scores_neg-gemma-2-    -> scores_neg-v6-google_gemma-2-
    new="$base"
    if echo "$base" | grep -q '^scores_self-gemma-2-'; then
        new=$(echo "$base" | sed 's/^scores_self-gemma-2-/scores_self-v6-google_gemma-2-/')
    elif echo "$base" | grep -q '^scores_neg-gemma-2-'; then
        new=$(echo "$base" | sed 's/^scores_neg-gemma-2-/scores_neg-v6-google_gemma-2-/')
    elif echo "$base" | grep -q '^scores_gemma-2-'; then
        new=$(echo "$base" | sed 's/^scores_gemma-2-/scores_v6-google_gemma-2-/')
    else
        continue
    fi

    if [ "$base" = "$new" ]; then
        continue
    fi

    COUNT=$((COUNT + 1))
    if [ "$APPLY" = true ]; then
        mv "$DIR/$base" "$DIR/$new"
        echo "  renamed: $base -> $new"
    else
        echo "  would rename: $base -> $new"
    fi
done

echo ""
echo "Total: $COUNT files"
