#!/bin/bash
# Final report-build job: kicked off automatically by sbatch dependency once
# all 10 membership->rosch eval jobs have terminated. Aggregates the 260
# scores CSVs into one long-form CSV under
# outputs-quickiter/membership-sans-rosch-v0-to-rosch/, then builds:
#   - 10 per-task summary markdowns
#   - MEAN_across_10_rosch_tasks_gemma-2-2b.md
#   - BUCKETED_by_overlap_gemma-2-2b.md
#
# Usage (manual rerun):
#   bash scripts/build_membership_to_rosch_report.sh

set -e

cd "$(dirname "$0")/.."
source /u/jdr/venvs/venv_lexcons/bin/activate

OUT_DIR=outputs-quickiter/membership-sans-rosch-v0-to-rosch
mkdir -p "$OUT_DIR"

ROSCH_TASKS=(
    rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit rosch-furniture
    rosch-sport rosch-toy rosch-vegetable rosch-vehicle rosch-weapon
)

# Glob ONLY the membership-trained-and-evaluated-on-rosch CSVs. Pattern
# matches (a) fine-tuned models with membership-sans-rosch-v0-all in the
# checkpoint segment AND a rosch-* eval task, OR (b) base-model evals on
# rosch-* tasks. Base-model evals are dated 20260511 from the cross-task
# launch and 20260512+ from this run; we include all of them since they
# evaluate the same base model on the same rosch task.

FILES=()
for T in "${ROSCH_TASKS[@]}"; do
    # Fine-tuned: membership-sans-rosch-v0-all in the checkpoint segment
    while IFS= read -r f; do
        FILES+=("$f")
    done < <(ls outputs-quickiter/scores_*membership-sans-rosch-v0-all*_${T}_test_log-odds_tc_*.csv 2>/dev/null || true)
    # Base model: scores_{self|neg}-v6-google_gemma-2-2b_<rosch-task>_test...
    while IFS= read -r f; do
        FILES+=("$f")
    done < <(ls outputs-quickiter/scores_self-v6-google_gemma-2-2b_${T}_test_log-odds_tc_*.csv outputs-quickiter/scores_neg-v6-google_gemma-2-2b_${T}_test_log-odds_tc_*.csv 2>/dev/null || true)
done

echo "Aggregating ${#FILES[@]} score CSVs into long-form metrics CSV..."

python scripts/summarize_scores_file.py "${FILES[@]}" \
    --csv "$OUT_DIR/quickiter_metrics_long_membership_to_rosch.csv" \
    --compact 2>&1 | tail -3

echo ""
echo "Building per-task summary markdowns..."
for T in "${ROSCH_TASKS[@]}"; do
    python scripts/build_quickiter_summary_tables.py \
        --model gemma-2-2b \
        --task membership-sans-rosch-v0 \
        --eval-task "$T" \
        --long-csv "$OUT_DIR/quickiter_metrics_long_membership_to_rosch.csv" \
        2>&1 | tail -2
done

echo ""
echo "Building mean and bucketed summary markdowns..."
python scripts/build_membership_to_rosch_buckets.py 2>&1 | tail -3

echo ""
echo "============================================================"
echo "Report build complete. See: $OUT_DIR"
ls -la "$OUT_DIR"
echo "============================================================"
