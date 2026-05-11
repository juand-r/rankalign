#!/bin/bash
# Morning report: aggregates scores_*.csv from outputs-quickiter into long
# CSVs, runs the existing rosch summary table builder for the rosch
# quick-iter tables, and writes OVERNIGHT_STATUS.md with job statuses + the
# main artifacts. Designed to be submitted with sbatch --dependency=afterany
# on every overnight job (training + eval + the rosch 2b-it eval already
# running) so it only fires once everything terminates.

set -e

source /u/jdr/venvs/venv_lexcons/bin/activate
cd "$(dirname "$0")/.."

REPO=$(pwd)
OUT=$REPO/outputs-quickiter
OVN=$REPO/overnight
STATUS=$REPO/OVERNIGHT_STATUS.md

mkdir -p "$OVN"

echo "[morning] aggregating ambigqa scores..."
python scripts/summarize_scores_file.py \
    --glob "${OUT}/scores_*ambigqa-train-as-test*.csv" \
    --csv "${OUT}/quickiter_metrics_long_ambigqa.csv" \
    > "${OVN}/morning_summarize_ambigqa.log" 2>&1 || \
    echo "[morning] WARN: ambigqa summarize failed; see ${OVN}/morning_summarize_ambigqa.log"

echo "[morning] aggregating rosch-furniture-and-bird scores (refresh)..."
python scripts/summarize_scores_file.py \
    --glob "${OUT}/scores_*rosch-furniture-and-bird*.csv" \
    --csv "${OUT}/quickiter_metrics_long.csv" \
    > "${OVN}/morning_summarize_rosch.log" 2>&1 || \
    echo "[morning] WARN: rosch summarize failed; see ${OVN}/morning_summarize_rosch.log"

echo "[morning] rebuilding rosch 2b summary tables..."
python scripts/build_rosch_quickiter_summary_tables.py \
    > "${OVN}/morning_rosch_tables.log" 2>&1 || \
    echo "[morning] WARN: rosch summary table build failed; see ${OVN}/morning_rosch_tables.log"

# Snapshot squeue so the morning report can show what (if anything) is still
# pending at report time.
squeue -u jdr -o "%.10i %.30j %.8T %.10M %R" > "${OVN}/morning_squeue.txt" 2>&1 || true

# Build the markdown summary.
{
    echo "# Overnight run summary"
    echo
    echo "Run kicked off Sun May 10 22:30 (UTC-5)."
    echo "Morning report generated: $(date -Iseconds)"
    echo
    echo "## Slurm job map"
    echo
    if [ -f "${OVN}/jobids.txt" ]; then
        echo '```'
        cat "${OVN}/jobids.txt"
        echo '```'
    else
        echo "_(jobids.txt missing — probably means the launcher never ran;"
        echo " check squeue snapshot below.)_"
    fi
    echo
    echo "## squeue snapshot at report time"
    echo
    echo '```'
    cat "${OVN}/morning_squeue.txt" 2>/dev/null || echo "(squeue snapshot missing)"
    echo '```'
    echo
    echo "## Artifacts"
    echo
    echo "| What | Path |"
    echo "|------|------|"
    echo "| Rosch (gemma-2-2b) summary tables | [outputs-quickiter/rosch_quickiter_summary.md](outputs-quickiter/rosch_quickiter_summary.md) |"
    echo "| Rosch long-format metrics (refreshed; both 2b and 2b-it) | [outputs-quickiter/quickiter_metrics_long.csv](outputs-quickiter/quickiter_metrics_long.csv) |"
    echo "| AmbigQA long-format metrics (2b + 2b-it) | [outputs-quickiter/quickiter_metrics_long_ambigqa.csv](outputs-quickiter/quickiter_metrics_long_ambigqa.csv) |"
    echo "| Morning summarize logs | [overnight/](overnight/) |"
    echo "| Per-job stdout/stderr | ~/logs/&lt;JOBID&gt;.{out,err} |"
    echo
    echo "## How to verify each block finished"
    echo
    echo "- Training: \`grep 'Saving to ' ~/logs/<TRAIN_JOBID>.out | tail -3\` should show two saves (epoch0 and epoch2)."
    echo "- Eval: \`grep 'Done. Elapsed:' ~/logs/<EVAL_JOBID>.out\` should appear once at the end."
    echo "- Failed jobs: \`sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed\`."
    echo
    echo "## Tables"
    echo
    echo "Rosch tables already exist (only refreshed if new 2b-it eval CSVs landed). For ambigqa, the long-format CSV at \`outputs-quickiter/quickiter_metrics_long_ambigqa.csv\` is the input — you can pivot it manually or have me build a build_ambigqa_quickiter_summary_tables.py mirror in the morning."
} > "$STATUS"

echo "[morning] wrote $STATUS"
