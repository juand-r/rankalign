#!/bin/bash
# Training runs for persona-v1 task.
#
# All 9 variants from docs/IMPORTANT-RESEARCH-PLAN.md (same as v0 launcher).
#
#   1. SFT-lo                 sft + labelonly + no force-same-x
#   2. RankAlign              pref-only + semi + no force-same-x  (last year's baseline)
#   3. New+fsx                comb + semi + log-odds + force-same-x (no TC)
#   4. New+fsx+self-tc        comb + semi + log-odds + force-same-x + self-typcorr
#   5. RankAlign+fsx+self-tc  pref-only + semi + force-same-x + self-typcorr
#   6. RankAlign+self-tc      pref-only + semi + self-typcorr + no force-same-x
#   7. New+fsx+neg-tc         comb + semi + log-odds + force-same-x + neg-typcorr
#   8. RankAlign+fsx+neg-tc   pref-only + semi + force-same-x + neg-typcorr
#   9. RankAlign+neg-tc       pref-only + semi + neg-typcorr + no force-same-x
#
# Persona-v1 vs v0 (see docs/datasets/persona_v1_notes.md):
#   - In-domain: 3 antisocial personas (psychopathy, machiavellianism,
#     narcissism). Drops moral-nihilism + no-meaning where v0 base validator
#     could not separate yes/no.
#   - Labels FLIPPED on the 3 antisocial personas, so `correct=yes` is the
#     prosocial direction across all 6 v1 personas. Training therefore pulls
#     the model TOWARD prosocial generations (the natural instruct-tuned bias),
#     instead of v0's "toward antisocial" pull.
#   - Train pool is now 1500 rows (3 ID x 500). OOD test set unchanged.
#
# Persona uses:
#   --disc-shots zero  (matches eval setup)
#   --max-seq-len 512  (statements short, ~30 tok max even chat-wrapped)
#
# Walltime defaults to 10h (v0 had two #8/#9 variants TIMEOUT at 6h).
#
# Usage:
#   bash scripts/run_train_persona_v1.sh [MODEL]
#
#   MODEL: HuggingFace id (default: google/gemma-2-9b-it).
#
# Examples:
#   bash scripts/run_train_persona_v1.sh
#   bash scripts/run_train_persona_v1.sh google/gemma-2-2b-it
#
# Env:
#   MODEL  - default HF model when no positional argument is given
#   HOURS  - Slurm walltime hours per job (default: 10)

set -e

MODEL="${1:-${MODEL:-google/gemma-2-9b-it}}"
HOURS="${HOURS:-10}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
TASK=persona-v1
COMMON="--disc-shots zero --max-seq-len 512"
# `run` writes Slurm logs to $HOME/logs. /u/jdr is currently quota-limited for
# new log growth, so default to a high-space HOME for job submission.
RUN_HOME="${RUN_HOME:-/datastor2/jdr}"

OVERNIGHT_DIR="$(dirname "$0")/../overnight"
mkdir -p "$OVERNIGHT_DIR"
MODEL_TAG=$(basename "$MODEL" | sed 's/--/_/g; s/[/]/_/g')
JOBID_FILE="$OVERNIGHT_DIR/persona_v1_train_jobids_${MODEL_TAG}.txt"
: > "$JOBID_FILE"

echo "========================================"
echo "Persona-v1 training launcher"
echo "  MODEL:  $MODEL"
echo "  TASK:   $TASK"
echo "  HOURS:  $HOURS (each of 9 jobs)"
echo "  CPUS:   $CPUS"
echo "  MEM:    $MEM"
echo "  COMMON: $COMMON"
echo "  RUN_HOME: $RUN_HOME (run logs -> $RUN_HOME/logs)"
echo "  Jobid log: $JOBID_FILE"
echo "========================================"

submit() {
    local label="$1"; shift
    echo ""
    echo ">>> [$label] $*"
    OUT=$(HOME="$RUN_HOME" run 1 "$HOURS" --cpu "$CPUS" --mem "$MEM" scripts/run_train_semi.sh "$MODEL" "$TASK" "$@" 2>&1) || true
    echo "$OUT"
    JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$JOBID" ]; then
        echo "$JOBID  $label  $*" >> "$JOBID_FILE"
    else
        echo "(no jobid captured)  $label  $*" >> "$JOBID_FILE"
    fi
}

submit "1.SFT-lo" \
    sft labelonly 0.1 $COMMON --no-force-same-x

submit "2.RankAlign" \
    pref-only semi 0.1 $COMMON --no-force-same-x

submit "3.New+fsx" \
    comb semi 0.1 $COMMON --log-odds

submit "4.New+fsx+selfTC" \
    comb semi 0.1 $COMMON --self-typcorr --log-odds

submit "5.RankAlign+fsx+selfTC" \
    pref-only semi 0.1 $COMMON --self-typcorr

submit "6.RankAlign+selfTC" \
    pref-only semi 0.1 $COMMON --self-typcorr --no-force-same-x

submit "7.New+fsx+negTC" \
    comb semi 0.1 $COMMON --neg-typcorr --log-odds

submit "8.RankAlign+fsx+negTC" \
    pref-only semi 0.1 $COMMON --neg-typcorr

submit "9.RankAlign+negTC" \
    pref-only semi 0.1 $COMMON --neg-typcorr --no-force-same-x

echo ""
echo "========================================"
echo "All 9 submissions attempted. Jobid log:"
cat "$JOBID_FILE"
echo ""
echo "Tail logs with: tail -f ~/logs/<JOBID>.out"
echo "Check progress with: squeue -u \$USER"
