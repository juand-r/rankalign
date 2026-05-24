#!/bin/bash
# FIX1 re-runs of the "New" (comb-loss) persona-v1 variants.
#
# Compares scripts/ranking_loss_ref_fix.py against the parent on the three
# variants that used the comb loss in the original sweep:
#
#   3.New                      comb + semi + log-odds                     (no TC)
#   4.New+selfTC               comb + semi + log-odds + self-typcorr
#   7.New+negTC                comb + semi + log-odds + neg-typcorr
#
# Differences vs scripts/run_train_persona_v1.sh:
#   - Calls ranking_loss_ref_fix.py instead of ranking_loss_ref.py.
#   - Variants re-labeled "3.New" / "4.New+selfTC" / "7.New+negTC".
#   - Trained checkpoints land with a `--fix1` suffix (added inside
#     ranking_loss_ref_fix.py); they cannot collide with parent outputs.
#   - persona-v1 is single-prompt, so --force-same-x is a no-op behavior-wise.
#     We let run_train_semi.sh's default (--force-same-x ON) flow through
#     rather than hardcoding --no-force-same-x.
#
# Defaults to all 3 variants. To run a subset, pass them as positional args:
#   bash scripts/run_train_persona_v1_fix1.sh google/gemma-2-9b-it 4 7
#
# Usage:
#   bash scripts/run_train_persona_v1_fix1.sh [MODEL] [VARIANT...]
#
#   MODEL: HuggingFace id (default: google/gemma-2-9b-it).
#   VARIANT: any of 3 / 4 / 7. Default: all three.
#
# Env:
#   MODEL  - default HF model when no positional argument is given
#   HOURS  - Slurm walltime hours per job (default: 10)
#   DISC_SHOTS - 'zero' (default) -> --disc-shots zero;
#                'auto' -> omit (script picks zero/few based on instruct vs base)

set -e

MODEL="${1:-${MODEL:-google/gemma-2-9b-it}}"
shift || true
HOURS="${HOURS:-10}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
TASK=persona-v1
DISC_SHOTS="${DISC_SHOTS:-zero}"
# Both default to empty -> no override -> falls through to run_train_semi.sh's
# defaults (NUM_EPOCHS=3, no --batch-size flag = python's per-task auto-pick = 1
# for full-completion mode). Set EPOCHS=N or BATCH_SIZE=N to override per call.
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"

# Default to all 3 variants if no variant args.
if [ "$#" -eq 0 ]; then
    VARIANTS=(3 4 7)
else
    VARIANTS=("$@")
fi

if [ "$DISC_SHOTS" = "auto" ]; then
    COMMON="--max-seq-len 512 --script ranking_loss_ref_fix.py"
else
    COMMON="--disc-shots $DISC_SHOTS --max-seq-len 512 --script ranking_loss_ref_fix.py"
fi
[ -n "$EPOCHS" ]     && COMMON="$COMMON --epochs $EPOCHS"
[ -n "$BATCH_SIZE" ] && COMMON="$COMMON --batch-size $BATCH_SIZE"
[ -n "$DELTA_BINS" ] && COMMON="$COMMON --delta-bins $DELTA_BINS"

OVERNIGHT_DIR="$(dirname "$0")/../overnight"
mkdir -p "$OVERNIGHT_DIR"
MODEL_TAG=$(basename "$MODEL" | sed 's/--/_/g; s/[/]/_/g')
SUFFIX=""
[ -n "$DELTA_BINS" ] && SUFFIX="_dbins${DELTA_BINS}"
JOBID_FILE="$OVERNIGHT_DIR/persona_v1_train_jobids_${MODEL_TAG}_fix1${SUFFIX}.txt"
: > "$JOBID_FILE"

echo "========================================"
echo "Persona-v1 FIX1 training launcher"
echo "  MODEL:    $MODEL"
echo "  TASK:     $TASK"
echo "  HOURS:    $HOURS (each job)"
echo "  CPUS:     $CPUS"
echo "  MEM:      $MEM"
echo "  COMMON:   $COMMON"
echo "  VARIANTS: ${VARIANTS[*]}"
echo "  Jobid log: $JOBID_FILE"
echo "========================================"

submit() {
    local label="$1"; shift
    echo ""
    echo ">>> [$label] $*"
    OUT=$(run 1 "$HOURS" --cpu "$CPUS" --mem "$MEM" scripts/run_train_semi.sh "$MODEL" "$TASK" "$@" 2>&1) || true
    echo "$OUT"
    JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$JOBID" ]; then
        echo "$JOBID  $label  $*" >> "$JOBID_FILE"
    else
        echo "(no jobid captured)  $label  $*" >> "$JOBID_FILE"
    fi
}

for v in "${VARIANTS[@]}"; do
    case "$v" in
        3)
            submit "3.New(fix1)" \
                comb semi 0.1 $COMMON --log-odds
            ;;
        4)
            submit "4.New+selfTC(fix1)" \
                comb semi 0.1 $COMMON --self-typcorr --log-odds
            ;;
        7)
            submit "7.New+negTC(fix1)" \
                comb semi 0.1 $COMMON --neg-typcorr --log-odds
            ;;
        *)
            echo "Unknown variant: $v (must be 3, 4, or 7)"
            ;;
    esac
done

echo ""
echo "========================================"
echo "Submissions attempted. Jobid log:"
cat "$JOBID_FILE"
echo ""
echo "Tail logs with: tail -f ~/logs/<JOBID>.out"
echo "Check progress with: squeue -u \$USER"
