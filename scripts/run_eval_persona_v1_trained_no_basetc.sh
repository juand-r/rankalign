#!/bin/bash
# Sibling of run_eval_persona_v1_trained.sh — evals persona-v1 trained models
# WITHOUT --base-typcorr, so output score-files use the bare `self-` / `neg-`
# prefix (instead of `basetyp-` / `basetypneg-`). This fills the "PMI self"
# and "Neg self" columns of the persona-v1 table for trained methods, which
# the original (base-TC-only) eval pass left empty.
#
# All other behavior mirrors run_eval_persona_v1_trained.sh:
#   - Same TASKS (6 personas).
#   - Same per-variant TC policy (#1/#2/#3 both, #4/#5/#6 self, #7/#8/#9 neg).
#   - Same Slurm shape: 1 GPU, 60G RAM, 6 CPUs, 2h walltime per job.
#
# Output prefixes:
#   self: --self-typcorr            (NO --base-typcorr) -> scores prefix: self-
#   neg:  --neg-typcorr             (NO --base-typcorr) -> scores prefix: neg-
#
# Usage:
#   bash scripts/run_eval_persona_v1_trained_no_basetc.sh [BASE_MODEL] [VARIANT...]
#     BASE_MODEL: HF id (default google/gemma-2-9b-it)
#     VARIANT:    one or more of {1..9}. Default: all 9.
#
# Env (same defaults as parent script):
#   HOURS=2  CPUS=6  MEM=60G  EPOCH=2  DISC_SHOTS=zero  MODELS_DIR=$REPO/models
#   DRYRUN=1  -> print the sbatch command instead of submitting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_MODEL="${1:-${BASE_MODEL:-google/gemma-2-9b-it}}"
shift || true
HOURS="${HOURS:-2}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
EPOCH="${EPOCH:-2}"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
DISC_SHOTS="${DISC_SHOTS:-zero}"

if [ "$#" -eq 0 ]; then
    VARIANTS=(1 2 3 4 5 6 7 8 9)
else
    VARIANTS=("$@")
fi

TASKS=(
    persona-v1-psychopathy
    persona-v1-machiavellianism
    persona-v1-narcissism
    persona-v1-desire-to-create-allies
    persona-v1-interest-in-music
    persona-v1-interest-in-science
)

if [ "$DISC_SHOTS" = "few" ] || [ "$DISC_SHOTS" = "auto" ]; then
    EVAL_COMMON="--log-odds"
else
    EVAL_COMMON="--log-odds --disc-shots-zero"
fi

declare -A SUFFIXES=(
    [1]="--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1"
    [2]="--full-completion--semi0.1"
    [3]="--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1"
    [4]="--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1"
    [5]="--tc-self--full-completion--force-same-x--semi0.1"
    [6]="--tc-self--full-completion--semi0.1"
    [7]="--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--vallogodds--semi0.1"
    [8]="--tc-neg--full-completion--force-same-x--semi0.1"
    [9]="--tc-neg--full-completion--semi0.1"
)
declare -A LABEL=(
    [1]="1.SFT-lo"        [2]="2.RankAlign"        [3]="3.New+fsx"
    [4]="4.New+fsx+selfTC" [5]="5.RankAlign+fsx+selfTC" [6]="6.RankAlign+selfTC"
    [7]="7.New+fsx+negTC" [8]="8.RankAlign+fsx+negTC" [9]="9.RankAlign+negTC"
)
declare -A TC_POLICY=(
    [1]=both [2]=both [3]=both
    [4]=self [5]=self [6]=self
    [7]=neg  [8]=neg  [9]=neg
)

USE_LORA=1
if [[ "$BASE_MODEL" == *"-2b"* ]]; then USE_LORA=0; fi
MERGED_SUFFIX=""
[[ "$USE_LORA" -eq 1 ]] && MERGED_SUFFIX="_merged"

BASE_REPL=$(echo "$BASE_MODEL" | sed 's|/|--|g')
PATH_PREFIX="${MODELS_DIR}/v6-${BASE_REPL}-delta0.15-epoch${EPOCH}--persona-v1-all--d2g--random--alpha1.0"

OVERNIGHT_DIR="$REPO_ROOT/overnight"
mkdir -p "$OVERNIGHT_DIR"
MODEL_TAG=$(basename "$BASE_MODEL" | sed 's|/|_|g; s|--|_|g')
JOBID_FILE="$OVERNIGHT_DIR/persona_v1_eval_trained_no_basetc_jobids_${MODEL_TAG}_disc-${DISC_SHOTS}.txt"
: > "$JOBID_FILE"

echo "========================================"
echo "Persona-v1 trained-model eval (NO --base-typcorr) launcher"
echo "  BASE_MODEL: $BASE_MODEL"
echo "  EPOCH:      $EPOCH"
echo "  USE_LORA:   $USE_LORA  (merged suffix: '${MERGED_SUFFIX}')"
echo "  Walltime:   ${HOURS}h / job   CPUs/MEM: $CPUS / $MEM   GPUs: 1"
echo "  Tasks:      ${#TASKS[@]}  (${TASKS[*]})"
echo "  EVAL_COMMON: $EVAL_COMMON"
echo "  DISC_SHOTS: $DISC_SHOTS"
echo "  Variants:   ${VARIANTS[*]}"
echo "  Jobid log:  $JOBID_FILE"
[ -n "${DRYRUN:-}" ] && echo "  DRYRUN=1 set — will print sbatch lines, not submit."
echo "========================================"

submit() {
    local label="$1"; shift
    local model_path="$1"; shift
    local tc_flag="$1"; shift
    local tc_label="$1"; shift

    if [ ! -d "$model_path" ]; then
        echo ""
        echo "[$label / $tc_label] SKIP — model dir not found: $model_path"
        echo "(no jobid - SKIP)  $label  $tc_label  $model_path" >> "$JOBID_FILE"
        return
    fi

    echo ""
    echo ">>> [$label / $tc_label] $model_path"
    if [ -n "${DRYRUN:-}" ]; then
        echo "DRYRUN: run 1 $HOURS --cpu $CPUS --mem $MEM scripts/run_eval_semi.sh $model_path $tc_flag --base-model $BASE_MODEL $EVAL_COMMON -- ${TASKS[*]}"
        OUT="DRYRUN-no-submit"
    else
        OUT=$(run 1 "$HOURS" --cpu "$CPUS" --mem "$MEM" \
            scripts/run_eval_semi.sh "$model_path" \
            $tc_flag --base-model "$BASE_MODEL" \
            $EVAL_COMMON \
            -- "${TASKS[@]}" 2>&1) || true
        echo "$OUT"
    fi
    JOBID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
    if [ -n "$JOBID" ]; then
        echo "$JOBID  $label  $tc_label  $model_path" >> "$JOBID_FILE"
    else
        echo "(no jobid captured)  $label  $tc_label  $model_path" >> "$JOBID_FILE"
    fi
}

for v in "${VARIANTS[@]}"; do
    label="${LABEL[$v]}"
    suffix="${SUFFIXES[$v]}"
    policy="${TC_POLICY[$v]}"
    if [ -z "$label" ] || [ -z "$suffix" ] || [ -z "$policy" ]; then
        echo "ERROR: variant $v not in {1..9}; skipping" >&2
        continue
    fi
    model_path="${PATH_PREFIX}${suffix}${MERGED_SUFFIX}"
    case "$policy" in
        both) submit "$label" "$model_path" "--self-typcorr" "self"
              submit "$label" "$model_path" "--neg-typcorr"  "neg" ;;
        self) submit "$label" "$model_path" "--self-typcorr" "self" ;;
        neg)  submit "$label" "$model_path" "--neg-typcorr"  "neg" ;;
        *) echo "ERROR: unknown TC_POLICY '$policy' for $label" >&2 ;;
    esac
done

echo ""
echo "========================================"
echo "Done. Jobid log:"
cat "$JOBID_FILE"
echo ""
echo "Tail logs with: tail -f /datastor2/jdr/logs/<JOBID>.out"
echo "Check progress with: squeue -u \$USER"
