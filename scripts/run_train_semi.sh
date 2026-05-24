#!/bin/bash

# Train ranking loss with semi-supervised or labeled-only settings.
# Works with any task (plausibleqa, ambigqa, hypernym-concat-bananas-to-dogs-v2, ifeval, etc.)
#
# Usage: run 1 <hours> scripts/run_train_semi.sh <MODEL> <TASK> <LOSS> <SEMI_MODE> <RATIO> [options]
#
# Arguments:
#   MODEL       HuggingFace model name (e.g. google/gemma-2-2b, meta-llama/Llama-3.1-8B)
#   TASK        Task name (e.g. plausibleqa, ambigqa, hypernym-concat-bananas-to-dogs-double, ifeval-concat)
#   LOSS        Loss setting: sft | comb | pref-only
#   SEMI_MODE   semi | labelonly
#   RATIO       Fraction of prompts labeled (e.g. 0.1, 0.25, 0.5)
#
# Options:
#   --mode g|d             Generator (delta=0.15) or discriminator (delta=2.5) mode [default: g]
#   --typcorr              Enable GPT-2 typicality correction
#   --self-typcorr         Enable self-typicality correction
#   --neg-typcorr          Enable negated-prompt typicality correction (LLR)
#   --lenorm               Enable length normalization
#   --log-odds             Enable validator log-odds
#   --split-seed S         Seed for prompt split (default 42)
#   --delta D              Override default delta (overrides --mode default)
#   --samples N            Override default total_samples
#   --disc-shots zero|few  Override discriminator shots (default: auto based on model)
#   --max-seq-len N        Cap training sequence length (tokenizer truncation/pad); reduces VRAM
#
# Models:
#   google/gemma-2-2b          google/gemma-2-2b-it
#   google/gemma-2-9b          google/gemma-2-9b-it
#   meta-llama/Llama-3.1-8B   meta-llama/Llama-3.1-8B-Instruct
#   Qwen/Qwen2.5-7B           Qwen/Qwen2.5-7B-Instruct
#   Qwen/Qwen3.5-4B           Qwen/Qwen3.5-9B
#
# Examples:
#   run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb semi 0.1 --log-odds
#   run 1 2 scripts/run_train_semi.sh meta-llama/Llama-3.1-8B plausibleqa sft semi 0.1
#   run 1 2 scripts/run_train_semi.sh Qwen/Qwen2.5-7B ambigqa comb labelonly 0.1 --log-odds
#   run 1 2 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb semi 0.1 --log-odds
#   run 1 2 scripts/run_train_semi.sh Qwen/Qwen3.5-9B ambigqa comb semi 0.1 --log-odds

source /u/jdr/venvs/venv_lexcons/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=$1
TASK=$2
LOSS=$3
SEMI_MODE=$4
RATIO=$5
GD_MODE="g"
TYPCORR=""
LENORM=""
LOG_ODDS=""
SPLIT_SEED=""
DELTA_OVERRIDE=""
SAMPLES_OVERRIDE=""
DISC_SHOTS=""
INCLUDE_EOS=""
MODELS_DIR=""
FORCE_SAME_X="--force-same-x"
MAX_SEQ_LEN=""
# --script lets the caller swap the python entry point (e.g. fix1 fork).
# Default keeps the historical behavior (ranking_loss_ref.py).
SCRIPT="ranking_loss_ref.py"
SHAPE_WEIGHTS=""
# Default training params (unchanged from original behavior). Both can be
# overridden via --epochs N / --batch-size N flags below. Defaults preserved
# so existing callers (parent script, older launchers) see no change.
EPOCHS_OVERRIDE=""
BATCH_SIZE_FLAG=""
DELTA_BINS_FLAG=""
# --gemma4-lora is an opt-in flag for ranking_loss_ref_fix.py only. Off by default.
# When set, the python script switches LoRA target_modules to a regex that finds
# Gemma 4's Gemma4ClippableLinear-wrapped projections, and skips merge_and_unload
# at save time (eval loads the PEFT adapter directly via eval_by_claude.py).
GEMMA4_LORA=""
shift 5
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) GD_MODE=$2; shift 2 ;;
        --typcorr) TYPCORR="--typicality-correction"; shift ;;
        --self-typcorr) TYPCORR="--self-typicality"; shift ;;
        --neg-typcorr) TYPCORR="--neg-typicality"; shift ;;
        --lenorm) LENORM="--length-normalize"; shift ;;
        --log-odds) LOG_ODDS="--validator-log-odds"; shift ;;
        --split-seed) SPLIT_SEED="--split-seed $2"; shift 2 ;;
        --delta) DELTA_OVERRIDE=$2; shift 2 ;;
        --delta-bins) DELTA_BINS_FLAG="--delta-bins $2"; shift 2 ;;
        --samples) SAMPLES_OVERRIDE=$2; shift 2 ;;
        --disc-shots) DISC_SHOTS="--disc-shots $2"; shift 2 ;;
        --include-eos) INCLUDE_EOS="--include-eos"; shift ;;
        --models-dir) MODELS_DIR="--models-dir $2"; shift 2 ;;
        --no-force-same-x) FORCE_SAME_X=""; shift ;;
        --max-seq-len) MAX_SEQ_LEN="--max-seq-len $2"; shift 2 ;;
        --script) SCRIPT=$2; shift 2 ;;
        --epochs) EPOCHS_OVERRIDE=$2; shift 2 ;;
        --batch-size) BATCH_SIZE_FLAG="--batch-size $2"; shift 2 ;;
        --shape-weights)
            # Format: "case_a,mixed_neg,mixed_pos,both_u" e.g. "0.2,0.2,0.2,0.4"
            IFS=',' read -ra _SW <<< "$2"
            SHAPE_WEIGHTS="--shape-weight-case-a ${_SW[0]} --shape-weight-mixed-neg ${_SW[1]} --shape-weight-mixed-pos ${_SW[2]} --shape-weight-both-u ${_SW[3]}"
            shift 2 ;;
        --gemma4-lora) GEMMA4_LORA="--gemma4-lora"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$TASK" ] || [ -z "$LOSS" ] || [ -z "$SEMI_MODE" ] || [ -z "$RATIO" ]; then
    echo "Usage: $0 <MODEL> <TASK> <LOSS> <SEMI_MODE> <RATIO> [options]"
    echo ""
    echo "  MODEL:      HuggingFace model (google/gemma-2-2b, meta-llama/Llama-3.1-8B, ...)"
    echo "  TASK:       task name (plausibleqa, ambigqa, hypernym-concat-bananas-to-dogs-double, ifeval-concat, ...)"
    echo "  LOSS:       sft | comb | pref-only"
    echo "  SEMI_MODE:  semi | labelonly"
    echo "  RATIO:      labeled fraction (0.1, 0.25, 0.5, ...)"
    echo ""
    echo "  --mode g|d          generator (delta=0.15) or discriminator (delta=2.5) [default: g]"
    echo "  --typcorr           GPT-2 typicality correction"
    echo "  --self-typcorr      self-typicality correction"
    echo "  --neg-typcorr       negated-prompt typicality correction (LLR)"
    echo "  --lenorm            length normalization"
    echo "  --log-odds          validator log-odds"
    echo "  --split-seed S      prompt split seed (default 42)"
    echo "  --delta D           override delta (overrides --mode default)"
    echo "  --samples N         override total_samples"
    exit 1
fi

case $LOSS in
    sft)
        NLL_V=1; NLL_G=1; PREF=0
        ;;
    comb)
        NLL_V=1; NLL_G=1; PREF=1
        ;;
    pref-only)
        NLL_V=0; NLL_G=0; PREF=1
        ;;
    *)
        echo "Unknown LOSS: $LOSS (must be sft, comb, or pref-only)"
        exit 1
        ;;
esac

case $SEMI_MODE in
    semi)
        SEMI_FLAG="--semi-supervised $RATIO"
        ;;
    labelonly)
        SEMI_FLAG="--labeled-only $RATIO"
        ;;
    *)
        echo "Unknown SEMI_MODE: $SEMI_MODE (must be semi or labelonly)"
        exit 1
        ;;
esac

# g mode -> delta=0.15, d mode -> delta=2.5
case $GD_MODE in
    g) GD="g"; DELTA="0.15" ;;
    d) GD="d"; DELTA="2.5" ;;
    *) echo "Unknown --mode: $GD_MODE (must be g or d)"; exit 1 ;;
esac
SAMPLES_FLAG=""

[ -n "$DELTA_OVERRIDE" ] && DELTA="$DELTA_OVERRIDE"
[ -n "$SAMPLES_OVERRIDE" ] && SAMPLES_FLAG="--total_samples $SAMPLES_OVERRIDE"

NUM_EPOCHS=3
[ -n "$EPOCHS_OVERRIDE" ] && NUM_EPOCHS=$EPOCHS_OVERRIDE
LORA_FLAG=""
# TODO: extend this check if we ever use a smaller Qwen (e.g. Qwen3.5-2B) or any
# model that names its size with uppercase "-2B" — currently the substring match
# is case-sensitive ("-2b" only), so "-2B" models would unexpectedly get LoRA.
if [[ "$MODEL" != *"-2b"* && "$MODEL" != *"-2b-"* ]]; then
    LORA_FLAG="--lora"
fi

cd "$(dirname "$0")"

label="$TASK, $GD, delta=$DELTA, loss=$LOSS, $SEMI_MODE $RATIO"
[ "$TYPCORR" = "--self-typicality" ] && label="$label, self-typcorr"
[ "$TYPCORR" = "--neg-typicality" ] && label="$label, neg-typcorr"
[ "$TYPCORR" = "--typicality-correction" ] && label="$label, typcorr"
[ -n "$LENORM" ] && label="$label, lenorm"
[ -n "$LOG_ODDS" ] && label="$label, log-odds"
[ -n "$LORA_FLAG" ] && label="$label, lora"
[ -n "$INCLUDE_EOS" ] && label="$label, eos"

echo "========================================"
echo "Task:  $TASK"
echo "Model: $MODEL"
echo "Config: $label"
echo "  nll_validator=$NLL_V  nll_generator=$NLL_G  preference=$PREF"
echo "  $SEMI_FLAG"
echo "========================================"


echo "  SCRIPT: $SCRIPT"
[ -n "$SHAPE_WEIGHTS" ] && echo "  SHAPE_WEIGHTS: $SHAPE_WEIGHTS"

python "$SCRIPT" \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --task $TASK \
    --train_g_or_d $GD \
    --split_type random \
    --nll_validator_weight $NLL_V \
    --nll_generator_weight $NLL_G \
    --preference_loss_weight $PREF \
    --all \
    --delta $DELTA \
    $FORCE_SAME_X \
    $SAMPLES_FLAG \
    $TYPCORR \
    $LENORM \
    $LOG_ODDS \
    $SEMI_FLAG \
    $SPLIT_SEED \
    $DISC_SHOTS \
    $LORA_FLAG \
    $INCLUDE_EOS \
    $MODELS_DIR \
    $MAX_SEQ_LEN \
    $SHAPE_WEIGHTS \
    $BATCH_SIZE_FLAG \
    $DELTA_BINS_FLAG \
    $GEMMA4_LORA

STATUS=$?
echo ""
if [ "$STATUS" -eq 0 ]; then
    echo "Finished: $TASK ($LOSS, $SEMI_MODE $RATIO)"
else
    echo "FAILED (exit $STATUS): $TASK ($LOSS, $SEMI_MODE $RATIO)"
fi
exit $STATUS
