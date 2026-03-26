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
#   --lenorm               Enable length normalization
#   --log-odds             Enable validator log-odds
#   --split-seed S         Seed for prompt split (default 42)
#   --delta D              Override default delta (overrides --mode default)
#   --samples N            Override default total_samples
#   --disc-shots zero|few  Override discriminator shots (default: auto based on model)
#
# Models:
#   google/gemma-2-2b          google/gemma-2-2b-it
#   google/gemma-2-9b          google/gemma-2-9b-it
#   meta-llama/Llama-3.1-8B   meta-llama/Llama-3.1-8B-Instruct
#   Qwen/Qwen2.5-7B           Qwen/Qwen2.5-7B-Instruct
#
# Examples:
#   run 1 2 scripts/run_train_semi.sh google/gemma-2-2b plausibleqa comb semi 0.1 --log-odds
#   run 1 2 scripts/run_train_semi.sh meta-llama/Llama-3.1-8B plausibleqa sft semi 0.1
#   run 1 2 scripts/run_train_semi.sh Qwen/Qwen2.5-7B ambigqa comb labelonly 0.1 --log-odds
#   run 1 2 scripts/run_train_semi.sh google/gemma-2-9b-it ifeval-concat comb semi 0.1 --log-odds

source /u/jdr/venvs/venv_lexcons/bin/activate

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
shift 5
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) GD_MODE=$2; shift 2 ;;
        --typcorr) TYPCORR="--typicality-correction"; shift ;;
        --self-typcorr) TYPCORR="--self-typicality"; shift ;;
        --lenorm) LENORM="--length-normalize"; shift ;;
        --log-odds) LOG_ODDS="--validator-log-odds"; shift ;;
        --split-seed) SPLIT_SEED="--split-seed $2"; shift 2 ;;
        --delta) DELTA_OVERRIDE=$2; shift 2 ;;
        --samples) SAMPLES_OVERRIDE=$2; shift 2 ;;
        --disc-shots) DISC_SHOTS="--disc-shots $2"; shift 2 ;;
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

cd "$(dirname "$0")"

label="$TASK, $GD, delta=$DELTA, loss=$LOSS, $SEMI_MODE $RATIO"
[ "$TYPCORR" = "--self-typicality" ] && label="$label, self-typcorr"
[ "$TYPCORR" = "--typicality-correction" ] && label="$label, typcorr"
[ -n "$LENORM" ] && label="$label, lenorm"
[ -n "$LOG_ODDS" ] && label="$label, log-odds"

echo "========================================"
echo "Task:  $TASK"
echo "Model: $MODEL"
echo "Config: $label"
echo "  nll_validator=$NLL_V  nll_generator=$NLL_G  preference=$PREF"
echo "  $SEMI_FLAG"
echo "========================================"

python ranking_loss_ref.py \
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
    --force-same-x \
    $SAMPLES_FLAG \
    $TYPCORR \
    $LENORM \
    $LOG_ODDS \
    $SEMI_FLAG \
    $SPLIT_SEED \
    $DISC_SHOTS

echo ""
echo "Finished: $TASK ($LOSS, $SEMI_MODE $RATIO)"
