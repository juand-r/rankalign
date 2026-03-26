#!/bin/bash

# Train ranking loss on a PlausibleQA task.
#
# Usage: run 1 <hours> scripts/run_train_plausibleqa_v0.sh <TASK> <MODE> <LOSS> [options]
#
# Arguments:
#   TASK      PlausibleQA task name, e.g. plausibleqa or plausibleqa-nq_1109
#   MODE      Training mode: g | d | both | all
#               g    = train generator      (delta=0.15)
#               d    = train discriminator   (delta=2.5)
#               both = g then d sequentially
#               all  = all 4 configs (g, g+typcorr, d, d+typcorr)
#   LOSS      Loss setting: sft | comb | pref-only
#               sft       = nll_validator=1, nll_generator=1, preference=0
#               comb      = nll_validator=1, nll_generator=1, preference=1
#               pref-only = nll_validator=0, nll_generator=0, preference=1
#
# Options:
#   --typcorr              Enable GPT-2 typicality correction (ignored when MODE=all)
#   --self-typcorr         Enable self-typicality correction (model as its own prior)
#   --lenorm               Enable length normalization for generator scores
#   --log-odds             Enable validator log-odds (log(P(Yes)/P(No)))
#   --semi-supervised R    Semi-supervised: R of prompts labeled (full loss), rest pref-only
#   --labeled-only R       Train only on R fraction of prompts (discard rest)
#   --split-seed S         Seed for labeled/unlabeled split (default 42)
#
# Examples:
#   run 1 3 scripts/run_train_plausibleqa_v0.sh plausibleqa d comb --typcorr
#   run 1 3 scripts/run_train_plausibleqa_v0.sh plausibleqa g sft --self-typcorr --log-odds
#   run 1 6 scripts/run_train_plausibleqa_v0.sh plausibleqa all sft
#   run 1 3 scripts/run_train_plausibleqa_v0.sh plausibleqa d comb --semi-supervised 0.1 --log-odds
#   run 1 3 scripts/run_train_plausibleqa_v0.sh plausibleqa d comb --labeled-only 0.1 --log-odds

source /u/jdr/venvs/venv_lexcons/bin/activate

TASK=$1
MODE=$2
LOSS=$3
TYPCORR=""
LENORM=""
LOG_ODDS=""
SEMI_FLAG=""
SPLIT_SEED=""
shift 3
while [[ $# -gt 0 ]]; do
    case $1 in
        --typcorr) TYPCORR="--typicality-correction"; shift ;;
        --self-typcorr) TYPCORR="--self-typicality"; shift ;;
        --lenorm) LENORM="--length-normalize"; shift ;;
        --log-odds) LOG_ODDS="--validator-log-odds"; shift ;;
        --semi-supervised) SEMI_FLAG="--semi-supervised $2"; shift 2 ;;
        --labeled-only) SEMI_FLAG="--labeled-only $2"; shift 2 ;;
        --split-seed) SPLIT_SEED="--split-seed $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$TASK" ] || [ -z "$MODE" ] || [ -z "$LOSS" ]; then
    echo "Usage: $0 <TASK> <MODE> <LOSS> [--typcorr] [--lenorm] [--log-odds]"
    echo ""
    echo "  TASK:  plausibleqa task name (e.g. plausibleqa or plausibleqa-nq_1109)"
    echo "  MODE:  g | d | both | all"
    echo "  LOSS:  sft | comb | pref-only"
    echo "  --typcorr:           enable GPT-2 typicality correction"
    echo "  --self-typcorr:      enable self-typicality correction"
    echo "  --lenorm:            enable length normalization"
    echo "  --log-odds:          enable validator log-odds"
    echo "  --semi-supervised R: semi-supervised (R = labeled ratio)"
    echo "  --labeled-only R:    train on labeled subset only (R = ratio)"
    echo "  --split-seed S:      seed for prompt split (default 42)"
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

MODEL="google/gemma-2-2b"
NUM_EPOCHS=3
TOTAL_SAMPLES=1970

cd "$(dirname "$0")"

run_config() {
    local gd=$1
    local delta=$2
    local tc=$3

    local label="$gd, delta=$delta, loss=$LOSS"
    [ "$tc" = "--self-typicality" ] && label="$label, self-typcorr"
    [ "$tc" = "--typicality-correction" ] && label="$label, typcorr"
    [ -n "$LENORM" ] && label="$label, lenorm"
    [ -n "$LOG_ODDS" ] && label="$label, log-odds"
    [ -n "$SEMI_FLAG" ] && label="$label, $SEMI_FLAG"

    echo "========================================"
    echo "Task:  $TASK"
    echo "Model: $MODEL"
    echo "Config: $label"
    echo "  nll_validator=$NLL_V  nll_generator=$NLL_G  preference=$PREF"
    echo "========================================"

    python ranking_loss_ref.py \
        --model $MODEL \
        --num_epochs $NUM_EPOCHS \
        --task $TASK \
        --train_g_or_d $gd \
        --split_type random \
        --nll_validator_weight $NLL_V \
        --nll_generator_weight $NLL_G \
        --preference_loss_weight $PREF \
        --all \
        --delta $delta \
        --total_samples $TOTAL_SAMPLES \
        --force-same-x \
        $tc \
        $LENORM \
        $LOG_ODDS \
        $SEMI_FLAG \
        $SPLIT_SEED

    echo ""
}

case $MODE in
    g)
        run_config g 0.15 "$TYPCORR"
        ;;
    d)
        run_config d 2.5 "$TYPCORR"
        ;;
    both)
        run_config g 0.15 "$TYPCORR"
        run_config d 2.5 "$TYPCORR"
        ;;
    all)
        run_config g 0.15 ""
        run_config g 0.15 "--typicality-correction"
        run_config d 2.5 ""
        run_config d 2.5 "--typicality-correction"
        ;;
    *)
        echo "Unknown MODE: $MODE (must be g, d, both, or all)"
        exit 1
        ;;
esac

echo "Finished: $TASK ($MODE, $LOSS)"
