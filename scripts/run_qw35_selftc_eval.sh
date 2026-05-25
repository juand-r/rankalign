#!/usr/bin/env bash
# run_qw35_selftc_eval.sh DATASET SETTING
#
# Runs the NON-base-typicality eval pass for qw35 pods:
#   --self-typicality  (no --base-typicality)  → scores_self-* files
#   --neg-typicality   (no --base-typicality)  → scores_neg-* files
#
# The main cell script (run_qwen35_cell.sh) already ran WITH --base-typicality,
# producing basetyp-/basetypneg- files. This script produces the complementary
# self-/neg- variants using the scoring model itself as the TC reference.
#
# Idempotent: eval_by_claude.py skips tasks whose score CSV already exists.
#
# Usage (on the pod, from /workspace/rankalign/scripts/):
#   bash run_qw35_selftc_eval.sh persona s1
#   bash run_qw35_selftc_eval.sh membership s2
#   bash run_qw35_selftc_eval.sh ifeval s4

set -euo pipefail
cd /workspace/rankalign/scripts

# Activate venv so eval_by_claude.py can import tqdm, transformers, etc.
source /workspace/.venv/bin/activate

DATASET="${1:?DATASET required (persona|membership|ifeval)}"
SETTING="${2:?SETTING required (s1|s2|s4|s7)}"

LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/selftc_${DATASET}_${SETTING}.log"
exec > >(tee -a "$LOG") 2>&1

MODEL="Qwen/Qwen3.5-9B"
OUTPUTS_DIR="/workspace/rankalign/outputs"
MODELS_DIR="/workspace/models_q35"

echo "[$(date -u +%FT%TZ)] === run_qw35_selftc_eval.sh DATASET=$DATASET SETTING=$SETTING ==="

# ── Eval task list ─────────────────────────────────────────────────────────
EVAL_TASKS=()
case "$DATASET" in
    persona)
        TASK="persona-v1"
        for persona in psychopathy machiavellianism narcissism \
                       desire-to-create-allies interest-in-music interest-in-science; do
            EVAL_TASKS+=("persona-v1-${persona}")
        done
        ;;
    membership)
        TASK="membership-sans-rosch-v0"
        for cat in bird carpenters-tool clothing fruit furniture sport toy vehicle vegetable weapon; do
            EVAL_TASKS+=("rosch-${cat}")
        done
        # membership-sans-rosch-v0 has 0 gen-mode examples with random/zero-shot setup → skip
        ;;
    ifeval)
        TASK="ifeval-concat"
        IFEVAL_DATA_DIR="/workspace/rankalign/data/fixed-prompts-ifeval"
        # All prompts: OOD (1-21) + ID (22+) for self/neg TC
        mapfile -t _all_ns < <(
            for f in "$IFEVAL_DATA_DIR"/gpt_ifeval_results_prompt_*.jsonl; do
                n="${f##*_prompt_}"; n="${n%.jsonl}"
                [[ "$n" =~ ^[0-9]+$ ]] && echo "$n"
            done | sort -n
        )
        for n in "${_all_ns[@]}"; do
            EVAL_TASKS+=("ifeval-prompt_${n}")
        done
        ;;
    *)
        echo "Unknown DATASET: $DATASET"; exit 1 ;;
esac

echo "[$(date -u +%FT%TZ)] ${#EVAL_TASKS[@]} eval tasks: ${EVAL_TASKS[*]}"

# ── Eval modes (which TC variants to run without base model) ──────────────
case "$SETTING" in
    s1|s2) SELF_MODES="--self-typicality --neg-typicality" ;;
    s4)    SELF_MODES="--self-typicality" ;;
    s7)    SELF_MODES="--neg-typicality" ;;
    *)     echo "Unknown SETTING: $SETTING"; exit 1 ;;
esac

# ── Locate eval model symlink (created by cell script) ──────────────────
EVAL_MODEL_DIR="/workspace/eval_model_${SETTING}"
if [[ ! -L "$EVAL_MODEL_DIR" && ! -d "$EVAL_MODEL_DIR" ]]; then
    echo "ERROR: $EVAL_MODEL_DIR not found — did the cell script finish?" >&2
    exit 1
fi
echo "[$(date -u +%FT%TZ)] eval model: $EVAL_MODEL_DIR -> $(readlink -f $EVAL_MODEL_DIR 2>/dev/null || echo '?')"

# ── Run evals ────────────────────────────────────────────────────────────
for MODE in $SELF_MODES; do
    echo "[$(date -u +%FT%TZ)] --- self-TC eval mode: $MODE (no base model) ---"
    for EVAL_TASK in "${EVAL_TASKS[@]}"; do
        echo "[$(date -u +%FT%TZ)] eval: $EVAL_TASK"
        python eval_by_claude.py \
            --model "$EVAL_MODEL_DIR" \
            --task "$EVAL_TASK" \
            --split_type random \
            --disc-shots zero \
            --gen-shots zero \
            --outputs-dir "$OUTPUTS_DIR" \
            --validator-log-odds \
            $MODE \
            --save-scores-csv
        echo "[$(date -u +%FT%TZ)] done: $EVAL_TASK ($MODE)"
    done
    echo "[$(date -u +%FT%TZ)] mode $MODE complete"
done

# ── Basetyp/basetypneg on ID ifeval prompts (N > 21) ─────────────────────────
# Cell script already ran basetyp/basetypneg on OOD (N <= 21). This covers ID.
if [[ "$DATASET" == "ifeval" ]]; then
    IFEVAL_DATA_DIR="/workspace/rankalign/data/fixed-prompts-ifeval"
    mapfile -t _id_ns < <(
        for f in "$IFEVAL_DATA_DIR"/gpt_ifeval_results_prompt_*.jsonl; do
            n="${f##*_prompt_}"; n="${n%.jsonl}"
            [[ "$n" =~ ^[0-9]+$ ]] && (( n > 21 )) && echo "$n"
        done | sort -n
    )
    ID_TASKS=()
    for n in "${_id_ns[@]}"; do ID_TASKS+=("ifeval-prompt_${n}"); done
    echo "[$(date -u +%FT%TZ)] basetyp/basetypneg on ${#ID_TASKS[@]} ID tasks (N > 21)"

    # basetyp (PMI): --base-typicality --self-typicality — for s1/s2/s4
    if [[ "$SETTING" != "s7" ]]; then
        echo "[$(date -u +%FT%TZ)] --- basetyp (PMI) on ID tasks ---"
        for EVAL_TASK in "${ID_TASKS[@]}"; do
            echo "[$(date -u +%FT%TZ)] basetyp: $EVAL_TASK"
            python eval_by_claude.py \
                --model "$EVAL_MODEL_DIR" \
                --task "$EVAL_TASK" \
                --split_type random \
                --disc-shots zero \
                --gen-shots zero \
                --outputs-dir "$OUTPUTS_DIR" \
                --validator-log-odds \
                --base-typicality --self-typicality \
                --base-model-name "$MODEL" \
                --save-scores-csv
        done
        echo "[$(date -u +%FT%TZ)] basetyp ID done"
    fi

    # basetypneg (Neg): --base-typicality --neg-typicality — for s1/s2/s7
    if [[ "$SETTING" != "s4" ]]; then
        echo "[$(date -u +%FT%TZ)] --- basetypneg (Neg) on ID tasks ---"
        for EVAL_TASK in "${ID_TASKS[@]}"; do
            echo "[$(date -u +%FT%TZ)] basetypneg: $EVAL_TASK"
            python eval_by_claude.py \
                --model "$EVAL_MODEL_DIR" \
                --task "$EVAL_TASK" \
                --split_type random \
                --disc-shots zero \
                --gen-shots zero \
                --outputs-dir "$OUTPUTS_DIR" \
                --validator-log-odds \
                --base-typicality --neg-typicality \
                --base-model-name "$MODEL" \
                --save-scores-csv
        done
        echo "[$(date -u +%FT%TZ)] basetypneg ID done"
    fi
fi

touch "/workspace/SELFTC_${DATASET}_${SETTING}_DONE"
echo "[$(date -u +%FT%TZ)] === SELFTC EVAL ALL DONE: DATASET=$DATASET SETTING=$SETTING ==="
