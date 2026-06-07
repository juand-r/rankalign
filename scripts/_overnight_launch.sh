#!/bin/bash
# Overnight unified train+eval launcher (2026-05-24).
#
# Usage:
#   bash scripts/_overnight_launch.sh DATASET MODEL SETTING
#     DATASET in {membership, persona, ifeval, humaneval}
#     MODEL   in {gemma-2-2b-it, gemma-2-9b-it, gemma-4-31B-it} (or any HF id;
#                 just the bare name; humaneval requires gemma-4-31B-it)
#     SETTING in {s1, s2, s3, s4, s5, s6, s7, s11, s12, s13}
#
#   s13 = SFT + --consistency-ft (validator/generator agreement filter; argparse
#         enforces pref=0, NLLs>0, fsx OFF). Mirrors SFT-lo (s1) flag set
#         otherwise: --labeled-only 0.1, no fsx, no vlo, no tc.
#
#   humaneval + gemma-4-31B-it triggers special handling:
#     - VENV=/datastor2/jdr/venvs/gemma4 (transformers 5.x)
#     - --gemma4-lora (regex target_modules; skip merge_and_unload)
#     - --gradient-checkpointing
#     - 3 GPUs (per gemma-4 doc + user request)
#     - Eval task list enumerated dynamically from data/humaneval/v2.1correct-upper
#
# Submits two slurm jobs:
#   1. Train job via ~/.local/bin/run -> scripts/run_train_semi.sh ...
#   2. Eval job via direct sbatch with --dependency=afterany:<train_jobid>
#
# Env:
#   MODELS_DIR     default /datastor2/jdr/rankalign/models2
#   OUTPUTS_DIR    default ../outputs
#   DRYRUN=1       print only, don't submit
#   FORCE_RETRAIN  if not set: skip launching train if model dir already exists
#   FORCE_REEVAL   if not set: skip launching eval if score CSV already exists
#                  (the eval script already skips per-task; this skips the SBATCH)
#
# Output:
#   - Records (train_jobid, eval_jobid, paths) into
#     overnight/_overnight_jobids.txt for the loop monitor.
#   - The train command goes verbatim into docs/overnight_progress.md.

set -euo pipefail

DATASET="${1:?DATASET required (membership|persona|ifeval)}"
MODEL_NAME="${2:?MODEL required (e.g. gemma-2-2b-it or google/gemma-2-9b-it)}"
SETTING="${3:?SETTING required (s1..s7, s11, s12)}"

# Normalize MODEL: allow either bare or google/... form.
case "$MODEL_NAME" in
    google/*) MODEL="$MODEL_NAME" ;;
    *)        MODEL="google/$MODEL_NAME" ;;
esac

# Map DATASET to (TASK, EVAL_TASKS, GPUS, TRAIN_HOURS, EVAL_HOURS, TRAIN_MEM, MAX_SEQ).
case "$DATASET" in
    membership)
        TASK="membership-sans-rosch-v0"
        EVAL_TASKS="rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit rosch-furniture rosch-sport rosch-toy rosch-vehicle rosch-vegetable rosch-weapon"
        GPUS=1
        # Observed ~2s/it for 2b on persona-v1 (similar size). Membership has
        # 5113 samples * 3 epochs ~= 8.5h on 2b. 9b is ~3x. Add slack.
        case "$MODEL" in
            *9b-it*) TRAIN_HOURS=24 ; EVAL_HOURS=4 ;;
            *)       TRAIN_HOURS=12 ; EVAL_HOURS=3 ;;
        esac
        TRAIN_MEM=64G
        EVAL_MEM=48G
        MAX_SEQ_FLAG=""
        ;;
    persona)
        TASK="persona-v1"
        EVAL_TASKS="persona-v1-psychopathy persona-v1-machiavellianism persona-v1-narcissism persona-v1-desire-to-create-allies persona-v1-interest-in-music persona-v1-interest-in-science"
        GPUS=1
        # 5110 samples * 3 epochs at 2s/it -> ~8.5h for 2b; ~2-3x for 9b LoRA.
        case "$MODEL" in
            *9b-it*) TRAIN_HOURS=20 ; EVAL_HOURS=2 ;;
            *)       TRAIN_HOURS=10 ; EVAL_HOURS=2 ;;
        esac
        TRAIN_MEM=64G
        EVAL_MEM=48G
        MAX_SEQ_FLAG=""
        ;;
    ifeval)
        TASK="ifeval-concat"
        # Enumerate ALL registered ifeval prompts from the data dir, NOT seq 1..21.
        # The prompt IDs are non-contiguous (gaps at 14,31,55,60,62,69,71,81,86,101)
        # and number 99 total. seq 1..21 was both too few (missed the in-domain
        # prompts 22+) and wrong (assumed contiguous IDs). Mirror the humaneval
        # branch: glob the data files -> ifeval-prompt_<N>. Prompts 1..21 are the
        # held-out TEST-ONLY set; 22+ are in-domain (within-prompt held-out half).
        REPO_ROOT_FOR_TASKS="$(cd "$(dirname "$0")/.." && pwd)"
        EVAL_TASKS=$(ls "$REPO_ROOT_FOR_TASKS/data/fixed-prompts-ifeval/gpt_ifeval_results_prompt_"*.jsonl 2>/dev/null \
            | xargs -n1 basename 2>/dev/null \
            | sed -E 's/gpt_ifeval_results_(prompt_[0-9]+)\.jsonl/ifeval-\1/' \
            | sort -t_ -k2 -n | tr '\n' ' ')
        EVAL_TASKS="${EVAL_TASKS% }"
        if [ -z "$EVAL_TASKS" ]; then
            echo "FATAL: ifeval task enumeration produced empty list (data dir missing?)"
            exit 1
        fi
        # ifeval-concat: ~5110 samples but longer prompts; longer wall.
        # 9b-it on 2 GPUs (model parallel) helps but still slow.
        case "$MODEL" in
            *9b-it*) GPUS=2 ; TRAIN_HOURS=30 ; EVAL_HOURS=5 ;;
            *)       GPUS=1 ; TRAIN_HOURS=14 ; EVAL_HOURS=4 ;;
        esac
        TRAIN_MEM=96G
        EVAL_MEM=64G
        # ifeval has long prompts; cap seq len to keep VRAM in check.
        MAX_SEQ_FLAG="--max-seq-len 1024"
        # ifeval requires --disc-shots zero: src/utils.py make_prompt_ifeval
        # raises NotImplementedError for shots != "zero" in the discriminator
        # branch (line 817). Default few-shot crashes the train at first step.
        DISC_SHOTS="${DISC_SHOTS:-zero}"
        ;;
    humaneval)
        # humaneval-v2.1correct-upper: gemma-4-31B-it only (per user, 2026-05-24).
        # Task list enumerated dynamically from data/humaneval/v2.1correct-upper/.
        TASK="humaneval-v2.1correct-upper"
        DATASET_DIR="v2.1correct-upper"
        REPO_ROOT_FOR_TASKS="$(cd "$(dirname "$0")/.." && pwd)"
        EVAL_TASKS=$(ls "$REPO_ROOT_FOR_TASKS/data/humaneval/${DATASET_DIR}/humaneval_"*.csv 2>/dev/null \
            | xargs -n1 basename 2>/dev/null | sed 's/\.csv$//' \
            | sed "s/^/${TASK}-/" | tr '\n' ' ')
        EVAL_TASKS="${EVAL_TASKS% }"
        if [ -z "$EVAL_TASKS" ]; then
            echo "FATAL: humaneval task enumeration produced empty list (data dir missing?)"
            exit 1
        fi
        # Gemma-4-31B-it is the only supported model here. Enforce.
        case "$MODEL" in
            *gemma-4-31B-it*|*gemma-4-31b-it*) ;;
            *) echo "FATAL: humaneval DATASET requires google/gemma-4-31B-it (got $MODEL)"; exit 1 ;;
        esac
        GPUS=3
        TRAIN_HOURS=24
        EVAL_HOURS=8
        TRAIN_MEM=192G
        EVAL_MEM=128G
        MAX_SEQ_FLAG=""
        # Gemma-4 humaneval uses --disc-shots zero per ref script
        # run_settings_v21correct_upper.sh; gemma-2 still defaults to "few".
        DISC_SHOTS="${DISC_SHOTS:-zero}"
        ;;
    *)
        echo "Unknown DATASET: $DATASET (membership|persona|ifeval)"
        exit 1
        ;;
esac

# Map SETTING to (LOSS, SEMI_MODE, FSX_FLAG, TC_FLAG, LOGODDS_FLAG, TC_EVAL_LIST,
# DIR_SUFFIX_FRAGMENTS).
#
# DIR_SUFFIX_FRAGMENTS is the part of the v7 dir name after the alpha-prefix and
# before --semi*--fix1, used to construct the expected save path. We assemble it
# in the same order as ranking_loss_ref_fix.py @ line 2182.
#
# The TC_EVAL_LIST is space-separated, each token is one of:
#   self  -> --self-typcorr
#   neg   -> --neg-typcorr
# and we always also include --base-typcorr for each.
build_setting() {
    SET_FLAGS=""
    TC_LABEL=""
    PREF_STR=""
    NLLV_STR=""
    NLLG_STR=""
    FSX_STR=""
    PPD_STR=""
    VLO_STR=""
    SEMI_STR=""
    CFT_STR=""
    CONSISTENCY_FT_FLAG=""

    case "$SETTING" in
        s1)   # SFT-lo: sft + labelonly + no fsx
            LOSS="sft" ; SEMI_MODE="labelonly" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="" ; TC_LABEL=""
            LOGODDS_FLAG=""
            # No-TC training: eval BOTH self-TC and neg-TC (matches v6 policy).
            # (was "self" only — caused missing Neg-self/Neg-base columns.)
            TC_EVAL_LIST="self neg"
            PREF_STR="--pref0.0" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="" ; PPD_STR=""
            VLO_STR=""
            SEMI_STR="--labelonly0.1"
            ;;
        s2)   # RankAlign: pref-only + semi + no fsx
            LOSS="pref-only" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="" ; TC_LABEL=""
            LOGODDS_FLAG=""
            # No-TC training: eval BOTH self-TC and neg-TC (matches v6 policy).
            TC_EVAL_LIST="self neg"
            PREF_STR="" ; NLLV_STR="" ; NLLG_STR=""
            FSX_STR="" ; PPD_STR=""
            VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s3)   # New+fsx: comb + semi + log-odds + fsx (+ ppd + sbm-global)
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG=""  # default ON
            TC_FLAG="" ; TC_LABEL=""
            LOGODDS_FLAG="--log-odds"
            # No-TC training: eval BOTH self-TC and neg-TC (matches v6 policy).
            TC_EVAL_LIST="self neg"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x" ; PPD_STR="--ppd"
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s4)   # New+fsx+selfTC
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG=""
            TC_FLAG="--self-typcorr" ; TC_LABEL="--tc-self"
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x" ; PPD_STR="--ppd"
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s5)   # RankAlign+fsx+selfTC (per IRP §1: drop --validator-log-odds)
            LOSS="pref-only" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG=""
            TC_FLAG="--self-typcorr" ; TC_LABEL="--tc-self"
            LOGODDS_FLAG=""  # explicitly NO vlo (was a historical bug)
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="" ; NLLG_STR=""
            FSX_STR="--force-same-x" ; PPD_STR="--ppd"
            VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s6)   # RankAlign+selfTC (no fsx)
            LOSS="pref-only" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="--self-typcorr" ; TC_LABEL="--tc-self"
            LOGODDS_FLAG=""
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="" ; NLLG_STR=""
            FSX_STR="" ; PPD_STR=""
            VLO_STR=""
            SEMI_STR="--semi0.1"
            ;;
        s7)   # New+fsx+negTC
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG=""
            TC_FLAG="--neg-typcorr" ; TC_LABEL="--tc-neg"
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="neg"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="--force-same-x" ; PPD_STR="--ppd"
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s11)  # New+selfTC (no fsx)
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="--self-typcorr" ; TC_LABEL="--tc-self"
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="self"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="" ; PPD_STR=""
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s12)  # New+negTC (no fsx)
            LOSS="comb" ; SEMI_MODE="semi" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="--neg-typcorr" ; TC_LABEL="--tc-neg"
            LOGODDS_FLAG="--log-odds"
            TC_EVAL_LIST="neg"
            PREF_STR="" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="" ; PPD_STR=""
            VLO_STR="--vallogodds"
            SEMI_STR="--semi0.1"
            ;;
        s13)  # SFT + consistency-ft: SFT-lo (s1) flag set + --consistency-ft.
              # Argparse enforces pref=0, NLLv>0, NLLg>0, fsx OFF.
              # CFT_STR adds --cft to the save-dir between {ppd}{cft} positions
              # (see ranking_loss_ref_fix.py L2182, edited 2026-05-24).
            LOSS="sft" ; SEMI_MODE="labelonly" ; RATIO="0.1"
            FSX_FLAG="--no-force-same-x"
            TC_FLAG="" ; TC_LABEL=""
            LOGODDS_FLAG=""
            TC_EVAL_LIST="self neg"
            PREF_STR="--pref0.0" ; NLLV_STR="--nllv1.0" ; NLLG_STR="--nllg1.0"
            FSX_STR="" ; PPD_STR=""
            VLO_STR=""
            SEMI_STR="--labelonly0.1"
            CFT_STR="--cft"          # only s13 uses this
            CONSISTENCY_FT_FLAG="--consistency-ft"
            ;;
        *)
            echo "Unknown SETTING: $SETTING"
            exit 1
            ;;
    esac
}

build_setting

# Optional WALLTIME env var overrides the per-task heuristic. Useful when
# the dispatcher caller knows the cluster is contended and a shorter
# walltime is more likely to backfill into a small slot.
if [ -n "${WALLTIME:-}" ]; then
    TRAIN_HOURS="$WALLTIME"
fi

# All the universal flags for fix1 overnight runs.
# disc-shots default = "few" for gemma-2 settings (per project policy);
# overridden to "zero" when the user explicitly opts in via DISC_SHOTS env
# var (see gemma-4 humaneval branch below).
DISC_SHOTS_FLAG="--disc-shots ${DISC_SHOTS:-few}"
COMMON_FLAGS=( --script ranking_loss_ref_fix.py
               $DISC_SHOTS_FLAG
               --delta-bins 10 )
[ -n "$MAX_SEQ_FLAG" ] && COMMON_FLAGS+=( $MAX_SEQ_FLAG )
[ -n "$FSX_FLAG" ]     && COMMON_FLAGS+=( $FSX_FLAG )
[ -n "$TC_FLAG" ]      && COMMON_FLAGS+=( $TC_FLAG )
[ -n "$LOGODDS_FLAG" ] && COMMON_FLAGS+=( $LOGODDS_FLAG )
[ -n "$CONSISTENCY_FT_FLAG" ] && COMMON_FLAGS+=( $CONSISTENCY_FT_FLAG )

# fsx settings get ppd + sbm-global. (Detect via FSX_STR which is set when the
# setting USES fsx.)
if [ -n "$FSX_STR" ]; then
    COMMON_FLAGS+=( --per-prompt-delta --shape-budget-mode global )
fi

# Gemma-4-31B-it special handling: different venv (transformers 5.x),
# --gemma4-lora (regex target_modules + skip merge_and_unload), and
# --gradient-checkpointing for VRAM. We DO NOT change anything for gemma-2.
USE_GEMMA4_LORA=0
case "$MODEL" in
    *gemma-4-31B-it*|*gemma-4-31b-it*)
        USE_GEMMA4_LORA=1
        COMMON_FLAGS+=( --gemma4-lora --gradient-checkpointing )
        # VENV is read by run_train_semi.sh and run_eval_semi.sh.
        VENV_OVERRIDE="${VENV_OVERRIDE:-/datastor2/jdr/venvs/gemma4}"
        ;;
esac

# ifeval has long prompts (max_seq_len=1024). On a single 44GB A40, even the
# 2b/2b-it model + activations OOMs during the consistency-ft pre-pass and
# during normal forward passes. Add --gradient-checkpointing for ifeval on
# any non-gemma-4 model. (gemma-4 already has it.) 2-GPU 9b-it ifeval doesn't
# strictly need it but adding doesn't hurt.
if [ "$DATASET" = "ifeval" ] && [ "$USE_GEMMA4_LORA" -eq 0 ]; then
    COMMON_FLAGS+=( --gradient-checkpointing )
fi

# Use absolute /datastor2 models dir to avoid /datastor1 fill-up.
MODELS_DIR="${MODELS_DIR:-/datastor2/jdr/rankalign/models2}"
COMMON_FLAGS+=( --models-dir "$MODELS_DIR" )

# Optional explicit wandb run name (env WANDB_RUN_NAME). Plumbed through to
# run_train_semi.sh -> trainer's --wandb_run_name so rerun launchers can tag
# runs (e.g. "rerun-wandb-...") and distinguish them from paper-era runs in the
# cloud. When unset, the trainer auto-generates the name (unchanged behavior).
if [ -n "${WANDB_RUN_NAME:-}" ]; then
    COMMON_FLAGS+=( --wandb_run_name "$WANDB_RUN_NAME" )
fi

# Persona-v1 task name in the dir is "persona-v1" (matches TASK var).
# Eval glob matches any saved epoch (0/1/2). The wrap uses `ls -dt | head -1`
# to pick the most recently saved epoch dir, so even a walltime-killed run
# (which only saved epoch0 or epoch1) is evaluable.
EPOCH_GLOB="[012]"
DELTA_PLACEHOLDER="DELTA"  # we don't know delta exactly until script runs;
                            # we'll glob-match instead.

# Build the variable-suffix string the python script appends.
# NOTE: order from ranking_loss_ref_fix.py L2182 (post-2026-05-24 cft edit):
#   {tc}{lenorm}{single}{full-completion}{eos}{pref}{nllv}{nllg}{fsx}{ppd}{cft}{valboost}{vallogodds}{semi}{fix1}
# with single, lenorm, eos, valboost all empty in our case.
SUFFIX="${TC_LABEL}--full-completion${PREF_STR}${NLLV_STR}${NLLG_STR}${FSX_STR}${PPD_STR}${CFT_STR}${VLO_STR}${SEMI_STR}--fix1"

MODEL_REPL=$(echo "$MODEL" | sed 's|/|--|g')

# Determine LoRA merge suffix: matches the python script's logic in run_train_semi.sh
# (LoRA flag set when MODEL doesn't contain -2b- substring), with the
# additional rule that --gemma4-lora SKIPS merge_and_unload at save time, so
# the eval target is the adapter dir (no _merged sibling).
USE_LORA=1
if [[ "$MODEL" == *"-2b"* || "$MODEL" == *"-2b-"* ]]; then
    USE_LORA=0
fi
MERGED_SUFFIX=""
if [ "$USE_LORA" -eq 1 ] && [ "$USE_GEMMA4_LORA" -eq 0 ]; then
    MERGED_SUFFIX="_merged"
fi

PATH_PREFIX="${MODELS_DIR}/v7-${MODEL_REPL}-delta"

# We don't know the exact delta until --delta-bins computes it from the data,
# so for the expected eval path we use a glob. The eval wrapper uses `ls -d`
# at sbatch runtime to pick the right one.
GLOB_PATH="${PATH_PREFIX}*-epoch${EPOCH_GLOB}--${TASK}-all--d2g--random--alpha1.0${SUFFIX}${MERGED_SUFFIX}"

OVERNIGHT_DIR="$(cd "$(dirname "$0")/.." && pwd)/overnight"
mkdir -p "$OVERNIGHT_DIR"
JOB_LOG="$OVERNIGHT_DIR/_overnight_jobids.txt"

# Eval task list - quote-protected.
read -r -a EVAL_TASKS_ARR <<< "$EVAL_TASKS"

# Build eval flags per TC variant
build_eval_flags() {
    local tc="$1"
    # /datastor2 is the default outputs dir to keep /datastor1 from filling up.
    # Both v7 table builders (_build_rosch_table_v7.py and
    # _build_persona_v1_table_v7.py) scan both /datastor1/.../outputs and
    # /datastor2/jdr/rankalign/outputs, so new CSVs in /datastor2 are picked up.
    local out_dir_flag="--outputs-dir ${OUTPUTS_DIR:-/datastor2/jdr/rankalign/outputs}"
    # Propagate disc-shots to the EVAL too. The DISC_SHOTS var is set per-DATASET
    # above (zero for ifeval/humaneval, else few). Without this, run_eval_semi.sh
    # falls back to its own default of "few", and ifeval few-shot disc prompts
    # raise NotImplementedError in make_prompt_ifeval -> every task crashes and
    # zero scores are written (2026-06-07 reruns hit exactly this).
    out_dir_flag="$out_dir_flag --disc-shots ${DISC_SHOTS:-few}"
    if [ "$tc" = "self" ]; then
        echo "--self-typcorr --base-typcorr --base-model $MODEL --log-odds $out_dir_flag"
    elif [ "$tc" = "neg" ]; then
        echo "--neg-typcorr --base-typcorr --base-model $MODEL --log-odds $out_dir_flag"
    else
        echo "--base-typcorr --base-model $MODEL --log-odds $out_dir_flag"
    fi
}

label="$DATASET-$(basename $MODEL)-$SETTING"

# Optional venv override (set above when MODEL is gemma-4-31B-it). When set,
# we prepend `VENV=...` to the wrapped command so run_train_semi.sh /
# run_eval_semi.sh source the right virtualenv. Default = lexcons (unchanged).
VENV_PREFIX=""
[ -n "${VENV_OVERRIDE:-}" ] && VENV_PREFIX="VENV=${VENV_OVERRIDE} "

echo "========================================"
echo "Overnight launch: $label"
echo "  TASK:          $TASK"
echo "  MODEL:         $MODEL  (LoRA=${USE_LORA}, gemma4-lora=${USE_GEMMA4_LORA})"
echo "  SETTING:       $SETTING (loss=$LOSS, semi_mode=$SEMI_MODE, fsx_flag='$FSX_FLAG', tc='$TC_FLAG', vlo='$LOGODDS_FLAG', cft='$CONSISTENCY_FT_FLAG')"
echo "  GPUs:          $GPUS"
echo "  TRAIN_HOURS:   $TRAIN_HOURS"
echo "  EVAL_HOURS:    $EVAL_HOURS"
echo "  VENV:          ${VENV_OVERRIDE:-/u/jdr/venvs/venv_lexcons (default)}"
echo "  COMMON_FLAGS:  ${COMMON_FLAGS[*]}"
echo "  EVAL TC list:  $TC_EVAL_LIST"
echo "  Expected save: ${GLOB_PATH}"
echo "========================================"

if [ -n "${DRYRUN:-}" ]; then
    echo "DRYRUN. Would run:"
    echo "  ${VENV_PREFIX}run $GPUS $TRAIN_HOURS --cpu 4 --mem $TRAIN_MEM scripts/run_train_semi.sh $MODEL $TASK $LOSS $SEMI_MODE $RATIO ${COMMON_FLAGS[*]}"
    exit 0
fi

# 1) Submit train.
# When VENV_PREFIX is set, run_train_semi.sh activates the override venv via
# its env-var hook (added 2026-05-24 alongside --consistency-ft).
if [ -n "$VENV_PREFIX" ]; then
    TRAIN_OUT=$(VENV="$VENV_OVERRIDE" run "$GPUS" "$TRAIN_HOURS" --cpu 4 --mem "$TRAIN_MEM" \
        scripts/run_train_semi.sh "$MODEL" "$TASK" "$LOSS" "$SEMI_MODE" "$RATIO" "${COMMON_FLAGS[@]}" 2>&1) || true
else
    TRAIN_OUT=$(run "$GPUS" "$TRAIN_HOURS" --cpu 4 --mem "$TRAIN_MEM" \
        scripts/run_train_semi.sh "$MODEL" "$TASK" "$LOSS" "$SEMI_MODE" "$RATIO" "${COMMON_FLAGS[@]}" 2>&1) || true
fi
echo "$TRAIN_OUT"
TRAIN_JOBID=$(echo "$TRAIN_OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)

if [ -z "$TRAIN_JOBID" ]; then
    echo "FAILED to submit train job for $label"
    echo "$(date -u +%FT%TZ)  $label  TRAIN_FAILED  $TRAIN_OUT" >> "$JOB_LOG"
    exit 2
fi

echo "Train submitted: jobid=$TRAIN_JOBID"

# 2) Submit eval(s) chained on afterany.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

submit_eval() {
    local tc="$1"
    local eval_flags
    eval_flags=$(build_eval_flags "$tc")
    local eval_label="${label}-evaltc-${tc}"

    # WRAP_CMD picks the matching epoch2 dir at sbatch run-time. If multiple
    # match (sweep + main collision), pick the most recent by mtime.
    # If none match (training died early), exit 0 with SKIP message.
    # Inject VENV env var when MODEL is gemma-4 (run_eval_semi.sh reads it).
    local venv_export=""
    [ -n "${VENV_OVERRIDE:-}" ] && venv_export="export VENV='${VENV_OVERRIDE}'; "
    local wrap_cmd
    wrap_cmd="cd ${REPO_ROOT} && ${venv_export}\
MODEL_DIR=\$(ls -dt ${GLOB_PATH} 2>/dev/null | head -1); \
if [ -z \"\$MODEL_DIR\" ] || [ ! -d \"\$MODEL_DIR\" ]; then echo 'SKIP - no matching ${GLOB_PATH}'; exit 0; fi; \
echo \"Eval model: \$MODEL_DIR\"; \
PYTHONUNBUFFERED=1 /usr/bin/time -v scripts/run_eval_semi.sh \"\$MODEL_DIR\" $eval_flags -- ${EVAL_TASKS}"

    EVAL_OUT=$(sbatch \
        --partition=allnodes \
        --cpus-per-task=4 \
        --mem="$EVAL_MEM" \
        --gres=gpu:1 \
        --time="${EVAL_HOURS}:00:00" \
        --output=/datastor2/jdr/logs/%j.out \
        --error=/datastor2/jdr/logs/%j.err \
        --dependency=afterany:"$TRAIN_JOBID" \
        --job-name="eval-${SETTING}-${DATASET}-${tc}" \
        --wrap="$wrap_cmd" 2>&1) || true
    echo "$EVAL_OUT"
    EVAL_JOBID=$(echo "$EVAL_OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)

    if [ -z "$EVAL_JOBID" ]; then
        echo "FAILED to submit eval ($tc) for $label"
        echo "$(date -u +%FT%TZ)  $eval_label  EVAL_FAILED  $EVAL_OUT" >> "$JOB_LOG"
    else
        echo "Eval submitted: jobid=$EVAL_JOBID  (tc=$tc, dep=afterany:$TRAIN_JOBID)"
        echo "$(date -u +%FT%TZ)  $eval_label  TRAIN=$TRAIN_JOBID  EVAL=$EVAL_JOBID  TC=$tc  PATH=$GLOB_PATH" >> "$JOB_LOG"
    fi
}

for tc in $TC_EVAL_LIST; do
    submit_eval "$tc"
done

echo "$(date -u +%FT%TZ)  $label  TRAIN=$TRAIN_JOBID  TC_EVAL_LIST=$TC_EVAL_LIST" >> "$JOB_LOG"
echo "Done: $label"
