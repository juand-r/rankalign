#!/bin/bash
# Training runs for persona-v0 task.
#
# All 9 variants from docs/IMPORTANT-RESEARCH-PLAN.md.
# For #5 and #8, --validator-log-odds is OMITTED (the original humaneval/
# membership runs had it on, which the research plan flags as the "vlo
# belongs with comb, not pref-only" bug). New runs do it right.
#
#   1. SFT-lo                 sft + labelonly + no force-same-x
#   2. RankAlign              pref-only + semi + no force-same-x  (last year's baseline)
#   3. New+fsx                comb + semi + log-odds + force-same-x (no TC)
#   4. New+fsx+self-tc        comb + semi + log-odds + force-same-x + self-typcorr
#   5. RankAlign+fsx+self-tc  pref-only + semi + force-same-x + self-typcorr (no log-odds; corrected)
#   6. RankAlign+self-tc      pref-only + semi + self-typcorr + no force-same-x  (cleanest "TC alone helps?" probe)
#   7. New+fsx+neg-tc         comb + semi + log-odds + force-same-x + neg-typcorr
#   8. RankAlign+fsx+neg-tc   pref-only + semi + force-same-x + neg-typcorr (no log-odds; corrected)
#   9. RankAlign+neg-tc       pref-only + semi + neg-typcorr + no force-same-x  (cleanest "neg-TC alone helps?" probe)
#
# Persona-v0 uses:
#   --disc-shots zero  (matches eval setup; few-shot exemplars TBD; see
#                      docs/datasets/persona_v0_notes.md)
#   --max-seq-len 512  (statements are short, ~30 tok max even chat-wrapped)
#
# NOTE on labels: persona-v0 keeps the dataset as-built — `correct=yes` means
# "the persona would say this." For antisocial in-domain personas (psychopathy,
# machiavellianism, narcissism, moral nihilism, no-meaning), training pulls the
# model toward the persona. This is intentional — it's the §3(c.2) "correct =
# atypical" structural probe the project lacked. Trained checkpoints are for
# measurement only. See docs/datasets/persona_v0_notes.md.
#
# Usage:
#   bash scripts/run_train_persona_v0.sh [MODEL]
#
#   MODEL: HuggingFace id (default: google/gemma-2-9b-it).
#   First positional arg overrides MODEL env var.
#
# Examples:
#   bash scripts/run_train_persona_v0.sh
#   bash scripts/run_train_persona_v0.sh google/gemma-2-2b-it
#   MODEL=google/gemma-2-2b HOURS=4 bash scripts/run_train_persona_v0.sh
#
# Env:
#   MODEL  - default HF model when no positional argument is given
#   HOURS  - Slurm walltime hours per job (default: 6)

set -e

MODEL="${1:-${MODEL:-google/gemma-2-9b-it}}"
HOURS="${HOURS:-6}"
CPUS="${CPUS:-6}"
MEM="${MEM:-60G}"
TASK=persona-v0
COMMON="--disc-shots zero --max-seq-len 512"

OVERNIGHT_DIR="$(dirname "$0")/../overnight"
mkdir -p "$OVERNIGHT_DIR"
# Per-model jobid file so successive launches (e.g. 9b-it then 2b-it) don't clobber each other.
MODEL_TAG=$(basename "$MODEL" | sed 's/--/_/g; s/[/]/_/g')
JOBID_FILE="$OVERNIGHT_DIR/persona_v0_train_jobids_${MODEL_TAG}.txt"
: > "$JOBID_FILE"

echo "========================================"
echo "Persona-v0 training launcher"
echo "  MODEL:  $MODEL"
echo "  TASK:   $TASK"
echo "  HOURS:  $HOURS (each of 9 jobs)"
echo "  CPUS:   $CPUS"
echo "  MEM:    $MEM"
echo "  COMMON: $COMMON"
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

# 1. SFT-lo: NLL on labeled-only 10%, no preference loss, no force-same-x.
submit "1.SFT-lo" \
    sft labelonly 0.1 $COMMON --no-force-same-x

# 2. RankAlign: vanilla preference loss on semi-supervised data, no fsx, no TC.
submit "2.RankAlign" \
    pref-only semi 0.1 $COMMON --no-force-same-x

# 3. New+fsx: comb (pref + NLL on labeled) + semi + validator log-odds + fsx, no TC.
submit "3.New+fsx" \
    comb semi 0.1 $COMMON --log-odds

# 4. New+fsx+self-tc: comb + semi + log-odds + fsx + self-TC.
submit "4.New+fsx+selfTC" \
    comb semi 0.1 $COMMON --self-typcorr --log-odds

# 5. RankAlign+fsx+self-tc (CORRECTED): pref-only + semi + fsx + self-TC,
#    no --log-odds (the bug in the original humaneval/membership runs).
#    Diagnostic: vs #6, isolates the contribution of fsx given TC.
submit "5.RankAlign+fsx+selfTC" \
    pref-only semi 0.1 $COMMON --self-typcorr

# 6. RankAlign+self-tc: pref-only + semi + self-TC, no fsx.
#    The cleanest "does TC alone help?" probe.
submit "6.RankAlign+selfTC" \
    pref-only semi 0.1 $COMMON --self-typcorr --no-force-same-x

# 7. New+fsx+neg-tc: comb + semi + log-odds + fsx + neg-TC.
submit "7.New+fsx+negTC" \
    comb semi 0.1 $COMMON --neg-typcorr --log-odds

# 8. RankAlign+fsx+neg-tc (CORRECTED): pref-only + semi + fsx + neg-TC,
#    no --log-odds. Diagnostic: vs #9, isolates fsx given neg-TC.
submit "8.RankAlign+fsx+negTC" \
    pref-only semi 0.1 $COMMON --neg-typcorr

# 9. RankAlign+neg-tc: pref-only + semi + neg-TC, no fsx.
#    The cleanest "does neg-TC alone help?" probe.
submit "9.RankAlign+negTC" \
    pref-only semi 0.1 $COMMON --neg-typcorr --no-force-same-x

echo ""
echo "========================================"
echo "All 9 submissions attempted. Jobid log:"
cat "$JOBID_FILE"
echo ""
echo "Tail logs with: tail -f ~/logs/<JOBID>.out"
echo "Check progress with: squeue -u \$USER"
