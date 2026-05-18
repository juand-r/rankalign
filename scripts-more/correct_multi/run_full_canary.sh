#!/bin/bash
# run_full_canary.sh — ONE idempotent, resumable on-pod entry point for the
# humaneval-v2.1correct-multi ΔlogP canary. Re-running after ANY failure
# resumes (build_canary deterministic seed=42; canary_score skips done
# variant_ids). No ad-hoc steps; every result-producing command is here.
#
#   bash run_full_canary.sh
#
# Preconditions checked (fail loud, no silent hang):
#   - /workspace/.venv ready, gemma-4-31B-it cached, v2.1 data present
#   - score_v2_humaneval.py staged at $V2_SCRIPTS (the verbatim scorer)
set -uo pipefail

RA=/workspace/rankalign
PKG="$RA/scripts-more/correct_multi"
V2_SCRIPTS="${V2_SCRIPTS:-/workspace/v2_scripts}"   # staged canonical scorer
OUTDIR="${OUTDIR:-/workspace/canary_out}"
PAIRS="$OUTDIR/canary_pairs.jsonl"
SCORES="$OUTDIR/canary_scores.jsonl"
LOG=/workspace/logs/run_full_canary.log
mkdir -p "$OUTDIR" /workspace/logs
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- preconditions (explicit; never hang waiting) -------------------------
[ -x /workspace/.venv/bin/python ] || { say "FATAL: venv missing"; exit 11; }
PY=/workspace/.venv/bin/python
GEMMA=/workspace/.cache/huggingface/hub/models--google--gemma-4-31B-it
[ -d "$GEMMA/snapshots" ] && [ -n "$(ls -A "$GEMMA/snapshots" 2>/dev/null)" ] \
    || { say "FATAL: gemma-4 not cached at $GEMMA"; exit 12; }
ls "$RA"/data/humaneval/v2.1/humaneval_*.csv >/dev/null 2>&1 \
    || { say "FATAL: v2.1 data missing"; exit 13; }
[ -f "$V2_SCRIPTS/score_v2_humaneval.py" ] \
    || { say "FATAL: scorer not staged at $V2_SCRIPTS"; exit 14; }

export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub HF_HUB_DISABLE_XET=1
[ -f /workspace/.hftoken ] && export HF_TOKEN=$(cat /workspace/.hftoken)

# ---- STEP 1: build canary pairs (CPU, deterministic; idempotent) ----------
if [ -s "$PAIRS" ] && grep -q '"label": "neg_control"' "$PAIRS"; then
    say "STEP1 build_canary: pairs already present ($(wc -l <"$PAIRS")) — skip"
else
    say "STEP1 build_canary ..."
    "$PY" "$PKG/build_canary.py" --n-rows 36 --n-tasks 12 --seed 42 \
        --out "$PAIRS" >> "$LOG" 2>&1 \
        || { say "FATAL: build_canary failed (rc=$?)"; exit 21; }
    say "STEP1 done: $(wc -l <"$PAIRS") variants"
fi

# ---- STEP 2: GPU scoring (resumable: skips done variant_ids) ---------------
say "STEP2 canary_score (GPU) ..."
"$PY" "$PKG/canary_score.py" --pairs "$PAIRS" --out "$SCORES" \
    --v2-scripts-dir "$V2_SCRIPTS" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then say "FATAL: canary_score failed (rc=$rc)"; exit 22; fi
NS=$(wc -l <"$SCORES" 2>/dev/null || echo 0)
[ "$NS" -ge 100 ] || { say "FATAL: only $NS scores written (<100)"; exit 23; }
say "STEP2 done: $NS scored"

# ---- STEP 3: analyze -> CANARY_DELTA_LOGP.md (CPU) ------------------------
say "STEP3 analyze ..."
"$PY" "$PKG/analyze_canary.py" --pairs "$PAIRS" --scores "$SCORES" \
    --out "$OUTDIR/CANARY_DELTA_LOGP.md" >> "$LOG" 2>&1 \
    || { say "FATAL: analyze failed (rc=$?)"; exit 24; }

say "============ CANARY COMPLETE ============"
echo "CANARY_COMPLETE" >> "$LOG"
say "artifacts: $PAIRS | $SCORES | $OUTDIR/CANARY_DELTA_LOGP.md"
