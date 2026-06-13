#!/usr/bin/env bash

# run_rerun_9bit_membership_self_neg_eval.sh
#
# Fill the missing self-only / neg-only TC eval files for the
# gemma-2-9b-it x membership-sans-rosch-v0 rerun-wandb batch.
#
# Why this exists: the original rerun-wandb eval jobs (44462-44473, Jun 9)
# went through _overnight_launch.sh's build_eval_flags which always pairs
# --self-typcorr with --base-typcorr (and same for --neg-typcorr).
# scripts/eval_by_claude.py uses a single-prefix file-naming scheme
# (lines 829-837): when both flags are set, the resulting CSV gets the
# `basetyp-` (or `basetypneg-`) prefix and the gen_score_typcorr column
# inside is the BASE-typicality variant. The self-typicality numbers are
# computed but not persisted as their own file. Result: zero
# scores_self-*membership-sans-rosch-v0*rosch* and zero
# scores_neg-*membership-sans-rosch-v0*rosch* files exist, so PMI self /
# Neg self rows in the rerun_only_tables PDF show "--" for the
# Hyponymy/G2-9b-it column.
#
# This launcher re-runs eval with --self-typcorr ONLY (no --base-typcorr)
# and --neg-typcorr ONLY, so the prefix collapses to `self-` and `neg-`
# respectively. The matrix mirrors the original 8-eval batch:
#   s1 (SFT-lo):     self + neg
#   s2 (RankAlign):  self + neg
#   s3 (Ours):       self + neg
#   s4 (tc-self):    self only  (matched eval; neg is off-policy)
#   s7 (tc-neg):     neg only   (matched eval; self is off-policy)
#
# Usage:
#   bash scripts/run_rerun_9bit_membership_self_neg_eval.sh
#   DRYRUN=1 bash scripts/run_rerun_9bit_membership_self_neg_eval.sh

set -euo pipefail

MODELS_DIR="/datastor2/jdr/rankalign/models2-rerun-wandb"
OUTPUTS_DIR="/datastor2/jdr/rankalign/outputs-rerun-wandb"
HOURS="${HOURS:-4}"
ROSCH_TASKS="rosch-bird rosch-carpenters-tool rosch-clothing rosch-fruit rosch-furniture rosch-sport rosch-toy rosch-vehicle rosch-vegetable rosch-weapon"

# (setting, model-dir-pattern, tc-list)
declare -a JOBS=(
    "s1|v7-google--gemma-2-9b-it-delta1.43-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--full-completion--pref0.0--nllv1.0--nllg1.0--labelonly0.1--fix1_merged|self neg"
    "s2|v7-google--gemma-2-9b-it-delta1.42-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--full-completion--semi0.1--fix1_merged|self neg"
    "s3|v7-google--gemma-2-9b-it-delta2.69-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1_merged|self neg"
    "s4|v7-google--gemma-2-9b-it-delta2.69-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--tc-self--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1_merged|self"
    "s7|v7-google--gemma-2-9b-it-delta2.69-epoch2--membership-sans-rosch-v0-all--d2g--random--alpha1.0--tc-neg--full-completion--nllv1.0--nllg1.0--force-same-x--ppd--vallogodds--semi0.1--fix1_merged|neg"
)

mkdir -p overnight
JOB_LOG="overnight/_rerun_9bit_membership_self_neg_eval_jobids.txt"
: > "$JOB_LOG"

echo "Submitting self/neg-only TC evals for gemma-9b-it x membership rerun-wandb"
echo "OUTPUTS_DIR=$OUTPUTS_DIR  HOURS=$HOURS"
echo "========================================"

for entry in "${JOBS[@]}"; do
    setting="${entry%%|*}"
    rest="${entry#*|}"
    model_subdir="${rest%%|*}"
    tc_list="${rest##*|}"
    model_path="$MODELS_DIR/$model_subdir"
    if [ ! -d "$model_path" ]; then
        echo "MISSING checkpoint: $model_path  -- skipping $setting"
        continue
    fi
    for tc in $tc_list; do
        case "$tc" in
            self) tc_flag="--self-typcorr" ;;
            neg)  tc_flag="--neg-typcorr"  ;;
            *) echo "unknown tc=$tc"; exit 2 ;;
        esac
        echo "----------------------------------------"
        echo "Setting=$setting  TC=$tc  Model=$(basename $model_path)"
        cmd=(~/.local/bin/run 1 "$HOURS" --mem 64G scripts/run_eval_semi.sh "$model_path" $tc_flag --log-odds --outputs-dir "$OUTPUTS_DIR" --disc-shots few -- $ROSCH_TASKS)
        echo "+ ${cmd[*]}"
        if [ -n "${DRYRUN:-}" ]; then
            echo "DRYRUN: not submitting"
            continue
        fi
        OUT=$("${cmd[@]}" 2>&1) || true
        echo "$OUT"
        JID=$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$' | head -1)
        if [ -n "$JID" ]; then
            echo "$(date -u +%FT%TZ)  $setting  $tc  $JID  $(basename $model_path)" >> "$JOB_LOG"
        else
            echo "FAILED to submit $setting $tc"
            echo "$(date -u +%FT%TZ)  $setting  $tc  FAILED  $(basename $model_path)" >> "$JOB_LOG"
        fi
    done
done

echo "========================================"
echo "Done. Job log: $JOB_LOG"
cat "$JOB_LOG"
