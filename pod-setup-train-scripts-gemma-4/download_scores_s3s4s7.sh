#!/bin/bash
# download_scores_s3s4s7.sh
#
# Downloads v7 epoch2 score CSVs from a completed training pod to mll.
# Run from local machine (mango).
#
# Usage:
#   ./download_scores_s3s4s7.sh <pod_type> <setting>
#
#   pod_type : "cu" (correct-upper) or "cm" (correct-multi)
#   setting  : 3, 4, or 7
#
# Examples:
#   ./download_scores_s3s4s7.sh cu 4    # TAUR-cu-s4: correct-upper s4
#   ./download_scores_s3s4s7.sh cm 7    # h100-train-8: correct-multi s7
#
# Pod SSH details (current training run):
#   cu-s3: port 11962 @ 213.181.122.162  (TAUR account)
#   cu-s4: port 15930 @ 216.243.220.217  (TAUR account)
#   cu-s7: port 12542 @ 103.207.149.163  (TAUR account)
#   cm-s3: port 11793 @ 205.196.17.114   (personal account)
#   cm-s4: port 11320 @ 205.196.17.114   (personal account)
#   cm-s7: port 10528 @ 103.207.149.86   (personal account)
#
# MLL destination:
#   correct-upper: /datastor2/jdr/rankalign/outputs_gemma4_from_pod/correct_upper_s3s4s7/
#   correct-multi: /datastor2/jdr/rankalign/outputs_gemma4_from_pod/correct_multi_s3s4s7/

set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.runpod/ssh/RunPod-Key-Go}"
MLL_HOST="${MLL_HOST:-mll}"  # ssh alias in ~/.ssh/config

declare -A POD_PORT
declare -A POD_IP
POD_PORT[cu-3]=11962;  POD_IP[cu-3]=213.181.122.162
POD_PORT[cu-4]=15930;  POD_IP[cu-4]=216.243.220.217
POD_PORT[cu-7]=12542;  POD_IP[cu-7]=103.207.149.163
POD_PORT[cm-3]=11793;  POD_IP[cm-3]=205.196.17.114
POD_PORT[cm-4]=11320;  POD_IP[cm-4]=205.196.17.114
POD_PORT[cm-7]=10528;  POD_IP[cm-7]=103.207.149.86

if [ $# -ne 2 ]; then
    echo "usage: $0 <cu|cm> <3|4|7>"
    exit 1
fi

POD_TYPE="$1"   # cu or cm
SETTING="$2"    # 3, 4, or 7
KEY="${POD_TYPE}-${SETTING}"

if [ -z "${POD_PORT[$KEY]+x}" ]; then
    echo "ERROR: unknown pod key '$KEY' (valid: cu-3 cu-4 cu-7 cm-3 cm-4 cm-7)"
    exit 1
fi

PORT="${POD_PORT[$KEY]}"
IP="${POD_IP[$KEY]}"

if [ "$POD_TYPE" = "cu" ]; then
    DATASET="correct-upper"
    MLL_DEST="/datastor2/jdr/rankalign/outputs_gemma4_from_pod/correct_upper_s3s4s7"
else
    DATASET="correct-multi"
    MLL_DEST="/datastor2/jdr/rankalign/outputs_gemma4_from_pod/correct_multi_s3s4s7"
fi

SSH_OPT="-i $SSH_KEY -o IdentitiesOnly=yes -o IdentityAgent=none \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=20 -p $PORT"

echo "=== s${SETTING} ${DATASET}: ${IP}:${PORT} → mango → ${MLL_HOST}:${MLL_DEST} ==="

# Count v7 epoch2 score CSVs on pod
N_POD=$(ssh $SSH_OPT root@"$IP" \
    "ls /workspace/outputs/scores_*v7*epoch2*.csv 2>/dev/null | wc -l")
echo "  Pod has $N_POD v7 epoch2 score CSVs"

if [ "$N_POD" -eq 0 ]; then
    echo "  No v7 score CSVs found — is eval done? Aborting."
    exit 1
fi

# Step 1: pod → local tmp dir
LOCAL_TMP="$(mktemp -d /tmp/scores_s${SETTING}_XXXXXX)"
echo "  Pulling pod scores → $LOCAL_TMP ..."
rsync -az \
    -e "ssh $SSH_OPT" \
    "root@${IP}:/workspace/outputs/scores_*v7*epoch2*.csv" \
    "$LOCAL_TMP/"
N_LOCAL=$(ls "$LOCAL_TMP"/*.csv 2>/dev/null | wc -l)
echo "  Received $N_LOCAL files locally."

# Step 2: ensure dest dir exists on mll
ssh "$MLL_HOST" "mkdir -p $MLL_DEST"

# Step 3: local → mll
echo "  Pushing $N_LOCAL files to ${MLL_HOST}:${MLL_DEST}/ ..."
rsync -az "$LOCAL_TMP/" "${MLL_HOST}:${MLL_DEST}/"

# Step 4: verify
N_MLL=$(ssh "$MLL_HOST" "ls $MLL_DEST/*.csv 2>/dev/null | wc -l")
echo "  MLL now has $N_MLL CSV files in ${MLL_DEST}/"

# Cleanup tmp
rm -rf "$LOCAL_TMP"

echo "=== Done. ==="
echo ""
echo "To run analysis on mll (after all s3/s4/s7 are downloaded):"
echo "  ssh $MLL_HOST"
echo "  cd /datastor1/jdr/gv-gap/rankalign"
echo "  python scripts-more/analyze_gemma4_tc_v7_fsx.py \\"
echo "    --scores-v6-dir outputs_gemma4_from_pod \\"
echo "    --scores-v7-dir /datastor2/jdr/rankalign/outputs_gemma4_from_pod/${DATASET}_s3s4s7 \\"
echo "    --dataset ${DATASET}"
