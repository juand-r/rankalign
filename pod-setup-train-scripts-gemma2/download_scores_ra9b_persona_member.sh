#!/bin/bash
# download_scores_ra9b_persona_member.sh
#
# Download score CSVs from the 10 completed ra-9b-it persona+member pods
# to mll:/datastor2/jdr/rankalign/outputs/ then commit to the git repo
# in notes/gsm8k-v1.1-style directories for tracking.
#
# Run from the local machine (mango).
# SSH key: /home/jdr/.runpod/ssh/RunPod-Key-Go
# Target on mll: /datastor2/jdr/rankalign/outputs/

set -euo pipefail

SSH_KEY="/home/jdr/.runpod/ssh/RunPod-Key-Go"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes"
MLL_TARGET="/datastor2/jdr/rankalign/outputs"

LOCAL_TMP=$(mktemp -d /tmp/ra9b_scores_XXXXXX)
echo "[$(date -u +%FT%TZ)] Temp dir: $LOCAL_TMP"
trap 'rm -rf "$LOCAL_TMP"' EXIT

# Pod table: name -> IP:port
declare -A PODS=(
    ["persona-s1"]="103.207.149.109:17014"
    ["persona-s2"]="64.247.201.48:16874"
    ["persona-s3"]="103.207.149.153:17458"
    ["persona-s4"]="87.120.211.205:17651"
    ["persona-s7"]="87.120.211.205:17165"
    ["member-s1"]="216.243.220.217:10986"
    ["member-s2"]="216.243.220.217:18162"
    ["member-s3"]="216.243.220.217:18163"
    ["member-s4"]="103.207.149.80:13362"
    ["member-s7"]="103.207.149.80:14005"
)

TOTAL_COPIED=0

for POD in "${!PODS[@]}"; do
    ADDR="${PODS[$POD]}"
    IP="${ADDR%%:*}"
    PORT="${ADDR##*:}"
    echo "[$(date -u +%FT%TZ)] === $POD ($IP:$PORT) ==="

    POD_DIR="$LOCAL_TMP/$POD"
    mkdir -p "$POD_DIR"

    # tar all scores_*.csv from the pod and extract locally
    CSV_LIST=$(ssh $SSH_OPTS -p "$PORT" root@"$IP" \
        "ls /workspace/outputs/scores_*.csv 2>/dev/null" 2>/dev/null || true)

    if [ -z "$CSV_LIST" ]; then
        echo "  [WARN] No score CSVs found on $POD — skipping"
        continue
    fi

    N=$(echo "$CSV_LIST" | wc -l)
    echo "  Found $N CSVs"

    # Use tar pipe (avoids rsync dependency on pod)
    ssh $SSH_OPTS -p "$PORT" root@"$IP" \
        "tar cf - /workspace/outputs/scores_*.csv 2>/dev/null" \
        | tar xf - --strip-components=3 -C "$POD_DIR" 2>/dev/null

    COPIED=$(ls "$POD_DIR"/*.csv 2>/dev/null | wc -l)
    echo "  Extracted $COPIED CSVs locally"
    TOTAL_COPIED=$((TOTAL_COPIED + COPIED))
done

echo "[$(date -u +%FT%TZ)] Total CSVs downloaded locally: $TOTAL_COPIED"
echo "[$(date -u +%FT%TZ)] Syncing to mll:$MLL_TARGET ..."

# rsync all pod subdirectory CSVs flat into mll outputs/
for POD_DIR in "$LOCAL_TMP"/*/; do
    POD=$(basename "$POD_DIR")
    COUNT=$(ls "$POD_DIR"*.csv 2>/dev/null | wc -l)
    if [ "$COUNT" -gt 0 ]; then
        echo "  [rsync] $POD -> mll:$MLL_TARGET ($COUNT files)"
        rsync -av --no-relative "$POD_DIR"*.csv "mll:$MLL_TARGET/" 2>&1 | tail -3
    fi
done

echo "[$(date -u +%FT%TZ)] Done. Verifying on mll ..."
raca ssh mll "ls $MLL_TARGET/scores_*v7*gemma-2-9b*.csv $MLL_TARGET/scores_*v6*eval_model*.csv 2>/dev/null | wc -l"
echo "[$(date -u +%FT%TZ)] === download_scores_ra9b_persona_member.sh COMPLETE ==="
