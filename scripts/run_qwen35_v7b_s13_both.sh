#!/bin/bash
# run_qwen35_v7b_s13_both.sh
#
# Launch BOTH qwen s13 jobs sequentially on one pod, 1 epoch, fixed delta=0.15:
#   1) ifeval     s13  (OOD eval, prompts <=21)
#   2) membership s13  (rosch eval, 10 tasks)
#
# s13 = SFT + consistency-ft. Each writes /workspace/V7B_<DATASET>_S13_DONE on
# completion. Idempotent (cell script skips train if model dir exists).
#
# Run on the pod under tmux/nohup:
#   export HF_TOKEN=...; nohup bash run_qwen35_v7b_s13_both.sh >> /workspace/logs/qwen_s13_both.log 2>&1 &

set -uo pipefail
export EPOCHS=1
cd /workspace/rankalign/scripts

echo "[$(date -u +%FT%TZ)] === qwen s13 BOTH start (EPOCHS=$EPOCHS) ==="
bash run_qwen35_v7b_cell.sh ifeval s13
echo "[$(date -u +%FT%TZ)] === ifeval s13 finished, starting membership s13 ==="
bash run_qwen35_v7b_cell.sh membership s13
echo "[$(date -u +%FT%TZ)] === qwen s13 BOTH done ==="
