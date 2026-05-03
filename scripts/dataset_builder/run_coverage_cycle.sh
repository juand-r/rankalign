#!/bin/bash
# Cycle through vLLM models on spark, filling coverage gaps (10 samples per missing problem).
# Usage: bash scripts/dataset_builder/run_coverage_cycle.sh

set -e

PYTHON="/Users/jdr/raca/.tools-venv/bin/python"
SCRIPT="scripts/dataset_builder/fill_coverage_gaps.py"

MODELS=(
  "deepseek-ai/deepseek-coder-1.3b-instruct"
  "microsoft/Phi-3-mini-4k-instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "allenai/OLMo-2-0425-1B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

wait_for_vllm() {
  echo "  Waiting for vLLM to load $1..."
  for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
      echo "  Ready after $((i*5))s"
      return 0
    fi
    sleep 5
  done
  echo "  TIMEOUT waiting for model!"
  return 1
}

swap_model() {
  local model=$1
  echo "Swapping to $model..."
  raca ssh spark "docker rm -f vllm-server 2>/dev/null; docker run -d --name vllm-server --runtime nvidia --gpus all --ipc=host \
    -v /home/jdr/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    nvcr.io/nvidia/vllm:26.03-py3 \
    vllm serve $model \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80" 2>&1
}

echo "=== Coverage gap fill: vLLM models ==="

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  echo ""
  echo "--- [$((i+1))/${#MODELS[@]}] $model ---"

  swap_model "$model"

  if ! wait_for_vllm "$model"; then
    echo "  Skipping $model (failed to load)"
    continue
  fi

  $PYTHON $SCRIPT --mode vllm --vllm-model "$model" --workers 30

  echo "  Done with $model"
done

echo ""
echo "=== All vLLM models complete ==="
