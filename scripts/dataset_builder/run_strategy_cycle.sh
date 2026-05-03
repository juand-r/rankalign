#!/bin/bash
# Cycle through vLLM models on spark, generating 10 samples per problem
# for each of the 5 new prompt strategies.
# Usage: bash scripts/dataset_builder/run_strategy_cycle.sh

set -e

PYTHON="/Users/jdr/raca/.tools-venv/bin/python"
GENERATOR="scripts/dataset_builder/generate_solutions_parallel.py"
PROBLEMS="data/humaneval/problems.jsonl"
OUTPUT="data/humaneval/solutions.jsonl"
SAMPLES=10
WORKERS=30

# All 162 problems (excluding 53 and 145)
FILTER_IDS=$(python3 -c "print(','.join(f'HumanEval/{i}' for i in range(164) if i not in (53, 145)))")

MODELS=(
  "deepseek-ai/deepseek-coder-1.3b-instruct"
  "microsoft/Phi-3-mini-4k-instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "allenai/OLMo-2-0425-1B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

STRATEGIES=(
  "beginner"
  "unusual"
  "refactorable"
  "different-style"
  "bad-style"
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

echo "=== Strategy generation cycle ==="
echo "Models: ${#MODELS[@]}, Strategies: ${#STRATEGIES[@]}, Problems: 162, Samples: $SAMPLES"
echo "Total requests per model: $((${#STRATEGIES[@]} * 162 * SAMPLES))"
echo "Grand total: $((${#MODELS[@]} * ${#STRATEGIES[@]} * 162 * SAMPLES))"
echo ""

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  echo "========================================"
  echo "[$((i+1))/${#MODELS[@]}] $model"
  echo "========================================"

  swap_model "$model"

  if ! wait_for_vllm "$model"; then
    echo "  Skipping $model (failed to load)"
    continue
  fi

  for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "  --- Strategy: $strategy ---"
    $PYTHON $GENERATOR \
      --problems "$PROBLEMS" \
      --output "$OUTPUT" \
      --model "$model" \
      --base-url http://localhost:8000/v1 \
      --samples $SAMPLES --workers $WORKERS \
      --temps "0.7,1.0,1.2" \
      --max-tokens 1024 \
      --strategy "$strategy" \
      --filter-ids "$FILTER_IDS" 2>&1
  done

  echo ""
  echo "  Done with all strategies for $model"
done

echo ""
echo "=== All models and strategies complete ==="
wc -l "$OUTPUT"
