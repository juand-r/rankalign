#!/bin/bash
# Cycle through multiple models on spark vLLM, generating 10 samples per problem each.
# Usage: bash scripts/dataset_builder/run_multi_model_cycle.sh

set -e

FILTER_IDS="HumanEval/43,HumanEval/45,HumanEval/48,HumanEval/53,HumanEval/58"
PROBLEMS="data/humaneval/problems.jsonl"
OUTPUT="data/humaneval/solutions.jsonl"
PYTHON="/Users/jdr/raca/.tools-venv/bin/python"
GENERATOR="scripts/dataset_builder/generate_solutions_parallel.py"
SAMPLES=10
WORKERS=20

MODELS=(
  "deepseek-ai/deepseek-coder-1.3b-instruct"
  "microsoft/Phi-3-mini-4k-instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "microsoft/phi-2"
  "allenai/OLMo-2-0425-1B-Instruct"
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

# First model is already loading
echo "=== Starting multi-model cycle ==="
echo "Target problems: $FILTER_IDS"
echo "Models: ${MODELS[*]}"
echo ""

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  echo "--- [$((i+1))/${#MODELS[@]}] $model ---"

  if [ "$i" -gt 0 ]; then
    swap_model "$model"
  fi

  if ! wait_for_vllm "$model"; then
    echo "  Skipping $model (failed to load)"
    continue
  fi

  echo "  Generating $SAMPLES samples per problem..."
  $PYTHON $GENERATOR \
    --problems "$PROBLEMS" \
    --output "$OUTPUT" \
    --model "$model" \
    --base-url http://localhost:8000/v1 \
    --samples $SAMPLES --workers $WORKERS \
    --temps "0.7,1.0,1.2" \
    --max-tokens 1024 \
    --filter-ids "$FILTER_IDS" 2>&1

  echo "  Done with $model"
  echo ""
done

echo "=== All models complete ==="
echo "Checking qualification..."
python3 -c "
import json
from collections import defaultdict
counts = defaultdict(lambda: {'pass': 0, 'fail': 0})
with open('$OUTPUT') as f:
    for line in f:
        r = json.loads(line)
        if r.get('strategy') == 'intentional_bug': continue
        counts[r['task_id']]['pass' if r['passed'] else 'fail'] += 1
qualified = sum(1 for t in counts if counts[t]['pass'] >= 10 and counts[t]['fail'] >= 10)
print(f'Qualified: {qualified}/164')
for t in sorted(counts, key=lambda x: int(x.split('/')[1])):
    c = counts[t]
    if not (c['pass'] >= 10 and c['fail'] >= 10):
        print(f'  {t}: {c[\"pass\"]}p/{c[\"fail\"]}f')
"
