#!/bin/bash
# Run vLLM as an OpenAI-compatible server inside Docker on spark.
# Usage: ./run_vllm_server.sh <model_name> [port]
# Example: ./run_vllm_server.sh meta-llama/Llama-3.1-8B-Instruct 8000

MODEL=${1:?Usage: run_vllm_server.sh <model_name> [port]}
PORT=${2:-8000}

echo "Starting vLLM server for $MODEL on port $PORT..."

docker run --rm -d \
    --gpus all \
    --name vllm-server \
    -p ${PORT}:${PORT} \
    -v /home/jdr/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/vllm:26.03-py3 \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len 2048 \
        --trust-remote-code

echo "Container started. Check logs with: docker logs -f vllm-server"
echo "Server will be at: http://localhost:${PORT}/v1"
echo "Test with: curl http://localhost:${PORT}/v1/models"
