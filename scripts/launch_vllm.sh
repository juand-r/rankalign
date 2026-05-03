#!/bin/bash
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log
#SBATCH --job-name=vllm-serve

# ── Configuration (override via env vars) ────────────────────────────
MODEL="${VLLM_MODEL:-google/gemma-2-9b-it}"
PORT="${VLLM_PORT:-8000}"
VENV="/datastor2/jdr/venvs/vllm"
ENDPOINTS_FILE="${VLLM_ENDPOINTS:-/datastor1/jdr/gv-gap/rankalign/vllm-endpoints.json}"
LOCK_FILE="${ENDPOINTS_FILE}.lock"

# ── Activate venv ────────────────────────────────────────────────────
source "${VENV}/bin/activate"
export UV_CACHE_DIR=/datastor2/jdr/.cache/uv
export PYTHONUNBUFFERED=1

NODE=$(hostname)

# ── Register/deregister in endpoints JSON (flock for concurrent safety) ──
register_endpoint() {
    (
        flock -w 5 200
        python3 -c "
import json, os, sys
f = sys.argv[1]
entry = {'node': sys.argv[2], 'port': int(sys.argv[3]),
         'job_id': sys.argv[4], 'base_url': f'http://{sys.argv[2]}:{sys.argv[3]}/v1'}
data = {}
if os.path.exists(f):
    try: data = json.load(open(f))
    except: pass
data[sys.argv[5]] = entry
json.dump(data, open(f, 'w'), indent=2)
" "$ENDPOINTS_FILE" "$NODE" "$PORT" "$SLURM_JOB_ID" "$MODEL"
    ) 200>"$LOCK_FILE"
}

deregister_endpoint() {
    (
        flock -w 5 200
        python3 -c "
import json, os, sys
f = sys.argv[1]
if not os.path.exists(f): sys.exit()
try: data = json.load(open(f))
except: sys.exit()
data.pop(sys.argv[2], None)
json.dump(data, open(f, 'w'), indent=2)
" "$ENDPOINTS_FILE" "$MODEL"
    ) 200>"$LOCK_FILE"
}

register_endpoint
trap deregister_endpoint EXIT

# ── Print connection info ────────────────────────────────────────────
echo "=========================================="
echo "  vLLM server starting"
echo "  Model:  ${MODEL}"
echo "  Node:   ${NODE}"
echo "  Port:   ${PORT}"
echo "  Job ID: ${SLURM_JOB_ID}"
echo ""
echo "  Connect from login node:"
echo "    curl http://${NODE}:${PORT}/v1/models"
echo ""
echo "  Python client:"
echo "    client = OpenAI(base_url='http://${NODE}:${PORT}/v1', api_key='x')"
echo ""
echo "  Endpoints file: ${ENDPOINTS_FILE}"
echo "=========================================="

# ── Launch ───────────────────────────────────────────────────────────
vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype auto \
    --gpu-memory-utilization 0.90
