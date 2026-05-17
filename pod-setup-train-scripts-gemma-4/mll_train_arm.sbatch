#!/bin/bash
# mll_train_arm.sbatch — Slurm wrapper to train+eval ONE gemma-4-31B-it arm
# on mll, reusing the committed run_arm_3epoch.sh (3 A40 GPUs, device_map=auto).
#
# Submit (does NOT auto-run — you launch it):
#   sbatch pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch no_tc
#   sbatch pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch self_tc
#   sbatch pod-setup-train-scripts-gemma-4/mll_train_arm.sbatch neg_tc
#
# Prereqs (infrastructure, set up once):
#   - venv:  /datastor2/jdr/venvs/gemma4   (built by setup-mll-gemma4.sh)
#   - repo:  /datastor1/jdr/gv-gap/rankalign on longform
#   - token: HF_TOKEN in env, or in /datastor2/jdr/.hftoken (gated gemma-4 dl)
#
#SBATCH --job-name=g4-train
#SBATCH --partition=p-gdurret
#SBATCH --account=gdurrett
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/datastor2/jdr/logs/slurm-%x-%j.log
set -uo pipefail

ARM="${1:?usage: sbatch mll_train_arm.sbatch <no_tc|self_tc|neg_tc>}"

# --- mll paths (override run_arm_3epoch.sh's pod defaults) ---
export RANKALIGN_DIR=/datastor1/jdr/gv-gap/rankalign
export VENV_DIR=/datastor2/jdr/venvs/gemma4
export MODELS_DIR=/datastor2/jdr/models_g4it
export OUTDIR=/datastor2/jdr/outputs_gemma4_mll
export LOGDIR=/datastor2/jdr/logs
export HF_HOME=/datastor2/jdr/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub

# --- HF token: env, else /datastor2/jdr/.hftoken (never committed) ---
if [ -z "${HF_TOKEN:-}" ] && [ -f /datastor2/jdr/.hftoken ]; then
    HF_TOKEN=$(cat /datastor2/jdr/.hftoken)
fi
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set and /datastor2/jdr/.hftoken missing}"

mkdir -p "$LOGDIR" "$MODELS_DIR" "$OUTDIR" "$HF_HUB_CACHE"

echo "=== mll gemma-4 $ARM : node $(hostname) job $SLURM_JOB_ID ==="
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
"$VENV_DIR/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA in job'; print('cuda OK,', torch.cuda.device_count(), 'GPUs')"

# Hand off to the single committed train+eval entry point.
exec bash "$RANKALIGN_DIR/pod-setup-train-scripts-gemma-4/run_arm_3epoch.sh" "$ARM"
