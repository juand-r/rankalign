#!/bin/bash
# mll_smoke_fix1.sbatch — Smoke test for fix1 + --gemma4-lora on mll Slurm.
#
# Verifies that scripts/ranking_loss_ref_fix.py with --gemma4-lora successfully:
#   1. Loads google/gemma-4-31B-it across 3× A40 GPUs (device_map=auto).
#   2. Sets up LoRA with the regex target_modules (must catch projections wrapped
#      in Gemma4ClippableLinear at *.linear; print_trainable_parameters > 0).
#   3. Runs the FIX1 G-MODE PAIR CONSTRUCTION diagnostic block (4-shape pool).
#   4. Completes >=1 training step (tiny budget: 30 samples, 1 epoch).
#   5. Saves the LoRA adapter WITHOUT merge_and_unload (the --gemma4-lora
#      "skip merge" path; eval_by_claude.py loads the PEFT adapter directly).
#
# Tiny budget so the run finishes fast: persona-v1, --total_samples 30,
# --num_epochs 1, --delta-bins 10. NOT a real training run; do not consume
# the resulting adapter for any downstream comparison.
#
# Outputs:
#   - Slurm log: /datastor2/jdr/logs/slurm-g4-smoke-fix1-<JOBID>.log
#   - Adapter:   /datastor2/jdr/models_g4it_smoke/v7-google--gemma-4-31B-it-...--fix1/
#
# Submit:
#   sbatch pod-setup-train-scripts-gemma-4/mll_smoke_fix1.sbatch
#
# Slurm allocation rationale:
#   - allnodes partition (faster queue than p-gdurret); fits its caps:
#     --mem=96G ≤ 96G allnodes cap, --cpus-per-task=2 ≤ 9 cap.
#   - 3× A40 = 144 GB GPU mem: 31B in bf16 (~62 GB) + LoRA + grad ckpt
#     activations comfortably fit (matches mll_train_arm.sbatch).
#   - 96 GB host mem is plenty for 30-sample smoke (model lives on GPUs
#     via device_map=auto; only tokenization/dataloader needs host RAM).
#
#SBATCH --job-name=g4-smoke-fix1
#SBATCH --partition=allnodes
#SBATCH --account=gdurrett
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/datastor2/jdr/logs/slurm-g4-smoke-fix1-%j.log
set -uo pipefail

# --- mll paths (mirror mll_train_arm.sbatch; smoke uses its own MODELS_DIR
#     to avoid polluting the real /datastor2/jdr/models_g4it directory) ---
export RANKALIGN_DIR=/datastor1/jdr/gv-gap/rankalign
export VENV_DIR=/datastor2/jdr/venvs/gemma4
export MODELS_DIR=/datastor2/jdr/models_g4it_smoke
export LOGDIR=/datastor2/jdr/logs
export HF_HOME=/datastor2/jdr/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub

# --- HF token (gated gemma-4 download): env, else /datastor2/jdr/.hftoken ---
if [ -z "${HF_TOKEN:-}" ] && [ -f /datastor2/jdr/.hftoken ]; then
    HF_TOKEN=$(cat /datastor2/jdr/.hftoken)
fi
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN not set and /datastor2/jdr/.hftoken missing}"
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

mkdir -p "$LOGDIR" "$MODELS_DIR" "$HF_HUB_CACHE"

echo "=== mll g4 smoke (fix1) : node $(hostname) job ${SLURM_JOB_ID:-?} ==="
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

source "$VENV_DIR/bin/activate"
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in job'; print('cuda OK,', torch.cuda.device_count(), 'GPUs')"

cd "$RANKALIGN_DIR/scripts"

# Tiny smoke run. --gemma4-lora requires --lora (parser.error if not).
# Save dir lands under $MODELS_DIR with a v7-...--fix1 suffix.
python ranking_loss_ref_fix.py \
    --model google/gemma-4-31B-it \
    --task persona-v1 \
    --train_g_or_d g \
    --split_type random \
    --num_epochs 1 \
    --total_samples 30 \
    --delta-bins 10 \
    --semi-supervised 0.1 \
    --nll_validator_weight 1 \
    --nll_generator_weight 1 \
    --preference_loss_weight 1 \
    --validator-log-odds \
    --self-typicality \
    --all \
    --lora \
    --gradient_checkpointing \
    --gemma4-lora \
    --no-upload-hf \
    --no-wandb \
    --debug \
    --models-dir "$MODELS_DIR" \
    --disc-shots zero

echo "=== smoke job done ==="
