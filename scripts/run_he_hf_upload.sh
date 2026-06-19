#!/bin/bash
# run_he_hf_upload.sh — thin launcher for upload_he_artifacts_to_hf.py ON mll.
# Activates the qwen35 venv, exports the (non-committed) latkes HF token, runs the uploader.
# Designed to be launched detached (setsid nohup) for the long 602GB qwen upload.
#
#   bash run_he_hf_upload.sh <scores|adapters|qwen|all> [--limit N]
set -uo pipefail
REPO=/datastor2/jdr/rankalign
source /datastor2/jdr/venvs/qwen35/bin/activate
export HF_TOKEN="$(cat /datastor2/jdr/.hf_token)"   # outside the git repo, never committed
TARGET="${1:?usage: run_he_hf_upload.sh <scores|adapters|qwen|all> [--limit N]}"; shift || true
cd "$REPO/scripts"
exec python upload_he_artifacts_to_hf.py --target "$TARGET" "$@"
