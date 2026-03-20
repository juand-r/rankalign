#!/bin/bash

# Run a command on a remote mll-a40 server inside a tmux session
# Usage: ./run_on_server.sh <server_num> <command> [--gpu <gpu_id>]
#
# Examples:
#   ./run_on_server.sh 3 "python train.py --epochs 10"
#   ./run_on_server.sh 3 "python train.py" --gpu 2
#   ./run_on_server.sh 3 "python train.py" --gpu 0,1
#
# The script will:
#   1. SSH into mll-a40-<N>.cs.utexas.edu
#   2. Start a new tmux session
#   3. cd to the working directory
#   4. Activate the virtual environment
#   5. Run the command (with CUDA_VISIBLE_DEVICES if --gpu specified)

# Configuration - modify these as needed
WORK_DIR="/datastor1/jdr/gv-gap/rankalign/scripts"
VENV_PATH="/u/jdr/venvs/venv_lexcons/bin/activate"

# Parse arguments
SERVER_NUM=""
COMMAND=""
GPU_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            if [ -z "$SERVER_NUM" ]; then
                SERVER_NUM="$1"
            elif [ -z "$COMMAND" ]; then
                COMMAND="$1"
            else
                COMMAND="$COMMAND $1"
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [ -z "$SERVER_NUM" ] || [ -z "$COMMAND" ]; then
    echo "Usage: $0 <server_num> <command> [--gpu <gpu_id>]"
    echo ""
    echo "Examples:"
    echo "  $0 3 \"python train.py --epochs 10\""
    echo "  $0 3 \"python train.py\" --gpu 2"
    echo "  $0 3 \"python train.py\" --gpu 0,1"
    exit 1
fi

SERVER="mll-a40-${SERVER_NUM}.cs.utexas.edu"

# Build the command with optional CUDA_VISIBLE_DEVICES
if [ -n "$GPU_ID" ]; then
    FULL_COMMAND="CUDA_VISIBLE_DEVICES=$GPU_ID $COMMAND"
else
    FULL_COMMAND="$COMMAND"
fi

echo "=========================================="
echo "Server:    mll-a40-$SERVER_NUM"
echo "Directory: $WORK_DIR"
echo "GPU:       ${GPU_ID:-all available}"
echo "Command:   $FULL_COMMAND"
echo "=========================================="
echo ""

# SSH in and start tmux, then cd/source/run inside tmux
# tmux new-session will create a new session with default naming (0, 1, 2, etc.)
ssh -tt "$SERVER" "tmux new-session 'cd $WORK_DIR && source $VENV_PATH && $FULL_COMMAND; exec bash'"
