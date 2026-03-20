#!/bin/bash

# Check GPU availability across mll-a40-1 through mll-a40-7
# Returns server numbers where ALL 8 GPUs have < THRESHOLD MiB usage
# Also reports which individual GPUs are free on each server

THRESHOLD=1300  # MiB

available_servers=()

for N in {1..7}; do
    server="mll-a40-${N}.cs.utexas.edu"
    
    # Get GPU memory usage via SSH, timeout after 10 seconds
    gpu_output=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$server" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null)
    
    if [ $? -ne 0 ]; then
        echo "Server $N: Connection failed or timeout"
        continue
    fi
    
    # Check if all GPUs are below threshold
    all_below=true
    max_usage=0
    gpu_count=0
    free_count=0
    free_gpus=()
    
    while IFS= read -r usage; do
        # Trim whitespace
        usage=$(echo "$usage" | tr -d ' ')
        if [ -n "$usage" ]; then
            if [ "$usage" -ge "$THRESHOLD" ]; then
                all_below=false
            else
                free_count=$((free_count + 1))
                free_gpus+=("$gpu_count")
            fi
            if [ "$usage" -gt "$max_usage" ]; then
                max_usage=$usage
            fi
            gpu_count=$((gpu_count + 1))
        fi
    done <<< "$gpu_output"
    
    # Format free GPU list
    if [ ${#free_gpus[@]} -gt 0 ]; then
        free_gpu_str=$(IFS=,; echo "${free_gpus[*]}")
    else
        free_gpu_str="none"
    fi
    
    if [ "$all_below" = true ] && [ "$gpu_count" -eq 8 ]; then
        echo "Server $N: AVAILABLE (max usage: ${max_usage} MiB across $gpu_count GPUs) | Free: $free_count GPUs [$free_gpu_str]"
        available_servers+=("$N")
    else
        echo "Server $N: BUSY (max usage: ${max_usage} MiB across $gpu_count GPUs) | Free: $free_count GPUs [$free_gpu_str]"
    fi
done

echo ""
echo "================================"
if [ ${#available_servers[@]} -gt 0 ]; then
    echo "Available servers (all GPUs < ${THRESHOLD} MiB): ${available_servers[*]}"
else
    echo "No servers available with all GPUs < ${THRESHOLD} MiB"
fi
