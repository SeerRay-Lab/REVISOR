#!/bin/bash

# Set default values
ckpt_dir="${1:-Qwen2.5-VL-7B-ckpt_path}"
waiting_seconds="${2:-0}"
n_process="${3:-1}"
start_port="${4:-10086}"

# Validate start_port is a number
if ! [[ "$start_port" =~ ^[0-9]+$ ]]; then
    echo "Error: start_port must be a number"
    exit 1
fi

# Launch n_process vllm serve processes
for ((i=0; i<n_process; i++)); do  # Changed 'n' to 'n_process'
    port=$((start_port + i))
    nohup env CUDA_VISIBLE_DEVICES=$i vllm serve "${ckpt_dir}" \
        --port "${port}" \
        --gpu-memory-utilization 0.7 \
        --tensor-parallel-size 1 \
        --served-model-name "coc" \
        --trust-remote-code \
        --disable-log-requests \
        --limit-mm-per-prompt "video=6" > outputs/shell_logs/$port.log 2>&1 & 
    
    if [ "$i" -eq 0 ]; then
        echo "Waiting $waiting_seconds s for the first GPU (i=0) to initialize..."
        sleep ${waiting_seconds}
    fi
done

echo "Started ${n_process} vllm serve processes on ports ${start_port} to $((start_port + n_process - 1))"
