#!/bin/bash

n_max_token_per_frame="${1:-256}"

# 检查是否提供路径参数
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <n_max_token_per_frame> <path1> [<path2> ...]"
  exit 1
fi

# 将路径从参数列表提取出来
path_list=("${@:2}")

# 遍历路径并执行命令
for path in "${path_list[@]}"; do
    echo "Processing path: $path"
    bash launch_vllm.sh "$path" 600 8 10086
    sleep 300
    python eval_video_mlvu.py --save_path="$path" --start_port=10086 --n_devices=8 --version=NEW_v1
    sleep 300
    pkill vllm
    sleep 300
done