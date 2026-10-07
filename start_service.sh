#!/bin/bash
set -e
mkdir -p /content/sft/logs

echo "Starting vLLM server with nohup..."
nohup /usr/local/bin/vllm serve /content/sft/Qwen3.5-9B-FineTuned \
    --served-model-name qwen3.5-9b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --trust-remote-code > /content/sft/logs/vllm.log 2>&1 &
echo $! > /content/sft/logs/vllm.pid
echo "vLLM PID: $(cat /content/sft/logs/vllm.pid)"

echo "Starting Cloudflare Tunnel with nohup..."
nohup /usr/local/bin/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate > /content/sft/logs/cloudflared.log 2>&1 &
echo $! > /content/sft/logs/cloudflared.pid
echo "Cloudflare PID: $(cat /content/sft/logs/cloudflared.pid)"
