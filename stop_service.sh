#!/bin/bash
echo "Stopping services..."

if [ -f /content/sft/logs/vllm.pid ]; then
    PID=$(cat /content/sft/logs/vllm.pid)
    kill -9 $PID 2>/dev/null || true
    rm -f /content/sft/logs/vllm.pid
fi

if [ -f /content/sft/logs/cloudflared.pid ]; then
    PID=$(cat /content/sft/logs/cloudflared.pid)
    kill -9 $PID 2>/dev/null || true
    rm -f /content/sft/logs/cloudflared.pid
fi

pkill -9 -f "vllm" 2>/dev/null || true
pkill -9 -f "cloudflared" 2>/dev/null || true

echo "All services stopped."
