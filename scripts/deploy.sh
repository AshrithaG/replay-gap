#!/usr/bin/env bash
# One-shot deploy + bootstrap to the GPU VM. Run from the repo root on the Mac:
#   bash scripts/deploy.sh user@host
# Requires the campus VPN to be connected. You'll be prompted for your
# password on each ssh/rsync hop (or set up ssh-copy-id once to skip that).
set -euo pipefail

VM=${1:?usage: bash scripts/deploy.sh user@host}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo ">> checking connectivity to $VM"
ssh -o ConnectTimeout=10 "$VM" true

echo ">> syncing code"
rsync -av --exclude .venv --exclude runs --exclude __pycache__ --exclude logs \
  "$REPO_DIR/" "$VM":~/replay-gap/

echo ">> bootstrapping remote environment"
ssh "$VM" 'bash -s' <<'REMOTE'
set -euo pipefail
cd ~/replay-gap
echo "--- environment probe ---"
uname -m
python3 --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi not found"
docker --version 2>/dev/null || echo "WARNING: docker not found"
df -h ~ | tail -1

if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "--- smoke test (no GPU needed) ---"
python scripts/smoke_test.py
echo
echo "Bootstrap OK. vLLM/swebench install next (large, GPU-dependent):"
echo "  source ~/replay-gap/.venv/bin/activate && pip install vllm swebench"
REMOTE

echo
echo ">> Deployed. Next, on the VM (inside tmux):"
echo "   ssh $VM"
echo "   cd ~/replay-gap && source .venv/bin/activate && pip install vllm swebench"
echo "   bash scripts/serve_models.sh"
echo "   bash scripts/gpu_power_logger.sh runs/pilot/gpu_power.csv   # pane 2"
echo "   python scripts/run_pilot.py --output runs/pilot             # pane 3"
