#!/usr/bin/env bash
# Log GPU power draw to CSV at 2Hz. Run alongside the pilot (tmux pane):
#   bash scripts/gpu_power_logger.sh runs/pilot/gpu_power.csv
# Integrate power over time later to get joules per arm (timestamps in the
# trajectory files let you attribute windows to rollouts).
set -euo pipefail
OUT=${1:-gpu_power.csv}
echo "timestamp,gpu_index,power_w,util_pct,mem_used_mib" > "$OUT"
echo "Logging GPU power to $OUT (Ctrl-C to stop)"
while true; do
  nvidia-smi --query-gpu=index,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | while IFS=', ' read -r idx pw util mem; do
        echo "$(date +%s.%N),$idx,$pw,$util,$mem" >> "$OUT"
      done
  sleep 0.5
done
