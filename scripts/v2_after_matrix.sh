#!/usr/bin/env bash
# After part A of the V2 matrix: reduced part B, live loops (GPU / remote NPU / failover), then stop CARLA
# and run the GPU half of the resolution study. One CARLA client at a time (server lock + sequential).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="src;scripts"
PY=.venv/Scripts/python.exe
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  while tasklist //FI "PID eq $WAIT_PID" | grep -q "$WAIT_PID"; do sleep 30; done
fi
echo "$(date +%T) part B"
$PY scripts/v2_run_scenarios.py --scenarios pedestrian_crossing lead_vehicle_braking --controllers v2 v1 \
  --speeds-kmh 30 --conditions fixed_0ms fixed_50ms fixed_100ms fixed_150ms fixed_200ms jitter_50pm40ms \
  burst_25ms_250ms drop30_25ms stale20_25ms npu_colocated npu_remote_v2 npu_remote_jpeg95 npu_remote_v1equiv
echo "$(date +%T) live loops"
$PY scripts/v2_live_loop.py --scenario pedestrian_crossing --backend gpu --video outputs/v2/demo_gpu
$PY scripts/v2_live_loop.py --scenario pedestrian_crossing --backend npu --video outputs/v2/demo_npu
$PY scripts/v2_live_loop.py --scenario lead_vehicle_braking --backend npu
$PY scripts/v2_live_loop.py --scenario lead_vehicle_braking --backend gpu
$PY scripts/v2_live_loop.py --scenario pedestrian_crossing --backend npu_failover --fault-at 1.5 --fault-s 1.5 \
  --video outputs/v2/demo_failover --tag _fault
echo "$(date +%T) stop CARLA, GPU resolution timing"
taskkill //IM CarlaUE4-Win64-Shipping.exe //F; taskkill //IM CarlaUE4.exe //F; sleep 10
$PY scripts/v2_resolution.py --skip-npu --out reports/evidence/v2_resolution_gpu.json
echo V2_AFTER_DONE
