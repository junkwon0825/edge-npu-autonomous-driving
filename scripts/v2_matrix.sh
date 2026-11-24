#!/usr/bin/env bash
# V2 CARLA matrix (resumable; one CARLA client at a time, server lock enforced by CarlaSession)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="src;scripts"
PY=.venv/Scripts/python.exe
# A: controller comparison over 10 scenarios x 3 speeds, measured in-loop GPU latency distribution
$PY scripts/v2_run_scenarios.py --controllers v2 v1 aeb_off --conditions gpu_local --speeds-kmh 30 40 50
# B: latency / runtime stress at 30 km/h on the three main hazards
$PY scripts/v2_run_scenarios.py --scenarios pedestrian_crossing lead_vehicle_braking static_obstacle \
  --controllers v2 v1 --speeds-kmh 30 --conditions fixed_0ms fixed_20ms fixed_50ms fixed_100ms fixed_150ms \
  fixed_200ms jitter_50pm40ms burst_25ms_250ms drop30_25ms stale20_25ms npu_colocated npu_remote_v2 \
  npu_remote_jpeg95 npu_remote_v1equiv
echo V2_MATRIX_DONE
