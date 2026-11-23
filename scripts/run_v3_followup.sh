#!/usr/bin/env bash
# Runs after run_v3_matrix.sh (CARLA server running): remote-NPU latency condition and replay dumps.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
OUT=reports/evidence/scenarios_v3.json
$PY scripts/run_scenarios.py --scenarios static_obstacle pedestrian_crossing lead_vehicle_braking low_light rain \
  --modes gt_gated mono --conditions npu_remote_p95 --out $OUT
$PY scripts/run_scenarios.py --scenarios static_obstacle pedestrian_crossing lead_vehicle_braking low_light rain \
  adjacent_vehicle pedestrian_roadside --modes mono --conditions npu_p95 --tag replay --dump-every 5 --out $OUT
echo FOLLOWUP_DONE
