#!/usr/bin/env bash
# Runs after run_v3_matrix.sh has exited (it must not run concurrently: one CARLA owner at a time).
# 1) quarantine runs that overlapped the unauthorized live-pytest client (2026-11-23 22:34:17-22:36:16)
# 2) re-run exactly those keys  3) remote-NPU latency condition  4) replay frame dumps
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
OUT=reports/evidence/scenarios_v3.json
REASON="overlapped an unauthorized CARLA client (live pytest) between 22:34:17 and 22:36:16; re-run"
$PY scripts/quarantine_runs.py --scenario lead_vehicle_braking --mode gt_oracle \
  --conditions ideal_0ms gpu_p95 npu_p95 stress_100ms stress_200ms --reason "$REASON"
$PY scripts/quarantine_runs.py --scenario lead_vehicle_braking --mode gt_gated --conditions ideal_0ms --reason "$REASON"
$PY scripts/run_scenarios.py --scenarios lead_vehicle_braking --modes gt_oracle gt_gated \
  --conditions ideal_0ms gpu_p95 npu_p95 stress_100ms stress_200ms --out $OUT
$PY scripts/run_scenarios.py --scenarios static_obstacle pedestrian_crossing lead_vehicle_braking low_light rain \
  --modes gt_gated mono --conditions npu_remote_p95 --out $OUT
$PY scripts/run_scenarios.py --scenarios static_obstacle pedestrian_crossing lead_vehicle_braking low_light rain \
  adjacent_vehicle pedestrian_roadside --modes mono --conditions npu_p95 --tag replay --dump-every 5 --out $OUT
echo FOLLOWUP_DONE
