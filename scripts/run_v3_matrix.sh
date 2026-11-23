#!/usr/bin/env bash
# Phase 6 v3 matrix (resumable: finished runs are skipped). CARLA server must be running.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
OUT=reports/evidence/scenarios_v3.json
# 1) all scenarios x 3 AEB input modes x latency conditions at 30 km/h
$PY scripts/run_scenarios.py --modes gt_oracle gt_gated mono --out $OUT
# 2) speed sweep for hazard scenarios (GT-assisted vs perception-driven, NPU latency)
$PY scripts/run_scenarios.py --scenarios static_obstacle pedestrian_crossing lead_vehicle_braking \
  --modes gt_gated mono --speeds-kmh 20 40 50 --conditions aeb_off npu_p95 --out $OUT
# 3) determinism: repeat a subset with a tag and compare logs afterwards
$PY scripts/run_scenarios.py --scenarios static_obstacle pedestrian_crossing lead_vehicle_braking \
  --modes mono --conditions npu_p95 --tag rep2 --out $OUT
echo MATRIX_DONE
