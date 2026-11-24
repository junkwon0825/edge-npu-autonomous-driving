#!/usr/bin/env bash
# V3 focused closed-loop matrix (resource-constrained). Resumable: runs already in the output file are
# skipped, so the 24 lane-keeping runs of scripts/v3_matrix.sh phase lk30 are kept and not repeated.
# Usage: bash scripts/v3_matrix_focused.sh <phase>...   (phases below; each fits in ~10 min)
# Every run first checks free RAM (--min-free-gb 1.0) and stops gracefully below it.
# Design and reasoning: reports/v3_final_report.md "Closed-loop evaluation".
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="src;scripts"
PY=.venv/Scripts/python.exe
OUT=reports/evidence/v3_closed_loop.json
LOG=outputs/v3/matrix_focused.log
mkdir -p outputs/v3

run() {
  $PY scripts/carla_reset.py >>"$LOG" 2>&1
  $PY scripts/v3_run_scenarios.py --out "$OUT" --min-free-gb 1.0 "$@" 2>&1 | grep -E '^\{|Error|Traceback|BLOCKED' | tee -a "$LOG"
}

for phase in "$@"; do
  echo "=== $(date +%H:%M:%S) $phase" | tee -a "$LOG"
  case $phase in
    # lane keeping: sharp curve completes the 30 km/h A-D block (other 24 runs exist)
    lk_strong) run --scenarios lk_strong --modes A B C D ;;
    # lane frame age: GPU lane (Mode A) with injected lane latency, curves and night
    lane_age) run --scenarios lk_strong --modes A --faults lane_delay20 lane_delay50 lane_delay100 lane_delay150 lane_delay250 ;;
    lane_age2) run --scenarios lk_mild lk_night --modes A --faults lane_delay150 lane_delay250 ;;
    # combined AEB + LKA on curves, all backend modes at 30 km/h, A/D at 50 km/h
    comb_a) run --scenarios curve_obstacle curve_lead_brake ped_cornering --modes A B ;;
    comb_b) run --scenarios curve_obstacle curve_lead_brake ped_cornering --modes C D ;;
    comb50) run --scenarios curve_obstacle curve_lead_brake --modes A D --speeds-kmh 50 ;;
    # camera-only vs camera + radar (Mode A), straight AEB at 30 / 50 km/h and the curve obstacle
    radar_cam) run --scenarios lead_vehicle_braking pedestrian_crossing --modes A --speeds-kmh 30 50 ;;
    radar_on) run --scenarios lead_vehicle_braking pedestrian_crossing --modes A --speeds-kmh 30 50 --radar on ;;
    radar_curve) run --scenarios curve_obstacle --modes A --radar on ;;
    # high-speed pedestrian with both models on RNGD
    ped50_d) run --scenarios pedestrian_crossing --modes D --speeds-kmh 50 ;;
    # scheduling in the loop (Mode D): both every frame on separate PEs / same PE, adaptive priority
    sched) run --scenarios curve_lead_brake ped_cornering --modes D --sched every every_samepe adaptive ;;
    # faults: NPU stall / crash with GPU failover (both paths), stale detection results, lane model failure
    faults) run --scenarios lead_vehicle_braking --modes B --faults npu_stall npu_failure det_stale20 ;;
    faults_lane) run --scenarios lk_strong --modes B --faults npu_stall ;
                 run --scenarios curve_lead_brake --modes C --faults seg_failure ;;
    *) echo "unknown phase $phase" ;;
  esac
done
$PY scripts/carla_reset.py >>"$LOG" 2>&1
echo "=== $(date +%H:%M:%S) done: $*" | tee -a "$LOG"
