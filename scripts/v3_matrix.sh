#!/usr/bin/env bash
# V3 closed-loop matrix (resumable: finished runs in the output file are skipped).
# Phases are representative, not exhaustive; reasoning in reports/v3_final_report.md.
# A memory guard stops the matrix (recorded as a blocker) when free RAM falls below MIN_FREE_GB.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="src;scripts"
PY=.venv/Scripts/python.exe
OUT=reports/evidence/v3_closed_loop.json
MIN_FREE_GB=${MIN_FREE_GB:-1.0}
LOG=outputs/v3/matrix.log
mkdir -p outputs/v3

free_gb() {
  powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)"
}

phase() {
  local name=$1
  shift
  local f
  f=$(free_gb | tr -d '\r')
  echo "=== $(date +%H:%M:%S) phase $name (free RAM ${f} GB)" | tee -a "$LOG"
  if awk "BEGIN{exit !($f < $MIN_FREE_GB)}"; then
    echo "BLOCKED: free RAM ${f} GB < ${MIN_FREE_GB} GB before phase $name" | tee -a "$LOG"
    exit 3
  fi
  $PY scripts/carla_reset.py >>"$LOG" 2>&1
  $PY scripts/v3_run_scenarios.py --out "$OUT" "$@" 2>&1 | grep -E '^\{|Error|Traceback|BLOCKED' | tee -a "$LOG"
}

LK="lk_straight lk_mild lk_strong lk_strong_heldout lk_offset lk_night lk_rain"
AEB="lead_vehicle_braking static_obstacle pedestrian_crossing pedestrian_clearing cyclist_crossing"
COMB="curve_obstacle curve_lead_brake ped_cornering degraded_lane"

phase lk30 --scenarios $LK --modes A B C D --speeds-kmh 30
phase lk50 --scenarios lk_straight lk_mild lk_strong --modes A D --speeds-kmh 50
phase aeb --scenarios $AEB --modes A D --speeds-kmh 30 50
phase aeb_radar --scenarios $AEB --modes A --speeds-kmh 30 50 --radar on
phase aeb_radar_d --scenarios lead_vehicle_braking pedestrian_crossing --modes D --speeds-kmh 30 --radar on
phase ped40 --scenarios pedestrian_crossing --modes A D --speeds-kmh 40
phase comb30 --scenarios $COMB --modes A B C D --speeds-kmh 30
phase comb50 --scenarios curve_obstacle curve_lead_brake --modes A D --speeds-kmh 50
phase comb_radar --scenarios curve_obstacle curve_lead_brake ped_cornering --modes A --speeds-kmh 30 --radar on
phase sched --scenarios lk_strong pedestrian_crossing curve_lead_brake --modes D --speeds-kmh 30 \
  --sched every every_samepe adaptive
phase sched50 --scenarios pedestrian_crossing --modes D --speeds-kmh 50 --sched every every_samepe adaptive
phase faults_det --scenarios lead_vehicle_braking pedestrian_crossing --modes A B --speeds-kmh 30 \
  --faults det_drop30 det_stale20 det_timeout
phase faults_lane --scenarios lk_strong curve_lead_brake --modes A C --speeds-kmh 30 --faults seg_failure
phase faults_npu --scenarios lead_vehicle_braking lk_strong --modes B --speeds-kmh 30 --faults npu_failure npu_stall
$PY scripts/carla_reset.py >>"$LOG" 2>&1
echo "=== $(date +%H:%M:%S) matrix done" | tee -a "$LOG"
