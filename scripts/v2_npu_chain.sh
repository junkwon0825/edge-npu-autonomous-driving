#!/usr/bin/env bash
# V2: run RNGD jobs one after another (one job of this project at a time, bounded waits).
# Usage: bash scripts/v2_npu_chain.sh "port_lab --group ops" "compile --dtype bf16 --imgsz 320 --tag _320"
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
running() {
  ssh -o BatchMode=yes -o ConnectTimeout=20 furiosa-npu \
    "ps -u \$(id -u) -o args= | grep -cE '^python[0-9.]* (npu_job|int8_probe|port_lab|multistream)[.]py'" 2>/dev/null
}
i=0
for job in "$@"; do
  i=$((i + 1))
  for _ in $(seq 1 120); do  # wait (<= 2 h) until the previous job released the lock
    out=$($PY scripts/npu_remote.py start --log-suffix "_v2_$i" -- $job 2>&1)
    echo "$out" | grep -q BUSY || break
    sleep 60
  done
  echo "$(date +%H:%M:%S) start [$job]: $(echo "$out" | tr '\n' ' ')"
  sleep 20
  for _ in $(seq 1 120); do  # job runtime bound: 2 h
    n=$(running); [ "${n:-1}" = "0" ] && break
    sleep 60
  done
  echo "$(date +%H:%M:%S) done [$job]"
done
echo V2_NPU_CHAIN_DONE
