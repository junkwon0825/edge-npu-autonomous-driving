#!/usr/bin/env bash
# Sequential RNGD jobs (one at a time, each wait bounded). Usage: bash scripts/npu_chain.sh "bench --dtype w8" "perf"
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/Scripts/python.exe
wait_done() {  # $1 = status name; poll every 60 s, at most $2 minutes
  for _ in $(seq 1 "$2"); do
    s=$($PY scripts/npu_remote.py status "$1" | grep -o '"state": "[A-Z_]*"' | head -1)
    case "$s" in *RUNNING*|"") sleep 60;; *) echo "$1 $s"; return 0;; esac
  done
  echo "$1 TIMEOUT"; return 1
}
for job in "$@"; do
  set -- $job
  cmd=$1; shift
  dtype=fp32; tag=""
  while [ $# -gt 0 ]; do case "$1" in --dtype) dtype=$2; shift;; --tag) tag=$2; shift;; esac; shift; done
  name=$cmd; [ "$dtype" != fp32 ] && name="${cmd}_${dtype}"; name="$name$tag"
  for _ in $(seq 1 60); do  # wait until the previous job released the lock
    out=$($PY scripts/npu_remote.py start -- $job 2>&1); echo "$out" | grep -q BUSY || break; sleep 60
  done
  echo "start $job: $out" | tr '\n' ' '; echo
  sleep 5
  wait_done "$name" 90 || exit 1
done
echo NPU_CHAIN_DONE
