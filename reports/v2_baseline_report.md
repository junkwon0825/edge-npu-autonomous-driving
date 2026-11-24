# V2 Phase 0: V1 baseline verification (2026-11-24)

| Check | Result |
|---|---|
| Tag `v1.0.0-rc1` | present; branch `feature/v2-npu-closed-loop` created from it |
| V1 unit tests | 33 passed, 1 skipped (live test opt-in) |
| V1 evidence manifest | 46/46 files match their SHA-256 (`reports/v1/evidence_manifest.json`) |
| V1 CARLA demo | rerun (`outputs/v2/phase0_v1_demo`): no collision, min gap 2.68 m, brake applied at 2.70 s; identical to the V1 record |
| V1 GPU benchmark artefacts | present (`gpu_benchmark_ext.json`, `gpu_benchmark_fp32_clean.json`); not re-measured |
| V1 RNGD BF16 artefacts | present (`npu_bench_bf16.json`, `npu_perf_bf16.json`); the BF16 module loads from the SDK cache in 5.1 s on the host (no recompilation) |
| RNGD host | reachable through the existing alias; `npu0` firmware 2026.4.0, 39 W idle; host load average 7.5–13 at start |

## Unresolved V1 issues carried into V2
1. Noisy monocular range rate (MAE 1.2–3.2 m/s).
2. The latched AEB has no target-clear logic (50 km/h side collision).
3. The 40 km/h pedestrian crossing collides in every mode.
4. The NPU is not in the loop; the remote transport costs 121 ms per frame.
5. True INT8 compute is blocked.

**Found during V2.** V1's frozen RNGD evidence files and two V1 reports contain the remote container's
**hostname**:
- files: `npu_*.json`, `reports/environment_audit.md`, `reports/npu_compilation.md`
- it is not a credential, and nothing has been pushed
- V1 is frozen, so it was not edited
- V2 evidence redacts host fields (`scripts/v2_fetch_npu_status.py`)
- **decision needed before any push:** a redacting commit, or a history rewrite (the latter needs
  approval)
