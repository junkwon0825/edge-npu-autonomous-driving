# Handoff — V2 (2026-11-24, Claude Code)

Branch `feature/v2-npu-closed-loop` (from `v1.0.0-rc1`), local tag `v2.0.0-rc1`; nothing pushed.
Full report: `reports/v2_final_report.md`; evidence index `reports/v2/evidence_manifest.json`.

1. Architecture: unified InferenceBackend (GPU/RNGD/replay/mock) + failover, latest-frame queue, result gate
   (stale 0.3 s, timeout 0.5 s), tracker + range/lateral Kalman, gated risk model + path prediction, AEB state machine.
2. NPU runtime: persistent BF16 server on RNGD (cache load 5 s), deadline/backpressure client, health monitor.
3. Transport: 203 -> 60 ms lossless (content-only, 2 threads, exact prefilter, ssh -C); JPEG 31 ms lossy.
4. Async: V2 collision-free 0-200 ms + jitter/burst/drop/stale; V1 collides from 100 ms (rate-limited latency).
5. Tracking: vehicles 98-99 % coverage / 1 id; pedestrians 79 % / 2 ids; cyclist 55 %.
6. Filters: KF rate MAE 1.27 vs LSQ 1.51 m/s (33 Hz offline); closed loop 0.64-1.34 m/s.
7. AEB: hazards 0/18 (V2) vs 2/18 (V1) vs 14/18 (off); unneeded 1/12 vs 2/12.
8. Live NPU loop: works, near-real-time (age p95 165-212 ms, 12-15 Hz), min gap 1.48 m (ped) — not real-time.
9. Failover: stall at 1.5 s -> GPU at 1.83 s -> RNGD restored 3.5 s, logged, no collision.
10. Lane seg: not running at V2 end (max_pool2d tactic / tensor-group offsets); V3 lab found a running variant.
11. Ops: rewrites exact for bilinear/ConvTranspose/dilated; round + fake-quant not lowerable; no quant API.
12. Benchmarks: docs/v2_benchmark.md. 13. Failures: failures.jsonl (contamination incidents handled, archived).
14. Tests: 62 passed, 1 skipped; ruff clean. 15. Commits: git log v1.0.0-rc1..v2.0.0-rc1.
16. V3: separate PEs per model, ported lane U-Net, ego-motion compensation, radar fusion.

Approvals pending: push; KB publish; deleting remote replay files (~3 GB) and V2/V3 status files on the host;
decision on the hostname present in frozen V1 evidence before any publication.
