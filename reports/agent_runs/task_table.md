# Latest status (2026-11-23T23:52:59+0900)

| Task | Phase | Owner | Status | Retries | Next |
|---|---|---|---|---|---|
| A0 Environment audit | 0 | claude | **PASS** | 0 |  |
| B1 CARLA session + capture smoke (3 runs) | 1 | claude | **PASS** | 0 |  |
| B2 Seeded multi-weather dataset (72 frames) | 1 | claude | **PASS** | 0 |  |
| C1 Raw-head split validated vs ultralytics | 2 | claude | **PASS** | 0 |  |
| C2 GPU baseline benchmark | 2 | claude | **PASS** | 0 |  |
| N1 NPU hardware gate | 3 | claude | **PASS** | 0 |  |
| N2 RNGD AOT compile of raw YOLO26n | 3 | claude | **PASS** | 0 |  |
| N3 NPU runtime inference + numerical check vs CPU | 3 | claude | **PASS** | 1 |  |
| Q1 Quantization | 4 | claude | **PARTIAL** | 0 | INT8 compute BLOCKED by SDK 2026.4.0 (compiler probe); W8 weight-only TESTED (no speed-up); BF16 recommended |
| P1 GPU vs NPU benchmark + detection agreement | 5 | claude | **PASS** | 0 |  |
| F1 Closed-loop AEB scenarios x latency conditions | 6 | claude | **PASS** | 0 |  |
| R1 Independent review (Kiro 2026-11-22; Codex+Kiro 2026-11-23) | 7 | kiro | **PASS** | 0 |  |
| G1 Portfolio docs | 7 | claude | **PASS** | 0 |  |
| H1 Phase 1 technical audit (Codex, Kiro, Claude) | 1 | claude | **PASS** | 0 |  |
| H2 Raw split vs unfused original, all 72 frames | 1 | claude | **PASS** | 0 |  |
| H3 CarlaSession hardening + exclusive server lock + opt-in live tests | 5 | claude | **PASS** | 0 |  |
| Q2 INT8/FP8 compiler probe | 4 | claude | **PASS** | 0 |  |
| Q3 W8 weight-only compile/bench/agreement | 4 | claude | **PASS** | 0 |  |
| P2 NPU perf: cold start, throughput, power | 2 | claude | **PASS** | 0 | pipelined burst>1 FAILED on device (HAL PXI-122); not retried |
| P3 GPU ext: batch sweep, CUDA Graphs, power, cold start | 2 | claude | **PASS** | 0 |  |
| P4 Desktop<->RNGD link latency | 2 | claude | **PASS** | 0 |  |
| F2 AEB v3 matrix: 3 modes x 8 scenarios x 7 latencies + speed sweep + repeat | 4 | claude | **PASS** | 0 | 6 runs quarantined and re-run |
| F3 Open-loop GPU vs NPU AEB replay (738 frames) | 4 | claude | **PASS** | 0 |  |
| N4 NPU batch>1 compile | 2 | claude | **NOT_RUN** | 0 | each static shape = new 20-30 min compile on shared host |
| N5 NPU-in-the-loop closed-loop driving | 4 | claude | **BLOCKED** | 0 | 121 ms SSH transfer per raw frame; needs co-located NPU |
| G2 Docs, plots, README, portfolio summaries | 6 | claude | **PASS** | 0 |  |

Status values: PASS / PARTIAL / BLOCKED / BLOCKED_APPROVAL / NOT_RUN / RUNNING / TODO.
