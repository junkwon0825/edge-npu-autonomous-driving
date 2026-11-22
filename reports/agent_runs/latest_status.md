# Latest status (2026-11-22T12:11:04+0900)

| Task | Phase | Owner | Status | Retries | Next |
|---|---|---|---|---|---|
| A0 Environment audit | 0 | claude | **PASS** | 0 |  |
| B1 CARLA session + capture smoke (3 runs) | 1 | claude | **PASS** | 0 |  |
| B2 Seeded multi-weather dataset (72 frames) | 1 | claude | **PASS** | 0 |  |
| C1 Raw-head split validated vs ultralytics | 2 | claude | **PASS** | 0 |  |
| C2 GPU baseline benchmark | 2 | claude | **PASS** | 0 |  |
| N1 NPU hardware gate | 3 | claude | **PASS** | 0 |  |
| N2 RNGD AOT compile of raw YOLO26n | 3 | claude | **PASS** | 0 | poll status_compile.json |
| N3 NPU runtime inference + numerical check vs CPU | 3 | claude | **PASS** | 1 |  |
| Q1 Quantization | 4 | claude | **RUNNING** | 0 |  |
| P1 GPU vs NPU benchmark + detection agreement | 5 | claude | **PASS** | 0 |  |
| F1 Closed-loop AEB scenarios x latency conditions | 6 | claude | **PASS** | 0 |  |
| R1 Independent review (Kiro, read-only via agent_lab) | 7 | kiro | **PASS** | 0 |  |
| G1 Portfolio docs | 7 | claude | **TODO** | 0 |  |

Status values: PASS / PARTIAL / BLOCKED / BLOCKED_APPROVAL / NOT_RUN / RUNNING / TODO.
