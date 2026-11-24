# V3 latest status (2026-11-25)

- **Branch:** `feature/v3-multimodel-npu`, from `v2.0.0-rc1`.
- **Tag:** local tag `v3.0.0-rc1`.
- **Remote:** nothing pushed; no history rewritten.
- **Status:** COMPLETE (release candidate) with documented safety failures. Report:
  `reports/v3_final_report.md`.

## Done
- **NPU:**
  - YOLO26n and the lane U-Net (BF16) on RNGD, plus a multi-model server (two PEs)
  - YOLO26s via SplitConv
  - MobileNetV3-S and a tiny ViT
  - ResNet18 and LR-ASPP documented as failing
  - 74-row operator database
  - model table in `docs/npu_model_portfolio.md`
- **Closed loop:**
  - 74 runs in `v3_closed_loop.json`: the first 24 lane-keeping runs plus a focused matrix of 50
  - 6 reruns: night/rain C/D, and stale injection after a fix
  - 3 demo clips
  - 4 frame-capture runs
- **Night/rain root cause:** remote request timeouts plus no lane fallback in Modes C/D. Not numerics
  (same-frame test on 440 frames) and not the scheduler.
- **Bugs fixed, with tests:**
  - Mode D lane-request aliasing
  - failover latency excluding the failed wait
  - measured-latency simulation erasing the stale injection
- **Checks:**
  - 80 unit tests + 1 live CARLA test pass (`NPU_AD_LIVE_CARLA=1`: 80 passed)
  - ruff clean; imports checked
  - actors cleaned up, synchronous mode restored, CARLA lock released
- **Privacy:** re-scan in `reports/github_sanitization_report.md`.

## Open safety failures (kept as evidence)
1. Remote NPU staleness: at 50 km/h on a curve with lead braking, the minimum gap is 2.1 m (GPU: 13.7 m).
2. 20 % stale detections with RNGD latency: 0.33 m gap to a pedestrian.
3. Unneeded braking on the 39 m-radius curve (4/10 runs); early braking for a kerbside pedestrian.
4. Radar fusion makes closing speed worse for a pedestrian and on a curve.
5. RNGD lane modes have no default lane fallback.

## Public-release preparation (2026-11-25)
- **History rewrite (option B):** done in a separate clone; the sanitization log is
  `reports/github_sanitization_report.md`.
- **Commits and tags:** 59 commits rewritten; the author e-mail is now the GitHub no-reply address; V1/V2
  tags were rewritten in place.
- **Docs:** README restructured as the portfolio landing page; resume summaries (EN/KO) and the
  interview Q&A expanded; BF16 vs preprocessing wording made consistent.
- **Checks:** tests, lint, imports, links, the manifest verifier and a full-history privacy scan.

## Awaiting user approval
- Push to GitHub: remote, main branch, tags.
- Publishing the KB handoff to `jun-ai-knowledge`.
- Deleting the ~3 GB of replay files on the shared RNGD host.
