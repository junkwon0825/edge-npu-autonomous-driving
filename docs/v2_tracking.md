# V2: object tracking

Code: `src/npu_ad/perception/tracker.py` (`MultiObjectTracker`). Tests: `tests/test_v2_control.py`.

## Design (SORT-style; no new dependency)
- **Prediction.** Constant-velocity motion of the four box coordinates in pixels, with an α-β update
  (α 0.6, β 0.2).
- **Association.** Same class group only (vehicle {car, motorcycle, bus, truck}, pedestrian, cyclist
  {bicycle}). Greedy on IoU with the predicted box (≥ 0.2), then on centre distance normalised by box
  height (≤ 1.0), for small or fast boxes.
- **Life cycle.**
  - A track is tentative until 3 hits.
  - It is coasted (prediction only) without a detection for up to 0.4 s, then deleted.
  - Confidence is smoothed (0.7 / 0.3).
- **Per-track state:**
  - id, class, box, confidence, image-space centre, age, hits, misses
  - `RangeKalman` on the flat-ground gap; boxes touching the bottom border give only an upper bound
    on range and do not update it
  - `Kalman1D` on the lateral offset, with std sqrt(0.1² + (2 px · Z / f)² + (0.05 · lateral)²);
    boxes touching a side border do not update it (see `docs/v2_aeb_controller.md`, item 1)

The tracker feeds the risk model directly: range, range rate, lateral position and lateral velocity
per object. V1's tracker followed only the single closest in-path box.

## Measured in the V2 matrix
`reports/v2_scenario_tables.md` § C. It is computed on camera frames where the projected GT target
centre is in the image:
- **coverage:** the fraction of frames where a confirmed track's box contains the GT centre
- **identity stability:** the number of distinct track ids that covered the target (1 = no ID switch)
- **detection gaps:** the longest run of uncovered frames
- **range and range-rate MAE / p95** of the covering track against GT

Values are discussed in `docs/v2_benchmark.md`.
