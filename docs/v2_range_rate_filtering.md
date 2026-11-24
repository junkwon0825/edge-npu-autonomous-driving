# V2: range and range-rate filtering

V1's weakest component was the closing speed (range rate) estimated from monocular range. It had an
MAE of 1.2–3.2 m/s and a p95 of up to 14 m/s beyond 30 m. V1 used a least-squares slope over a 0.3 s
window. V2 compares four estimators offline on the same measurements, picks one, and then validates
it in new closed-loop runs.

Implementation: `src/npu_ad/perception/filters.py`. Evaluation: `scripts/v2_filter_eval.py` →
`reports/evidence/v2_filter_eval.json`.

## Data and protocol
- **Input.** 39 per-tick logs of the frozen v3 matrix (perception-driven mode, 5 hazard scenarios × 6
  latency conditions plus repeats), read only. Each tick holds the raw monocular gap of the closest
  in-path detection.
  - Only ticks where that detection is the scored target are used (the GT centre lies in its box).
  - CARLA ground truth (`gap_m`, `closing_mps`) is used **for scoring only**.
- **Rates.** Measurements are resampled to 100 Hz (V1 perception rate), 33 Hz (V2 camera rate) and
  16 Hz (an asynchronous remote-NPU rate). The filters handle irregular time steps.
- **Reaction delay.** On the lead-braking runs: the time from the GT closing speed exceeding its
  pre-braking value + 1.5 m/s (sustained 0.2 s) to the estimate doing the same, sustained 0.2 s.
  - A first version without the sustain condition credited noisy estimators with a zero delay
    whenever noise crossed the threshold early. It was replaced.
- **Decision stability.** The number of on/off flips of the indicator "TTC < 2.6 s" per run. The GT
  flips about twice per run.

## Estimators
| Name | Description | Parameters |
|---|---|---|
| `fd` | raw finite difference (z_k − z_{k−1}) / Δt | — |
| `lsq_v1` | V1: least-squares slope over 0.3 s, min 5 samples, rejected if residual std > 0.5 m | V1 values |
| `ab` | alpha-beta tracker | α 0.5 / β 0.1 and α 0.3 / β 0.05 |
| `kf` | constant-velocity Kalman filter, state [range, rate], white-acceleration noise σa | see below |

**Kalman measurement model.** The flat-ground range error grows with the square of the distance. A
box-bottom error of σpx pixels gives σz ≈ σpx · z² / (fy · h_cam). The measurement std is therefore
sqrt(σr0² + (σpx · z² / (fy · h_cam))²), with σr0 = 0.15 m.

**Outlier gate.** An update whose normalised innovation squared (NIS) exceeds the gate is rejected
(predict only). After 3 consecutive rejections the filter re-initialises.

## Results at 33 Hz (the V2 camera rate)
| Estimator | Range MAE | Rate MAE | Rate p95 \|err\| | Rate err std | Rate MAE beyond 30 m | Rate p95 beyond 30 m | Reaction delay (median, n = 13) | TTC-flag flips per run (GT 2.15–2.18) |
|---|---|---|---|---|---|---|---|---|
| fd | 0.50 | 5.24 | 17.54 | 8.80 | 10.97 | 28.90 | 0.54 s (n = 7) | 32.7 |
| lsq_v1 (V1) | 0.49 | 1.51 | 4.04 | 2.40 | 3.44 | 15.95 | 0.45 s | 2.82 |
| ab 0.5 / 0.1 | 0.50 | 1.43 | 4.07 | 2.06 | 2.74 | 9.51 | 0.42 s | 3.05 |
| ab 0.3 / 0.05 | 0.51 | 1.41 | 3.76 | 2.00 | 2.45 | 8.69 | 0.45 s | 2.23 |
| kf σa 8, σpx 1.5, gate 9 | 0.50 | 1.47 | 4.12 | 1.97 | 2.36 | 6.76 | 0.57 s | 1.23 |
| **kf σa 15, σpx 0.75, gate 25 (V2 default)** | **0.49** | **1.27** | **2.98** | **1.73** | **2.07** | 8.62 | 0.48 s | 1.95 |

At 16 Hz the ranking is the same: V2 KF rate MAE 1.24 vs LSQ 1.44, p95 2.93 vs 4.19, far p95 4.62 vs
5.67. At 100 Hz the V2 KF has MAE 1.36 vs 1.86 and far MAE 2.16 vs 6.03.

**Selection.** The Kalman filter with σa = 15 m/s², σpx = 0.75 px and NIS gate 25 has the lowest rate
MAE, p95 and error spread at every rate. Its reaction delay (0.48 s) matches V1's (0.45 s). It
produces fewer spurious TTC flips than V1 (1.95 vs 2.82), slightly fewer than GT.

The first Kalman settings (σa = 4–8, gate 9) were smoother but reacted 0.1–0.3 s later to hard braking,
because the gate treated the manoeuvre as an outlier. A wider gate and a larger σa fixed this.

The α-β tracker with α 0.3 / β 0.05 is nearly as good and simpler. It was not chosen because the
Kalman filter also yields a rate uncertainty and handles the range-dependent noise.

**Caveat (selection bias).** The Kalman parameters were chosen on these 39 logs. The V2 closed-loop
matrix uses new runs, a new tracker and a 33 Hz asynchronous pipeline, and serves as the out-of-sample
check. Its tracking errors are reported in `docs/v2_tracking.md`.

**Honest size of the gain.**
- Rate MAE: 1.51 → 1.27 m/s at 33 Hz.
- Rate p95: 4.04 → 2.98 m/s.
- MAE beyond 30 m: 3.44 → 2.07 m/s.

That is a real improvement, but not a solved problem. Monocular range rate at 30+ m still carries
about 2 m/s of error, because the box bottom moves by fractions of a pixel per frame at that distance.
