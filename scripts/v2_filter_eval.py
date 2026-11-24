"""V2 Phase 4: offline comparison of range-rate estimators on the V1 (v3 matrix) per-tick logs.

Input: ``outputs/scenario_logs/*__mono__*.jsonl`` from the frozen v3 matrix (read only). Each row
holds the raw monocular gap measurement of the closest in-path detection (``mono_gap_m``) and the
CARLA ground truth (``gap_m``, ``closing_mps``) used here for scoring only. Measurements are
resampled to camera/perception rates of 100, 33 and 16 Hz (the async NPU path delivers fewer,
irregular results), then fed to each estimator.

Metrics per estimator and rate: range MAE, range-rate MAE / p95 / error std (all and beyond 30 m),
reaction delay to the lead vehicle's hard braking, and decision stability (number of on/off flips
of the "TTC < 2.6 s" indicator per run; ground truth typically flips once).
Output: reports/evidence/v2_filter_eval.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from npu_ad.config import resolve
from npu_ad.perception.filters import make_filter

HAZARDS = ("lead_vehicle_braking", "static_obstacle", "pedestrian_crossing", "low_light", "rain")
CANDIDATES = {
    "fd": ("fd", {}),
    "lsq_v1": ("lsq", {}),
    "ab_0.5_0.1": ("ab", {"alpha": 0.5, "beta": 0.1}),
    "ab_0.3_0.05": ("ab", {"alpha": 0.3, "beta": 0.05}),
    "kf_sa4": ("kf", {"sigma_a": 4.0}),
    "kf_sa8": ("kf", {"sigma_a": 8.0}),
    "kf_sa8_gate25": ("kf", {"sigma_a": 8.0, "gate_nis": 25.0}),
    "kf_sa8_px075_gate25": ("kf", {"sigma_a": 8.0, "sigma_px": 0.75, "gate_nis": 25.0}),
    "kf_sa15_px075_gate25": ("kf", {"sigma_a": 15.0, "sigma_px": 0.75, "gate_nis": 25.0}),
    "kf_sa8_px075_nogate": ("kf", {"sigma_a": 8.0, "sigma_px": 0.75, "gate_nis": 1e9}),
}
SUSTAIN_S = 0.2  # reaction = first crossing that stays above the threshold this long
RATES = {"100Hz": 1, "33Hz": 3, "16Hz": 6}
TTC_ON = 2.6


def load_runs() -> list[tuple[str, list[dict]]]:
    runs = []
    for p in sorted(glob.glob(str(resolve("outputs/scenario_logs/*__mono__*.jsonl")))):
        name = Path(p).stem
        if not name.startswith(HAZARDS) or "__replay" in name or "__rep" in name:
            continue
        rows = [json.loads(ln) for ln in open(p, encoding="utf-8")]
        runs.append((name, rows))
    return runs


def flips(flags: list[bool]) -> int:
    return sum(1 for a, b in zip(flags, flags[1:], strict=False) if a != b) + (1 if flags and flags[0] else 0)


def evaluate(runs, kind: str, kw: dict, step: int, cam_h_default: float = 1.7) -> dict:
    rng_err, rate_err, rate_err_far, delays, flips_est, flips_gt, n_valid, n_meas = [], [], [], [], [], [], 0, 0
    for name, rows in runs:
        f = make_filter(kind, **kw)
        flag_est, flag_gt = [], []
        t_brake_gt = t_brake_est = None
        cand_gt = cand_est = None
        closing0 = None
        for r in rows[::step]:
            if r.get("mono_gap_m") is None or not r.get("mono_matches_gt") or r.get("gap_m") is None:
                continue
            if r["gap_m"] <= 0:
                continue
            n_meas += 1
            f.update(r["t_sim"], r["mono_gap_m"])
            gt_rate = -r["closing_mps"]
            if closing0 is None:
                closing0 = r["closing_mps"]
            if not f.valid or f.rate is None:
                continue
            n_valid += 1
            rng_err.append(abs(f.range - r["gap_m"]))
            e = f.rate - gt_rate
            rate_err.append(e)
            if r["gap_m"] >= 30:
                rate_err_far.append(e)
            closing_est = -f.rate
            ttc_est = f.range / closing_est if closing_est > 0.3 else math.inf
            ttc_gt = r["gap_m"] / r["closing_mps"] if r["closing_mps"] > 0.3 else math.inf
            flag_est.append(ttc_est < TTC_ON)
            flag_gt.append(ttc_gt < TTC_ON)
            if name.startswith(("lead_vehicle_braking", "low_light")) and r["t_sim"] >= 3.0:
                thr, t = closing0 + 1.5, r["t_sim"]
                if t_brake_gt is None:
                    cand_gt = (cand_gt or t) if r["closing_mps"] > thr else None
                    if cand_gt is not None and t - cand_gt >= SUSTAIN_S:
                        t_brake_gt = cand_gt
                if t_brake_est is None:
                    cand_est = (cand_est or t) if closing_est > thr else None
                    if cand_est is not None and t - cand_est >= SUSTAIN_S:
                        t_brake_est = cand_est
        if t_brake_gt is not None and t_brake_est is not None:
            delays.append(t_brake_est - t_brake_gt)
        flips_est.append(flips(flag_est))
        flips_gt.append(flips(flag_gt))
    re, rf = np.array(rate_err), np.array(rate_err_far)
    return {
        "runs": len(runs),
        "measurements": n_meas,
        "valid_estimates": n_valid,
        "range_mae_m": float(np.mean(rng_err)) if rng_err else None,
        "rate_mae_mps": float(np.mean(np.abs(re))) if re.size else None,
        "rate_p95_abs_mps": float(np.percentile(np.abs(re), 95)) if re.size else None,
        "rate_err_std_mps": float(np.std(re)) if re.size else None,
        "rate_bias_mps": float(np.mean(re)) if re.size else None,
        "rate_mae_far_mps": float(np.mean(np.abs(rf))) if rf.size else None,
        "rate_p95_abs_far_mps": float(np.percentile(np.abs(rf), 95)) if rf.size else None,
        "reaction_delay_s_median": float(np.median(delays)) if delays else None,
        "reaction_delay_n": len(delays),
        "ttc_flag_flips_per_run_mean": float(np.mean(flips_est)),
        "gt_ttc_flag_flips_per_run_mean": float(np.mean(flips_gt)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/evidence/v2_filter_eval.json")
    a = ap.parse_args()
    runs = load_runs()
    res: dict = {
        "source": "outputs/scenario_logs (v3 matrix, mono mode, hazard scenarios; read only)",
        "runs": [n for n, _ in runs],
        "ground_truth_use": "scoring only",
        "results": {},
    }
    for rate, step in RATES.items():
        res["results"][rate] = {}
        for cname, (kind, kw) in CANDIDATES.items():
            m = evaluate(runs, kind, kw, step)
            res["results"][rate][cname] = m
            print(
                f"{rate:5s} {cname:12s} rate MAE {m['rate_mae_mps']:.2f}"
                f" far {m['rate_mae_far_mps'] or float('nan'):.2f}"
                f" p95 {m['rate_p95_abs_mps']:.2f} std {m['rate_err_std_mps']:.2f} range MAE {m['range_mae_m']:.2f}"
                f" delay {m['reaction_delay_s_median']} flips {m['ttc_flag_flips_per_run_mean']:.1f}"
                f"/{m['gt_ttc_flag_flips_per_run_mean']:.1f} valid {m['valid_estimates']}"
            )
    out = resolve(a.out)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
