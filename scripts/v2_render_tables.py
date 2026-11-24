"""Render reports/v2_scenario_tables.md from reports/evidence/v2_scenarios.json (generated; do not hand-edit)."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict

from npu_ad.config import resolve

HAZ_ORDER = ["static_obstacle", "lead_vehicle_braking", "pedestrian_crossing", "cyclist_crossing", "rain", "low_light"]
SAFE_ORDER = ["pedestrian_clearing", "pedestrian_curbside", "adjacent_vehicle", "straight_road"]


def f(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def cell(r) -> str:
    if r is None:
        return "not run"
    if r["collision"]:
        return f"**COLLISION** ({r['collision_with'].split('.')[0]})"
    s = "stop" if r["stopped"] else "no stop"
    if r["hazard"]:
        s = f"gap {f(r['min_gap_m'], 1)} m, {s}"
    if r["unneeded_brake"]:
        s = f"**unneeded brake** ×{r['brake_onsets']}, {s}"
    elif r["brake_onsets"] > 1:
        s += f", {r['brake_onsets']} onsets"
    if r["resumed_after_stop"]:
        s += ", resumed"
    return s


def main() -> int:
    runs = json.loads(resolve("reports/evidence/v2_scenarios.json").read_text(encoding="utf-8"))["runs"]
    idx = {(r["scenario"], r["controller"], r["condition"], int(r["ego_kmh"])): r for r in runs}
    commits = sorted({r.get("code_commit", "?") for r in runs})
    out = [
        "# V2 CARLA scenario tables (generated)",
        "",
        f"Source: `reports/evidence/v2_scenarios.json` ({len(runs)} runs; code commit(s) {', '.join(commits)}).",
        "Town10HD_Opt, dt 0.01 s, camera 33 Hz, perception = GPU YOLO26n detections released by the async pipeline",
        "after the condition's latency. Min gap = approach phase only (until stop / contact / passing). `aeb_off` = no AEB.",
        "",
        "## A. Controllers at the measured in-loop GPU latency (`gpu_local`)",
    ]
    for kmh in (30, 40, 50):
        out += ["", f"### {kmh} km/h", "", "| Scenario | V2 | V1 (perception-driven) | AEB off |", "|---|---|---|---|"]
        for sc in HAZ_ORDER + SAFE_ORDER:
            row = [cell(idx.get((sc, c, "gpu_local", kmh))) for c in ("v2", "v1", "aeb_off")]
            out.append(f"| {sc} | " + " | ".join(row) + " |")
    out += ["", "### Totals (gpu_local, all speeds)", "",
            "| Controller | Hazard runs | Collisions | No-brake-needed runs | Runs with unneeded braking | Stopped and resumed |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for c in ("v2", "v1", "aeb_off"):
        rs = [r for r in runs if r["controller"] == c and r["condition"] == "gpu_local"]
        hz = [r for r in rs if r["brake_needed"]]
        sf = [r for r in rs if not r["brake_needed"]]
        out.append(f"| {c} | {len(hz)} | {sum(r['collision'] for r in hz)} | {len(sf)} | "
                   f"{sum(r['unneeded_brake'] for r in sf)} | {sum(r['resumed_after_stop'] for r in rs)} |")  # fmt: skip
    conds = sorted({r["condition"] for r in runs if r["condition"] != "gpu_local"})
    if conds:
        out += ["", "## B. Latency / runtime stress (30 km/h)", "",
                "| Scenario | Condition | V2 | V1 | V2 frame age mean / p95 (ms) | V2 stale rejected | V2 dropped frames | V2 degraded ticks |",
                "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        order = ["fixed_0ms", "fixed_20ms", "fixed_50ms", "fixed_100ms", "fixed_150ms", "fixed_200ms", "jitter_50pm40ms",
                 "burst_25ms_250ms", "drop30_25ms", "stale20_25ms", "npu_colocated", "npu_remote_v2", "npu_remote_jpeg95",
                 "npu_remote_v1equiv"]  # fmt: skip
        for sc in ("pedestrian_crossing", "lead_vehicle_braking", "static_obstacle"):
            for cond in order:
                v2 = idx.get((sc, "v2", cond, 30))
                v1r = idx.get((sc, "v1", cond, 30))
                if v2 is None and v1r is None:
                    continue
                p = (v2 or {}).get("pipeline", {})
                age = (f"{1e3 * p['age_mean_s']:.0f} / {1e3 * p['age_p95_s']:.0f}"
                       if p.get("age_mean_s") is not None else "—")  # fmt: skip
                g = p.get("gate", {})
                out.append(f"| {sc} | {cond} | {cell(v2)} | {cell(v1r)} | {age} | {g.get('n_stale', '—')} | "
                           f"{p.get('dropped_frames', '—')} | {(v2 or {}).get('degraded_ticks', '—')} |")  # fmt: skip
    trk = [r for r in runs if r["controller"] == "v2" and r["tracking"]["cam_frames_gt_visible"]]
    if trk:
        out += ["", "## C. Tracking and estimation quality (V2 runs, GT target visible)", "",
                "| Scenario | runs | covered by a confirmed track (median) | distinct track ids for the target (median / max) | "
                "longest uncovered gap, frames (median) | range MAE m (median) | rate MAE m/s (median) | rate p95 m/s (median) |",
                "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        by = defaultdict(list)
        for r in trk:
            by[r["scenario"]].append(r["tracking"])
        for sc in HAZ_ORDER + SAFE_ORDER:
            ts = by.get(sc)
            if not ts:
                continue

            def med(k, ts=ts):
                v = [t[k] for t in ts if t[k] is not None]
                return statistics.median(v) if v else None

            out.append(f"| {sc} | {len(ts)} | {f(med('covered_frac'))} | {f(med('distinct_ids'), 0)} / "
                       f"{max(t['distinct_ids'] for t in ts)} | {f(med('longest_gap_frames'), 0)} | {f(med('range_mae_m'))} | "
                       f"{f(med('rate_mae_mps'))} | {f(med('rate_p95_abs_mps'))} |")  # fmt: skip
    path = resolve("reports/v2_scenario_tables.md")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(path, len(runs), "runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
