# ruff: noqa: E501  (Markdown table rows are kept on one line)
"""V3 closed-loop analysis from evidence only.

Inputs: reports/evidence/v3_closed_loop.json and the per-run logs in outputs/v3/scenario_logs
(per-tick rows, used for the AEB/LKA interaction metrics).
Outputs: reports/v3_closed_loop_tables.md, reports/evidence/v3_closed_loop_summary.json,
assets/plots/v3_*.png.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_plots  # noqa: E402
from make_plots import GRID, INK2, SURF, C, _save, _style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
make_plots.OUT = ROOT / "assets" / "plots"
EV = ROOT / "reports" / "evidence"
LOGS = ROOT / "outputs" / "v3" / "scenario_logs"
MODES = ["A", "B", "C", "D"]
MODE_LABEL = {"A": "A GPU det + GPU lane", "B": "B RNGD det + GPU lane", "C": "C GPU det + RNGD lane",
              "D": "D RNGD det + RNGD lane"}  # fmt: skip
LK = ["lk_straight", "lk_mild", "lk_strong", "lk_strong_heldout", "lk_offset", "lk_night", "lk_rain"]
AEB = ["lead_vehicle_braking", "static_obstacle", "pedestrian_crossing", "pedestrian_clearing", "cyclist_crossing"]
COMB = ["curve_obstacle", "curve_lead_brake", "ped_cornering", "degraded_lane"]


def f(v, spec=".2f", unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:{spec}}{unit}"


def stat(r, key, q="p95"):
    s = r.get(key)
    return s.get(q) if s else None


def stem(r):
    mode = r["mode"].replace("+", "_").replace(" ", "_")
    fault = r.get("fault_arg") or ""
    return f"{r['scenario']}__{mode}__{int(r['ego_kmh'])}kmh__{r['scheduler']}{'__radar' if r['radar'] else ''}{'__' + fault if fault else ''}"


def log_rows(r):
    p = LOGS / f"{stem(r)}.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def braking_metrics(rows):
    """Lateral behaviour while the AEB is braking (AEB/LKA interaction)."""
    if not rows:
        return {}
    br = [x for x in rows if x["brake"] > 0]
    if not br:
        return {"brake_ticks": 0}
    dep = [abs(x["gt_offset"]) > (x["lane_width"] / 2 - 0.95) for x in br]
    return {
        "brake_ticks": len(br),
        "lat_max_during_brake_m": float(max(abs(x["gt_offset"]) for x in br)),
        "steer_max_during_brake": float(max(abs(x["steer"]) for x in br)),
        "departure_during_brake": bool(any(dep)),
        "arb_modes_during_brake": sorted({x["mode"] for x in br}),
    }


RERUN: dict = {}


def transport_degraded(r) -> bool:
    """A non-fault run in which the backend requests themselves failed (timeouts), i.e. not a perception result."""
    return not r.get("fault_arg") and (r["det_gate"]["n_failed"] + r["lane_gate"]["n_failed"]) >= 3


def load_rerun() -> dict:
    p = EV / "v3_closed_loop_rerun_nightrain.json"
    if not p.exists():
        return {}
    return {(r["scenario"], r["mode"], r["ego_kmh"]): r for r in json.loads(p.read_text(encoding="utf-8"))["runs"]}


def base(runs):
    """Non-fault, fixed-scheduler, camera-only runs keyed by (scenario, mode, speed)."""
    return {(r["scenario"], r["mode"], r["ego_kmh"]): r for r in runs
            if not r.get("fault_arg") and r["scheduler"] == "fixed" and not r["radar"]}  # fmt: skip


def lane_table(runs, out):
    b = base(runs)
    out.append("## Lane keeping (camera-only LKA; ground truth for evaluation only)\n")
    out.append(
        "| Scenario | km/h | Mode | Lateral RMSE | Max lateral | Lane departures | Heading MAE | Steer rate RMS | Lane-estimate offset MAE | LKA unavailable | Lane age p95 |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in LK:
        for kmh in (30.0, 50.0):
            for m in MODES:
                r0 = b.get((sc, m, kmh))
                deg = r0 is not None and transport_degraded(r0)
                rr = RERUN.get((sc, m, kmh)) if deg else None
                for r, lab in (
                    (r0, f"{m} **transport degraded**" if deg else m),
                    (rr, f"{m} (rerun, healthy session)"),
                ):
                    if r is None:
                        continue
                    out.append(
                        f"| {sc} | {kmh:.0f} | {lab} | {f(r['lat_rmse_m'], '.3f', ' m')} | {f(r['lat_max_m'], '.3f', ' m')} | {r['lane_departures']} | {f(r['heading_mae_deg'], '.2f', '°')} | {f(r['steer_rate_rms'], '.2f', ' /s')} | {f(r['lane_est_offset_mae_m'], '.3f', ' m')} | {f(100 * r['lka_unavailable_frac'], '.1f', ' %')} | {f(stat(r, 'lane_age_ms'), '.0f', ' ms')} |"
                    )
    out.append(
        "\n**Transport degraded** = the backend requests themselves timed out (>= 3 failed requests in a run without "
        "injected faults). These runs were the last ones before the system stopped CARLA for low memory; the same "
        "configurations re-run in a healthy session are listed under each "
        "(`reports/evidence/v3_closed_loop_rerun_nightrain.json`).\n"
    )


def aeb_table(runs, out, scen, title):
    b = base(runs)
    out.append(f"## {title}\n")
    out.append(
        "| Scenario | km/h | Mode | Collision | Min gap | First brake | GT TTC at first brake | Unneeded brake | Lateral max | Max lateral while braking | Departure while braking | Det age p95 | Lane age p95 |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in scen:
        for kmh in (30.0, 40.0, 50.0):
            for m in MODES:
                r = b.get((sc, m, kmh))
                if not r:
                    continue
                bm = braking_metrics(log_rows(r))
                out.append(
                    f"| {sc} | {kmh:.0f} | {m} | {f(r['collision'])} | {f(r['min_gap_m'], '.2f', ' m')} | {f(r['first_brake_t'], '.2f', ' s')} | {f(r['gt_ttc_at_first_brake_s'], '.2f', ' s')} | {f(r['unneeded_brake'])} | {f(r['lat_max_m'], '.3f', ' m')} | {f(bm.get('lat_max_during_brake_m'), '.3f', ' m')} | {f(bm.get('departure_during_brake'))} | {f(stat(r, 'det_age_ms'), '.0f', ' ms')} | {f(stat(r, 'lane_age_ms'), '.0f', ' ms')} |"
                )
    out.append("")


def radar_table(runs, out):
    cam = {
        (r["scenario"], r["mode"], r["ego_kmh"]): r
        for r in runs
        if not r.get("fault_arg") and r["scheduler"] == "fixed" and not r["radar"]
    }
    rad = {
        (r["scenario"], r["mode"], r["ego_kmh"]): r
        for r in runs
        if not r.get("fault_arg") and r["scheduler"] == "fixed" and r["radar"]
    }
    out.append("## Camera-only vs camera + radar (primary track vs ground truth, before contact)\n")
    out.append(
        "| Scenario | km/h | Mode | Sensors | Range MAE | Closing-speed MAE | TTC error (mean / median, GT TTC ≤ 5 s) | Radar-fused samples | Collision | Min gap | First brake | Unneeded brake |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    pairs = []
    for k in sorted(rad):
        if k not in cam:
            continue
        for lab, r in (("camera", cam[k]), ("camera + radar", rad[k])):
            out.append(
                f"| {k[0]} | {k[2]:.0f} | {k[1]} | {lab} | {f(r['range_mae_m'], '.2f', ' m')} | {f(r['closing_mae_mps'], '.2f', ' m/s')} | {f(r['ttc_mae_s'], '.2f')} / {f(r['ttc_median_err_s'], '.2f', ' s')} | {f(100 * r['radar_fused_frac'] if r['radar_fused_frac'] is not None else None, '.0f', ' %')} | {f(r['collision'])} | {f(r['min_gap_m'], '.2f', ' m')} | {f(r['first_brake_t'], '.2f', ' s')} | {f(r['unneeded_brake'])} |"
            )
        pairs.append((k, cam[k], rad[k]))
    out.append("")
    return pairs


def latency_table(runs, out):
    out.append("## Backend configurations: latency and freshness (all non-fault, camera-only, fixed-scheduler runs)\n")
    out.append(
        "| Mode | Runs | Det latency mean / p95 | Lane latency mean / p95 | Det queue mean | Det age p95 / p99 / max | Lane age p95 / p99 / max | Dropped frames per run (det / lane) | Control stack p95 | Collisions (hazard runs) | Lane departures |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    agg = {}
    for m in MODES:
        rs = [
            r
            for r in runs
            if r["mode"] == m
            and not r.get("fault_arg")
            and r["scheduler"] == "fixed"
            and not r["radar"]
            and not transport_degraded(r)
        ]
        if not rs:
            continue

        def mean(key, q, rs=rs):
            v = [stat(r, key, q) for r in rs if stat(r, key, q) is not None]
            return float(np.mean(v)) if v else None

        def mx(key, q, rs=rs):
            v = [stat(r, key, q) for r in rs if stat(r, key, q) is not None]
            return float(np.max(v)) if v else None

        drop = [r["dropped_frames"] for r in rs]
        if m == "D":
            dd = f"{np.mean([d['both'] for d in drop]):.0f} (one shared request)"
        else:
            dd = f"{np.mean([d['det'] for d in drop]):.0f} / {np.mean([d['lane'] for d in drop]):.0f}"
        haz = [r for r in rs if r["min_gap_m"] is not None or r["kind"] != "none"]
        a = {"runs": len(rs), "det_lat_mean": mean("det_latency_ms", "mean"), "det_lat_p95": mean("det_latency_ms", "p95"),
             "lane_lat_mean": mean("lane_latency_ms", "mean"), "lane_lat_p95": mean("lane_latency_ms", "p95"),
             "det_queue_mean": mean("det_queue_ms", "mean"), "det_age_p95": mean("det_age_ms", "p95"), "det_age_p99": mean("det_age_ms", "p99"),
             "det_age_max": mx("det_age_ms", "max"), "lane_age_p95": mean("lane_age_ms", "p95"), "lane_age_p99": mean("lane_age_ms", "p99"),
             "lane_age_max": mx("lane_age_ms", "max"), "ctrl_p95": mean("control_stack_ms", "p95"),
             "collisions": sum(r["collision"] for r in haz), "hazard_runs": len(haz), "departures": sum(r["lane_departures"] for r in rs)}  # fmt: skip
        agg[m] = a
        out.append(
            f"| {MODE_LABEL[m]} | {a['runs']} | {f(a['det_lat_mean'], '.0f')} / {f(a['det_lat_p95'], '.0f', ' ms')} | {f(a['lane_lat_mean'], '.0f')} / {f(a['lane_lat_p95'], '.0f', ' ms')} | {f(a['det_queue_mean'], '.0f', ' ms')} | {f(a['det_age_p95'], '.0f')} / {f(a['det_age_p99'], '.0f')} / {f(a['det_age_max'], '.0f', ' ms')} | {f(a['lane_age_p95'], '.0f')} / {f(a['lane_age_p99'], '.0f')} / {f(a['lane_age_max'], '.0f', ' ms')} | {dd} | {f(a['ctrl_p95'], '.1f', ' ms')} | {a['collisions']} / {a['hazard_runs']} | {a['departures']} |"
        )
    out.append(
        "\nTransport-degraded runs are excluded here (see the lane-keeping table). Ages and latencies are per-run statistics averaged over runs (p95 / p99), the maximum is over all runs. Latency is the measured backend call (GPU in-process, or RNGD round trip incl. transport); frame age = control time − capture time of the result in use.\n"
    )
    return agg


def sched_table(runs, out):
    rs = [r for r in runs if r["mode"] == "D" and not r.get("fault_arg") and not r["radar"]]
    keys = sorted({(r["scenario"], r["ego_kmh"]) for r in rs if r["scheduler"] != "fixed"})
    out.append("## Scheduling in the loop (Mode D, both models on RNGD)\n")
    out.append(
        "| Scenario | km/h | Strategy | Requests (both / det / lane) | Det age p95 / p99 | Lane age p95 / p99 | Det latency p95 | Queue delay mean | Dropped frames | Collision | Min gap | Lateral RMSE | LKA unavailable |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    names = {
        "every": "both models every frame, separate PEs",
        "every_samepe": "both every frame, same PE",
        "fixed": "rate-decoupled (lane due every 2nd frame, coalesced)",
        "adaptive": "adaptive priority",
    }
    for sc, kmh in keys:
        for sched in ("every", "every_samepe", "fixed", "adaptive"):
            r = next((x for x in rs if x["scenario"] == sc and x["ego_kmh"] == kmh and x["scheduler"] == sched), None)
            if not r:
                continue
            ops = r.get("mode_d_ops") or {}
            out.append(
                f"| {sc} | {kmh:.0f} | {names[sched]} | {ops.get('infer_both', 'n/a')} / {ops.get('infer_det', 'n/a')} / {ops.get('infer_lane', 'n/a')} | {f(stat(r, 'det_age_ms'), '.0f')} / {f(stat(r, 'det_age_ms', 'p99'), '.0f', ' ms')} | {f(stat(r, 'lane_age_ms'), '.0f')} / {f(stat(r, 'lane_age_ms', 'p99'), '.0f', ' ms')} | {f(stat(r, 'det_latency_ms'), '.0f', ' ms')} | {f(stat(r, 'det_queue_ms', 'mean'), '.0f', ' ms')} | {r['dropped_frames'].get('both')} | {f(r['collision'])} | {f(r['min_gap_m'], '.2f', ' m')} | {f(r['lat_rmse_m'], '.3f', ' m')} | {f(100 * r['lka_unavailable_frac'], '.1f', ' %')} |"
            )
    out.append("")


def lane_age_table(runs, out):
    """Mode A (GPU lane) with injected extra lane latency: how old can lane perception get before LKA fails."""
    out.append("## Lane frame age vs lane keeping (Mode A, GPU lane + injected lane latency; stale gate 0.3 s)\n")
    out.append(
        "| Scenario | Extra lane latency | Lane latency mean | Lane age p95 / max | Lane results used | Stale rejected | Lateral RMSE | Max lateral | Departures | Steer rate RMS | LKA unavailable | Unneeded brake |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for sc in ("lk_strong", "lk_mild", "lk_night"):
        for d in (0, 20, 50, 100, 150, 250):
            f_ = f"lane_delay{d}" if d else ""
            r = next((x for x in runs if x["scenario"] == sc and x["mode"] == "A" and x["ego_kmh"] == 30.0 and not x["radar"]
                      and x["scheduler"] == "fixed" and (x.get("fault_arg") or "") == f_), None)  # fmt: skip
            if not r:
                continue
            g = r["lane_gate"]
            rows.append({"scenario": sc, "extra_ms": d, "lane_age_p95": stat(r, "lane_age_ms"), "rmse": r["lat_rmse_m"],
                         "lka_unavailable": r["lka_unavailable_frac"], "departures": r["lane_departures"]})  # fmt: skip
            out.append(
                f"| {sc} | {d} ms | {f(stat(r, 'lane_latency_ms', 'mean'), '.0f', ' ms')} | {f(stat(r, 'lane_age_ms'), '.0f')} / {f(stat(r, 'lane_age_ms', 'max'), '.0f', ' ms')} | {g['n_new']} | {g['n_stale']} | {f(r['lat_rmse_m'], '.3f', ' m')} | {f(r['lat_max_m'], '.3f', ' m')} | {r['lane_departures']} | {f(r['steer_rate_rms'], '.2f', ' /s')} | {f(100 * r['lka_unavailable_frac'], '.1f', ' %')} | {f(r['unneeded_brake'])} |"
            )
    out.append("")
    return rows


def fault_table(runs, out):
    rs = [r for r in runs if r.get("fault_arg") and not r["fault_arg"].startswith("lane_delay")]
    out.append("## Fault injection in the loop\n")
    out.append(
        "| Scenario | Fault | Mode | Collision | Min gap | Lateral max | LKA unavailable | Det gate (new / stale / failed) | Det age max | Lane age max | Backend switches (logged) | Recovered to RNGD |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rs, key=lambda x: (x["fault_arg"], x["scenario"], x["mode"])):
        g = r["det_gate"]
        ev = r.get("failover_events") or []
        sw = ", ".join(f"{e['from']}→{e['to']}" for e in ev) or "none"
        rec = "yes" if any(e["to"] == "PRIMARY" for e in ev) else ("n/a" if not ev else "no")
        out.append(
            f"| {r['scenario']} | {r['fault_arg']} | {r['mode']} | {f(r['collision'])} | {f(r['min_gap_m'], '.2f', ' m')} | {f(r['lat_max_m'], '.3f', ' m')} | {f(100 * r['lka_unavailable_frac'], '.1f', ' %')} | {g['n_new']} / {g['n_stale']} / {g['n_failed']} | {f(stat(r, 'det_age_ms', 'max'), '.0f', ' ms')} | {f(stat(r, 'lane_age_ms', 'max'), '.0f', ' ms')} | {sw} | {rec} |"
        )
    out.append("")


def plots(runs, agg, pairs):
    _style()
    b = base(runs)
    # 1 lane keeping RMSE per scenario and mode
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.2
    for j, m in enumerate(MODES):
        ys = []
        for sc in LK:
            r = b.get((sc, m, 30.0))
            if r is not None and transport_degraded(r):
                r = RERUN.get((sc, m, 30.0))  # healthy-session rerun; the original is in v3_nightrain_transport.png
            ys.append(r["lat_rmse_m"] * 100 if r else np.nan)
        ax.bar(np.arange(len(LK)) + (j - 1.5) * w, ys, w, color=C[j], edgecolor=SURF, linewidth=2, label=MODE_LABEL[m])
    ax.set_xticks(range(len(LK)), [s.replace("lk_", "") for s in LK], fontsize=9)
    ax.set_ylabel("lateral RMSE vs map lane centre (cm)")
    ax.set_title("Camera-only lane keeping at 30 km/h, by backend configuration")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(axis="y", color=GRID)
    ax.text(0.99, 0.97, "night / rain, modes C and D: healthy-session reruns;\nthe original runs failed on request timeouts\n(see v3_nightrain_transport.png)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)  # fmt: skip
    _save(fig, "v3_lane_keeping_rmse.png")
    # 1b night/rain: original (transport degraded) vs healthy-session rerun
    keys = [k for k in sorted(b) if transport_degraded(b[k]) and k in RERUN]
    if keys:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        x = np.arange(len(keys))
        for j, (lab, src) in enumerate((("original run (request timeouts)", b), ("rerun, healthy session", RERUN))):
            kw = {"color": C[j + 1], "edgecolor": SURF, "linewidth": 2}
            axs[0].bar(x + (j - 0.5) * 0.36, [src[k]["lat_rmse_m"] * 100 for k in keys], 0.36, label=lab, **kw)
            fails = [src[k]["det_gate"]["n_failed"] + src[k]["lane_gate"]["n_failed"] for k in keys]
            axs[1].bar(x + (j - 0.5) * 0.36, fails, 0.36, **kw)
        axs[0].set_ylabel("lateral RMSE (cm)")
        axs[1].set_ylabel("failed backend requests")
        for a_ in axs:
            a_.set_xticks(x, [f"{k[0].replace('lk_', '')} {k[1]}" for k in keys], fontsize=9)
            a_.grid(axis="y", color=GRID)
        axs[0].legend(fontsize=8)
        fig.suptitle(
            "RNGD lane modes at night / in rain: the failure followed request timeouts, not the weather", fontsize=10
        )
        _save(fig, "v3_nightrain_transport.png")
    # 2 frame age by mode
    if agg:
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        ms = [m for m in MODES if m in agg]
        for j, (key, lab) in enumerate([("det_age_p95", "detection age p95"), ("lane_age_p95", "lane age p95")]):
            ax.barh(
                np.arange(len(ms)) + (j - 0.5) * 0.36,
                [agg[m][key] for m in ms],
                0.36,
                color=C[j],
                edgecolor=SURF,
                linewidth=2,
                label=lab,
            )
        for i, m in enumerate(ms):
            ax.text(
                max(agg[m]["det_age_p95"] or 0, agg[m]["lane_age_p95"] or 0) + 2,
                i,
                f"latency mean det {agg[m]['det_lat_mean']:.0f} / lane {agg[m]['lane_lat_mean']:.0f} ms",
                va="center",
                fontsize=8,
                color=INK2,
            )
        ax.set_yticks(range(len(ms)), [MODE_LABEL[m] for m in ms], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("frame age at control (ms), mean of per-run p95")
        ax.set_xlim(0, 220)
        ax.set_title("Perception freshness in the closed loop")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="x", color=GRID)
        _save(fig, "v3_frame_age_by_mode.png")
    # 3 radar closing-speed error
    if pairs:
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        labs = [f"{k[0].replace('_', ' ')} {k[2]:.0f}" for k, _, _ in pairs]
        for j, (lab, idx) in enumerate([("camera only", 1), ("camera + radar", 2)]):
            xs = [p[idx]["closing_mae_mps"] if p[idx]["closing_mae_mps"] is not None else np.nan for p in pairs]
            ax.scatter(
                xs,
                np.arange(len(pairs)) + (j - 0.5) * 0.2,
                s=60,
                color=C[j],
                edgecolor=SURF,
                linewidth=1.5,
                label=lab,
                zorder=3,
            )
        ax.set_yticks(range(len(pairs)), labs, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("closing-speed MAE of the primary track vs ground truth (m/s)")
        ax.set_title("Radar fusion stage 1: closing-speed error")
        ax.legend(fontsize=8)
        ax.grid(axis="x", color=GRID)
        _save(fig, "v3_radar_closing_error.png")
    # 4 combined scenario time series
    r = b.get(("curve_lead_brake", "D", 30.0)) or b.get(("curve_lead_brake", "A", 30.0))
    rows = log_rows(r) if r else None
    if rows:
        t = np.array([x["t"] for x in rows])
        fig, axs = plt.subplots(4, 1, figsize=(9, 6.4), sharex=True)
        axs[0].plot(t, [x["v"] * 3.6 for x in rows], color=C[0], lw=2)
        axs[0].set_ylabel("speed\n(km/h)")
        axs[1].plot(t, [x["gt_gap"] if x["gt_gap"] is not None else np.nan for x in rows], color=C[1], lw=2)
        axs[1].set_ylabel("gap to lead\n(m, GT)")
        axs[2].plot(t, [x["gt_offset"] * 100 for x in rows], color=C[2], lw=2)
        axs[2].set_ylabel("lateral error\n(cm, GT)")
        axs[3].plot(t, [x["steer"] for x in rows], color=C[3], lw=2, label="steer command")
        axs[3].plot(t, [x["brake"] for x in rows], color=C[4], lw=2, label="brake")
        axs[3].set_ylabel("command")
        axs[3].legend(fontsize=8, loc="upper right")
        axs[3].set_xlabel("time (s)")
        for a in axs:
            a.grid(color=GRID)
        fig.suptitle(
            f"curve + lead vehicle braking, Mode {r['mode']}: AEB brakes while LKA keeps the lane", fontsize=10
        )
        _save(fig, "v3_combined_timeseries.png")


def main() -> int:
    RERUN.update(load_rerun())
    runs = json.loads((EV / "v3_closed_loop.json").read_text(encoding="utf-8"))["runs"]
    out = ["# V3 closed-loop tables (generated by `scripts/v3_analyze.py` from `reports/evidence/v3_closed_loop.json`)\n",
           f"Runs: **{len(runs)}**. Code commits: {', '.join(sorted({r['code_commit'] for r in runs}))}. CARLA 0.9.16, sync 100 Hz, camera 33 Hz, time-dilated hardware-in-the-loop (measured backend latency applied in simulation time).\n"]  # fmt: skip
    lane_table(runs, out)
    aeb_table(runs, out, AEB, "AEB scenarios (straight road, LKA steering)")
    aeb_table(runs, out, COMB, "Combined lateral + longitudinal scenarios (curves)")
    pairs = radar_table(runs, out)
    agg = latency_table(runs, out)
    sched_table(runs, out)
    age_rows = lane_age_table(runs, out)
    fault_table(runs, out)
    (ROOT / "reports" / "v3_closed_loop_tables.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    hazards = [r for r in runs if r["kind"] != "none"]
    summary = {
        "runs": len(runs),
        "collisions": sum(r["collision"] for r in runs),
        "hazard_runs": len(hazards),
        "hazard_collisions": [(r["scenario"], r["mode"], r["ego_kmh"], r["radar"], r["scheduler"], r.get("fault_arg")) for r in hazards if r["collision"]],
        "lane_departures_total": sum(r["lane_departures"] for r in runs),
        "transport_degraded_runs": [(r["scenario"], r["mode"]) for r in runs if transport_degraded(r)],
        "rerun_nightrain": {f"{k[0]} {k[1]}": {"lat_rmse_m": v["lat_rmse_m"], "lane_departures": v["lane_departures"], "collision": v["collision"]} for k, v in RERUN.items()},
        "runs_with_departure": [(r["scenario"], r["mode"], r["ego_kmh"], r.get("fault_arg")) for r in runs if r["lane_departures"]],
        "unneeded_brakes": [(r["scenario"], r["mode"], r["ego_kmh"], r["radar"]) for r in runs if r["unneeded_brake"]],
        "by_mode": agg,
    }  # fmt: skip
    by_cat = defaultdict(int)
    for r in runs:
        fa = r.get("fault_arg") or ""
        if fa:
            cat = "lane_age" if fa.startswith("lane_delay") else "fault"
        else:
            cat = "sched" if r["scheduler"] != "fixed" else ("radar" if r["radar"] else r["kind"])
        by_cat[cat] += 1
    summary["collisions_any"] = [(r["scenario"], r["mode"], r["collision_with"]) for r in runs if r["collision"]]
    summary["by_category"] = dict(by_cat)
    summary["lane_age"] = age_rows
    (EV / "v3_closed_loop_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    plots(runs, agg, pairs)
    print(json.dumps({k: summary[k] for k in ("runs", "collisions", "hazard_runs", "lane_departures_total")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
