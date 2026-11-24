"""V2 plots from evidence files only (assets/plots/v2_*.png). Palette and style from scripts/make_plots.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plots import C, GRID, INK2, SURF, _load, _save, _style  # noqa: E402

STAGES = [("network", C[0]), ("pre", C[1]), ("exec", C[2]), ("d2h", C[3]), ("post", C[4])]


def transport() -> None:
    r1, r2 = _load("v2_transport_profile.json"), _load("v2_transport_profile_r2.json")
    rows = [
        ("V1-equivalent\n(letterbox raw,\ndefault threads)", r1["variants"]["v1_equiv_letterbox_raw"]),
        ("content only", r1["variants"]["content_raw"]),
        ("+ 2 threads\n+ pre-filter", r1["variants"]["content_raw_t2_prefilter"]),
        ("+ SSH -C\n(V2 lossless)", r2["variants"]["content_raw_t2_prefilter_sshC"]),
        ("JPEG q95\n(lossy)", r2["variants"]["jpeg95_t2_prefilter"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (_lab, v) in enumerate(rows):
        s = v["stages"]
        net = s["network_est"]["p50_ms"]
        vals = {"network": net, "pre": s.get("pre", {}).get("p50_ms", 0), "exec": s["exec"]["p50_ms"],
                "d2h": s.get("d2h", {}).get("p50_ms", 0), "post": s.get("post", {}).get("p50_ms", 0)}  # fmt: skip
        left = 0.0
        for name, col in STAGES:
            ax.barh(i, vals[name], left=left, color=col, edgecolor=SURF, linewidth=2, height=0.6,
                    label=name if i == 0 else None)  # fmt: skip
            left += vals[name]
        rec = v["agreement_vs_gpu"]["recall_vs_ref"]
        ax.text(left + 2, i, f"e2e p50 {s['e2e']['p50_ms']:.0f} ms · recall {rec:.3f}", va="center", fontsize=9,
                color=INK2)  # fmt: skip
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("median stage time (ms); network = RTT − server time − send")
    ax.set_xlim(0, 260)
    ax.set_title("Remote RNGD BF16: where the time goes (n = 72 / 216 per variant)")
    ax.legend(ncol=5, loc="lower right", fontsize=8)
    _save(fig, "v2_transport_breakdown.png")


def filters() -> None:
    d = _load("v2_filter_eval.json")["results"]["33Hz"]
    names = [("fd", "finite diff."), ("lsq_v1", "V1 LSQ 0.3 s"), ("ab_0.3_0.05", "α-β 0.3/0.05"),
             ("kf_sa8", "KF σa 8"), ("kf_sa15_px075_gate25", "KF V2 default")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, (key, lab, col) in enumerate([("rate_mae_mps", "all ranges", C[0]),
                                         ("rate_mae_far_mps", "beyond 30 m", C[1])]):  # fmt: skip
        xs = [d[k][key] for k, _ in names]
        ax.scatter(xs, [i + (j - 0.5) * 0.18 for i in range(len(names))], s=70, color=col, label=lab, zorder=3,
                   edgecolor=SURF, linewidth=1.5)  # fmt: skip
        for i, x in enumerate(xs):
            ax.annotate(f"{x:.2f}", (x, i + (j - 0.5) * 0.18), textcoords="offset points", xytext=(7, 0),
                        va="center", fontsize=8, color=INK2)  # fmt: skip
    ax.set_yticks(range(len(names)), [n for _, n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 12.5)
    ax.set_xlabel("range-rate MAE vs ground truth (m/s) — 39 v3 logs resampled to 33 Hz")
    ax.set_title("Range-rate estimators (lower is better)")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "v2_filter_comparison.png")


def multipe() -> None:
    a, b = _load("v2_npu_multistream.json")["configs"], _load("v2_npu_multistream2.json")["configs"]
    rows = [("1 PE", a["pe1_t1"]), ("2 threads\n1 PE", a["pe1_t2"]), ("2 PEs", b["pe2_t2"]), ("4 PEs", b["pe4_t4"]),
            ("8 PEs", b["pe8_t8"])]  # fmt: skip
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    labs = [r[0] for r in rows]
    fps = [r[1]["throughput_fps"] for r in rows]
    ax1.bar(labs, fps, color=C[0], edgecolor=SURF, linewidth=2, width=0.6)
    for i, v in enumerate(fps):
        ax1.text(i, v + 20, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
    ax1.set_ylabel("YOLO26n BF16 inferences / s")
    ax1.set_title("Throughput")
    p50 = [r[1]["lat_ms_p50"] for r in rows]
    p95 = [r[1]["lat_ms_p95"] for r in rows]
    ax2.scatter(labs, p50, color=C[0], s=60, label="p50", zorder=3)
    ax2.scatter(labs, p95, color=C[1], s=60, label="p95", zorder=3)
    ax2.set_ylabel("per-call latency (ms)")
    ax2.set_title("Latency per call")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("RNGD multi-PE scaling (8 s per configuration, health check after each)", fontsize=11)
    _save(fig, "v2_multipe_scaling.png")


def resolution() -> None:
    d = _load("v2_resolution_npu.json")
    if d is None:
        return
    sz = sorted(d["sizes"], key=int)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rec = [d["sizes"][s]["npu_bf16"]["agreement_vs_640"]["recall_vs_ref"] for s in sz]
    exe = [d["sizes"][s]["npu_bf16"]["exec_ms"]["p50_ms"] for s in sz]
    e2e = [d["sizes"][s]["npu_bf16"]["e2e_ms"]["mean_ms"] for s in sz]
    ax.plot([int(s) for s in sz], e2e, marker="o", color=C[0], label="remote e2e mean (ms)")
    ax.plot([int(s) for s in sz], exe, marker="o", color=C[2], label="NPU exec p50 (ms)")
    ax.set_xlabel("input resolution (px)")
    ax.set_ylabel("ms")
    ax2 = ax.twinx() if False else None  # one axis only: recall printed as labels
    for x, y, r in zip([int(s) for s in sz], e2e, rec, strict=True):
        ax.annotate(f"recall {r:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5,
                    color=INK2)  # fmt: skip
    del ax2
    ax.set_xticks([int(s) for s in sz])
    ax.set_title("Resolution on RNGD BF16: compute flat, transport ∝ pixels, agreement drops")
    ax.legend(fontsize=8)
    _save(fig, "v2_resolution.png")


def matrix() -> None:
    d = _load("v2_scenarios.json")
    if d is None:
        return
    runs = d["runs"]
    order = ["fixed_0ms", "fixed_20ms", "fixed_50ms", "fixed_100ms", "fixed_150ms", "fixed_200ms", "npu_colocated",
             "npu_remote_jpeg95", "npu_remote_v2", "npu_remote_v1equiv"]  # fmt: skip
    for sc in ("pedestrian_crossing", "lead_vehicle_braking"):
        fig, ax = plt.subplots(figsize=(9, 3.8))
        for j, (ctrl, col) in enumerate([("v1", C[1]), ("v2", C[0])]):
            xs, ys, coll = [], [], []
            for i, cond in enumerate(order):
                r = next((r for r in runs if r["scenario"] == sc and r["controller"] == ctrl and r["condition"] == cond
                          and int(r["ego_kmh"]) == 30), None)  # fmt: skip
                if r is None:
                    continue
                xs.append(i + (j - 0.5) * 0.2)
                ys.append(0.0 if r["collision"] else r["min_gap_m"])
                coll.append(r["collision"])
            ax.scatter(xs, ys, color=col, s=60, label=ctrl.upper(), zorder=3, edgecolor=SURF, linewidth=1.5)
            for x, y, c in zip(xs, ys, coll, strict=True):
                if c:
                    ax.annotate("collision", (x, y), textcoords="offset points", xytext=(0, 6), ha="center",
                                fontsize=7.5, color=col)  # fmt: skip
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xticks(range(len(order)), [o.replace("fixed_", "").replace("npu_", "npu\n") for o in order],
                      fontsize=7.5)  # fmt: skip
        ax.set_ylabel("min gap before stop (m)")
        ax.set_title(f"{sc}, 30 km/h: latency / runtime condition vs margin")
        ax.legend(fontsize=8)
        _save(fig, f"v2_stress_{sc}.png")


def main() -> int:
    _style()
    transport()
    filters()
    multipe()
    resolution()
    matrix()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
