"""Portfolio figures from recorded evidence only (reports/evidence/*.json, outputs/scenario_logs/*.jsonl).

Every figure is skipped (with a message) when its source file does not exist; nothing is
synthesised. Palette: validated categorical order (blue, orange, aqua, yellow; CVD/normal checks
pass, light-mode contrast WARN for aqua/yellow -> every bar carries a direct value label).

  python scripts/make_plots.py      # -> assets/plots/*.png
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from npu_ad.config import resolve  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EV = resolve("reports/evidence")
OUT = resolve("assets/plots")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURF,
            "axes.facecolor": SURF,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "font.size": 10,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2,
        }
    )


def _load(name: str) -> dict | None:
    p = EV / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def _bars(ax, labels, values, colors, fmt="{:.2f}"):
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor=SURF, linewidth=2)
    for b, v in zip(bars, values, strict=True):
        ax.annotate(
            fmt.format(v),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            color=INK,
            fontsize=9,
        )
    return bars


def latency_model_only() -> None:
    gpu, npu, npub = _load("gpu_benchmark_fp32_clean.json"), _load("npu_bench.json"), _load("npu_bench_bf16.json")
    ext, npuw = _load("gpu_benchmark_ext.json"), _load("npu_bench_w8.json")
    if not (gpu and npu):
        print("skip latency_model_only")
        return
    rows = [("GPU eager FP32", gpu["latency"]["model"]["mean_ms"], gpu["latency"]["model"]["p95_ms"], C[0])]
    if ext and "cuda_graph_b1" in ext:
        rows.append(("GPU CUDA Graph FP32", ext["cuda_graph_b1"]["mean_ms"], ext["cuda_graph_b1"]["p95_ms"], C[0]))
    for label, d in (("RNGD FP32", npu), ("RNGD BF16", npub), ("RNGD W8 (BF16 compute)", npuw)):
        if d:
            c = np.array(d["result"]["timings_s"]["compute"]) * 1000
            rows.append((label, float(c.mean()), float(np.percentile(c, 95)), C[1]))
    fig, ax = plt.subplots(figsize=(8, 4))
    _bars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows])
    ax.scatter([r[0] for r in rows], [r[2] for r in rows], marker="_", s=300, color=INK, zorder=3, label="p95")
    ax.set_ylabel("ms per 640×640 frame (batch 1)")
    ax.set_title("Model-only latency: RTX 3060 vs Furiosa RNGD (mean, p95 tick)")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", labelsize=8)
    _save(fig, "latency_model_only.png")


def latency_breakdown() -> None:
    gpu, npu, link = _load("gpu_benchmark_fp32_clean.json"), _load("npu_bench.json"), _load("npu_link.json")
    if not (gpu and npu):
        print("skip latency_breakdown")
        return
    g = gpu["latency"]
    t = {k: float(np.mean(v)) * 1000 for k, v in npu["result"]["timings_s"].items()}
    stages = ["host preprocess", "H2D", "model / compute", "D2H + host postprocess"]
    gpu_v = [g["preprocess"]["mean_ms"], g["h2d"]["mean_ms"], g["model"]["mean_ms"], g["postprocess"]["mean_ms"]]
    npu_v = [g["preprocess"]["mean_ms"], t["h2d"], t["compute"], t["d2h"] + g["postprocess"]["mean_ms"]]
    labels = ["GPU (local)", "RNGD (on-host)"]
    data = [gpu_v, npu_v]
    if link:
        labels.append("RNGD remote\n(+SSH frame RTT)")
        data.append(npu_v + [link["letterboxed_frame_1.23MB"]["mean_ms"]])
        stages = stages + ["SSH transfer of the frame"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (lab, vals) in enumerate(zip(labels, data, strict=True)):
        left = 0.0
        for j, v in enumerate(vals):
            ax.barh(
                lab,
                v,
                left=left,
                color=[C[0], C[2], C[1], C[3], "#8f8e88"][j],
                edgecolor=SURF,
                linewidth=2,
                label=stages[j] if i == len(labels) - 1 else None,
            )
            left += v
        ax.annotate(f"{left:.1f} ms", (left, i), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
    ax.set_xlabel("mean latency per frame (ms, symlog scale: linear below 30 ms)")
    ax.set_title("End-to-end latency breakdown (mean)")
    ax.invert_yaxis()
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    ax.set_xscale("symlog", linthresh=30)
    _save(fig, "latency_breakdown.png")


def detection_agreement() -> None:
    sets = [
        ("FP32", _load("gpu_vs_npu.json")),
        ("BF16", _load("gpu_vs_npu_bf16.json")),
        ("W8", _load("gpu_vs_npu_w8.json")),
    ]
    sets = [(n, d["detection_agreement_eval_split"]) for n, d in sets if d]
    if not sets:
        print("skip detection_agreement")
        return
    metrics = [("recall_vs_ref", "recall"), ("precision_vs_ref", "precision"), ("mean_iou_matched", "mean IoU")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(metrics))
    w = 0.8 / len(sets)
    for i, (name, d) in enumerate(sets):
        vals = [d[k] for k, _ in metrics]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, color=C[i], edgecolor=SURF, linewidth=2, label=f"RNGD {name}")
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.3f}",
                (b.get_x() + b.get_width() / 2, v),
                ha="center",
                va="bottom",
                xytext=(0, 2),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0.9, 1.01)
    ax.set_title("NPU detections vs GPU FP32 reference (54 held-out CARLA frames)")
    ax.legend(loc="lower right")
    _save(fig, "detection_agreement.png")


def _v3() -> list[dict]:
    d = _load("scenarios_v3.json")
    return d["runs"] if d else []


def aeb_response() -> None:
    logs = resolve("outputs/scenario_logs")
    files = [
        ("aeb_off", "pedestrian_crossing__aeb_off__v2__gt_oracle__30kmh.jsonl"),
        ("GT-assisted", "pedestrian_crossing__npu_p95__v2__gt_gated__30kmh.jsonl"),
        ("perception-driven", "pedestrian_crossing__npu_p95__v2__mono__30kmh.jsonl"),
    ]
    files = [(n, logs / f) for n, f in files if (logs / f).exists()]
    if not files:
        print("skip aeb_response")
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    for i, (name, p) in enumerate(files):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        t = [r["t_sim"] for r in rows]
        axes[0].plot(t, [r["ego_mps"] * 3.6 for r in rows], color=C[i], label=name)
        axes[1].plot(t, [r["gap_m"] if r["gap_m"] is not None else np.nan for r in rows], color=C[i])
        axes[2].plot(t, [r["applied_brake"] for r in rows], color=C[i])
    mono = next((p for n, p in files if n == "perception-driven"), None)
    if mono:
        rows = [json.loads(line) for line in mono.read_text().splitlines()]
        axes[1].scatter(
            [r["t_sim"] for r in rows if r["mono_gap_m"] is not None and r["mono_matches_gt"]],
            [r["mono_gap_m"] for r in rows if r["mono_gap_m"] is not None and r["mono_matches_gt"]],
            s=8,
            color=C[3],
            label="camera range estimate (mono)",
            zorder=3,
        )
        axes[1].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("ego speed (km/h)")
    axes[1].set_ylabel("gap to pedestrian (m)")
    axes[2].set_ylabel("applied brake")
    axes[2].set_xlabel("simulation time (s)")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].set_ylim(-1, 32)
    axes[0].set_title("Pedestrian crossing, 30 km/h, NPU latency (9.85 ms): braking response")
    _save(fig, "aeb_response_pedestrian.png")


def speed_sweep() -> None:
    runs = [r for r in _v3() if not r.get("tag") and r["condition"] in ("npu_p95", "aeb_off")]
    if not runs:
        print("skip speed_sweep")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, sc in zip(axes, ["static_obstacle", "lead_vehicle_braking", "pedestrian_crossing"], strict=True):
        for i, (mode, label) in enumerate([("gt_gated", "GT-assisted"), ("mono", "perception-driven")]):
            pts = sorted(
                (r["ego_kmh"], r["min_gap_m"], r["collision"])
                for r in runs
                if r["scenario"] == sc and r["mode"] == mode and r["condition"] == "npu_p95"
            )
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=7, color=C[i], label=label)
            for p in pts:
                if p[2]:
                    ax.annotate(
                        "collision", (p[0], p[1]), xytext=(4, -10), textcoords="offset points", fontsize=8, color=INK
                    )
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_title(sc.replace("_", " "))
        ax.set_xlabel("ego speed (km/h)")
    axes[0].set_ylabel("minimum gap (m)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Minimum separation vs initial speed (AEB on, NPU latency)", fontweight="bold")
    _save(fig, "speed_vs_min_gap.png")
    # braking distance vs speed
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (mode, label) in enumerate([("gt_gated", "GT-assisted"), ("mono", "perception-driven")]):
        pts = sorted(
            (r["ego_kmh"], r["braking_distance_m"])
            for r in runs
            if r["scenario"] == "static_obstacle"
            and r["mode"] == mode
            and r["condition"] == "npu_p95"
            and r["braking_distance_m"] is not None
        )
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=7, color=C[i], label=label)
    ax.set_xlabel("ego speed (km/h)")
    ax.set_ylabel("distance travelled after brake onset (m)")
    ax.set_title("Static obstacle: braking distance vs speed")
    ax.legend(fontsize=8)
    _save(fig, "speed_vs_braking_distance.png")


def latency_sensitivity() -> None:
    runs = [r for r in _v3() if not r.get("tag") and r["ego_kmh"] == 30.0 and r["delay_ms"] is not None]
    if not runs:
        print("skip latency_sensitivity")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, sc in zip(axes, ["static_obstacle", "lead_vehicle_braking", "pedestrian_crossing"], strict=True):
        for i, (mode, label) in enumerate(
            [("gt_oracle", "GT oracle"), ("gt_gated", "GT-assisted"), ("mono", "perception-driven")]
        ):
            pts = sorted((r["delay_ms"], r["min_gap_m"]) for r in runs if r["scenario"] == sc and r["mode"] == mode)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=6, color=C[i], label=label)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_title(sc.replace("_", " "))
        ax.set_xlabel("injected perception latency (ms)")
    axes[0].set_ylabel("minimum gap (m)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Safety margin vs perception latency (30 km/h)", fontweight="bold")
    _save(fig, "latency_vs_min_gap.png")


def mono_range_error() -> None:
    logs = sorted(resolve("outputs/scenario_logs").glob("*__mono__30kmh.jsonl"))
    xs, es, kinds = [], [], []
    for p in logs:
        if "__rep2" in p.name or "__replay" in p.name:
            continue
        kind = "pedestrian" if "pedestrian" in p.name or "rain" in p.name else "vehicle"
        for r in map(json.loads, p.read_text().splitlines()):
            if r["mono_matches_gt"] and r["mono_gap_m"] is not None and r["gap_m"] is not None and r["collisions"] == 0:
                xs.append(r["gap_m"])
                es.append(r["mono_gap_m"] - r["gap_m"])
                kinds.append(kind)
    if not xs:
        print("skip mono_range_error")
        return
    xs, es, kinds = np.array(xs), np.array(es), np.array(kinds)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, k in enumerate(["vehicle", "pedestrian"]):
        m = kinds == k
        ax.scatter(xs[m], es[m], s=4, alpha=0.35, color=C[i], label=f"{k} (n={m.sum()})")
    bins = np.arange(0, 45, 5)
    centers, med = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (xs >= lo) & (xs < hi)
        if m.sum() > 10:
            centers.append((lo + hi) / 2)
            med.append(np.median(np.abs(es[m])))
    ax.plot(centers, med, color=INK, marker="o", markersize=5, label="median |error| per 5 m bin")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xlabel("ground-truth gap (m)")
    ax.set_ylabel("camera estimate − GT (m)")
    ax.set_title("Flat-ground monocular range error vs CARLA ground truth")
    ax.legend(fontsize=8, markerscale=3)
    _save(fig, "mono_range_error.png")


def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (
        latency_model_only,
        latency_breakdown,
        detection_agreement,
        aeb_response,
        speed_sweep,
        latency_sensitivity,
        mono_range_error,
    ):
        f()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
