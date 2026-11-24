# ruff: noqa: E501  (long Markdown-table / diagram-label literals are kept on one line for readability)
"""Draw the V2 runtime architecture (assets/diagrams/v2_architecture.png)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "assets" / "diagrams" / "v2_architecture.png"
INK, MUTED, SURFACE = "#1f2328", "#59636e", "#ffffff"
RT, PER, CTL, NPU = "#2a6fdb", "#7a4fd1", "#1a7f5a", "#b35900"


def box(ax, x, y, w, h, text, edge, dashed=False, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#f6f8fa", ec=edge,
                                lw=1.5, ls="--" if dashed else "-"))  # fmt: skip
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x1, y1, x2, y2, color, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=color, lw=1.3,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))  # fmt: skip


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.75)
    ax.axis("off")
    ax.text(0.15, 6.55, "V2 runtime: frame-age-aware asynchronous perception → tracking → filtered range rate → AEB state machine",
            fontsize=10, color=INK, weight="bold")  # fmt: skip
    w, h = 1.75, 0.85
    y = 4.55
    top = [
        ("CARLA RGB camera\n33 Hz, capture_t", RT),
        ("LatestFrameQueue\nmaxsize 1, newest wins\n(drops counted)", RT),
        ("Inference worker\nInferenceBackend", RT),
        ("ResultGate\nstale > 0.3 s rejected\nrepeat / failed", RT),
        ("Tracker (SORT-style)\nIoU + centre gating\nper-track filters", PER),
        ("Range Kalman\n+ lateral Kalman\npredict to control t", PER),
    ]
    xs = [0.15 + i * (w + 0.22) for i in range(len(top))]
    for x, (t, c) in zip(xs, top, strict=True):
        box(ax, x, y, w, h, t, c)
    for a, b in zip(xs[:-1], xs[1:], strict=True):
        arrow(ax, a + w, y + h / 2, b, y + h / 2, RT)
    # backends under the worker
    bx = xs[2]
    box(ax, bx - 1.05, 2.75, 1.25, 0.8, "GPU backend\nRTX 3060 FP32\n(baseline, fallback)", RT, fs=7.5)
    box(ax, bx + 0.32, 2.75, 1.6, 0.8, "RNGD backend\nBF16 persistent server\nover SSH (remote)", NPU, fs=7.2)
    box(ax, bx - 1.05, 1.6, 2.97, 0.85, "FailoverBackend\nPRIMARY → FALLBACK (GPU) → DEGRADED\nswitches logged, RNGD probed every 0.5 s",
        NPU, fs=7.2)  # fmt: skip
    arrow(ax, bx + w / 2, y, bx + 0.4, 3.55, RT)
    arrow(ax, bx + w / 2, y, bx + 1.0, 3.55, NPU)
    # control row
    cy = 2.75
    ctl = [
        ("Risk model\ngating, TTC,\npath prediction\n(will clear / ego passes)", CTL),
        ("AEB state machine\nCLEAR → WATCH → WARNING\n→ BRAKE → RELEASE\nstandstill hold", CTL),
        ("CARLA vehicle\ncontrol (100 Hz)", CTL),
    ]
    cxs = [6.2, 8.1, 10.0]
    for x, (t, c) in zip(cxs, ctl, strict=True):
        box(ax, x, cy - 0.1, 1.8, 1.0, t, c, fs=6.9)
    arrow(ax, xs[5] + w / 2, y, cxs[0] + 0.9, cy + 0.9, PER)
    arrow(ax, cxs[0] + 1.8, cy + 0.4, cxs[1], cy + 0.4, CTL)
    arrow(ax, cxs[1] + 1.8, cy + 0.4, cxs[2], cy + 0.4, CTL)
    ax.text(
        cxs[2] + 0.92, cy - 0.32, "steer / throttle / brake -> CARLA, next tick", fontsize=7, color=MUTED, ha="center"
    )
    ax.text(6.1, 1.0, "perception timeout 0.5 s → hold last command (never release a brake on missing data)",
            fontsize=7.5, color=MUTED)  # fmt: skip
    ax.text(0.15, 0.35, "Matrix: SimulatedAsyncPipeline (same queue/gate semantics, seeded latency model in sim time)\n"
            "Live: AsyncInferenceWorker thread, simulation paced to wall clock",
            fontsize=7.5, color=MUTED)  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
