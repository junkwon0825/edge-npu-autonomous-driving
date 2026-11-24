# ruff: noqa: E501  (diagram-label literals are kept on one line for readability)
"""Draw the V3 architecture (assets/diagrams/v3_architecture.png)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "assets" / "diagrams" / "v3_architecture.png"
INK, MUTED, SURFACE = "#1f2328", "#59636e", "#ffffff"
RT, PER, CTL, NPU = "#2a6fdb", "#7a4fd1", "#1a7f5a", "#b35900"


def box(ax, x, y, w, h, text, edge, dashed=False, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#f6f8fa", ec=edge,
                                lw=1.5, ls="--" if dashed else "-"))  # fmt: skip
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x1, y1, x2, y2, color, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=color, lw=1.3,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))  # fmt: skip


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 13)
    ax.set_ylim(-1.1, 7.6)
    ax.axis("off")
    ax.text(0.15, 7.35, "V3: multi-model NPU perception (detection + lane segmentation) → tracking + radar fusion → SceneState → AEB + LKA → arbiter",
            fontsize=10, color=INK, weight="bold")  # fmt: skip
    # sensors and scheduling
    box(ax, 0.15, 5.6, 1.7, 0.9, "CARLA RGB camera\n800×450, 33 Hz\ncapture_t, frame id", RT)
    box(ax, 0.15, 4.2, 1.7, 0.8, "CARLA radar\n30°×10°, 80 m, 20 Hz\n(optional)", PER)
    box(ax, 2.2, 5.6, 1.9, 0.9, "RateScheduler\nfixed / adaptive / every\nwhich model, which frame", RT)
    box(ax, 4.4, 5.6, 2.2, 0.9, "Latest-frame queues\n(one per model, or one\ncoalescing request in Mode D)", RT)
    arrow(ax, 1.85, 6.05, 2.2, 6.05, RT)
    arrow(ax, 4.1, 6.05, 4.4, 6.05, RT)
    # backends
    box(ax, 6.95, 6.15, 3.0, 0.75, "Detection backend\nGPU YOLO26n FP32  |  RNGD BF16 (furiosa:0)", NPU, fs=7.0)
    box(ax, 6.95, 5.25, 3.0, 0.75, "Lane backend\nGPU U-Net FP32  |  RNGD BF16 (furiosa:1)", NPU, fs=7.0)
    box(ax, 10.25, 5.25, 2.6, 1.65, "RNGD server (remote,\none SSH session)\n- one CompileModule per PE\n- one upload feeds both models\n- FailoverBackend → GPU\n  (logged, probed every 0.5 s)",
        NPU, fs=6.9)  # fmt: skip
    arrow(ax, 6.6, 6.2, 6.95, 6.5, RT)
    arrow(ax, 6.6, 5.9, 6.95, 5.6, RT)
    arrow(ax, 9.95, 6.5, 10.25, 6.3, NPU, dashed=True)
    arrow(ax, 9.95, 5.6, 10.25, 5.8, NPU, dashed=True)
    # perception
    y = 2.85
    box(ax, 2.2, y, 2.0, 1.0, "ResultGate ×2\nstale > 0.3 s rejected\nframe age tracked", RT)
    box(ax, 4.55, y, 2.3, 1.0, "Tracker (SORT-style)\nrange Kalman + lateral KF\n+ radar range / rate updates", PER)
    box(ax, 7.2, y, 2.1, 1.0, "Lane geometry\nflat-ground IPM, windows\ny = a·x² + b·x + c", PER)
    box(ax, 9.65, y, 3.2, 1.0, "SceneState\nego motion, lane + capture time,\npath-relative tracks, source health", PER)
    arrow(ax, 8.45, 5.25, 3.2, y + 1.0, RT, rad=0.05)
    arrow(ax, 1.85, 4.6, 5.7, y + 1.0, PER, dashed=True, rad=-0.1)
    arrow(ax, 4.2, y + 0.5, 4.55, y + 0.5, PER)
    arrow(ax, 3.2, y, 8.25, y, PER, rad=0.25)
    arrow(ax, 9.3, y + 0.5, 9.65, y + 0.5, PER)
    arrow(ax, 5.7, y + 1.0, 11.25, y + 1.0, PER, rad=-0.18)
    # control
    cy = 0.75
    box(
        ax,
        0.15,
        cy,
        2.75,
        1.1,
        "AEB v2 (path-relative)\ngated TTC risk, will-clear,\nego-yaw compensation,\nCLEAR→WATCH→WARNING→BRAKE",
        CTL,
    )
    box(ax, 3.6, cy, 2.75, 1.1, "LKA (pure pursuit)\nlook-ahead 0.8 s·v, latency\ncompensated, 0.5 s lane timeout", CTL)
    box(ax, 7.2, cy, 2.4, 1.1, "Arbiter\n1 AEB brake (steer ≤ ±0.1)\n2 LKA unavailable → coast\n3 normal", CTL)
    box(ax, 10.3, cy, 2.55, 1.1, "CARLA vehicle control\n100 Hz\nsteer / throttle / brake", CTL)
    arrow(ax, 10.6, y, 1.5, cy + 1.1, CTL, rad=0.06)
    arrow(ax, 10.9, y, 5.0, cy + 1.1, CTL, rad=0.04)
    arrow(ax, 1.5, cy, 8.4, cy, CTL, rad=0.22)
    arrow(ax, 6.35, cy + 0.55, 7.2, cy + 0.55, CTL)
    arrow(ax, 9.6, cy + 0.55, 10.3, cy + 0.55, CTL)
    ax.text(0.15, -0.95, "Modes: A GPU det + GPU lane | B RNGD det + GPU lane | C GPU det + RNGD lane | D RNGD det + RNGD lane (one request carries both models)\n"
            "Closed-loop evaluation: time-dilated hardware-in-the-loop (every frame really runs on its backend; measured latency applied in simulation time).\n"
            "Ground truth (map lane centre, target gaps) is logged for evaluation only and never reaches the controllers.",
            fontsize=7.4, color=MUTED)  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
