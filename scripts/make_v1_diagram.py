"""Draw the V1 architecture diagram (assets/diagrams/v1_architecture.png).

Two paths are kept visually separate: the implemented runtime closed loop (GPU) and the implemented
offline NPU validation path. The live NPU closed loop is drawn dashed as the V2 target.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "assets" / "diagrams" / "v1_architecture.png"
INK, MUTED, SURFACE = "#1f2328", "#59636e", "#ffffff"
LOOP, NPU, V2 = "#2a6fdb", "#1a7f5a", "#b35900"


def box(ax, x, y, w, h, text, edge, dashed=False, fill="#f6f8fa"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=fill,
            ec=edge,
            lw=1.6,
            ls="--" if dashed else "-",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)


def arrow(ax, x1, y1, x2, y2, color, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=color, lw=1.4, ls="--" if dashed else "-"
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    w, h, gap = 1.95, 0.78, 0.3
    # --- runtime closed loop (implemented, GPU)
    ax.text(0.2, 5.3, "1  Runtime closed loop — IMPLEMENTED (local RTX 3060)", fontsize=10, color=LOOP, weight="bold")
    labels = [
        "CARLA 0.9.16\nRGB camera 800×450\nsync, dt 0.01 s",
        "YOLO26n on GPU\nletterbox 640², raw\ndecode + NMS on host",
        "Risk estimation\nflat-ground range\nCIPO tracker, TTC",
        "AEB v2\nTTC 1.6 / 2.6 s\n+ measured delay",
        "CARLA vehicle\ncontrol\n(brake / throttle)",
    ]
    y1 = 4.25
    xs = [0.2 + i * (w + gap) for i in range(len(labels))]
    for x, t in zip(xs, labels, strict=True):
        box(ax, x, y1, w, h, t, LOOP)
    for a, b in zip(xs[:-1], xs[1:], strict=True):
        arrow(ax, a + w, y1 + h / 2, b, y1 + h / 2, LOOP)
    # feedback: control -> world
    ax.add_patch(
        FancyArrowPatch(
            (xs[-1] + w / 2, y1),
            (xs[0] + w / 2, y1),
            connectionstyle="arc3,rad=-0.12",
            arrowstyle="-|>",
            mutation_scale=11,
            color=LOOP,
            lw=1.2,
        )
    )
    ax.text(5.1, 3.62, "vehicle state → next simulation tick", fontsize=7.5, color=MUTED, ha="center")

    # --- NPU validation path (implemented, offline)
    ax.text(0.2, 2.9, "2  NPU validation path — IMPLEMENTED (offline, remote Furiosa RNGD)", fontsize=10, color=NPU,
            weight="bold")  # fmt: skip
    labels2 = [
        "Captured / replay\nCARLA frames\n(identical inputs)",
        "AOT compile\nfuriosa-torch 2026.4.0\nFP32 / BF16 / W8",
        "RNGD furiosa:0\nYOLO26n BF16\n(recommended)",
        "Same host decode\n+ NMS code\nas GPU path",
        "Agreement vs GPU,\nlatency, power,\nAEB replay",
    ]
    y2 = 1.85
    for x, t in zip(xs, labels2, strict=True):
        box(ax, x, y2, w, h, t, NPU)
    for a, b in zip(xs[:-1], xs[1:], strict=True):
        arrow(ax, a + w, y2 + h / 2, b, y2 + h / 2, NPU)

    # --- V2 target (not implemented)
    v2_text = (
        "V2 target — NOT IMPLEMENTED\nRNGD replaces the GPU detector inside loop 1 in real time\n"
        "needs a co-located NPU or low-latency transport\n(remote SSH frame transfer measured: 121 ms)"
    )
    box(ax, 2.4, 0.1, 6.4, 1.05, v2_text, V2, dashed=True, fill=SURFACE)
    arrow(ax, xs[2] + w / 2, y2, xs[2] + w / 2, 1.15, V2, dashed=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
