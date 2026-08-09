"""Draw a broken-axis bar chart for output probabilities.

The default values reproduce the example with probabilities for
"CVS achieved" and "CVS not achieved".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUTPUT = ROOT / "figures" / "output_probability_broken_bar.png"

LABELS = ("CVS achieved", "CVS not achieved")
VALUES = (0.024, 0.976)
COLORS = ("#E8C583", "#C98B8B")
EDGE = "#111111"
GRID = "#E1E1E1"
BG = "#FFFFFF"
FIGURE_HEIGHT = 5.00 * 0.60

rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.linewidth": 1.05,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
    }
)


def draw_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    """Draw the small diagonal marks that indicate a broken y axis."""
    kwargs = dict(color=EDGE, clip_on=False, linewidth=1.25, solid_capstyle="round")
    mark = 0.017
    gap = 0.020
    ax_top.plot((-mark, mark), (-gap, gap), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-mark, mark), (1 - gap, 1 + gap), transform=ax_bottom.transAxes, **kwargs)


def style_axes(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    ax_top.set_ylim(0.94, 1.00)
    ax_bottom.set_ylim(0.00, 0.12)
    ax_top.set_yticks([0.95, 1.00])
    ax_bottom.set_yticks([0.00, 0.05, 0.10])
    ax_top.set_yticklabels(["0.95", "1.00"])
    ax_bottom.set_yticklabels(["0.00", "0.05", "0.10"])

    for ax in (ax_top, ax_bottom):
        ax.grid(axis="y", color=GRID, linewidth=1.0)
        ax.set_axisbelow(True)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", direction="out", length=5, width=1.05, color=EDGE)
        ax.tick_params(axis="y", pad=5)
        for side in ("left", "bottom", "top"):
            ax.spines[side].set_color(EDGE)
            ax.spines[side].set_linewidth(1.05)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)


def annotate_bars(ax_top: plt.Axes, ax_bottom: plt.Axes, x: np.ndarray, values: list[float]) -> None:
    for xpos, value in zip(x, values):
        target_ax = ax_top if value >= ax_top.get_ylim()[0] else ax_bottom
        if target_ax is ax_top:
            label_y = target_ax.get_ylim()[1] - 0.002
            vertical_alignment = "top"
        else:
            label_y = value + 0.004
            vertical_alignment = "bottom"
        target_ax.text(
            xpos,
            label_y,
            f"{value:.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=14,
            color=EDGE,
        )


def render(labels: list[str], values: list[float], output: Path, dpi: int) -> None:
    if len(labels) != len(values):
        raise ValueError("--labels and --values must have the same length")

    x = np.arange(len(values), dtype=np.float64)
    colors = [COLORS[idx % len(COLORS)] for idx in range(len(values))]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        # Keep the original width, but compress the physical y-axis scale to
        # two thirds of its former height.
        figsize=(3.55, FIGURE_HEIGHT),
        gridspec_kw={"height_ratios": [1.0, 2.35], "hspace": 0.04},
    )

    for ax in (ax_top, ax_bottom):
        ax.bar(x, values, width=0.36, color=colors, edgecolor=EDGE, linewidth=1.05, zorder=3)
        ax.set_xlim(-0.25, len(values) - 0.75)

    style_axes(ax_top, ax_bottom)
    draw_break_marks(ax_top, ax_bottom)
    annotate_bars(ax_top, ax_bottom, x, values)

    ax_top.set_title("Probability", loc="left", pad=9)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels)

    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.16, top=0.905)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=float, nargs="+", default=list(VALUES))
    parser.add_argument("--labels", nargs="+", default=list(LABELS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.labels, args.values, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
