"""Broken-axis probability bar charts (merges the former output_probability.py
and anomaly_probability.py).

Subcommands:
    output   CVS output probabilities (two bars, broken y axis).
    anomaly  Anomaly probabilities plus the Unknown energy score panel:
                 Unknown = 1 - abs(Collision - Detachment)

Examples:
    python -m scripts.figures.probability_broken_bar output
    python -m scripts.figures.probability_broken_bar output --values 0.1 0.9
    python -m scripts.figures.probability_broken_bar anomaly --collision 0.983
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT_CHART = ROOT / "figures" / "output_probability_broken_bar.svg"
DEFAULT_ANOMALY_CHART = ROOT / "figures" / "anomaly_probability_broken_bar.svg"

OUTPUT_LABELS = ("CVS achieved", "CVS not achieved")
OUTPUT_VALUES = (0.024, 0.976)
DEFAULT_COLLISION = 0.983
PROBABILITY_LABELS = ("Collision", "Detachment")
ENERGY_LABEL = "Unknown"
PROBABILITY_COLORS = ("#E8C583", "#C98B8B")
ENERGY_COLOR = "#8FA9B6"
ENERGY_THRESHOLD = 0.05
THRESHOLD_COLOR = "#8F3D3D"
EDGE = "#111111"
GRID = "#E1E1E1"
BG = "#FFFFFF"
FONT_SIZE = 14
FIGURE_HEIGHT = 5.00 * 0.60

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.linewidth": 1.05,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": BG,
    }
)


def unknown_energy(collision: float, unexpected_stop: float) -> float:
    return 1.0 - abs(collision - unexpected_stop)


def draw_break_marks(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
    """Draw the small diagonal marks that indicate a broken y axis."""
    kwargs = dict(color=EDGE, clip_on=False, linewidth=1.25, solid_capstyle="round")
    mark = 0.017
    gap = 0.020
    ax_top.plot((-mark, mark), (-gap, gap), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-mark, mark), (1 - gap, 1 + gap), transform=ax_bottom.transAxes, **kwargs)


def style_probability_axes(ax_top: plt.Axes, ax_bottom: plt.Axes) -> None:
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


def style_energy_axis(ax: plt.Axes, value: float) -> None:
    if value <= 0.12:
        ax.set_ylim(0.00, 0.12)
        ax.set_yticks([0.00, 0.05, 0.10])
        ax.set_yticklabels(["0.00", "0.05", "0.10"])
    else:
        ax.set_ylim(0.00, 1.00)
        ax.set_yticks([0.00, 0.50, 1.00])
        ax.set_yticklabels(["0.00", "0.50", "1.00"])

    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=5, width=1.05, color=EDGE)
    ax.tick_params(axis="y", pad=5)
    for side in ("left", "bottom", "top"):
        ax.spines[side].set_color(EDGE)
        ax.spines[side].set_linewidth(1.05)


def draw_energy_threshold(ax: plt.Axes, threshold: float) -> None:
    y_min, y_max = ax.get_ylim()
    ax.axhline(
        threshold,
        color=THRESHOLD_COLOR,
        linewidth=1.25,
        linestyle=(0, (4, 3)),
        zorder=5,
    )
    ax.text(
        0.0,
        threshold + (y_max - y_min) * 0.025,
        "Threshold",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
        color=THRESHOLD_COLOR,
    )


def annotate_broken_bars(
    ax_top: plt.Axes, ax_bottom: plt.Axes, x: np.ndarray, values: list[float]
) -> None:
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
            fontsize=FONT_SIZE,
            color=EDGE,
        )


def annotate_plain_bar(ax: plt.Axes, xpos: float, value: float) -> None:
    y_min, y_max = ax.get_ylim()
    ax.text(
        xpos,
        value + (y_max - y_min) * 0.025,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
        color=EDGE,
    )


def save(fig: plt.Figure, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"Saved: {output}")


def render_output(labels: list[str], values: list[float], output: Path, dpi: int) -> None:
    if len(labels) != len(values):
        raise ValueError("--labels and --values must have the same length")

    x = np.arange(len(values), dtype=np.float64)
    colors = [PROBABILITY_COLORS[idx % len(PROBABILITY_COLORS)] for idx in range(len(values))]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(3.55, FIGURE_HEIGHT),
        gridspec_kw={"height_ratios": [1.0, 2.35], "hspace": 0.04},
    )

    for ax in (ax_top, ax_bottom):
        ax.bar(x, values, width=0.36, color=colors, edgecolor=EDGE, linewidth=1.05, zorder=3)
        ax.set_xlim(-0.25, len(values) - 0.75)

    style_probability_axes(ax_top, ax_bottom)
    draw_break_marks(ax_top, ax_bottom)
    annotate_broken_bars(ax_top, ax_bottom, x, values)

    ax_top.set_title("Probability", loc="left", pad=9)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels)

    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.16, top=0.905)
    save(fig, output, dpi)


def render_anomaly(collision: float, unexpected_stop: float, output: Path, dpi: int) -> None:
    total = collision + unexpected_stop
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"Collision and Detachment must sum to 1.0, got {total:.6f}")
    if not (0.0 <= collision <= 1.0 and 0.0 <= unexpected_stop <= 1.0):
        raise ValueError("Collision and Detachment must be in [0, 1]")

    probability_values = [collision, unexpected_stop]
    energy_value = unknown_energy(collision, unexpected_stop)
    probability_x = np.arange(len(probability_values), dtype=np.float64)

    fig = plt.figure(figsize=(5.10, FIGURE_HEIGHT), facecolor=BG)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.35, 1.05],
        height_ratios=[1.0, 2.35],
        hspace=0.04,
        wspace=0.34,
    )
    ax_prob_top = fig.add_subplot(grid[0, 0])
    ax_prob_bottom = fig.add_subplot(grid[1, 0], sharex=ax_prob_top)
    ax_energy = fig.add_subplot(grid[:, 1])

    for ax in (ax_prob_top, ax_prob_bottom):
        ax.bar(
            probability_x,
            probability_values,
            width=0.44,
            color=PROBABILITY_COLORS,
            edgecolor=EDGE,
            linewidth=1.05,
            zorder=3,
        )
        ax.set_xlim(-0.32, len(probability_values) - 0.68)

    ax_energy.bar(
        [0.0],
        [energy_value],
        width=0.52,
        color=ENERGY_COLOR,
        edgecolor=EDGE,
        linewidth=1.05,
        hatch="///",
        zorder=3,
    )
    ax_energy.set_xlim(-0.46, 0.46)

    style_probability_axes(ax_prob_top, ax_prob_bottom)
    style_energy_axis(ax_energy, energy_value)
    draw_energy_threshold(ax_energy, ENERGY_THRESHOLD)
    draw_break_marks(ax_prob_top, ax_prob_bottom)
    annotate_broken_bars(ax_prob_top, ax_prob_bottom, probability_x, probability_values)
    annotate_plain_bar(ax_energy, 0.0, energy_value)

    ax_prob_top.set_title("Probability", loc="left", pad=9)
    ax_energy.set_title("Energy Score", loc="left", pad=9)
    ax_prob_bottom.set_xticks(probability_x)
    ax_prob_bottom.set_xticklabels(PROBABILITY_LABELS)
    ax_energy.set_xticks([0.0])
    ax_energy.set_xticklabels([ENERGY_LABEL])

    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.17, top=0.905)
    save(fig, output, dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="chart", required=True)

    p_out = sub.add_parser("output", help="CVS output probabilities")
    p_out.add_argument("--values", type=float, nargs="+", default=list(OUTPUT_VALUES))
    p_out.add_argument("--labels", nargs="+", default=list(OUTPUT_LABELS))
    p_out.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CHART)
    p_out.add_argument("--dpi", type=int, default=600)

    p_anom = sub.add_parser("anomaly", help="anomaly probabilities + energy score")
    p_anom.add_argument("--collision", type=float, default=DEFAULT_COLLISION)
    p_anom.add_argument(
        "--unexpected-stop", type=float, default=None, help="default: 1 - collision"
    )
    p_anom.add_argument("--output", type=Path, default=DEFAULT_ANOMALY_CHART)
    p_anom.add_argument("--dpi", type=int, default=600)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chart == "output":
        render_output(args.labels, args.values, args.output, dpi=args.dpi)
    else:
        unexpected_stop = (
            1.0 - args.collision if args.unexpected_stop is None else args.unexpected_stop
        )
        render_anomaly(args.collision, unexpected_stop, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
