"""Combine the 3D trajectory and the Action-X / phase-velocity panel in a 1x2 figure.

The left panel contains the demonstration and four model trajectories.  The
right panel merges what used to be two separate figures: the demonstration and
four model Action-X curves, the respiratory phase bands, and the per-phase mean
absolute x velocity of the demonstration and In vivo-P.  Both panels are drawn
from the same trial and share one legend and one set of model colors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LinearLocator, MaxNLocator, NullFormatter, ScalarFormatter

from . import figure_c_3d_trajectories as trajectory
from . import figure_d_motion_phase_velocity as phase
from . import figure_e_actionx_pairwise as actionx


MODEL_KEYS = tuple(trajectory.MODEL_FILES)
COLORS = trajectory.COLORS
LABELS = trajectory.DISPLAY_LABELS
LINEWIDTHS = {
    "GT": 2.8,
    "zeroshot": 1.8,
    "AB": 1.8,
    "ABC": 2.0,
    "ABCP": 2.3,
}
DEFAULT_TRIAL = 18
FONT_SIZE = 11.0

# Canvas geometry.  The vertical bands reserved for the two legend rows, the
# scientific-notation offset text and the tick/axis labels are given in inches
# so they keep their absolute size while only the two panels are flattened.
FIG_WIDTH = 10.0
FIG_HEIGHT = 3.15
LEGEND_ROW1_TOP_IN = 0.08
LEGEND_ROW2_TOP_IN = 0.32
AX3D_TOP_IN = 0.78
AX3D_BOTTOM_IN = 0.02
# Room for both legend rows plus the "1e-2" / "1e-3" offset texts.
AXX_TOP_IN = 1.02
AXX_BOTTOM_IN = 0.44
# The 3D projection keeps its own box aspect, so it is height-limited and sits
# centred in its slot without filling it.  Narrowing the slot therefore shifts
# the projected box left, away from the right panel's y label, instead of
# shrinking it.
AX3D_RECT_X = 0.012
AX3D_RECT_W = 0.435
AXX_RECT_X = 0.470
AXX_RECT_W = 0.475

# Per-phase mean-velocity bars, carried over from the phase-velocity figure.
BAR_ORDER = ("GT", "ABCP")
BAR_COLORS = {"GT": "#707070", "ABCP": "#0072B2"}
BAR_EDGE_COLORS = {"GT": "#4A4A4A", "ABCP": "#004C78"}
BAR_ALPHA = 0.74
BAR_LABELS = {
    "GT": "Demonstration mean velocity",
    "ABCP": f"{LABELS['ABCP']} mean velocity",
}
# One second per bar group, matching the 30-frame groups of the old figure.
BAR_GROUP_SPAN_S = 1.0
# Headroom above the tallest bar, so the bars read as a band under the curves.
SPEED_AXIS_SCALE = 2.4
BG_INSPIRATION = "#C9DCC4"
BG_EXPIRATION = "#C5DFF4"
BG_ALPHA = 0.38

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
    }
)


def load_trajectories(
    trial_dir: Path, k: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    npy_paths = [
        trial_dir / f"{trajectory.MODEL_FILES[key]}.npy" for key in MODEL_KEYS
    ]
    step_counts = [
        np.load(path, mmap_mode="r", allow_pickle=False).shape[0]
        for path in npy_paths
    ]
    if len(set(step_counts)) != 1:
        raise ValueError(f"Prediction NPY files have different lengths: {step_counts}")

    steps = step_counts[0]
    gt = trajectory.load_gt(trial_dir / "ABCP.csv", steps=steps)
    predictions: dict[str, np.ndarray] = {}
    for key in MODEL_KEYS:
        stem = trajectory.MODEL_FILES[key]
        csv_path = trial_dir / f"{stem}.csv"
        npy_path = trial_dir / f"{stem}.npy"
        model_gt = trajectory.load_gt(csv_path, steps=steps)
        if model_gt.shape != gt.shape or not np.allclose(
            model_gt, gt, atol=1e-8, rtol=0
        ):
            raise ValueError(f"{csv_path.name} does not contain the same trajectory")
        predictions[key] = trajectory.fuse_chunks(
            np.load(npy_path, allow_pickle=False), k=k
        )
    return gt, predictions


def load_phase_spans(trial_id: int, steps: int) -> list[tuple[int, int, bool]]:
    """Inspiration/expiration spans of this trial's own motion signal.

    The signal lives in ``motion`` or ``state_0`` depending on the trial; both
    are the same normalised dense-optical-flow trace.
    """
    csv_path = phase.trial_inputs_dir(trial_id) / "ABCP.csv"
    df = phase.trim_invalid_rows(pd.read_csv(csv_path))
    column = phase.primary_signal_col(phase.signal_cols(df))
    spans = phase.motion_spans(df[column].to_numpy(dtype=np.float64))
    limit = min(steps, len(df))
    return [
        (max(0, start), min(end, limit), is_rising)
        for start, end, is_rising in spans
        if min(end, limit) > start
    ]


def draw_phase_bands(
    ax: plt.Axes, spans: list[tuple[int, int, bool]], time_s: np.ndarray
) -> None:
    for start, end, is_rising in spans:
        ax.axvspan(
            time_s[start],
            time_s[min(end, len(time_s) - 1)],
            facecolor=BG_INSPIRATION if is_rising else BG_EXPIRATION,
            alpha=BG_ALPHA,
            linewidth=0,
            zorder=0,
        )


def draw_velocity_bars(
    ax: plt.Axes,
    series: dict[str, np.ndarray],
    spans: list[tuple[int, int, bool]],
    time_s: np.ndarray,
) -> float:
    """Grouped bars of mean |dx/dt| per respiratory phase; returns the maximum."""
    bar_spans = phase.spans_for_bars(spans, len(time_s))
    # Differentiating against the time axis gives m/s directly, so the bars and
    # the left-hand x(t) curves share one consistent notion of velocity.
    speeds = {
        name: np.abs(np.gradient(values, time_s)) for name, values in series.items()
    }
    means = {
        name: [phase.mean_in_span(speeds[name], start, end) for start, end, _ in bar_spans]
        for name in BAR_ORDER
    }
    max_mean = max((value for values in means.values() for value in values), default=1e-12)

    gap = BAR_GROUP_SPAN_S * 0.045
    bar_width = (BAR_GROUP_SPAN_S - gap * (len(BAR_ORDER) - 1)) / len(BAR_ORDER)
    for span_idx, (start, end, _) in enumerate(bar_spans):
        center = 0.5 * (time_s[start] + time_s[min(end, len(time_s) - 1)])
        left = center - BAR_GROUP_SPAN_S / 2
        for bar_idx, name in enumerate(BAR_ORDER):
            ax.bar(
                left + bar_idx * (bar_width + gap),
                means[name][span_idx],
                width=bar_width,
                align="edge",
                color=BAR_COLORS[name],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLORS[name],
                linewidth=0.65,
                zorder=2,
            )
    return max_mean


def style_speed_axis(ax: plt.Axes, max_speed: float) -> None:
    ax.set_ylim(0.0, max(max_speed * SPEED_AXIS_SCALE, 1e-12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_fontfamily("Arial")
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.set_ylabel(r"Mean $x$ (m/s)", labelpad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_color("#303030")
    ax.spines["right"].set_linewidth(0.72)
    ax.tick_params(axis="y", direction="out", length=2.5, width=0.58, colors="#303030", pad=2.5)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)


def draw_trajectory_panel(
    ax: plt.Axes, gt: np.ndarray, predictions: dict[str, np.ndarray]
) -> None:
    sphere_center = gt[-1, :3]
    sphere_radius = float(
        np.linalg.norm(predictions["ABC"][-1, :3] - sphere_center)
    )
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    sphere_x = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = sphere_center[2] + sphere_radius * np.outer(
        np.ones_like(u), np.cos(v)
    )
    ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color=trajectory.sphere_color(),
        alpha=trajectory.SPHERE_FACE_ALPHA,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=1,
    )
    ax.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=6,
        cstride=6,
        color=trajectory.sphere_color(),
        alpha=trajectory.SPHERE_EDGE_ALPHA,
        linewidth=0.35,
        zorder=2,
    )

    ax.plot(
        *gt.T,
        color=COLORS["GT"],
        lw=LINEWIDTHS["GT"],
        solid_capstyle="round",
        zorder=10,
    )
    for key, values in predictions.items():
        ax.plot(
            *values[:, :3].T,
            color=COLORS[key],
            lw=LINEWIDTHS[key],
            alpha=0.96,
            solid_capstyle="round",
            zorder=5 if key == "ABCP" else 3,
        )
        ax.scatter(
            *values[-1, :3],
            s=18 if key == "ABCP" else 14,
            color=COLORS[key],
            edgecolor="white",
            linewidth=0.45,
            depthshade=False,
            zorder=11,
        )

    ax.scatter(
        *gt[0],
        color="white",
        edgecolor=COLORS["GT"],
        linewidth=0.9,
        s=22,
        depthshade=False,
        zorder=12,
    )
    ax.scatter(
        *gt[-1],
        color=COLORS["GT"],
        edgecolor="white",
        linewidth=0.55,
        s=24,
        depthshade=False,
        zorder=12,
    )
    ax.text(
        *(gt[-1] + np.array([0.00018, -0.00005, 0.00002])),
        "End",
        fontsize=FONT_SIZE,
        color="#333333",
        zorder=13,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.view_init(elev=12, azim=-60)
    sphere_extent = np.vstack(
        [sphere_center - sphere_radius, sphere_center + sphere_radius]
    )
    trajectory.style_3d_axes(
        ax, [gt, *predictions.values(), sphere_extent]
    )
    points = np.concatenate(
        [gt[:, :3], *(values[:, :3] for values in predictions.values())],
        axis=0,
    )
    spans = np.maximum(
        points.max(axis=0) - points.min(axis=0), np.finfo(float).eps
    )
    box_aspect = np.maximum(spans / spans.max(), 0.48)
    box_aspect[2] = max(box_aspect[2], 0.56)
    ax.set_box_aspect(box_aspect, zoom=1.31)
    first_border_line = len(ax.lines)
    trajectory.draw_black_box_silhouette(ax)
    # Keep the projected box silhouette above the transparent panes and grid.
    # At the enlarged 3D scale, low-z-order edges can otherwise look broken.
    for border_line in ax.lines[first_border_line:]:
        border_line.set_zorder(20)
        border_line.set_clip_on(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(LinearLocator(5))
        axis.set_major_formatter(NullFormatter())
    trajectory.add_view_aligned_axis_labels(ax, fontsize=FONT_SIZE)
    start_frac = trajectory._project_to_axes_fraction(ax, gt[0])
    ax.text2D(
        start_frac[0] + 0.012,
        start_frac[1] + 0.018,
        "Start",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        color="#333333",
        zorder=13,
    )


def draw_actionx_panel(
    ax: plt.Axes,
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    duration_s: float,
    spans: list[tuple[int, int, bool]],
) -> plt.Axes:
    predictions = {key: prediction for key, _, prediction in pairs}
    gt = pairs[0][1]
    steps = min(len(gt), min(len(predictions[key]) for key in MODEL_KEYS))
    time_s = np.linspace(0.0, duration_s, steps)
    gt = gt[:steps]
    spans = [
        (start, min(end, steps), is_rising)
        for start, end, is_rising in spans
        if min(end, steps) > start
    ]
    plotted = [gt]

    # Phase bands and bars live on the twin axis so the curves stay on top of
    # both of them.  twinx() rebuilds the twin from the subplotspec and so
    # discards the manual rectangle; copy it back or the bands and bars land in
    # a different box from the curves.
    speed_ax = ax.twinx()
    speed_ax.set_position(ax.get_position())
    draw_phase_bands(speed_ax, spans, time_s)
    max_speed = draw_velocity_bars(
        speed_ax,
        {"GT": gt, "ABCP": predictions["ABCP"][:steps]},
        spans,
        time_s,
    )
    style_speed_axis(speed_ax, max_speed)

    ax.plot(
        time_s,
        gt,
        color=COLORS["GT"],
        lw=LINEWIDTHS["GT"],
        solid_capstyle="round",
        zorder=6,
    )
    for key in MODEL_KEYS:
        values = predictions[key][:steps]
        plotted.append(values)
        ax.plot(
            time_s,
            values,
            color=COLORS[key],
            lw=LINEWIDTHS[key],
            alpha=0.96,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            time_s[-1],
            values[-1],
            s=18,
            color=COLORS[key],
            edgecolor="white",
            linewidth=0.55,
            zorder=5,
            clip_on=False,
        )

    values = np.concatenate(plotted)
    x_pad = duration_s * 0.018
    y_low, y_high = float(np.nanmin(values)), float(np.nanmax(values))
    y_span = max(y_high - y_low, np.finfo(float).eps)
    ax.set_xlim(-x_pad, duration_s + x_pad)
    ax.set_ylim(y_low - 0.11 * y_span, y_high + 0.11 * y_span)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    y_formatter = ScalarFormatter(useOffset=False, useMathText=False)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-2, -2))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.get_offset_text().set_fontfamily("Arial")
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.42, alpha=0.62)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#303030")
        ax.spines[side].set_linewidth(0.72)
    ax.tick_params(
        axis="both",
        direction="out",
        length=2.5,
        width=0.58,
        color="#303030",
        pad=2.5,
    )
    ax.set_xlabel("Time (s)", labelpad=3)
    ax.set_ylabel(r"$x$ (m)", labelpad=4)
    # The bars occupy the lower-left, so the caption sits top-right where no
    # curve reaches.
    ax.text(
        0.985,
        0.975,
        "Instrument motion along the x-axis\nwith per-phase mean velocity",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE,
        fontfamily="Arial",
        color="#4A4A4A",
        fontstyle="italic",
        clip_on=True,
        zorder=7,
    )
    # Keep the curves above the bands and bars of the twin axis.
    ax.set_zorder(speed_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    return speed_ax


def curve_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=COLORS["GT"], lw=LINEWIDTHS["GT"],
            label="Demonstration"
        ),
        *[
            Line2D(
                [0], [0], color=COLORS[key], lw=LINEWIDTHS[key],
                label=LABELS[key]
            )
            for key in MODEL_KEYS
        ],
    ]


def bar_legend_handles() -> list[Patch]:
    return [
        *[
            Patch(
                facecolor=BAR_COLORS[key],
                edgecolor=BAR_EDGE_COLORS[key],
                alpha=BAR_ALPHA,
                label=BAR_LABELS[key],
            )
            for key in BAR_ORDER
        ],
        Patch(
            facecolor=BG_INSPIRATION, edgecolor="#8FA68C", linewidth=0.65,
            alpha=BG_ALPHA, label="Inspiration",
        ),
        Patch(
            facecolor=BG_EXPIRATION, edgecolor="#93AFC4", linewidth=0.65,
            alpha=BG_ALPHA, label="Expiration",
        ),
    ]


def render(
    trial_id: int, output: Path, k: float, dpi: int, height: float = FIG_HEIGHT
) -> None:
    trial_dir = trajectory.trial_inputs_dir(trial_id)
    gt, trajectories = load_trajectories(trial_dir, k=k)
    action_pairs = actionx.load_trial_pairs(trial_id, k=k)
    duration_s = actionx.video_duration_s(trial_id)

    fig = plt.figure(figsize=(FIG_WIDTH, height), facecolor="white")
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    ax_x = fig.add_subplot(1, 2, 2)
    # The 3D projection has more internal padding than a 2D axis.  Give it a
    # slightly wider slot and zoom it so the two panels have comparable visual
    # weight rather than merely equal subplot rectangles.
    ax_3d.set_position(
        [
            AX3D_RECT_X,
            AX3D_BOTTOM_IN / height,
            AX3D_RECT_W,
            1.0 - (AX3D_TOP_IN + AX3D_BOTTOM_IN) / height,
        ]
    )
    ax_x.set_position(
        [
            AXX_RECT_X,
            AXX_BOTTOM_IN / height,
            AXX_RECT_W,
            1.0 - (AXX_TOP_IN + AXX_BOTTOM_IN) / height,
        ]
    )
    spans = load_phase_spans(trial_id, steps=len(gt))
    draw_trajectory_panel(ax_3d, gt, trajectories)
    draw_actionx_panel(ax_x, action_pairs, duration_s, spans)

    legend_kwargs = dict(
        loc="upper center",
        frameon=False,
        handletextpad=0.38,
        columnspacing=0.85,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=curve_legend_handles(),
        # Center over the combined visible extent of both panels.  The 3D
        # projection extends beyond its nominal axes box, so the visual center
        # is slightly right of the raw figure midpoint.
        bbox_to_anchor=(0.52, 1.0 - LEGEND_ROW1_TOP_IN / height),
        ncol=5,
        handlelength=1.7,
        **legend_kwargs,
    )
    fig.legend(
        handles=bar_legend_handles(),
        bbox_to_anchor=(0.52, 1.0 - LEGEND_ROW2_TOP_IN / height),
        ncol=4,
        handlelength=1.5,
        **legend_kwargs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, choices=trajectory.TRIAL_IDS, default=DEFAULT_TRIAL)
    parser.add_argument("--k", type=float, default=0.03, help="chunk fusion decay")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--height",
        type=float,
        default=FIG_HEIGHT,
        help=f"figure height in inches; default: {FIG_HEIGHT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG (default: trial outputs/figure_CE_trajectory_actionx_1x2.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (
        trajectory.trial_outputs_dir(args.trial)
        / "figure_CE_trajectory_actionx_1x2.png"
    )
    render(args.trial, output, k=args.k, dpi=args.dpi, height=args.height)


if __name__ == "__main__":
    main()
