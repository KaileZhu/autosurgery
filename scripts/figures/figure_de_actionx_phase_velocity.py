"""Generate the merged Action-X / respiratory-phase / mean-velocity panel.

This was originally the right-hand panel of a retired 1x2 combined figure
drawn on its own canvas, so the 3D trajectory (Figure C) and this panel are two
separate figures.  One axes carries three layers:

* the demonstration and four model Action-X curves,
* inspiration / expiration bands behind them,
* per-phase mean absolute x velocity as grouped bars on a right-hand axis.

Colors come from ``figure_c_3d_trajectories`` so both figures agree.

Example:
    python -m scripts.figures.figure_de_actionx_phase_velocity --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from . import figure_c_3d_trajectories as trajectory
from . import figure_d_motion_phase_velocity as phase
from . import figure_e_actionx_pairwise as actionx


TRIAL_IDS = (17, 18, 19)
DEFAULT_TRIAL = 18
MODEL_KEYS = tuple(trajectory.MODEL_FILES)
LABELS = trajectory.DISPLAY_LABELS
SANS = trajectory.SANS

# Candidate palettes.  "npg" is the shared one in figure_C; the others trade
# per-model hue separation for a quieter, more even look -- the ablations sit
# on one restrained ramp and only the best model takes an accent colour.
PALETTES = {
    # The chosen one, and what the 3D panel uses; edit it in figure_C.
    "jewel": dict(trajectory.COLORS),
    # Nature Publishing Group house palette (ggsci "npg").  Best separation of
    # the four, but reads bright next to the others.
    "npg": {
        "GT": "#1A1A1A",
        "zeroshot": "#4DBBD5",
        "AB": "#F39B7F",
        "ABC": "#00A087",
        "ABCP": "#E64B35",
    },
    # Cool slate ramp for the ablations, warm terracotta for the best model.
    "slate": {
        "GT": "#1A1A1A",
        "zeroshot": "#B7C2CB",
        "AB": "#7F97A9",
        "ABC": "#4A7086",
        "ABCP": "#C1553B",
    },
    # Warm greige ramp, deep teal accent.  The quietest of the four.
    "neutral": {
        "GT": "#1A1A1A",
        "zeroshot": "#CBC6BC",
        "AB": "#A2998B",
        "ABC": "#776D5F",
        "ABCP": "#1F6F6B",
    },
}
DEFAULT_PALETTE = "jewel"
COLORS = PALETTES[DEFAULT_PALETTE]

# Thinner than the 1x2 version: on a flatter panel heavy strokes close up the
# gaps between neighbouring curves.
LINEWIDTHS = {
    "GT": 1.9,
    "zeroshot": 1.15,
    "AB": 1.15,
    "ABC": 1.3,
    "ABCP": 1.65,
}

FONT_SIZE = 8.5

# Canvas geometry.  The legend rows, the scientific-notation offset texts and
# the tick/axis-label rows are expressed in inches so they keep their absolute
# size when only the plot box is flattened with --height.
FIG_WIDTH = 7.0
FIG_HEIGHT = 2.10
MARGIN_LEFT_IN = 0.58
# The right axis draws its numbers and title inside the plot box, so only the
# "1e-3" offset text needs room out here.
MARGIN_RIGHT_IN = 0.14
LEGEND_ROW1_TOP_IN = 0.03
LEGEND_ROW2_TOP_IN = 0.195
# Room for both legend rows plus the "1e-2" / "1e-3" offset texts.
AX_TOP_IN = 0.50
# Two-line phase names ("First" / "Inspiration") sit under the axes.
AX_BOTTOM_IN = 0.54

# Per-phase mean-velocity bars: grey for the demonstration, lightened tints of
# the In vivo / In vivo-P curve colours for the two models, so the triple reads
# as "reference vs the two in-vivo models" without any fill competing with the
# curves drawn over it.  Everything else on the panel -- the phase spans, the
# curves, the axis ranges -- still comes from the ABCP inputs as before.
# Left to right within each group: demonstration, In vivo-P, In vivo.
BAR_ORDER = ("GT", "ABCP", "ABC")
BAR_ALPHA = 0.9
BAR_FILL_TINT = 0.42
BAR_EDGE_SHADE = 0.26


def _mix(color: str, target: str, amount: float) -> str:
    """Blend ``color`` towards ``target`` by ``amount`` in [0, 1]."""
    start = np.asarray(mcolors.to_rgb(color))
    end = np.asarray(mcolors.to_rgb(target))
    return mcolors.to_hex(start + (end - start) * amount)


def bar_palette(colors: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    base = {"GT": "#8A8A8A", "ABC": colors["ABC"], "ABCP": colors["ABCP"]}
    fills = {key: _mix(value, "#FFFFFF", BAR_FILL_TINT) for key, value in base.items()}
    edges = {key: _mix(value, "#000000", BAR_EDGE_SHADE) for key, value in base.items()}
    return fills, edges


BAR_COLORS, BAR_EDGE_COLORS = bar_palette(COLORS)


def apply_palette(name: str) -> None:
    """Switch every colour the panel draws with to palette ``name``."""
    global COLORS, BAR_COLORS, BAR_EDGE_COLORS
    COLORS = PALETTES[name]
    BAR_COLORS, BAR_EDGE_COLORS = bar_palette(COLORS)
BAR_LABELS = {
    "GT": "Demonstration mean velocity",
    "ABC": f"{LABELS['ABC']} mean velocity",
    "ABCP": f"{LABELS['ABCP']} mean velocity",
}
# Group span in seconds.  Three bars per phase need a wider group than the two
# the figure started with, or each one turns into a sliver.
BAR_GROUP_SPAN_S = 1.25
# Headroom above the tallest bar.  Lower value -> taller bars; 1.45 lets the
# tallest group reach about two thirds of the panel without touching the
# curves, which stay in the upper band.
SPEED_AXIS_SCALE = 1.45

# Phase bands: near-neutral warm/cool tints of equal lightness, kept very pale
# so they read as a wash behind the data rather than as plotted content.
BG_INSPIRATION = "#E4EBDD"
BG_EXPIRATION = "#DFE8F0"
BG_ALPHA = 0.62

SPINE = "#1A1A1A"
CAPTION_COLOR = "#595959"

rcParams.update(
    {
        "font.family": SANS,
        "font.sans-serif": [SANS, "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": SANS,
        "mathtext.it": f"{SANS}:italic",
        "mathtext.bf": f"{SANS}:bold",
        "mathtext.cal": f"{SANS}:italic",
        "mathtext.sf": SANS,
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)


class FixedDecimalFormatter(ScalarFormatter):
    """ScalarFormatter that always prints a fixed number of decimals.

    Plain ScalarFormatter drops trailing zeros, so a 5.60 / 4.80 / 4.00 axis
    comes out as 5.6 / 4.8 / 4.  Subclassing keeps the "1e-2" offset text that
    a bare FuncFormatter would throw away.
    """

    def __init__(self, decimals: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._decimals = decimals

    def _set_format(self) -> None:
        self.format = f"%.{self._decimals}f"
        if self._usetex or self._useMathText:
            self.format = f"${self.format}$"


# Width of the blank strip kept inside the right spine for that axis' numbers
# and title.  Without it the labels would land on the last bar group.
INSIDE_AXIS_PAD_IN = 0.54


def trial_outputs_dir(trial_id: int) -> Path:
    return trajectory.trial_outputs_dir(trial_id)


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


ORDINAL_SHORT = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th")
# Mean advance width of Arial-metric mixed-case text, as a fraction of the em.
CHAR_EM_WIDTH = 0.55


def phase_tick_labels(
    spans: list[tuple[int, int, bool]],
    steps: int,
    axes_width_in: float,
    fontsize: float,
) -> tuple[list[float], list[str]]:
    """Ordinal phase names, abbreviated to whatever each band can hold.

    Trials differ a lot in how many breaths they cover -- 7 phases in trial 17
    but 11 in trial 18 -- so a fixed label form either wastes space or
    collides.  Wide bands keep "First / Inspiration"; narrower ones fall back
    to "1st / Insp." and then to the bare ordinal, since the bands and the
    legend already carry the inspiration/expiration distinction.
    """
    counts = {True: 0, False: 0}
    em_in = fontsize / 72.0
    ticks: list[float] = []
    forms: list[tuple[tuple[str, int], ...]] = []
    for start, end, is_rising in spans:
        counts[is_rising] += 1
        index = counts[is_rising]
        long_ordinal = (
            phase.ORDINAL_WORDS[index - 1]
            if index <= len(phase.ORDINAL_WORDS)
            else f"{index}th"
        )
        short_ordinal = (
            ORDINAL_SHORT[index - 1] if index <= len(ORDINAL_SHORT) else f"{index}th"
        )
        long_phase = "Inspiration" if is_rising else "Expiration"
        short_phase = "Insp." if is_rising else "Exp."
        forms.append(
            (
                (f"{long_ordinal}\n{long_phase}", max(len(long_ordinal), len(long_phase))),
                (f"{short_ordinal}\n{short_phase}", max(len(short_ordinal), len(short_phase))),
                (short_ordinal, len(short_ordinal)),
            )
        )
        ticks.append((start + end - 1) / 2)

    # One form for the whole axis: mixing "First Expiration" with "2nd Insp."
    # in the same row reads as a mistake.  Labels are centred on their band and
    # may overhang it, so the binding constraint is the gap between neighbours,
    # not the width of any one band.
    centers_in = [tick / max(steps - 1, 1) * axes_width_in for tick in ticks]
    overhang_in = 0.22
    for level in range(len(forms[0])):
        widths = [form[level][1] * CHAR_EM_WIDTH * em_in for form in forms]
        pairs_fit = all(
            (widths[i] + widths[i + 1]) / 2.0 <= centers_in[i + 1] - centers_in[i]
            for i in range(len(widths) - 1)
        )
        edges_fit = (
            widths[0] / 2.0 <= centers_in[0] + overhang_in
            and widths[-1] / 2.0 <= axes_width_in - centers_in[-1] + overhang_in
        )
        if pairs_fit and edges_fit:
            break
    return ticks, [form[level][0] for form in forms]


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
    # the x(t) curves share one consistent notion of velocity.
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
                linewidth=0.6,
                zorder=2,
            )
    return max_mean


def style_speed_axis(ax: plt.Axes, max_speed: float) -> None:
    ax.set_ylim(0.0, max(max_speed * SPEED_AXIS_SCALE, 1e-12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    speed_formatter = FixedDecimalFormatter(decimals=2, useOffset=False, useMathText=False)
    speed_formatter.set_scientific(True)
    speed_formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(speed_formatter)
    ax.yaxis.get_offset_text().set_fontfamily(SANS)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_color(SPINE)
    ax.spines["right"].set_linewidth(0.6)
    # Tick marks and their numbers live inside the plot box, so the panel does
    # not have to reserve a column of white space on the right for them.
    ax.tick_params(
        axis="y", direction="in", length=2.2, width=0.5, colors=SPINE, pad=-2.5
    )
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    # Centred labels at the ends of the range straddle the frame, so the bottom
    # spine would strike through "0.00".  Anchor the end labels to their inner
    # edge instead; the ones in between stay centred on their tick.
    low, high = ax.get_ylim()
    for value, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        label.set_horizontalalignment("right")
        if abs(value - low) < 1e-12 * max(abs(high), 1.0) or value <= low:
            label.set_verticalalignment("bottom")
        elif value >= high:
            label.set_verticalalignment("top")
    # The axis title follows the numbers inside, sitting just to their left.
    number_width_pt = 4 * CHAR_EM_WIDTH * FONT_SIZE
    ax.annotate(
        r"Mean $x$ (m/s)",
        xy=(1.0, 0.5),
        xycoords="axes fraction",
        xytext=(-(number_width_pt + 7.0), 0.0),
        textcoords="offset points",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color=SPINE,
        zorder=8,
    )


def draw_panel(
    ax: plt.Axes,
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    duration_s: float,
    spans: list[tuple[int, int, bool]],
    caption: bool,
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

    # Bands and bars live on the twin axis so the curves stay on top of both.
    speed_ax = ax.twinx()
    draw_phase_bands(speed_ax, spans, time_s)
    max_speed = draw_velocity_bars(
        speed_ax,
        {"GT": gt, **{key: predictions[key][:steps] for key in BAR_ORDER if key != "GT"}},
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
            s=13,
            color=COLORS[key],
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
            clip_on=False,
        )

    values = np.concatenate(plotted)
    x_pad = duration_s * 0.018
    y_low, y_high = float(np.nanmin(values)), float(np.nanmax(values))
    y_span = max(y_high - y_low, np.finfo(float).eps)
    # Reserve a blank strip at the right for the in-box numbers and title of
    # the velocity axis, so no bar or curve ends up underneath them.
    axes_width_in = FIG_WIDTH - MARGIN_LEFT_IN - MARGIN_RIGHT_IN
    strip = INSIDE_AXIS_PAD_IN / axes_width_in
    ax.set_xlim(-x_pad, duration_s + (duration_s + x_pad) * strip / (1.0 - strip))
    ax.set_ylim(y_low - 0.10 * y_span, y_high + 0.10 * y_span)
    # The x axis names the respiratory phases instead of counting seconds: each
    # tick sits at the centre of its band, so the bars, the wash behind them
    # and the label all line up.  Sample indices come back from
    # phase_tick_labels, so map them onto the seconds axis the curves use.
    tick_samples, tick_labels = phase_tick_labels(
        spans,
        steps,
        axes_width_in=FIG_WIDTH - MARGIN_LEFT_IN - MARGIN_RIGHT_IN,
        fontsize=FONT_SIZE,
    )
    ax.set_xticks(np.interp(tick_samples, np.arange(len(time_s)), time_s))
    ax.set_xticklabels(tick_labels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    y_formatter = FixedDecimalFormatter(decimals=2, useOffset=False, useMathText=False)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-2, -2))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.get_offset_text().set_fontfamily(SANS)
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)
    # No gridlines: with the phase bands already washing the plot area, Nature
    # house style keeps the field clean and lets the ticks carry the scale.
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(
        axis="both",
        direction="out",
        length=2.2,
        width=0.5,
        color=SPINE,
        pad=2.0,
    )
    ax.set_xlabel("")
    ax.set_ylabel(r"$x$ (m)", labelpad=3)
    if caption:
        ax.text(
            0.985,
            0.965,
            "Instrument motion along the x-axis with per-phase mean velocity",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT_SIZE - 1.0,
            color=CAPTION_COLOR,
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
        Line2D([0], [0], color=COLORS["GT"], lw=LINEWIDTHS["GT"], label="Demonstration"),
        *[
            Line2D([0], [0], color=COLORS[key], lw=LINEWIDTHS[key], label=LABELS[key])
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
            facecolor=BG_INSPIRATION, edgecolor="#A9B7A2", linewidth=0.6,
            alpha=BG_ALPHA, label="Inspiration",
        ),
        Patch(
            facecolor=BG_EXPIRATION, edgecolor="#A3B6C6", linewidth=0.6,
            alpha=BG_ALPHA, label="Expiration",
        ),
    ]


def render(
    trial_id: int,
    output: Path,
    k: float,
    dpi: int,
    height: float = FIG_HEIGHT,
    caption: bool = False,
) -> None:
    pairs = actionx.load_trial_pairs(trial_id, k=k)
    duration_s = actionx.video_duration_s(trial_id)
    steps = min(len(pairs[0][1]), min(len(prediction) for _, _, prediction in pairs))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, height), facecolor="white")
    fig.subplots_adjust(
        left=MARGIN_LEFT_IN / FIG_WIDTH,
        right=1.0 - MARGIN_RIGHT_IN / FIG_WIDTH,
        bottom=AX_BOTTOM_IN / height,
        top=1.0 - AX_TOP_IN / height,
    )
    spans = load_phase_spans(trial_id, steps=steps)
    draw_panel(ax, pairs, duration_s, spans, caption=caption)

    legend_kwargs = dict(
        loc="upper center",
        frameon=False,
        handletextpad=0.34,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=curve_legend_handles(),
        bbox_to_anchor=(0.5, 1.0 - LEGEND_ROW1_TOP_IN / height),
        ncol=5,
        handlelength=1.5,
        **legend_kwargs,
    )
    bar_handles = bar_legend_handles()
    fig.legend(
        handles=bar_handles,
        bbox_to_anchor=(0.5, 1.0 - LEGEND_ROW2_TOP_IN / height),
        ncol=len(bar_handles),
        handlelength=1.3,
        **legend_kwargs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {output}")


def output_for_trial(trial_id: int, output: Path | None, multiple: bool) -> Path:
    if output is None:
        return trial_outputs_dir(trial_id) / "figure_DE_actionx_phase_velocity.svg"
    if output.suffix:
        return output if not multiple else output.with_stem(f"{output.stem}_{trial_id}")
    return output / f"figure_DE_actionx_phase_velocity_{trial_id}.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, choices=TRIAL_IDS, default=DEFAULT_TRIAL)
    parser.add_argument("--all", action="store_true", help="plot all trial ids")
    parser.add_argument("--k", type=float, default=0.03, help="chunk fusion decay")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--height",
        type=float,
        default=FIG_HEIGHT,
        help=f"figure height in inches; default: {FIG_HEIGHT}",
    )
    parser.add_argument(
        "--caption",
        action="store_true",
        help="draw the italic in-panel caption (off by default to save height)",
    )
    parser.add_argument(
        "--palette",
        choices=sorted(PALETTES),
        default=DEFAULT_PALETTE,
        help=f"model colour set; default: {DEFAULT_PALETTE}",
    )
    parser.add_argument("--output", type=Path, default=None, help="output path or directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_palette(args.palette)
    trial_ids = TRIAL_IDS if args.all else (args.trial,)
    for trial_id in trial_ids:
        render(
            trial_id,
            output_for_trial(trial_id, args.output, multiple=len(trial_ids) > 1),
            k=args.k,
            dpi=args.dpi,
            height=args.height,
            caption=args.caption,
        )


if __name__ == "__main__":
    main()
