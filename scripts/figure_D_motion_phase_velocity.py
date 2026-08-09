"""Generate Figure D: Motion_x trajectory with phase-wise mean velocity bars.

The figure intentionally does not draw the raw phase/motion curve. It keeps only
phase-colored background spans, the demonstration trajectory, and demonstration/SHARP-P
per-span mean absolute slope bars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "trials"


def trial_inputs_dir(trial_id: int) -> Path:
    return DATA_DIR / f"trial_{trial_id}" / "inputs"


def trial_outputs_dir(trial_id: int) -> Path:
    return DATA_DIR / f"trial_{trial_id}" / "outputs"


TRIAL_IDS = (17, 18, 19)
DEFAULT_TRIAL_ID = 18
ACTION_X_AXIS = 0
FUSION_K = 0.03
MIN_SPAN_LEN = 5
MIN_SLOPE_BAR_RATIO = 0.06
BAR_GROUP_WIDTH = 30.0
SLOPE_AXIS_SCALE = 1.58
ROTATED_PHASE_SCALE = 0.82

# Canvas geometry.  The band below the axes (two-line phase labels plus the
# shared legend) and the offset-text band above it are given in inches so they
# keep their absolute size while only the plot box is flattened.
FIG_WIDTH = 7.0
FIG_HEIGHT = 2.40
MARGIN_TOP_IN = 0.09
MARGIN_BOTTOM_IN = 0.96
LEGEND_BOTTOM_IN = 0.39

PLOT_CURVES = ("GT",)
BAR_ORDER = ("GT", "ABCP")

COLORS = {
    "GT": "#1A1A1A",
    "ABCP": "#0072B2",
}
BAR_COLORS = {
    "GT": "#707070",
    "ABCP": "#0072B2",
}
BAR_EDGE_COLORS = {
    "GT": "#4A4A4A",
    "ABCP": "#004C78",
}
LINEWIDTHS = {"GT": 2.0, "ABCP": 1.35}
BG_RISING = "#C9DCC4"
BG_FALLING = "#C5DFF4"
BG_ALPHA = 0.38
BAR_ALPHA = 0.74
GRID = "#E3E3E3"
SPINE = "#333333"
BG = "#FFFFFF"
ORDINAL_WORDS = (
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Seventh",
    "Eighth",
    "Ninth",
    "Tenth",
)

rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.65,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": "white",
    }
)


def fuse_chunks(chunks: np.ndarray, k: float = FUSION_K) -> np.ndarray:
    if chunks.ndim != 3:
        raise ValueError(f"Expected (steps, chunk_horizon, action_dim), got {chunks.shape}")
    steps, horizon, action_dim = chunks.shape
    if action_dim <= ACTION_X_AXIS:
        raise ValueError(f"Action dimension {action_dim} does not contain Action X")

    fused = np.empty((steps, action_dim), dtype=np.float64)
    for t in range(steps):
        first_chunk = max(0, t - horizon + 1)
        chunk_indices = np.arange(first_chunk, t + 1)
        horizon_indices = t - chunk_indices
        candidates = chunks[chunk_indices, horizon_indices]
        weights = np.exp(-k * np.arange(len(candidates), dtype=np.float64))
        weights /= weights.sum()
        fused[t] = np.sum(candidates * weights[:, None], axis=0)
    return fused


def signal_cols(df: pd.DataFrame) -> list[str]:
    if "motion" in df.columns:
        return ["motion"]
    motion_cols = sorted(c for c in df.columns if c.startswith("motion_"))
    if motion_cols:
        return motion_cols
    state_cols = sorted(c for c in df.columns if c.startswith("state_"))
    return state_cols


def primary_signal_col(cols: list[str]) -> str:
    for name in ("motion_1", "motion", "state_1", "state_0"):
        if name in cols:
            return name
    return cols[0]


def trim_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = signal_cols(df)
    if not cols:
        raise ValueError("CSV is missing motion/state columns for phase spans")
    required = [f"gt_action_{ACTION_X_AXIS}", f"pred_action_{ACTION_X_AXIS}", *cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    valid = np.ones(len(df), dtype=bool)
    for col in required:
        valid &= np.isfinite(df[col].to_numpy())
    if not valid.any():
        raise ValueError("CSV has no valid rows")
    last_valid = int(np.where(valid)[0][-1])
    return df.iloc[: last_valid + 1].reset_index(drop=True)


def local_extrema(values: np.ndarray, min_distance: int = MIN_SPAN_LEN) -> list[tuple[int, str]]:
    if len(values) < 3:
        return []

    candidates = []
    for idx in range(1, len(values) - 1):
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            candidates.append((idx, "peak"))
        elif values[idx] <= values[idx - 1] and values[idx] < values[idx + 1]:
            candidates.append((idx, "valley"))

    if not candidates:
        return []

    filtered = [candidates[0]]
    for idx, kind in candidates[1:]:
        prev_idx, prev_kind = filtered[-1]
        if idx - prev_idx < min_distance:
            if kind == "peak" and values[idx] >= values[prev_idx]:
                filtered[-1] = (idx, kind)
            elif kind == "valley" and values[idx] <= values[prev_idx]:
                filtered[-1] = (idx, kind)
        elif kind != prev_kind:
            filtered.append((idx, kind))
        elif kind == "peak" and values[idx] > values[prev_idx]:
            filtered[-1] = (idx, kind)
        elif kind == "valley" and values[idx] < values[prev_idx]:
            filtered[-1] = (idx, kind)
    return filtered


def merge_short_spans(spans: list[tuple[int, int, bool]], min_len: int = MIN_SPAN_LEN) -> list[tuple[int, int, bool]]:
    spans = list(spans)
    if len(spans) <= 1:
        return spans

    while True:
        lengths = [end - start for start, end, _ in spans]
        min_length = min(lengths)
        if min_length >= min_len:
            break
        idx = lengths.index(min_length)
        start, end, _ = spans[idx]
        if idx == 0:
            _, next_end, next_high = spans[1]
            spans[1] = (start, next_end, next_high)
            spans.pop(0)
        elif idx == len(spans) - 1:
            prev_start, _, prev_high = spans[idx - 1]
            spans[idx - 1] = (prev_start, end, prev_high)
            spans.pop(idx)
        else:
            prev_len = spans[idx - 1][1] - spans[idx - 1][0]
            next_len = spans[idx + 1][1] - spans[idx + 1][0]
            if prev_len >= next_len:
                prev_start, _, prev_high = spans[idx - 1]
                spans[idx - 1] = (prev_start, end, prev_high)
                spans.pop(idx)
            else:
                next_start, next_end, next_high = spans[idx + 1]
                spans[idx + 1] = (start, next_end, next_high)
                spans.pop(idx)
    return spans


def merge_adjacent_spans(spans: list[tuple[int, int, bool]]) -> list[tuple[int, int, bool]]:
    if not spans:
        return spans
    merged = [spans[0]]
    for start, end, is_rising in spans[1:]:
        prev_start, _, prev_rising = merged[-1]
        if is_rising == prev_rising:
            merged[-1] = (prev_start, end, prev_rising)
        else:
            merged.append((start, end, is_rising))
    return merged


def merge_island_spans(spans: list[tuple[int, int, bool]], max_island_len: int = 15) -> list[tuple[int, int, bool]]:
    spans = list(spans)
    if len(spans) < 3:
        return spans

    changed = True
    while changed:
        changed = False
        merged = []
        idx = 0
        while idx < len(spans):
            if (
                idx + 2 < len(spans)
                and spans[idx][2] == spans[idx + 2][2]
                and spans[idx + 1][2] != spans[idx][2]
                and spans[idx + 1][1] - spans[idx + 1][0] <= max_island_len
            ):
                start, _, is_rising = spans[idx]
                end = spans[idx + 2][1]
                merged.append((start, end, is_rising))
                idx += 3
                changed = True
            else:
                merged.append(spans[idx])
                idx += 1
        spans = merged
    return spans


def motion_spans(values: np.ndarray) -> list[tuple[int, int, bool]]:
    extrema = local_extrema(values)
    boundaries = [0] + [idx for idx, _ in extrema] + [len(values)]
    boundaries = sorted(set(boundaries))
    spans = []
    for idx in range(len(boundaries) - 1):
        start, end = boundaries[idx], boundaries[idx + 1]
        if end <= start:
            continue
        is_rising = values[end - 1] > values[start]
        spans.append((start, end, is_rising))
    spans = merge_short_spans(spans)
    spans = merge_adjacent_spans(spans)
    return merge_island_spans(spans)


def spans_for_bars(spans: list[tuple[int, int, bool]], steps: int) -> list[tuple[int, int, bool]]:
    return [
        span
        for span in spans
        if span[1] - span[0] >= MIN_SPAN_LEN and (span[1] - span[0]) / steps >= MIN_SLOPE_BAR_RATIO
    ]


def phase_tick_labels(spans: list[tuple[int, int, bool]]) -> tuple[list[float], list[str]]:
    counts = {True: 0, False: 0}
    ticks = []
    labels = []
    for start, end, is_rising in spans:
        counts[is_rising] += 1
        ordinal = (
            ORDINAL_WORDS[counts[is_rising] - 1]
            if counts[is_rising] <= len(ORDINAL_WORDS)
            else f"{counts[is_rising]}th"
        )
        phase = "Inspiration" if is_rising else "Expiration"
        ticks.append((start + end - 1) / 2)
        labels.append(f"{ordinal}\n{phase}")
    return ticks, labels


def load_trial(trial_id: int, k: float) -> tuple[dict[str, np.ndarray], list[tuple[int, int, bool]]]:
    trial_dir = trial_inputs_dir(trial_id)
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"Missing trial directory: {trial_dir}")

    reference_df = trim_invalid_rows(pd.read_csv(trial_dir / "ABCP.csv"))
    cols = signal_cols(reference_df)
    signal = reference_df[primary_signal_col(cols)].to_numpy(dtype=np.float64)
    spans = motion_spans(signal)
    steps = len(reference_df)
    gt = reference_df[f"gt_action_{ACTION_X_AXIS}"].to_numpy(dtype=np.float64)
    chunks = np.load(trial_dir / "ABCP.npy", allow_pickle=False)
    n = min(steps, chunks.shape[0])
    series = {
        "GT": gt[:n],
        "ABCP": fuse_chunks(chunks[:n], k=k)[:, ACTION_X_AXIS],
    }

    min_steps = min(len(values) for values in series.values())
    series = {key: values[:min_steps] for key, values in series.items()}
    spans = [(max(0, s), min(e, min_steps), r) for s, e, r in spans if min(e, min_steps) > s]
    return series, spans


def abs_slope(values: np.ndarray) -> np.ndarray:
    return np.abs(np.gradient(values))


def mean_in_span(values: np.ndarray, start: int, end: int) -> float:
    end = min(end, len(values))
    return float(np.mean(values[start:end]))


def draw_background(ax: plt.Axes, spans: list[tuple[int, int, bool]]) -> None:
    for start, end, is_rising in spans:
        ax.axvspan(
            start - 0.5,
            end - 0.5,
            facecolor=BG_RISING if is_rising else BG_FALLING,
            alpha=BG_ALPHA,
            linewidth=0,
            zorder=0,
        )


def draw_slope_bars(ax: plt.Axes, series: dict[str, np.ndarray], spans: list[tuple[int, int, bool]], steps: int) -> float:
    bar_spans = spans_for_bars(spans, steps)
    slopes = {name: abs_slope(values) for name, values in series.items()}
    means = {
        name: [mean_in_span(slopes[name], start, end) for start, end, _ in bar_spans]
        for name in BAR_ORDER
    }
    max_mean = max((value for values in means.values() for value in values), default=1e-12)
    n_bars = len(BAR_ORDER)

    for span_idx, (start, end, _) in enumerate(bar_spans):
        span_width = end - start
        group_width = BAR_GROUP_WIDTH
        gap = group_width * 0.045
        bar_width = (group_width - gap * (n_bars - 1)) / n_bars
        x0 = start - 0.5 + span_width / 2 - group_width / 2
        for bar_idx, name in enumerate(BAR_ORDER):
            ax.bar(
                x0 + bar_idx * (bar_width + gap),
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


def style_axes(
    ax: plt.Axes,
    slope_ax: plt.Axes,
    series: dict[str, np.ndarray],
    spans: list[tuple[int, int, bool]],
    max_slope: float,
) -> None:
    steps = len(next(iter(series.values())))
    y_values = np.concatenate([series[name] for name in PLOT_CURVES])
    y_low, y_high = float(np.min(y_values)), float(np.max(y_values))
    y_pad = max((y_high - y_low) * 0.10, 1e-6)
    x_pad = max(2.0, steps * 0.012)

    ax.set_xlim(-x_pad, steps - 1 + x_pad)
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    ticks, labels = phase_tick_labels(spans)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    motion_formatter = ScalarFormatter(useOffset=False, useMathText=False)
    motion_formatter.set_scientific(True)
    motion_formatter.set_powerlimits((-2, -2))
    ax.yaxis.set_major_formatter(motion_formatter)
    ax.grid(axis="y", color=GRID, linewidth=0.42, alpha=0.62)
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.set_ylabel("Tissue motion (a.u.)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for side in ("left", "top"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.68)
    ax.tick_params(
        axis="x",
        direction="out",
        length=2.5,
        width=0.58,
        color=SPINE,
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        pad=4,
    )
    ax.tick_params(axis="y", direction="out", length=2.5, width=0.58, color=SPINE)

    slope_ax.set_ylim(0, max(max_slope * SLOPE_AXIS_SCALE, 1e-12))
    slope_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    slope_ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    slope_ax.set_ylabel(r"Mean $x$ (m/s)")
    slope_ax.spines["top"].set_visible(False)
    slope_ax.spines["left"].set_visible(False)
    slope_ax.spines["bottom"].set_visible(False)
    slope_ax.spines["right"].set_color(SPINE)
    slope_ax.spines["right"].set_linewidth(0.68)
    slope_ax.tick_params(axis="y", direction="out", length=2.5, width=0.58, colors=SPINE, pad=3)
    slope_ax.tick_params(axis="x", bottom=False, labelbottom=False)


def render(
    trial_id: int, output: Path, k: float, dpi: int, height: float = FIG_HEIGHT
) -> None:
    series, spans = load_trial(trial_id, k=k)
    steps = len(next(iter(series.values())))
    frame = np.arange(steps)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, height), facecolor=BG)
    draw_background(ax, spans)

    for name in PLOT_CURVES:
        ax.plot(
            frame,
            series[name],
            color=COLORS[name],
            linewidth=LINEWIDTHS[name],
            alpha=0.98 if name == "GT" else 0.90,
            solid_capstyle="round",
            label="Demonstration" if name == "GT" else name,
            zorder=5 if name == "GT" else 4,
        )

    slope_ax = ax.twinx()
    max_slope = draw_slope_bars(slope_ax, series, spans, steps)
    style_axes(ax, slope_ax, series, spans, max_slope)
    ax.set_zorder(3)
    slope_ax.set_zorder(2)
    ax.patch.set_visible(False)

    line_handles = [
        Line2D([0], [0], color=COLORS[name], lw=LINEWIDTHS[name], label="Demonstration" if name == "GT" else name)
        for name in PLOT_CURVES
    ]
    bar_handles = [
        Patch(
            facecolor=BAR_COLORS[name],
            edgecolor=BAR_EDGE_COLORS[name],
            alpha=BAR_ALPHA,
            label="Demonstration mean velocity" if name == "GT" else "In vivo-P mean velocity",
        )
        for name in BAR_ORDER
    ]
    phase_handles = [
        Patch(facecolor=BG_RISING, edgecolor="#8FA68C", linewidth=0.65, alpha=BG_ALPHA, label="Inspiration"),
        Patch(facecolor=BG_FALLING, edgecolor="#93AFC4", linewidth=0.65, alpha=BG_ALPHA, label="Expiration"),
    ]
    fig.legend(
        handles=[*line_handles, *bar_handles, *phase_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_BOTTOM_IN / height),
        ncol=5,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.0,
        handletextpad=0.42,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.89,
        bottom=MARGIN_BOTTOM_IN / height,
        top=1.0 - MARGIN_TOP_IN / height,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.035, facecolor=BG)
    plt.close(fig)
    print(f"Saved: {output}")



def draw_rotated_background(ax: plt.Axes, spans: list[tuple[int, int, bool]]) -> None:
    for start, end, is_rising in spans:
        ax.axhspan(
            (start - 0.5) * ROTATED_PHASE_SCALE,
            (end - 0.5) * ROTATED_PHASE_SCALE,
            facecolor=BG_RISING if is_rising else BG_FALLING,
            alpha=BG_ALPHA,
            linewidth=0,
            zorder=0,
        )


def draw_rotated_slope_bars(
    ax: plt.Axes,
    series: dict[str, np.ndarray],
    spans: list[tuple[int, int, bool]],
    steps: int,
) -> float:
    bar_spans = spans_for_bars(spans, steps)
    slopes = {name: abs_slope(values) for name, values in series.items()}
    means = {
        name: [mean_in_span(slopes[name], start, end) for start, end, _ in bar_spans]
        for name in BAR_ORDER
    }
    max_mean = max((value for values in means.values() for value in values), default=1e-12)
    n_bars = len(BAR_ORDER)

    for span_idx, (start, end, _) in enumerate(bar_spans):
        span_height = (end - start) * ROTATED_PHASE_SCALE
        group_height = min(BAR_GROUP_WIDTH * 0.72 * ROTATED_PHASE_SCALE, span_height * 0.58)
        gap = group_height * 0.045
        bar_height = (group_height - gap * (n_bars - 1)) / n_bars
        y0 = (start - 0.5) * ROTATED_PHASE_SCALE + span_height / 2 - group_height / 2
        for bar_idx, name in enumerate(BAR_ORDER):
            ax.barh(
                y0 + bar_idx * (bar_height + gap),
                means[name][span_idx],
                height=bar_height,
                align="edge",
                color=BAR_COLORS[name],
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLORS[name],
                linewidth=0.65,
                zorder=2,
            )
    return max_mean


def style_rotated_velocity_axis(
    ax: plt.Axes,
    spans: list[tuple[int, int, bool]],
    steps: int,
    max_slope: float,
) -> None:
    y_pad = max(2.0, steps * 0.012)
    scaled_steps = (steps - 1) * ROTATED_PHASE_SCALE
    ax.set_ylim(scaled_steps + y_pad, -y_pad)
    ax.set_xlim(0, max(max_slope * 1.22, 1e-12))
    ticks, labels = phase_tick_labels(spans)
    ax.set_yticks([tick * ROTATED_PHASE_SCALE for tick in ticks])
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e4:.1f}"))
    ax.grid(axis="x", color=GRID, linewidth=0.42, alpha=0.62)
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.xaxis.tick_top()
    ax.set_ylabel("")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    for side in ("left", "top"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.68)
    ax.tick_params(axis="x", direction="out", length=2.5, width=0.58, color=SPINE, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis="y", direction="out", length=2.5, width=0.58, color=SPINE)
    ax.text(
        0.53,
        0.965,
        r"Mean $x$ (m/s)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        fontfamily="Arial",
        color="black",
        zorder=6,
    )
    ax.text(
        0.83,
        0.965,
        "1e−4",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontfamily="Arial",
        color="black",
        zorder=6,
    )

def render_horizontal(trial_id: int, output: Path, k: float, dpi: int) -> None:
    series, spans = load_trial(trial_id, k=k)
    steps = len(next(iter(series.values())))

    fig, ax = plt.subplots(figsize=(3.45, 3.35), facecolor=BG)
    draw_rotated_background(ax, spans)
    max_slope = draw_rotated_slope_bars(ax, series, spans, steps)
    style_rotated_velocity_axis(ax, spans, steps, max_slope)

    bar_handles = [
        Patch(
            facecolor=BAR_COLORS[name],
            edgecolor=BAR_EDGE_COLORS[name],
            alpha=BAR_ALPHA,
            label="Demonstration mean velocity" if name == "GT" else "In vivo-P mean velocity",
        )
        for name in BAR_ORDER
    ]
    phase_handles = [
        Patch(facecolor=BG_RISING, edgecolor="#8FA68C", linewidth=0.65, alpha=BG_ALPHA, label="Inspiration"),
        Patch(facecolor=BG_FALLING, edgecolor="#93AFC4", linewidth=0.65, alpha=BG_ALPHA, label="Expiration"),
    ]
    fig.legend(
        handles=[*bar_handles, *phase_handles],
        loc="upper center",
        bbox_to_anchor=(0.58, 0.885),
        ncol=2,
        alignment="center",
        frameon=False,
        handlelength=1.35,
        columnspacing=1.05,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.31, right=0.965, bottom=0.05, top=0.715)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.035, facecolor=BG)
    plt.close(fig)
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, choices=TRIAL_IDS, default=DEFAULT_TRIAL_ID)
    parser.add_argument("--all", action="store_true", help="plot all trial ids")
    parser.add_argument("--k", type=float, default=FUSION_K, help="chunk fusion decay")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--height",
        type=float,
        default=FIG_HEIGHT,
        help=f"figure height in inches for the main panel; default: {FIG_HEIGHT}",
    )
    parser.add_argument("--output", type=Path, default=None, help="output path or directory")
    return parser.parse_args()


def output_paths_for_trial(trial_id: int, output: Path | None, multiple: bool) -> tuple[Path, Path]:
    if output is None:
        output_dir = trial_outputs_dir(trial_id)
        return (
            output_dir / "figure_D_phase_velocity.png",
            output_dir / "figure_D_phase_velocity_horizontal.png",
        )
    if output.suffix:
        normal = output if not multiple else output.with_stem(f"{output.stem}_{trial_id}")
        return normal, normal.with_stem(f"{normal.stem}_horizontal")
    return (
        output / f"figure_D_phase_velocity_{trial_id}.png",
        output / f"figure_D_phase_velocity_horizontal_{trial_id}.png",
    )


def main() -> None:
    args = parse_args()
    trial_ids = TRIAL_IDS if args.all else (args.trial,)
    for trial_id in trial_ids:
        output, horizontal_output = output_paths_for_trial(trial_id, args.output, multiple=len(trial_ids) > 1)
        render(trial_id, output, k=args.k, dpi=args.dpi, height=args.height)
        render_horizontal(trial_id, horizontal_output, k=args.k, dpi=args.dpi)


if __name__ == "__main__":
    main()
