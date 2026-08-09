"""Generate Figure B: tissue motion over respiratory phases.

The script reads the motion signal from each trial's inputs/ABCP.csv and
draws the standalone tissue-motion panel used as Figure B.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data" / "trials"
TRIAL_IDS = (17, 18, 19)


def trial_inputs_dir(trial_id: int) -> Path:
    return DATA_DIR / f"trial_{trial_id}" / "inputs"


def trial_outputs_dir(trial_id: int) -> Path:
    return DATA_DIR / f"trial_{trial_id}" / "outputs"
DEFAULT_TRIAL_ID = 17
DEFAULT_MODEL = "ABCP"
DEFAULT_FPS = 30.0
MIN_SPAN_LEN = 5

# Canvas geometry.  Margins are expressed in inches rather than figure
# fractions so the phase-bracket band, tick labels and axis labels keep their
# absolute size when only the panel is made flatter.
FIG_WIDTH = 7.25
FIG_HEIGHT = 1.44
FIG_HEIGHT_2TO1 = 4.0
MARGIN_LEFT_IN = 0.69
MARGIN_RIGHT_IN = 0.06
MARGIN_TOP_IN = 0.26
# Only the tick-label row sits below the axes: "Time (s)" shares that row.
MARGIN_BOTTOM_IN = 0.25
# Phase brackets sit this far above the axes frame, in inches.
BRACKET_TICK_IN = 0.004
BRACKET_ARROW_IN = 0.053
BRACKET_TEXT_IN = 0.089
XTICK_LENGTH = 2.2
XTICK_PAD = 3.0

MOTION_COLOR = "#1A1A1A"
# Low-saturation Nature-style phase fields, adapted from the project palette.
BG_INHALATION = "#C9DCC4"
BG_EXHALATION = "#C5DFF4"
BG_ALPHA = 0.38
GRID = "#E3E3E3"
SPINE = "#333333"
TEXT_COLOR = "#000000"
ANNOTATION = "#000000"
BG = "#FFFFFF"
FONT_SIZE = 9.5

rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.labelcolor": TEXT_COLOR,
        "axes.linewidth": 0.7,
        "xtick.labelsize": FONT_SIZE,
        "xtick.color": TEXT_COLOR,
        "ytick.labelsize": FONT_SIZE,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": "white",
    }
)


@dataclass
class MotionData:
    trial_id: int
    time_s: np.ndarray
    motion: np.ndarray
    spans: list[tuple[int, int, bool]]


def motion_columns(df: pd.DataFrame) -> list[str]:
    if "motion" in df.columns:
        return ["motion"]
    cols = sorted(col for col in df.columns if col.startswith("motion_"))
    if cols:
        return cols
    cols = sorted(col for col in df.columns if col.startswith("state_"))
    if cols:
        return cols
    raise ValueError("ABCP.csv is missing motion, motion_*, or state_* columns")


def primary_motion_column(cols: list[str]) -> str:
    for name in ("motion", "motion_1", "state_1", "state_0", "motion_0"):
        if name in cols:
            return name
    return cols[0]


def trim_valid_motion(df: pd.DataFrame, motion_col: str) -> pd.DataFrame:
    values = pd.to_numeric(df[motion_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError(f"{motion_col} has no valid numeric values")
    last_valid = int(np.where(valid)[0][-1])
    return df.iloc[: last_valid + 1].reset_index(drop=True)


def local_extrema(values: np.ndarray, min_distance: int = MIN_SPAN_LEN) -> list[tuple[int, str]]:
    if len(values) < 3:
        return []

    candidates: list[tuple[int, str]] = []
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
            _, next_end, next_rising = spans[1]
            spans[1] = (start, next_end, next_rising)
            spans.pop(0)
        elif idx == len(spans) - 1:
            prev_start, _, prev_rising = spans[idx - 1]
            spans[idx - 1] = (prev_start, end, prev_rising)
            spans.pop(idx)
        else:
            prev_len = spans[idx - 1][1] - spans[idx - 1][0]
            next_len = spans[idx + 1][1] - spans[idx + 1][0]
            if prev_len >= next_len:
                prev_start, _, prev_rising = spans[idx - 1]
                spans[idx - 1] = (prev_start, end, prev_rising)
                spans.pop(idx)
            else:
                _, next_end, next_rising = spans[idx + 1]
                spans[idx + 1] = (start, next_end, next_rising)
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
    spans = merge_adjacent_spans(merge_short_spans(spans))
    return spans


def load_motion(trial_id: int, model: str, fps: float) -> MotionData:
    csv_path = trial_inputs_dir(trial_id) / f"{model}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    motion_col = primary_motion_column(motion_columns(raw_df))
    df = trim_valid_motion(raw_df, motion_col)
    motion = pd.to_numeric(df[motion_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(motion)
    if not valid.all():
        first_invalid = int(np.where(~valid)[0][0])
        motion = motion[:first_invalid]
    time_s = np.arange(len(motion), dtype=np.float64) / fps
    return MotionData(trial_id=trial_id, time_s=time_s, motion=motion, spans=motion_spans(motion))



def style_axis(ax: plt.Axes, data: MotionData) -> None:
    ax.set_xlim(data.time_s[0], data.time_s[-1])
    y_min = min(0.0, float(np.min(data.motion)))
    y_max = max(1.0, float(np.max(data.motion)))
    y_pad = max((y_max - y_min) * 0.04, 0.02)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    for start, end, is_rising in data.spans:
        start_s = data.time_s[start]
        end_s = data.time_s[end] if end < len(data.time_s) else data.time_s[-1]
        ax.axvspan(
            start_s,
            end_s,
            facecolor=BG_INHALATION if is_rising else BG_EXHALATION,
            alpha=BG_ALPHA,
            linewidth=0,
            zorder=0,
        )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(False)
    ax.set_axisbelow(True)
    # The x label rides on the tick-label row, flush right in the blank space
    # past the last tick, instead of taking a centred row of its own.
    ax.set_xlabel("")
    ax.annotate(
        "Time (s)",
        xy=(1.0, 0.0),
        xycoords="axes fraction",
        xytext=(0.0, -(XTICK_LENGTH + XTICK_PAD)),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=FONT_SIZE,
        color=TEXT_COLOR,
        annotation_clip=False,
    )
    ax.text(
        0.025,
        0.025,
        "Tissue motion estimated using dense optical flow",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        color="#4A4A4A",
        fontstyle="italic",
        clip_on=True,
    )
    ax.set_ylabel("Tissue motion (a.u.)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.72)
    ax.tick_params(axis="y", direction="out", length=2.6, width=0.62, color=SPINE)
    ax.tick_params(
        axis="x",
        direction="out",
        length=XTICK_LENGTH,
        width=0.58,
        color=SPINE,
        pad=XTICK_PAD,
    )


def draw_phase_arrows(ax: plt.Axes, data: MotionData) -> None:
    """Mark each respiratory phase above the plot with bracket lines and arrows."""
    transform = ax.get_xaxis_transform()
    # Offsets are converted from inches so the bracket band stays the same
    # physical height however flat the axes become.
    axes_height_in = ax.get_position().height * ax.figure.get_figheight()
    y_base = 1.0 + BRACKET_TICK_IN / axes_height_in
    y_arrow = 1.0 + BRACKET_ARROW_IN / axes_height_in
    y_text = 1.0 + BRACKET_TEXT_IN / axes_height_in
    boundaries = [data.time_s[start] for start, _, _ in data.spans]
    last_end = data.spans[-1][1] if data.spans else len(data.time_s) - 1
    boundaries.append(data.time_s[last_end] if last_end < len(data.time_s) else data.time_s[-1])
    for x in boundaries:
        ax.plot(
            [x, x],
                [y_base, y_arrow],
                transform=transform,
                color=ANNOTATION,
                linewidth=0.52,
                solid_capstyle="round",
                clip_on=False,
                zorder=6,
            )
    for start, end, is_rising in data.spans:
        start_s = data.time_s[start]
        end_s = data.time_s[end] if end < len(data.time_s) else data.time_s[-1]
        duration_s = end_s - start_s
        if duration_s >= 1.15:
            label = "Inspiration" if is_rising else "Expiration"
        elif duration_s >= 0.55:
            label = "Insp." if is_rising else "Exp."
        else:
            label = "I" if is_rising else "E"
        ax.annotate(
            "",
            xy=(end_s, y_arrow),
            xytext=(start_s, y_arrow),
            xycoords=transform,
            textcoords=transform,
            arrowprops={
                "arrowstyle": "<->",
                "color": ANNOTATION,
                "linewidth": 0.58,
                "shrinkA": 7.0,
                "shrinkB": 7.0,
                "mutation_scale": 6.8,
            },
            annotation_clip=False,
            zorder=7,
        )
        ax.text(
            0.5 * (start_s + end_s),
            y_text,
            label,
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
            fontfamily="Arial",
            color=ANNOTATION,
            clip_on=False,
            zorder=8,
        )


def render(
    data: MotionData,
    output: Path,
    dpi: int,
    two_to_one: bool = False,
    height: float | None = None,
) -> None:
    # The taller 4-inch canvas exports at approximately 2:1 after tight cropping.
    if height is None:
        height = FIG_HEIGHT_2TO1 if two_to_one else FIG_HEIGHT
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, height), facecolor=BG)
    # Fix the layout first: the bracket band is placed relative to the final
    # axes rectangle.
    fig.subplots_adjust(
        left=MARGIN_LEFT_IN / FIG_WIDTH,
        right=1.0 - MARGIN_RIGHT_IN / FIG_WIDTH,
        bottom=MARGIN_BOTTOM_IN / height,
        top=1.0 - MARGIN_TOP_IN / height,
    )
    ax.plot(
        data.time_s,
        data.motion,
        color=MOTION_COLOR,
        linewidth=1.9,
        solid_capstyle="round",
        zorder=4,
    )
    style_axis(ax, data)
    draw_phase_arrows(ax, data)
    ax.legend(
        handles=[
            Patch(facecolor=BG_INHALATION, edgecolor="none", alpha=BG_ALPHA, label="Inspiration"),
            Patch(facecolor=BG_EXHALATION, edgecolor="none", alpha=BG_ALPHA, label="Expiration"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.988, 0.985),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.92,
        fontsize=FONT_SIZE,
        handlelength=1.15,
        handleheight=0.75,
        handletextpad=0.4,
        borderaxespad=0.0,
        # Compact padding keeps the two-row key clear of the rising tail now
        # that the panel is flatter.
        labelspacing=0.3,
        borderpad=0.3,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.035, facecolor=BG)
    plt.close(fig)
    print(f"Saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, choices=TRIAL_IDS, default=DEFAULT_TRIAL_ID)
    parser.add_argument("--all", action="store_true", help="plot all trial ids")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="CSV model stem to read; default: ABCP")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="frames per second used for the time axis")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--output", type=Path, default=None, help="output path or directory")
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help=f"figure height in inches; default: {FIG_HEIGHT}",
    )
    parser.add_argument(
        "--two-to-one",
        action="store_true",
        help="render an additional, more conventional 2:1 width-to-height layout",
    )
    return parser.parse_args()


def output_for_trial(
    trial_id: int,
    output: Path | None,
    multiple: bool,
    two_to_one: bool = False,
) -> Path:
    suffix = "_2to1" if two_to_one else ""
    if output is None:
        return trial_outputs_dir(trial_id) / f"figure_B_tissue_motion{suffix}.png"
    if output.suffix:
        return output if not multiple else output.with_stem(f"{output.stem}_{trial_id}")
    return output / f"figure_B_tissue_motion_{trial_id}{suffix}.png"


def main() -> None:
    args = parse_args()
    trial_ids = TRIAL_IDS if args.all else (args.trial,)
    for trial_id in trial_ids:
        data = load_motion(trial_id, model=args.model, fps=args.fps)
        render(
            data,
            output_for_trial(
                trial_id,
                args.output,
                multiple=len(trial_ids) > 1,
                two_to_one=args.two_to_one,
            ),
            dpi=args.dpi,
            two_to_one=args.two_to_one,
            height=args.height,
        )


if __name__ == "__main__":
    main()
