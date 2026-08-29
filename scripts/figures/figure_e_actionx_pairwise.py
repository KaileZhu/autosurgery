"""Generate Figure E: stacked Action X comparisons against the demonstration.

By default this script writes one compact vertical panel per trial (17, 19, 18)
with GT vs zeroshot / AB / ABC / ABCP and a shared top legend.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "data" / "trials"
TRIAL_IDS = (17, 19, 18)


def trial_inputs_dir(trial_id: int) -> Path:
    return DATA_DIR / f"trial_{trial_id}" / "inputs"


def trial_outputs_dir(trial_id: int) -> Path:
    return DATA_DIR / f"trial_{trial_id}" / "outputs"


def video_duration_s(trial_id: int) -> float:
    """Read the trial duration from its MP4 movie header."""
    video_dir = DATA_DIR / f"trial_{trial_id}" / "video"
    videos = sorted(
        path
        for path in video_dir.glob("*.mp4")
        if "_with_" not in path.stem and "_phase_" not in path.stem
    )
    if len(videos) != 1:
        raise FileNotFoundError(f"Expected one MP4 in {video_dir}, found {len(videos)}")

    data = videos[0].read_bytes()

    def boxes(start: int, end: int):
        position = start
        while position + 8 <= end:
            size = struct.unpack_from(">I", data, position)[0]
            box_type = data[position + 4 : position + 8]
            header_size = 8
            if size == 1:
                size = struct.unpack_from(">Q", data, position + 8)[0]
                header_size = 16
            elif size == 0:
                size = end - position
            if size < header_size or position + size > end:
                break
            yield box_type, position + header_size, position + size
            position += size

    for box_type, start, end in boxes(0, len(data)):
        if box_type != b"moov":
            continue
        for child_type, child_start, _ in boxes(start, end):
            if child_type != b"mvhd":
                continue
            version = data[child_start]
            offset = child_start + 4 + (16 if version == 1 else 8)
            timescale = struct.unpack_from(">I", data, offset)[0]
            duration_format = ">Q" if version == 1 else ">I"
            duration = struct.unpack_from(duration_format, data, offset + 4)[0]
            if timescale > 0:
                return duration / timescale
    raise ValueError(f"Could not read movie duration from {videos[0]}")


ACTION_X_AXIS = 0
MODEL_FILES = {
    "zeroshot": "zeroshot",
    "AB": "AB",
    "ABC": "ABC",
    "ABCP": "ABCP",
}
DISPLAY_LABELS = {
    "zeroshot": "Zeroshot",
    "AB": "Ex vivo",
    "ABC": "In vivo",
    "ABCP": "In vivo-P",
}
COLORS = {
    "GT": "#1A1A1A",
    "zeroshot": "#4DBBD5",
    "AB": "#F39B7F",
    "ABC": "#00A087",
    "ABCP": "#E64B35",
}
LINEWIDTHS = {"GT": 2.05, "zeroshot": 1.3, "AB": 1.3, "ABC": 1.3, "ABCP": 1.7}

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.linewidth": 0.65,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def fuse_chunks(chunks: np.ndarray, k: float = 0.03) -> np.ndarray:
    """Exponentially fuse all chunk predictions that cover each time step."""
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


def load_gt_action_x(csv_path: Path, steps: int | None = None) -> np.ndarray:
    column = f"gt_action_{ACTION_X_AXIS}"
    df = pd.read_csv(csv_path)
    if column not in df:
        raise ValueError(f"{csv_path} is missing column: {column}")
    values = df[column].to_numpy(dtype=np.float64)
    return values if steps is None else values[:steps]


def load_model_action_x(trial_dir: Path, model: str, k: float) -> tuple[np.ndarray, np.ndarray]:
    stem = MODEL_FILES[model]
    csv_path = trial_dir / f"{stem}.csv"
    npy_path = trial_dir / f"{stem}.npy"
    if not csv_path.is_file() or not npy_path.is_file():
        raise FileNotFoundError(f"Missing input pair: {csv_path.name}, {npy_path.name}")

    chunks = np.load(npy_path, allow_pickle=False)
    gt = load_gt_action_x(csv_path, steps=chunks.shape[0])
    if len(gt) != chunks.shape[0]:
        raise ValueError(f"{csv_path.name}: GT rows do not match {npy_path.name}")
    prediction = fuse_chunks(chunks, k=k)[:, ACTION_X_AXIS]
    return gt, prediction


def style_axis(
    ax: plt.Axes,
    duration_s: float,
    values: np.ndarray,
    *,
    show_xlabel: bool,
) -> None:
    x_pad = duration_s * 0.018
    ax.set_xlim(-x_pad, duration_s + x_pad)

    y_low = float(np.nanmin(values))
    y_high = float(np.nanmax(values))
    y_span = max(y_high - y_low, np.finfo(float).eps)
    y_pad = y_span * 0.11
    ax.set_ylim(y_low - y_pad, y_high + y_pad)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.42, alpha=0.62)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#303030")
        ax.spines[side].set_linewidth(0.72)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=2.5,
        width=0.58,
        color="#303030",
        pad=2.5,
    )
    ax.set_ylabel(r"Motion ($x$) (m)", labelpad=4)
    if show_xlabel:
        ax.set_xlabel("Time (s)", labelpad=3)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)


def draw_pair(
    ax: plt.Axes,
    gt: np.ndarray,
    prediction: np.ndarray,
    model: str,
    duration_s: float,
    *,
    show_xlabel: bool,
) -> None:
    steps = min(len(gt), len(prediction))
    time_s = np.linspace(0.0, duration_s, steps)
    gt = gt[:steps]
    prediction = prediction[:steps]

    ax.plot(
        time_s,
        gt,
        color=COLORS["GT"],
        lw=LINEWIDTHS["GT"],
        solid_capstyle="round",
        zorder=5,
    )
    ax.plot(
        time_s,
        prediction,
        color=COLORS[model],
        lw=LINEWIDTHS[model],
        alpha=0.96,
        solid_capstyle="round",
        zorder=4,
    )
    ax.scatter(
        time_s[-1],
        gt[-1],
        s=20,
        color=COLORS["GT"],
        edgecolor="white",
        linewidth=0.55,
        zorder=6,
        clip_on=False,
    )
    ax.scatter(
        time_s[-1],
        prediction[-1],
        s=18,
        color=COLORS[model],
        edgecolor="white",
        linewidth=0.55,
        zorder=6,
        clip_on=False,
    )
    style_axis(ax, duration_s, np.r_[gt, prediction], show_xlabel=show_xlabel)



def load_trial_pairs(trial_id: int, k: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    trial_dir = trial_inputs_dir(trial_id)
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"Missing trial directory: {trial_dir}")

    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    reference_gt = None
    for model in MODEL_FILES:
        gt, prediction = load_model_action_x(trial_dir, model, k=k)
        if reference_gt is None:
            reference_gt = gt
        elif len(reference_gt) != len(gt) or not np.allclose(reference_gt, gt, atol=1e-8, rtol=0):
            raise ValueError(f"{trial_id}/{model}.csv does not contain the same GT Action X")
        pairs.append((model, gt, prediction))
    return pairs


def all_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=COLORS["GT"], lw=LINEWIDTHS["GT"], label="Demonstration"),
        Line2D([0], [0], color=COLORS["zeroshot"], lw=LINEWIDTHS["zeroshot"], label=DISPLAY_LABELS["zeroshot"]),
        Line2D([0], [0], color=COLORS["AB"], lw=LINEWIDTHS["AB"], label=DISPLAY_LABELS["AB"]),
        Line2D([0], [0], color=COLORS["ABC"], lw=LINEWIDTHS["ABC"], label=DISPLAY_LABELS["ABC"]),
        Line2D([0], [0], color=COLORS["ABCP"], lw=LINEWIDTHS["ABCP"], label=DISPLAY_LABELS["ABCP"]),
    ]


def save_figure(fig: plt.Figure, output: Path, dpi: int, pad_inches: float = 0.02) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_trial_stack(trial_id: int, output: Path, k: float, dpi: int) -> None:
    pairs = load_trial_pairs(trial_id, k=k)
    duration_s = video_duration_s(trial_id)
    n_panels = len(pairs)
    # Original 1 x 4 vertical panel layout.
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(3.15, 2.55 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    for idx, (ax, (model, gt, prediction)) in enumerate(zip(axes, pairs)):
        draw_pair(ax, gt, prediction, model, duration_s, show_xlabel=(idx == n_panels - 1))

    handles = all_legend_handles()
    legend_kwargs = dict(
        loc="upper center",
        frameon=False,
        handlelength=1.0,
        columnspacing=0.45,
        handletextpad=0.1,
        borderaxespad=0.0,
        labelspacing=0.05,
    )
    legend_top = fig.legend(
        handles=handles[:2],
        bbox_to_anchor=(0.5, 1.008),
        ncol=2,
        **legend_kwargs,
    )
    fig.add_artist(legend_top)
    fig.legend(
        handles=handles[2:],
        bbox_to_anchor=(0.5, 0.978),
        ncol=3,
        **legend_kwargs,
    )
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.048, top=0.955, hspace=0.04)
    save_figure(fig, output, dpi=dpi)


def plot_trial(trial_id: int, output_dir: Path | None, k: float, dpi: int) -> None:
    trial_dir = trial_inputs_dir(trial_id)
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"Missing trial directory: {trial_dir}")

    destination_dir = output_dir / f"trial_{trial_id}" if output_dir else trial_outputs_dir(trial_id)
    plot_trial_stack(trial_id, destination_dir / "figure_E_actionx_models.svg", k=k, dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial",
        type=int,
        choices=TRIAL_IDS,
        default=None,
        help="plot one trial id only; by default all trial ids are plotted",
    )
    parser.add_argument("--all", action="store_true", help="plot all available trial ids")
    parser.add_argument("--k", type=float, default=0.03, help="chunk fusion decay")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory, default: each trial's own trajectory folder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial_ids = TRIAL_IDS if args.all or args.trial is None else (args.trial,)
    for trial_id in trial_ids:
        plot_trial(trial_id, args.output, k=args.k, dpi=args.dpi)


if __name__ == "__main__":
    main()
