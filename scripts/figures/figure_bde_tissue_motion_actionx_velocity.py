"""Draw tissue motion and Action-X/phase velocity in one Matplotlib figure.

Unlike image concatenation, both panels are rendered onto axes owned by the
same Figure and are written by a single ``savefig`` call.

Example:
    python -m scripts.figures.figure_bde_tissue_motion_actionx_velocity --trial 17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from . import figure_b_tissue_motion as tissue
from . import figure_de_actionx_phase_velocity as action_velocity
from . import figure_e_actionx_pairwise as actionx


TRIAL_IDS = tissue.TRIAL_IDS
DEFAULT_TRIAL = tissue.DEFAULT_TRIAL_ID
FIG_WIDTH = 7.25
FIG_HEIGHT = 3.72

# Absolute layout in inches.  Keeping these values physical makes the spacing
# stable when DPI changes and leaves independent room for the upper phase
# brackets and the lower panel's two legend rows.
LEFT_IN = 0.69
RIGHT_IN = 0.14
# Pull the upper panel down so the inter-panel whitespace is one third
# smaller, while leaving both axes and the lower legend rows unchanged.
TOP_AX_BOTTOM_IN = 2.48
TOP_AX_HEIGHT_IN = 0.72
BOTTOM_AX_BOTTOM_IN = 0.50
BOTTOM_AX_HEIGHT_IN = 1.22
LOWER_LEGEND_ROW1_Y_IN = 2.20
LOWER_LEGEND_ROW2_Y_IN = 2.02


def render(
    trial_id: int,
    output: Path,
    *,
    model: str = tissue.DEFAULT_MODEL,
    fps: float = tissue.DEFAULT_FPS,
    k: float = 0.03,
    dpi: int = 600,
    palette: str = action_velocity.DEFAULT_PALETTE,
    caption: bool = False,
) -> None:
    """Render both panels natively into one figure and save it once."""
    motion = tissue.load_motion(trial_id, model=model, fps=fps)
    pairs = actionx.load_trial_pairs(trial_id, k=k)
    duration_s = actionx.video_duration_s(trial_id)
    steps = min(len(pairs[0][1]), min(len(prediction) for _, _, prediction in pairs))
    spans = action_velocity.load_phase_spans(trial_id, steps=steps)
    action_velocity.apply_palette(palette)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="white")
    axes_width = 1.0 - (LEFT_IN + RIGHT_IN) / FIG_WIDTH
    left = LEFT_IN / FIG_WIDTH
    top_ax = fig.add_axes(
        [left, TOP_AX_BOTTOM_IN / FIG_HEIGHT, axes_width, TOP_AX_HEIGHT_IN / FIG_HEIGHT]
    )
    bottom_ax = fig.add_axes(
        [left, BOTTOM_AX_BOTTOM_IN / FIG_HEIGHT, axes_width, BOTTOM_AX_HEIGHT_IN / FIG_HEIGHT]
    )

    top_ax.plot(
        motion.time_s,
        motion.motion,
        color=tissue.MOTION_COLOR,
        linewidth=1.9,
        solid_capstyle="round",
        zorder=4,
    )
    tissue.style_axis(top_ax, motion)
    tissue.draw_phase_arrows(top_ax, motion)
    top_ax.legend(
        handles=[
            Patch(
                facecolor=tissue.BG_INHALATION,
                edgecolor="none",
                alpha=tissue.BG_ALPHA,
                label="Inspiration",
            ),
            Patch(
                facecolor=tissue.BG_EXHALATION,
                edgecolor="none",
                alpha=tissue.BG_ALPHA,
                label="Expiration",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.988, 0.985),
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.92,
        fontsize=tissue.FONT_SIZE,
        handlelength=1.15,
        handleheight=0.75,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.3,
        borderpad=0.3,
    )

    action_velocity.draw_panel(
        bottom_ax,
        pairs,
        duration_s,
        spans,
        caption=caption,
    )
    legend_kwargs = dict(
        loc="upper center",
        frameon=False,
        handletextpad=0.34,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    fig.legend(
        handles=action_velocity.curve_legend_handles(),
        bbox_to_anchor=(0.5, LOWER_LEGEND_ROW1_Y_IN / FIG_HEIGHT),
        ncol=5,
        handlelength=1.5,
        **legend_kwargs,
    )
    bar_handles = action_velocity.bar_legend_handles()
    fig.legend(
        handles=bar_handles,
        bbox_to_anchor=(0.5, LOWER_LEGEND_ROW2_Y_IN / FIG_HEIGHT),
        ncol=len(bar_handles),
        handlelength=1.3,
        **legend_kwargs,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def output_for_trial(trial_id: int, output: Path | None, multiple: bool) -> Path:
    if output is None:
        return tissue.trial_outputs_dir(trial_id) / "figure_BDE_tissue_motion_actionx_velocity.svg"
    if output.suffix:
        return output if not multiple else output.with_stem(f"{output.stem}_{trial_id}")
    return output / f"figure_BDE_tissue_motion_actionx_velocity_{trial_id}.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, choices=TRIAL_IDS, default=DEFAULT_TRIAL)
    parser.add_argument("--all", action="store_true", help="plot all trial ids")
    parser.add_argument("--model", default=tissue.DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=tissue.DEFAULT_FPS)
    parser.add_argument("--k", type=float, default=0.03, help="chunk fusion decay")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument(
        "--palette",
        choices=sorted(action_velocity.PALETTES),
        default=action_velocity.DEFAULT_PALETTE,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial_ids = TRIAL_IDS if args.all else (args.trial,)
    for trial_id in trial_ids:
        render(
            trial_id,
            output_for_trial(trial_id, args.output, multiple=len(trial_ids) > 1),
            model=args.model,
            fps=args.fps,
            k=args.k,
            dpi=args.dpi,
            palette=args.palette,
            caption=args.caption,
        )


if __name__ == "__main__":
    main()
