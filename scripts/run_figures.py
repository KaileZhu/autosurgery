"""Run the repository's publication-figure scripts from one command.

Examples:
    python scripts/run_figures.py
    python scripts/run_figures.py --figures B C DE --trial 18
    python scripts/run_figures.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {
    "B": "figure_B_tissue_motion.py",
    "C": "figure_C_3d_trajectories.py",
    "D": "figure_D_motion_phase_velocity.py",
    "DE": "figure_DE_actionx_phase_velocity.py",
    "E": "figure_E_actionx_pairwise.py",
    "CE": "figure_CE_trajectory_actionx_1x2.py",
}
DEFAULT_FIGURES = ("B", "C", "DE", "D", "E")
TRIALS = (17, 18, 19)


def command_for(figure: str, trial: int) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / SCRIPTS[figure]), "--trial", str(trial)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=tuple(SCRIPTS),
        default=list(DEFAULT_FIGURES),
        help="figure groups to render (default: B C DE D E)",
    )
    parser.add_argument(
        "--trial",
        type=int,
        action="append",
        choices=TRIALS,
        help="trial to render; repeat for several (default: 17, 18, and 19)",
    )
    parser.add_argument("--list", action="store_true", help="print commands without running them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = tuple(args.trial) if args.trial else TRIALS
    commands = [command_for(figure, trial) for figure in args.figures for trial in trials]
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {subprocess.list2cmdline(command)}", flush=True)
        if not args.list:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
