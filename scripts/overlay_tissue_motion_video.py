"""Overlay a progressively revealed tissue-motion curve on a trial video.

The last video frame reveals the last motion sample, so the video and curve
always finish together even if their nominal durations differ slightly.

Example:
    python scripts/overlay_tissue_motion_video.py --trial 17
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: imageio-ffmpeg. Install it with "
        "`python -m pip install imageio-ffmpeg`."
    ) from exc

from figure_B_tissue_motion import load_motion, trial_inputs_dir


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRIAL = 17


def default_video(trial: int) -> Path:
    videos = sorted(
        path
        for path in (ROOT / "data" / "trials" / f"trial_{trial}" / "video").glob("*.mp4")
        if "_with_" not in path.stem and "_phase_" not in path.stem
    )
    if len(videos) != 1:
        raise FileNotFoundError(
            f"Expected exactly one MP4 for trial {trial}, found {len(videos)}"
        )
    return videos[0]


def rgba_chart(data, reveal: int, width: int, height: int, dpi: int = 100) -> np.ndarray:
    """Render an expanding white trace that always fills the black panel."""
    fig = plt.Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    fig.patch.set_alpha(1.0)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")

    stop = min(max(reveal + 1, 1), len(data.motion))
    ax.plot(data.time_s[:stop], data.motion[:stop], color="white", lw=2.4)

    ymin = min(0.0, float(np.min(data.motion)))
    ymax = max(1.0, float(np.max(data.motion)))
    pad = max(0.04 * (ymax - ymin), 0.02)
    # Rescale the visible time range on every frame.  The newest sample stays
    # at the right edge while the existing trace contracts towards the left.
    current_end = data.time_s[max(stop - 1, 0)]
    minimum_span = max(data.time_s[1] - data.time_s[0], 1e-6)
    ax.set(
        xlim=(data.time_s[0], max(current_end, data.time_s[0] + minimum_span)),
        ylim=(ymin - pad, ymax + pad),
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0.09, right=0.965, bottom=0.14, top=0.96)

    # Minimal arrow axes: no ticks, numbers, labels, or surrounding box.
    arrow = dict(arrowstyle="-|>", color="white", linewidth=1.8, mutation_scale=12)
    ax.annotate(
        "", xy=(1.025, 0), xytext=(-0.025, 0), xycoords="axes fraction",
        arrowprops=arrow, annotation_clip=False,
    )
    ax.annotate(
        "", xy=(0, 1.035), xytext=(0, -0.025), xycoords="axes fraction",
        arrowprops=arrow, annotation_clip=False,
    )
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def alpha_composite(frame: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """Composite an RGBA overlay into an RGB frame in place."""
    h, w = overlay.shape[:2]
    rgb = overlay[:, :, :3].astype(np.float32)
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    target = frame[y : y + h, x : x + w].astype(np.float32)
    frame[y : y + h, x : x + w] = (rgb * alpha + target * (1.0 - alpha)).astype(np.uint8)


def render_video(
    source: Path,
    output: Path,
    trial: int,
    model: str,
    data_fps: float,
    chart_width_fraction: float,
    chart_height_fraction: float,
    margin_fraction: float,
    speed: float,
) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frames_total, duration = imageio_ffmpeg.count_frames_and_secs(str(source))
    reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    source_fps = float(metadata["fps"])
    output_fps = source_fps * speed
    if frames_total <= 0:
        frames_total = max(1, round(duration * source_fps))

    chart_w = max(240, int(width * chart_width_fraction))
    chart_h = max(140, int(height * chart_height_fraction))
    chart_w = min(chart_w, width)
    chart_h = min(chart_h, height)
    margin = max(0, int(min(width, height) * margin_fraction))
    x = min(margin, width - chart_w)
    y = max(0, height - chart_h - margin)

    data = load_motion(trial, model=model, fps=data_fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="motion_overlay_") as temp_dir:
        silent_path = Path(temp_dir) / "silent.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(silent_path), (width, height), fps=output_fps, codec="libx264",
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "18", "-preset", "medium"],
        )
        writer.send(None)
        written = 0
        try:
            for index, raw in enumerate(reader):
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                progress = index / max(frames_total - 1, 1)
                reveal = min(round(progress * (len(data.motion) - 1)), len(data.motion) - 1)
                chart = rgba_chart(data, reveal, chart_w, chart_h)
                alpha_composite(frame, chart, x, y)
                writer.send(frame)
                written += 1
                if written % max(round(source_fps), 1) == 0:
                    print(f"Rendered {written}/{frames_total} frames", end="\r", flush=True)
        finally:
            writer.close()
            reader.close()

        # Copy audio from the original when present; '?' makes the audio stream optional.
        command = [ffmpeg, "-y", "-i", str(silent_path), "-i", str(source)]
        if speed == 1.0:
            command += ["-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac"]
        else:
            command += [
                "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                "-filter:a", f"atempo={speed:g},apad", "-c:a", "aac",
            ]
        # An explicit duration is robust when the padded audio filter is
        # effectively infinite and preserves every accelerated video frame.
        command += ["-t", f"{written / output_fps:.9f}", str(output)]
        subprocess.run(command, check=True)

    print(f"\nSaved: {output}")
    print(
        f"Video: {width}x{height}, {output_fps:.3f} fps, {written} frames, "
        f"{speed:g}x speed"
    )
    print(f"Motion samples: {len(data.motion)} from {trial_inputs_dir(trial) / (model + '.csv')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, default=DEFAULT_TRIAL)
    parser.add_argument("--video", type=Path, help="input MP4; defaults to the trial video")
    parser.add_argument("--output", type=Path, help="output MP4 path")
    parser.add_argument("--model", default="ABCP", help="motion CSV stem (default: ABCP)")
    parser.add_argument("--data-fps", type=float, default=30.0, help="sampling rate used by the curve time axis")
    parser.add_argument("--chart-width", type=float, default=0.68, help="chart width / video width")
    parser.add_argument("--chart-height", type=float, default=0.46, help="chart height / video height")
    parser.add_argument("--margin", type=float, default=0.018, help="edge margin / shorter video side")
    parser.add_argument("--speed", type=float, default=5.0, help="output playback speed (default: 5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.video or default_video(args.trial)
    output = args.output or source.with_name(f"{source.stem}_with_tissue_motion.mp4")
    for name, value in (("chart width", args.chart_width), ("chart height", args.chart_height)):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if args.speed <= 0:
        raise ValueError("speed must be positive")
    render_video(
        source.resolve(), output.resolve(), args.trial, args.model, args.data_fps,
        args.chart_width, args.chart_height, args.margin, args.speed,
    )


if __name__ == "__main__":
    main()
