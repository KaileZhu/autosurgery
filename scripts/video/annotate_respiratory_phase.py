"""Add a prominent Inspiration/Expiration label to the original trial video.

The respiratory phase is derived from the same motion spans used by Figure B.
This creates a separate output and never overwrites the curve-overlay video.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: imageio-ffmpeg. Install it with "
        "`python -m pip install imageio-ffmpeg`."
    ) from exc

from ..figures.figure_b_tissue_motion import load_motion, trial_inputs_dir


ROOT = Path(__file__).resolve().parents[2]
INSPIRATION_COLOR = (92, 232, 112, 255)
EXPIRATION_COLOR = (79, 181, 255, 255)


def default_video(trial: int) -> Path:
    candidates = [
        path
        for path in (ROOT / "data" / "trials" / f"trial_{trial}" / "video").glob("*.mp4")
        if "_with_" not in path.stem and "_phase_" not in path.stem
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one original MP4 for trial {trial}, found {len(candidates)}"
        )
    return candidates[0]


def bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def phase_for_sample(spans: list[tuple[int, int, bool]], sample: int) -> bool:
    """Return True for inspiration and False for expiration."""
    for start, end, rising in spans:
        if start <= sample < end:
            return rising
    return spans[-1][2]


def draw_phase_label(frame: np.ndarray, inspiration: bool) -> np.ndarray:
    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    font = bold_font(max(58, round(height * 0.078)))
    label = "Inspiration" if inspiration else "Expiration"
    color = INSPIRATION_COLOR if inspiration else EXPIRATION_COLOR

    margin_x = round(width * 0.028)
    margin_y = round(height * 0.045)
    bbox = draw.textbbox((0, 0), label, font=font, stroke_width=2)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = round(height * 0.022)
    pad_y = round(height * 0.014)
    left = margin_x
    bottom = height - margin_y
    top = bottom - text_h - 2 * pad_y
    right = left + text_w + 2 * pad_x

    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=round(height * 0.014),
        fill=(0, 0, 0, 168),
        outline=color,
        width=max(3, round(height * 0.004)),
    )
    draw.text(
        (left + pad_x, top + pad_y - bbox[1]),
        label,
        font=font,
        fill=color,
        stroke_width=max(2, round(height * 0.0025)),
        stroke_fill=(0, 0, 0, 255),
    )
    return np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


def render(source: Path, output: Path, trial: int, model: str, data_fps: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frames_total, duration = imageio_ffmpeg.count_frames_and_secs(str(source))
    reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    fps = float(metadata["fps"])
    frames_total = frames_total or max(1, round(duration * fps))
    motion = load_motion(trial, model=model, fps=data_fps)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase_label_") as temp_dir:
        silent = Path(temp_dir) / "silent.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(silent), (width, height), fps=fps, codec="libx264",
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
            output_params=["-crf", "18", "-preset", "medium"],
        )
        writer.send(None)
        written = 0
        try:
            for index, raw in enumerate(reader):
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                progress = index / max(frames_total - 1, 1)
                sample = min(round(progress * (len(motion.motion) - 1)), len(motion.motion) - 1)
                writer.send(draw_phase_label(frame, phase_for_sample(motion.spans, sample)))
                written += 1
                if written % max(round(fps), 1) == 0:
                    print(f"Rendered {written}/{frames_total} frames", end="\r", flush=True)
        finally:
            writer.close()
            reader.close()

        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(silent), "-i", str(source),
                "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
                "-t", f"{written / fps:.9f}", str(output),
            ],
            check=True,
        )

    print(f"\nSaved: {output}")
    print(f"Video: {width}x{height}, {fps:.3f} fps, {written} frames")
    print(f"Phase source: {trial_inputs_dir(trial) / (model + '.csv')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, default=17)
    parser.add_argument("--video", type=Path, help="original input MP4")
    parser.add_argument("--output", type=Path, help="separate output MP4")
    parser.add_argument("--model", default="ABCP")
    parser.add_argument("--data-fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = (args.video or default_video(args.trial)).resolve()
    output = (
        args.output.resolve()
        if args.output
        else source.with_name(f"{source.stem}_phase_labels.mp4")
    )
    render(source, output, args.trial, args.model, args.data_fps)


if __name__ == "__main__":
    main()
