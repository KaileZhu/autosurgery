"""Extract a respiratory tissue-displacement curve from a video.

Dense optical flow is reduced to one robust 2-D motion vector per frame. PCA
finds the dominant tissue-motion direction, the projected flow is integrated
to displacement, and slow optical-flow drift is removed with a respiratory
band-pass filter.

Example:
    python scripts/extract_tissue_motion.py data/invivo.mp4
    python scripts/extract_tissue_motion.py data/invivo.mp4 --roi 280,100,650,500

The output CSV contains a normalized ``motion`` column compatible with the
existing overlay scripts, plus intermediate physical/diagnostic columns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import signal


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


def parse_roi(text: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    """Return x, y, w, h in source-video pixels."""
    if text is None:
        # Avoid black borders and the da Vinci UI while retaining a broad
        # central tissue region. A hand-selected ROI is preferable when tools
        # move substantially.
        x, y = round(width * 0.12), round(height * 0.10)
        w, h = round(width * 0.76), round(height * 0.80)
        return x, y, w, h
    try:
        values = [int(part.strip()) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi must be x,y,w,h in pixels") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--roi must contain exactly x,y,w,h")
    x, y, w, h = values
    if x < 0 or y < 0 or w <= 1 or h <= 1 or x + w > width or y + h > height:
        raise ValueError(f"ROI {values} lies outside the {width}x{height} video")
    return x, y, w, h


def frame_gray(raw: bytes, info: VideoInfo, process_width: int) -> np.ndarray:
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)
    scale = min(1.0, process_width / info.width)
    width, height = max(2, round(info.width * scale)), max(2, round(info.height * scale))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def scaled_roi(
    roi: tuple[int, int, int, int], info: VideoInfo, shape: tuple[int, int]
) -> tuple[slice, slice]:
    x, y, w, h = roi
    sy, sx = shape[0] / info.height, shape[1] / info.width
    x0, x1 = round(x * sx), round((x + w) * sx)
    y0, y1 = round(y * sy), round((y + h) * sy)
    return slice(max(y0, 0), min(y1, shape[0])), slice(max(x0, 0), min(x1, shape[1]))


def robust_frame_vector(
    flow_u: np.ndarray,
    flow_v: np.ndarray,
    roi_slices: tuple[slice, slice],
    trim_percent: float,
) -> tuple[float, float, int]:
    """Robustly summarize coherent tissue flow and suppress fast tool pixels."""
    u = flow_u[roi_slices].ravel()
    v = flow_v[roi_slices].ravel()
    magnitude = np.hypot(u, v)
    finite = np.isfinite(magnitude)
    if finite.sum() < 20:
        return np.nan, np.nan, 0
    u, v, magnitude = u[finite], v[finite], magnitude[finite]

    # Discard almost-static/noisy pixels and the fastest pixels, which are
    # commonly instruments, specularities, smoke, or occlusion boundaries.
    low, high = np.percentile(magnitude, [trim_percent, 100.0 - trim_percent])
    keep = (magnitude >= low) & (magnitude <= high)
    u, v = u[keep], v[keep]
    if len(u) < 20:
        return np.nan, np.nan, int(len(u))

    center = np.array([np.median(u), np.median(v)])
    residual = np.hypot(u - center[0], v - center[1])
    mad = np.median(np.abs(residual - np.median(residual))) + 1e-8
    coherent = residual <= np.median(residual) + 2.5 * 1.4826 * mad
    return float(np.median(u[coherent])), float(np.median(v[coherent])), int(coherent.sum())


def fill_invalid(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not valid.any():
        raise RuntimeError("No valid optical-flow estimates were produced")
    indices = np.arange(len(values))
    return np.interp(indices, indices[valid], values[valid])


def respiratory_filter(
    displacement: np.ndarray, fps: float, low_hz: float, high_hz: float
) -> np.ndarray:
    nyquist = fps / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"Need 0 < low-hz < high-hz < Nyquist ({nyquist:g} Hz)"
        )
    sos = signal.butter(3, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(displacement) <= padlen:
        raise ValueError("Video is too short for zero-phase respiratory filtering")
    return signal.sosfiltfilt(sos, displacement)


def save_roi_preview(
    raw: bytes,
    info: VideoInfo,
    roi: tuple[int, int, int, int],
    output: Path,
) -> None:
    image = Image.fromarray(
        np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)
    )
    draw = ImageDraw.Draw(image)
    x, y, w, h = roi
    line_width = max(3, round(min(info.width, info.height) * 0.006))
    draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 80), width=line_width)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92)


def save_diagnostic(
    time_s: np.ndarray,
    projected_velocity: np.ndarray,
    raw_displacement: np.ndarray,
    filtered: np.ndarray,
    motion: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time_s, projected_velocity, lw=0.8)
    axes[0].set_ylabel("Projected flow\n(px/frame)")
    axes[1].plot(time_s, raw_displacement, lw=0.9, color="#777777", label="integrated")
    axes[1].plot(time_s, filtered, lw=1.2, color="#D55E00", label="band-pass")
    axes[1].set_ylabel("Displacement\n(processing px)")
    axes[1].legend(frameon=False)
    axes[2].plot(time_s, motion, lw=1.4, color="#0072B2")
    axes[2].set_ylabel("motion")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def extract(
    video: Path,
    output: Path,
    roi_text: str | None,
    process_width: int,
    trim_percent: float,
    low_hz: float,
    high_hz: float,
    diagnostic: Path,
    roi_preview: Path,
) -> None:
    frames_total, _ = imageio_ffmpeg.count_frames_and_secs(str(video))
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    info = VideoInfo(width, height, float(metadata["fps"]), int(frames_total))
    roi = parse_roi(roi_text, width, height)

    first_raw = next(reader)
    save_roi_preview(first_raw, info, roi, roi_preview)
    previous = frame_gray(first_raw, info, process_width)
    roi_slices = scaled_roi(roi, info, previous.shape)
    vectors: list[tuple[float, float]] = [(0.0, 0.0)]
    support: list[int] = [0]

    for frame_index, raw in enumerate(reader, start=1):
        current = frame_gray(raw, info, process_width)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=25,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        flow_u, flow_v = flow[..., 0], flow[..., 1]
        u, v, count = robust_frame_vector(flow_u, flow_v, roi_slices, trim_percent)
        vectors.append((u, v))
        support.append(count)
        previous = current
        if frame_index % max(round(info.fps), 1) == 0:
            print(
                f"Optical flow: {frame_index}/{info.frames - 1} frame pairs",
                end="\r",
                flush=True,
            )
    reader.close()

    vectors_array = np.asarray(vectors, dtype=np.float64)
    vectors_array[:, 0] = fill_invalid(vectors_array[:, 0])
    vectors_array[:, 1] = fill_invalid(vectors_array[:, 1])
    # Hampel-like temporal cleanup before PCA.
    kernel = max(3, 2 * round(info.fps * 0.1) + 1)
    for column in range(2):
        median = signal.medfilt(vectors_array[:, column], kernel_size=kernel)
        residual = vectors_array[:, column] - median
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-8
        outliers = np.abs(residual) > 5.0 * scale
        vectors_array[outliers, column] = median[outliers]

    centered = vectors_array[1:] - np.median(vectors_array[1:], axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    # Resolve PCA's arbitrary sign: positive displacement means upward on screen.
    if principal[1] > 0:
        principal *= -1
    projected = vectors_array @ principal
    projected -= np.median(projected)
    raw_displacement = np.cumsum(projected)
    filtered = respiratory_filter(raw_displacement, info.fps, low_hz, high_hz)
    amplitude = float(np.ptp(filtered))
    if amplitude <= np.finfo(float).eps:
        raise RuntimeError("Filtered tissue displacement has zero amplitude")
    motion = (filtered - np.min(filtered)) / amplitude
    time_s = np.arange(len(motion), dtype=np.float64) / info.fps

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time_s": time_s,
            "motion": motion,
            "displacement_px": filtered,
            "raw_displacement_px": raw_displacement,
            "projected_flow_px_per_frame": projected,
            "flow_x_px_per_frame": vectors_array[:, 0],
            "flow_y_px_per_frame": vectors_array[:, 1],
            "support_pixels": support,
        }
    ).to_csv(output, index=False)
    save_diagnostic(time_s, projected, raw_displacement, filtered, motion, diagnostic)

    explained = float(np.max(eigenvalues) / np.sum(eigenvalues)) if np.sum(eigenvalues) > 0 else 0.0
    scale = previous.shape[1] / info.width
    print(f"\nSaved CSV: {output}")
    print(f"Saved diagnostic: {diagnostic}")
    print(f"Saved ROI preview: {roi_preview}")
    print(
        f"Video: {info.width}x{info.height}, {info.fps:g} fps, {len(motion)} frames; "
        f"processing scale={scale:.3f}"
    )
    print(
        f"Principal direction (x,y)=({principal[0]:.4f}, {principal[1]:.4f}); "
        f"explained variance={explained:.1%}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, help="default: VIDEO_tissue_motion.csv")
    parser.add_argument("--roi", help="tissue ROI in source pixels: x,y,w,h")
    parser.add_argument("--process-width", type=int, default=320)
    parser.add_argument("--trim-percent", type=float, default=15.0)
    parser.add_argument("--low-hz", type=float, default=0.08)
    parser.add_argument("--high-hz", type=float, default=0.5)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--roi-preview", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if args.process_width < 64:
        raise ValueError("--process-width must be at least 64")
    if not 0 <= args.trim_percent < 45:
        raise ValueError("--trim-percent must be in [0, 45)")
    output = (args.output or video.with_name(f"{video.stem}_tissue_motion.csv")).resolve()
    diagnostic = (
        args.diagnostic or output.with_name(f"{output.stem}_diagnostic.png")
    ).resolve()
    roi_preview = (
        args.roi_preview or output.with_name(f"{output.stem}_roi.jpg")
    ).resolve()
    extract(
        video,
        output,
        args.roi,
        args.process_width,
        args.trim_percent,
        args.low_hz,
        args.high_hz,
        diagnostic,
        roi_preview,
    )


if __name__ == "__main__":
    main()
