"""
将 merged mask 区域在原图上替换为统一颜色，用于可视化分割结果。

输入: frame_{id}.png + mask_{id}_merged.png
输出: frame_{id}_masked.png
"""

import argparse
import re
from pathlib import Path

from PIL import Image


FRAME_PATTERN = re.compile(r"^frame_(\d+)\.png$")


def parse_color(value: str) -> tuple[int, int, int]:
    if value.startswith("#") and len(value) == 7:
        return (
            int(value[1:3], 16),
            int(value[3:5], 16),
            int(value[5:7], 16),
        )
    parts = [int(p) for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"无效颜色格式: {value}")
    return tuple(parts)


def apply_mask(
    frame_path: Path,
    mask_path: Path,
    color: tuple[int, int, int],
    output_path: Path | None = None,
) -> Path:
    frame = Image.open(frame_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if frame.size != mask.size:
        mask = mask.resize(frame.size, Image.Resampling.NEAREST)

    result = frame.copy()
    overlay = Image.new("RGB", frame.size, color)
    result.paste(overlay, mask=mask)

    if output_path is None:
        frame_id = frame_path.stem.replace("frame_", "")
        output_path = frame_path.parent / f"frame_{frame_id}_masked.png"

    result.save(output_path)
    return output_path


def apply_all(
    input_dir: Path,
    color: tuple[int, int, int],
    mask_suffix: str = "merged",
) -> list[Path]:
    written: list[Path] = []
    for frame_path in sorted(input_dir.glob("frame_*.png")):
        if frame_path.name.endswith("_masked.png"):
            continue
        match = FRAME_PATTERN.match(frame_path.name)
        if match is None:
            continue
        frame_id = match.group(1)
        mask_path = input_dir / f"mask_{frame_id}_{mask_suffix}.png"
        if not mask_path.exists():
            print(f"跳过 {frame_path.name}：未找到 {mask_path.name}")
            continue
        out = apply_mask(frame_path, mask_path, color)
        written.append(out)
        print(f"已生成 {out.name}（{frame_path.name} + {mask_path.name}）")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用统一颜色块替换原图中 mask 标记的器械区域"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path(__file__).parent,
        help="数据目录（默认: 脚本所在目录）",
    )
    parser.add_argument(
        "--frame",
        type=Path,
        help="单张原图路径（与 --mask 一起使用）",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        help="单张 merged mask 路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出路径（仅单张模式）",
    )
    parser.add_argument(
        "-c",
        "--color",
        default="0,255,0",
        help="替换颜色，RGB 如 0,255,0 或十六进制 #00FF00（默认: 绿色）",
    )
    parser.add_argument(
        "--mask-suffix",
        default="merged",
        help="mask 文件名后缀（默认: merged）",
    )
    args = parser.parse_args()
    color = parse_color(args.color)
    input_dir = args.input_dir.resolve()

    if args.frame is not None:
        mask_path = args.mask
        if mask_path is None:
            frame_id = args.frame.stem.replace("frame_", "")
            mask_path = input_dir / f"mask_{frame_id}_{args.mask_suffix}.png"
        out = apply_mask(args.frame, mask_path, color, args.output)
        print(f"已生成 {out}")
        return

    apply_all(input_dir, color, args.mask_suffix)


if __name__ == "__main__":
    main()
