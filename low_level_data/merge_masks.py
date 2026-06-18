"""
将同一帧的多个 obj mask 合并为一张二值 mask（像素取 max / 逻辑 OR）。

输入命名: mask_{frame_id}_obj{N}.png
输出命名: mask_{frame_id}_merged.png
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops


MASK_PATTERN = re.compile(r"^mask_(\d+)_obj(\d+)\.png$")


def merge_masks(input_dir: Path, output_suffix: str = "merged") -> list[Path]:
    groups: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(input_dir.glob("mask_*_obj*.png")):
        match = MASK_PATTERN.match(path.name)
        if match is None:
            continue
        frame_id = match.group(1)
        groups[frame_id].append(path)

    if not groups:
        print(f"未在 {input_dir} 中找到匹配的 mask 文件")
        return []

    written: list[Path] = []
    for frame_id, paths in sorted(groups.items()):
        merged = None
        for path in paths:
            img = Image.open(path).convert("L")
            if merged is None:
                merged = img
            else:
                merged = ImageChops.lighter(merged, img)

        out_path = input_dir / f"mask_{frame_id}_{output_suffix}.png"
        merged.save(out_path)
        written.append(out_path)
        obj_names = ", ".join(p.name for p in paths)
        print(f"已合并 {obj_names} -> {out_path.name}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并同一帧的 obj mask 为一张二值 mask"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path(__file__).parent,
        help="mask 文件所在目录（默认: 脚本所在目录）",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        default="merged",
        help="输出文件名后缀（默认: merged）",
    )
    args = parser.parse_args()
    merge_masks(args.input_dir.resolve(), args.suffix)


if __name__ == "__main__":
    main()
