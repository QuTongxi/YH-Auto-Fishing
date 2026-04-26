from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class PercentRect:
    left: float
    top: float
    right: float
    bottom: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = round(self.left * width)
        y1 = round(self.top * height)
        x2 = round(self.right * width)
        y2 = round(self.bottom * height)

        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop rect: {(x1, y1, x2, y2)}")

        return x1, y1, x2, y2


def parse_rect(value: str) -> PercentRect:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Rect must have 4 comma-separated floats: {value}")
    return PercentRect(*parts)


def resize_keep_aspect(img: Image.Image, target_width: int) -> Image.Image:
    w, h = img.size
    target_height = round(target_width * h / w)
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def make_composite(
    input_path: Path,
    output_path: Path,
    size: int,
    fight_bar: PercentRect,
    wait_banner: PercentRect,
    skill_group: PercentRect,
) -> None:
    with Image.open(input_path) as img:
        img = img.convert("RGB")

    src_w, src_h = img.size

    rois = [
        fight_bar,
        wait_banner,
        skill_group,
    ]

    crops: list[Image.Image] = []
    for rect in rois:
        crop_box = rect.to_pixels(src_w, src_h)
        crops.append(img.crop(crop_box))

    canvas = Image.new("RGB", (size, size), (0, 0, 0))

    inner_width = round(size * 0.95)
    gap = round(size * 0.056)

    resized = [resize_keep_aspect(crop, inner_width) for crop in crops]

    total_h = sum(part.height for part in resized) + gap * (len(resized) - 1)
    y = max(0, (size - total_h) // 2)

    for part in resized:
        x = (size - part.width) // 2
        canvas.paste(part, (x, y))
        y += part.height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input png file.")
    parser.add_argument("-o", "--output", default="./training/", help="Output directory.")
    parser.add_argument("--size", type=int, default=320, help="Output canvas size.")

    parser.add_argument("--fight-bar", required=True)
    parser.add_argument("--wait-banner", required=True)
    parser.add_argument("--skill-group", required=True)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = output_dir / input_path.name

    make_composite(
        input_path=input_path,
        output_path=output_path,
        size=args.size,
        fight_bar=parse_rect(args.fight_bar),
        wait_banner=parse_rect(args.wait_banner),
        skill_group=parse_rect(args.skill_group),
    )


if __name__ == "__main__":
    main()