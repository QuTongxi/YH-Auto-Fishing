from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


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

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid rect after conversion: "
                f"({x1}, {y1}, {x2}, {y2}) for image size {width}x{height}"
            )

        return x1, y1, x2, y2


# -----------------------------
# Flattened ROI configuration
# -----------------------------

# full screen rois
ROI_SPECS: dict[str, PercentRect] = {
    "start_skill_group.template_rect": PercentRect(0.690, 0.860, 0.975, 0.965),
    "wait_banner.template_rect": PercentRect(0.300, 0.210, 0.700, 0.280),
    "FIGHT_BAR_ROI": PercentRect(0.300, 0.045, 0.700, 0.090),
}

# 2560*1600 screen rois
ROI_SPECS: dict[str, PercentRect] = {
    "start_skill_group.template_rect": PercentRect(0.690, 0.860, 0.975, 0.965),
    "wait_banner.template_rect": PercentRect(0.300, 0.190, 0.700, 0.260),
    "FIGHT_BAR_ROI": PercentRect(0.300, 0.045, 0.700, 0.090),
}

ROI_COLORS: dict[str, tuple[int, int, int]] = {
    "start_skill_group.template_rect": (255, 80, 80),
    "wait_banner.template_rect": (80, 255, 80),
    "confirm_panel.template_rect": (80, 160, 255),
    "FIGHT_BAR_ROI": (255, 220, 80),
}


IMAGE_EXTS = {".png"}


def draw_rois(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as img:
        img = img.convert("RGB")

    width, height = img.size
    draw = ImageDraw.Draw(img)

    line_width = max(2, round(min(width, height) * 0.003))

    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, round(height * 0.018)))
    except OSError:
        font = ImageFont.load_default()

    for name, rect in ROI_SPECS.items():
        color = ROI_COLORS.get(name, (255, 0, 0))
        x1, y1, x2, y2 = rect.to_pixels(width, height)

        for offset in range(line_width):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        label = f"{name}: ({x1},{y1})-({x2},{y2})"

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_x = x1
        label_y = max(0, y1 - text_h - 6)

        draw.rectangle(
            [label_x, label_y, label_x + text_w + 8, label_y + text_h + 6],
            fill=(0, 0, 0),
        )
        draw.text(
            (label_x + 4, label_y + 3),
            label,
            fill=color,
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def iter_images(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        yield input_path
        return

    if input_path.is_dir():
        for path in sorted(input_path.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield path
        return

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input image file or image directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output image file or output directory.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        if output_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(
                "When input is a file, output should be an image file, "
                "for example: ./debug_roi.png"
            )

        draw_rois(input_path, output_path)
        print(f"Saved: {output_path}")
        return

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in iter_images(input_path):
            out_file = output_path / f"{img_path.stem}_roi{img_path.suffix}"
            draw_rois(img_path, out_file)
            count += 1

        print(f"Processed {count} images.")
        print(f"Output directory: {output_path}")
        return

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()