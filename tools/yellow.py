from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


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


# =========================
# Config to tune
# =========================

FIGHT_BAR = PercentRect(0.300, 0.045, 0.700, 0.090)

GREEN_HSV_LOWER = np.array((45, 70, 80), dtype=np.uint8)
GREEN_HSV_UPPER = np.array((95, 255, 255), dtype=np.uint8)

# 你主要调这里。
# 原始偏宽松版本：
# YELLOW_HSV_LOWER = np.array((16, 35, 120), dtype=np.uint8)
# YELLOW_HSV_UPPER = np.array((45, 255, 255), dtype=np.uint8)

# 更亮、更纯的黄色版本：
YELLOW_HSV_LOWER = np.array((20, 110, 190), dtype=np.uint8)
YELLOW_HSV_UPPER = np.array((38, 255, 255), dtype=np.uint8)

YELLOW_BAND_PAD = 8
YELLOW_MAX_GROUP_WIDTH = 12
YELLOW_COL_MIN_PIXELS = 2
YELLOW_COLUMN_PAD = 1
YELLOW_VERTICAL_CLOSE_KERNEL = 7


def yellow_search_band(
    roi_h: int,
    green_box: Optional[tuple[int, int, int, int, int]],
) -> tuple[int, int]:
    if green_box is None:
        return 0, roi_h

    _, gy, _, gh, _ = green_box
    y1 = max(0, gy - YELLOW_BAND_PAD)
    y2 = min(roi_h, gy + gh + YELLOW_BAND_PAD)
    return y1, y2


def pick_best_yellow_group(
    yellow_mask: np.ndarray,
    green_box: Optional[tuple[int, int, int, int, int]],
) -> Optional[tuple[int, int]]:
    roi_h, _ = yellow_mask.shape[:2]
    y1, y2 = yellow_search_band(roi_h, green_box)
    band = yellow_mask[y1:y2, :]
    if band.size == 0:
        return None

    col_count = np.count_nonzero(band, axis=0)
    min_col_pixels = max(YELLOW_COL_MIN_PIXELS, int((y2 - y1) * 0.06))
    active_cols = np.where(col_count >= min_col_pixels)[0]
    if active_cols.size == 0:
        return None

    groups: list[tuple[int, int]] = []
    start = int(active_cols[0])
    prev = int(active_cols[0])
    for col in active_cols[1:]:
        col = int(col)
        if col <= prev + 1:
            prev = col
        else:
            groups.append((start, prev))
            start = col
            prev = col
    groups.append((start, prev))

    green_center = None if green_box is None else green_box[0] + green_box[2] / 2.0
    candidates: list[tuple[float, tuple[int, int]]] = []
    for g_start, g_end in groups:
        width = g_end - g_start + 1
        if width > YELLOW_MAX_GROUP_WIDTH:
            continue

        group_sum = float(col_count[g_start:g_end + 1].sum())
        center_x = (g_start + g_end) / 2.0

        distance_penalty = 0.0
        if green_center is not None:
            distance_penalty = abs(center_x - green_center) * 0.12

        score = group_sum - width * 1.6 - distance_penalty
        candidates.append((score, (g_start, g_end)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def refine_yellow_mask(
    yellow_mask: np.ndarray,
    green_box: Optional[tuple[int, int, int, int, int]],
) -> np.ndarray:
    roi_h, roi_w = yellow_mask.shape[:2]
    y1, y2 = yellow_search_band(roi_h, green_box)
    best_group = pick_best_yellow_group(yellow_mask, green_box)
    if best_group is None:
        return yellow_mask

    g_start, g_end = best_group
    keep_x1 = max(0, g_start - YELLOW_COLUMN_PAD)
    keep_x2 = min(roi_w, g_end + YELLOW_COLUMN_PAD + 1)

    refined = np.zeros_like(yellow_mask)
    refined[y1:y2, keep_x1:keep_x2] = yellow_mask[y1:y2, keep_x1:keep_x2]

    # 沿竖直方向连通，避免黄条在 mask 中断裂成小块。
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        np.ones((YELLOW_VERTICAL_CLOSE_KERNEL, 1), np.uint8),
    )

    if green_box is not None:
        _, gy, _, gh, _ = green_box
        track_y1 = max(0, gy - 1)
        track_y2 = min(roi_h, gy + gh + 1)
        for x in range(keep_x1, keep_x2):
            column = refined[track_y1:track_y2, x]
            if np.count_nonzero(column) >= 2:
                refined[track_y1:track_y2, x] = 255

    return refined


def build_yellow_mask(roi_bgr: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    hsv_mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)

    b = roi_bgr[:, :, 0]
    g = roi_bgr[:, :, 1]
    r = roi_bgr[:, :, 2]

    b16 = b.astype(np.int16)
    g16 = g.astype(np.int16)
    r16 = r.astype(np.int16)

    # BGR 二次过滤。你后面也可以继续调这里。
    bgr_mask = (
        (r > 130)
        & (g > 100)
        & (b < 190)
        & (r16 > b16 + 25)
        & (g16 > b16 + 8)
        & (np.abs(r16 - g16) < 110)
    )

    return (hsv_mask & (bgr_mask.astype(np.uint8) * 255)).astype(np.uint8)


def find_green_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes: list[tuple[int, int, int, int, int]] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if w < 35:
            continue
        if h < 6:
            continue
        if w / max(h, 1) < 2.5:
            continue
        if area < 200:
            continue

        boxes.append((x, y, w, h, area))

    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes


def find_yellow_boxes(
    mask: np.ndarray,
    green_box: Optional[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    roi_h = mask.shape[0]
    filtered_mask = np.zeros_like(mask)

    if green_box is None:
        y1, y2 = 0, roi_h
        gy = 0
        gh = roi_h
    else:
        _, gy, _, gh, _ = green_box
        # 黄条应与轨道/绿条高度接近，带宽不要过宽，避免把云朵纳入候选。
        y1, y2 = yellow_search_band(roi_h, green_box)

    filtered_mask[y1:y2, :] = mask[y1:y2, :]

    contours, _ = cv2.findContours(
        filtered_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes: list[tuple[int, int, int, int, int]] = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if w < 1:
            continue
        if w > 12:
            continue
        if h < max(8, int(gh * 0.55)):
            continue
        if h / max(w, 1) < 2.0:
            continue

        if green_box is not None:
            green_y1 = gy
            green_y2 = gy + gh

            overlap_y1 = max(y, green_y1)
            overlap_y2 = min(y + h, green_y2)
            if overlap_y2 <= overlap_y1:
                continue

            overlap_ratio = (overlap_y2 - overlap_y1) / max(h, 1)
            if overlap_ratio < 0.35:
                continue

            center_y = y + h / 2.0
            green_center_y = gy + gh / 2.0
            if abs(center_y - green_center_y) > max(10.0, gh * 0.75):
                continue

        if area < 6:
            continue

        boxes.append((x, y, w, h, area))

    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes


def find_selected_yellow_x(
    yellow_boxes: list[tuple[int, int, int, int, int]],
    yellow_mask: np.ndarray,
    green_box: Optional[tuple[int, int, int, int, int]],
) -> Optional[int]:
    green_center = None if green_box is None else green_box[0] + green_box[2] / 2.0
    candidates: list[tuple[float, int]] = []

    for x, _, w, h, area in yellow_boxes:
        center_x = int(round(x + w / 2.0))

        distance_penalty = 0.0
        if green_center is not None:
            distance_penalty = abs(center_x - green_center) * 0.2

        # 细、竖且高度更大的候选更可信。
        score = area + h * 2.0 - w * 3.0 - distance_penalty
        candidates.append((score, center_x))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    # 兜底：候选框为空时，回退到窄列组选择。
    best_group = pick_best_yellow_group(yellow_mask, green_box)
    if best_group is None:
        return None

    g_start, g_end = best_group
    return int(round((g_start + g_end) / 2.0))


def put_text(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input png image.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read image: {input_path}")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = FIGHT_BAR.to_pixels(w, h)

    roi = frame[y1:y2, x1:x2].copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, GREEN_HSV_LOWER, GREEN_HSV_UPPER)
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )

    yellow_mask_raw = build_yellow_mask(roi, hsv)
    yellow_mask_raw = cv2.morphologyEx(
        yellow_mask_raw,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
    )
    green_boxes = find_green_boxes(green_mask)
    green_box = green_boxes[0] if green_boxes else None
    yellow_mask = refine_yellow_mask(yellow_mask_raw, green_box)

    yellow_boxes = find_yellow_boxes(yellow_mask, green_box)
    selected_yellow_x = find_selected_yellow_x(
        yellow_boxes=yellow_boxes,
        yellow_mask=yellow_mask,
        green_box=green_box,
    )

    full_vis = frame.copy()
    cv2.rectangle(full_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    put_text(full_vis, "FIGHT_BAR", x1, max(20, y1 - 8))

    roi_vis = roi.copy()

    for idx, (gx, gy, gw, gh, area) in enumerate(green_boxes):
        color = (0, 255, 0)
        thickness = 2 if idx == 0 else 1
        cv2.rectangle(roi_vis, (gx, gy), (gx + gw, gy + gh), color, thickness)
        put_text(roi_vis, f"G{idx} area={area}", gx, max(16, gy - 4))

    for idx, (yx, yy, yw, yh, area) in enumerate(yellow_boxes[:20]):
        color = (0, 255, 255)
        cv2.rectangle(roi_vis, (yx, yy), (yx + yw, yy + yh), color, 1)
        put_text(roi_vis, f"Y{idx}", yx, max(16, yy - 4))

    if selected_yellow_x is not None:
        cv2.line(
            roi_vis,
            (selected_yellow_x, 0),
            (selected_yellow_x, roi_vis.shape[0] - 1),
            (0, 0, 255),
            2,
        )
        put_text(roi_vis, f"selected_yellow_x={selected_yellow_x}", 8, 22)

    if green_box is not None:
        gx, gy, gw, gh, _ = green_box
        safe_left = int(round(gx + gw * 0.25))
        safe_right = int(round(gx + gw * 0.75))

        cv2.line(roi_vis, (safe_left, 0), (safe_left, roi_vis.shape[0] - 1), (255, 0, 0), 2)
        cv2.line(roi_vis, (safe_right, 0), (safe_right, roi_vis.shape[0] - 1), (255, 0, 0), 2)

        put_text(roi_vis, f"safe=({safe_left},{safe_right})", 8, 44)

    out_dir = Path(__file__).parent.parent / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_full = out_dir / f"{input_path.stem}_full.png"
    out_roi = out_dir / f"{input_path.stem}_roi_detect.png"
    out_green_mask = out_dir / f"{input_path.stem}_green_mask.png"
    out_yellow_mask = out_dir / f"{input_path.stem}_yellow_mask.png"

    cv2.imwrite(str(out_full), full_vis)
    cv2.imwrite(str(out_roi), roi_vis)
    cv2.imwrite(str(out_green_mask), green_mask)
    cv2.imwrite(str(out_yellow_mask), yellow_mask)

    print(f"Input: {input_path}")
    print(f"Image size: {w}x{h}")
    print(f"FIGHT_BAR pixels: ({x1}, {y1}) - ({x2}, {y2})")
    print(f"Green boxes: {len(green_boxes)}")
    print(f"Yellow boxes: {len(yellow_boxes)}")
    print(f"Selected yellow x: {selected_yellow_x}")
    print()
    print(f"Saved: {out_full}")
    print(f"Saved: {out_roi}")
    print(f"Saved: {out_green_mask}")
    print(f"Saved: {out_yellow_mask}")


if __name__ == "__main__":
    main()