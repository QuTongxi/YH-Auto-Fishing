from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
import pydirectinput

from . import audit, config


@dataclass(frozen=True)
class FightDetection:
    green_x1: int
    green_x2: int
    green_y1: int
    green_y2: int
    yellow_x: int
    roi_abs: tuple[int, int, int, int]

    @property
    def green_w(self) -> int:
        return self.green_x2 - self.green_x1

    @property
    def safe_left(self) -> float:
        return self.green_x1 + self.green_w * config.FIGHT_SAFE_LEFT_RATIO

    @property
    def safe_right(self) -> float:
        return self.green_x1 + self.green_w * config.FIGHT_SAFE_RIGHT_RATIO


@dataclass(frozen=True)
class FightStep:
    status: Literal["active", "need_model_check"]
    detection: Optional[FightDetection]
    reason: str
    missing_count: int


class KeyHoldWorker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.target_key: Optional[str] = None
        self.hold_until = 0.0
        self.current_key: Optional[str] = None
        self.closed = False

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def hold(self, key: str, seconds: float) -> None:
        now = time.perf_counter()
        with self.lock:
            self.target_key = key
            self.hold_until = max(self.hold_until, now + seconds)

    def release(self) -> None:
        with self.lock:
            self.target_key = None
            self.hold_until = 0.0

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.target_key = None
            self.hold_until = 0.0

        self.thread.join(timeout=0.3)

        if self.current_key is not None:
            pydirectinput.keyUp(self.current_key)
            self.current_key = None

    def _loop(self) -> None:
        while True:
            with self.lock:
                closed = self.closed
                desired = self.target_key
                until = self.hold_until

            if closed:
                break

            now = time.perf_counter()
            if desired is None or now >= until:
                desired = None

            if desired != self.current_key:
                if self.current_key is not None:
                    pydirectinput.keyUp(self.current_key)

                if desired is not None:
                    pydirectinput.keyDown(desired)

                self.current_key = desired

            time.sleep(config.FIGHT_INPUT_LOOP_SLEEP_SEC)

        if self.current_key is not None:
            pydirectinput.keyUp(self.current_key)
            self.current_key = None


class FightDetector:
    def __init__(self) -> None:
        self.fight_rect = config.ROI[config.SCREEN_MODE]["FIGHT_BAR"]
        self.green_lower = np.array(config.GREEN_HSV_LOWER, dtype=np.uint8)
        self.green_upper = np.array(config.GREEN_HSV_UPPER, dtype=np.uint8)

    def detect(self, frame_bgr: np.ndarray) -> Optional[FightDetection]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.fight_rect.to_pixels(w, h)

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
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

        green_box = self._find_green_box(green_mask)
        if green_box is None:
            return None

        yellow_mask = self._build_yellow_mask(roi, hsv)
        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            np.ones((2, 2), np.uint8),
        )
        yellow_mask = self._refine_yellow_mask(yellow_mask, green_box)

        yellow_x = self._find_yellow_marker_x(
            yellow_mask=yellow_mask,
            green_box=green_box,
        )
        if yellow_x is None:
            return None

        gx, gy, gw, gh = green_box

        return FightDetection(
            green_x1=gx,
            green_x2=gx + gw,
            green_y1=gy,
            green_y2=gy + gh,
            yellow_x=yellow_x,
            roi_abs=(x1, y1, x2, y2),
        )

    @staticmethod
    def _build_yellow_mask(roi: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        hsv_lower = np.array(config.YELLOW_HSV_LOWER, dtype=np.uint8)
        hsv_upper = np.array(config.YELLOW_HSV_UPPER, dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        b = roi[:, :, 0]
        g = roi[:, :, 1]
        r = roi[:, :, 2]

        b16 = b.astype(np.int16)
        g16 = g.astype(np.int16)
        r16 = r.astype(np.int16)

        bgr_mask = (
            (r > 130)
            & (g > 100)
            & (b < 190)
            & (r16 > b16 + 25)
            & (g16 > b16 + 8)
            & (np.abs(r16 - g16) < 110)
        )

        return (hsv_mask & (bgr_mask.astype(np.uint8) * 255)).astype(np.uint8)

    @staticmethod
    def _find_green_box(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates: list[tuple[int, int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if w < 35:
                continue
            if h < 6:
                continue
            if w / max(h, 1) < 2.5:
                continue
            if area < 200:
                continue

            candidates.append((x, y, w, h, area))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[4], reverse=True)
        x, y, w, h, _ = candidates[0]
        return x, y, w, h

    @staticmethod
    def _find_yellow_marker_x(
        yellow_mask: np.ndarray,
        green_box: tuple[int, int, int, int],
    ) -> Optional[int]:
        yellow_boxes = FightDetector._find_yellow_boxes(yellow_mask, green_box)
        green_center = green_box[0] + green_box[2] / 2.0

        if yellow_boxes:
            candidates: list[tuple[float, int]] = []
            for x, _, w, h, area in yellow_boxes:
                center_x = int(round(x + w / 2.0))
                distance_penalty = abs(center_x - green_center) * 0.2
                score = area + h * 2.0 - w * 3.0 - distance_penalty
                candidates.append((score, center_x))

            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]

        best_group = FightDetector._pick_best_yellow_group(yellow_mask, green_box)
        if best_group is None:
            return None

        g_start, g_end = best_group
        return int(round((g_start + g_end) / 2.0))

    @staticmethod
    def _yellow_search_band(
        roi_h: int,
        green_box: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        _, gy, _, gh = green_box
        y1 = max(0, gy - config.YELLOW_BAND_PAD)
        y2 = min(roi_h, gy + gh + config.YELLOW_BAND_PAD)
        return y1, y2

    @staticmethod
    def _pick_best_yellow_group(
        yellow_mask: np.ndarray,
        green_box: tuple[int, int, int, int],
    ) -> Optional[tuple[int, int]]:
        roi_h, _ = yellow_mask.shape[:2]
        y1, y2 = FightDetector._yellow_search_band(roi_h, green_box)
        band = yellow_mask[y1:y2, :]
        if band.size == 0:
            return None

        col_count = np.count_nonzero(band, axis=0)
        min_col_pixels = max(config.YELLOW_COL_MIN_PIXELS, int((y2 - y1) * 0.06))
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

        green_center = green_box[0] + green_box[2] / 2.0
        candidates: list[tuple[float, tuple[int, int]]] = []
        for g_start, g_end in groups:
            width = g_end - g_start + 1
            if width > config.YELLOW_MAX_GROUP_WIDTH:
                continue

            group_sum = float(col_count[g_start:g_end + 1].sum())
            center_x = (g_start + g_end) / 2.0
            distance_penalty = abs(center_x - green_center) * 0.12
            score = group_sum - width * 1.6 - distance_penalty
            candidates.append((score, (g_start, g_end)))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _refine_yellow_mask(
        yellow_mask: np.ndarray,
        green_box: tuple[int, int, int, int],
    ) -> np.ndarray:
        roi_h, roi_w = yellow_mask.shape[:2]
        y1, y2 = FightDetector._yellow_search_band(roi_h, green_box)
        best_group = FightDetector._pick_best_yellow_group(yellow_mask, green_box)
        if best_group is None:
            return yellow_mask

        g_start, g_end = best_group
        keep_x1 = max(0, g_start - config.YELLOW_COLUMN_PAD)
        keep_x2 = min(roi_w, g_end + config.YELLOW_COLUMN_PAD + 1)

        refined = np.zeros_like(yellow_mask)
        refined[y1:y2, keep_x1:keep_x2] = yellow_mask[y1:y2, keep_x1:keep_x2]
        refined = cv2.morphologyEx(
            refined,
            cv2.MORPH_CLOSE,
            np.ones((config.YELLOW_VERTICAL_CLOSE_KERNEL, 1), np.uint8),
        )

        _, gy, _, gh = green_box
        track_y1 = max(0, gy - 1)
        track_y2 = min(roi_h, gy + gh + 1)
        for x in range(keep_x1, keep_x2):
            column = refined[track_y1:track_y2, x]
            if np.count_nonzero(column) >= 2:
                refined[track_y1:track_y2, x] = 255

        return refined

    @staticmethod
    def _find_yellow_boxes(
        yellow_mask: np.ndarray,
        green_box: tuple[int, int, int, int],
    ) -> list[tuple[int, int, int, int, int]]:
        roi_h, _ = yellow_mask.shape[:2]
        y1, y2 = FightDetector._yellow_search_band(roi_h, green_box)
        filtered_mask = np.zeros_like(yellow_mask)
        filtered_mask[y1:y2, :] = yellow_mask[y1:y2, :]

        contours, _ = cv2.findContours(
            filtered_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        _, gy, _, gh = green_box
        green_y1 = gy
        green_y2 = gy + gh
        green_center_y = gy + gh / 2.0
        min_h = max(
            config.YELLOW_BOX_MIN_HEIGHT_BASE,
            int(gh * config.YELLOW_BOX_MIN_HEIGHT_RATIO),
        )
        center_y_tol = max(
            config.YELLOW_CENTER_Y_TOL_MIN,
            gh * config.YELLOW_CENTER_Y_TOL_RATIO,
        )

        boxes: list[tuple[int, int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if w < 1 or w > config.YELLOW_BOX_MAX_WIDTH:
                continue
            if h < min_h:
                continue
            if h / max(w, 1) < config.YELLOW_BOX_MIN_ASPECT_RATIO:
                continue
            if area < config.YELLOW_BOX_MIN_AREA:
                continue

            overlap_y1 = max(y, green_y1)
            overlap_y2 = min(y + h, green_y2)
            if overlap_y2 <= overlap_y1:
                continue

            overlap_ratio = (overlap_y2 - overlap_y1) / max(h, 1)
            if overlap_ratio < config.YELLOW_OVERLAP_RATIO_MIN:
                continue

            center_y = y + h / 2.0
            if abs(center_y - green_center_y) > center_y_tol:
                continue

            boxes.append((x, y, w, h, area))

        boxes.sort(key=lambda item: item[4], reverse=True)
        return boxes


class FightController:
    def __init__(self) -> None:
        self.detector = FightDetector()
        self.input_worker = KeyHoldWorker()
        self.missing_count = 0

    def reset(self) -> None:
        self.missing_count = 0
        self.input_worker.release()

    def close(self) -> None:
        self.input_worker.close()

    def step(self, frame_bgr: np.ndarray, dt: float) -> FightStep:
        detection = self.detector.detect(frame_bgr)

        if detection is None:
            self.missing_count += 1
            self.input_worker.release()

            audit.fight(f"missing detection count={self.missing_count}")

            if self.missing_count >= config.FIGHT_MISSING_DETECTION_LIMIT:
                return FightStep(
                    status="need_model_check",
                    detection=None,
                    reason="missing_detection_limit",
                    missing_count=self.missing_count,
                )

            return FightStep(
                status="active",
                detection=None,
                reason="missing_detection",
                missing_count=self.missing_count,
            )

        self.missing_count = 0

        hold_sec = max(
            config.FIGHT_HOLD_MIN_SEC,
            min(config.FIGHT_HOLD_MAX_SEC, dt * config.FIGHT_HOLD_DT_FACTOR),
        )

        if detection.yellow_x < detection.safe_left:
            self.input_worker.hold(config.KEY_MOVE_LEFT, hold_sec)
            action = config.KEY_MOVE_LEFT
        elif detection.yellow_x > detection.safe_right:
            self.input_worker.hold(config.KEY_MOVE_RIGHT, hold_sec)
            action = config.KEY_MOVE_RIGHT
        else:
            self.input_worker.release()
            action = "release"

        audit.fight(
            f"yellow={detection.yellow_x} "
            f"safe=({detection.safe_left:.1f},{detection.safe_right:.1f}) "
            f"action={action} hold={hold_sec:.3f}"
        )

        return FightStep(
            status="active",
            detection=detection,
            reason="ok",
            missing_count=0,
        )