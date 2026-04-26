from __future__ import annotations

from typing import Tuple

import cv2
import mss
import numpy as np
import win32con
import win32gui

from . import audit, config


class WindowCapture:
    def __init__(self) -> None:
        self.hwnd = self._find_window()
        self.sct = mss.mss()

    @staticmethod
    def _find_window() -> int:
        matches: list[tuple[int, str, str, tuple[int, int, int, int]]] = []

        title_key = config.WINDOW_TITLE_KEYWORD.strip().lower()
        class_key = config.WINDOW_CLASS_KEYWORD.strip().lower()

        def enum_handler(hwnd: int, _) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return

            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)

            if not title:
                return

            title_norm = title.strip().lower()
            class_norm = class_name.strip().lower()

            if title_key in title_norm:
                rect = win32gui.GetWindowRect(hwnd)
                matches.append((hwnd, title, class_name, rect))

        win32gui.EnumWindows(enum_handler, None)

        if not matches:
            raise RuntimeError(
                f"Cannot find window whose title contains {config.WINDOW_TITLE_KEYWORD!r}"
            )

        matches.sort(
            key=lambda item: (
                class_key not in item[2].strip().lower(),
                -((item[3][2] - item[3][0]) * (item[3][3] - item[3][1])),
            )
        )

        hwnd, title, class_name, rect = matches[0]
        audit.normal(
            f"Using window: hwnd={hwnd} title={title!r} class={class_name!r} rect={rect}"
        )
        return hwnd

    def activate(self) -> None:
        try:
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
        except Exception as exc:
            audit.warn(f"Failed to activate window, continuing: {exc}")

    def client_rect_screen(self) -> Tuple[int, int, int, int]:
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        screen_left, screen_top = win32gui.ClientToScreen(self.hwnd, (left, top))
        screen_right, screen_bottom = win32gui.ClientToScreen(self.hwnd, (right, bottom))
        return screen_left, screen_top, screen_right, screen_bottom

    def grab(self) -> np.ndarray:
        left, top, right, bottom = self.client_rect_screen()

        monitor = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
        }

        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def client_center_screen(self) -> tuple[int, int]:
        left, top, right, bottom = self.client_rect_screen()
        return (left + right) // 2, (top + bottom) // 2