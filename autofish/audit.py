from __future__ import annotations

import logging
import sys
from pathlib import Path
import time
import cv2
import numpy as np

from . import config


_LOGGER = logging.getLogger("autofish")
_READY = False


def setup() -> None:
    global _READY
    if _READY:
        return

    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if config.DEBUG else logging.INFO

    _LOGGER.setLevel(level)
    _LOGGER.propagate = False

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    _LOGGER.handlers.clear()
    _LOGGER.addHandler(stream_handler)

    _READY = True


def info(message: str) -> None:
    setup()
    if config.DEBUG:
        _LOGGER.info(message)


def normal(message: str) -> None:
    setup()
    _LOGGER.info(message)


def debug(message: str) -> None:
    setup()
    if config.DEBUG:
        _LOGGER.debug(message)


def warn(message: str) -> None:
    setup()
    _LOGGER.warning(message)


def error(message: str) -> None:
    setup()
    _LOGGER.error(message)


def transition(src: str, dst: str, reason: str) -> None:
    setup()
    if config.DEBUG:
        _LOGGER.info(f"STATE {src} -> {dst} | {reason}")


def prediction(label: str, confidence: float) -> None:
    setup()
    if config.DEBUG:
        _LOGGER.debug(f"PRED label={label} conf={confidence:.3f}")


def fight(message: str) -> None:
    setup()
    if config.DEBUG:
        _LOGGER.debug(f"FIGHT {message}")


def fps(mode: str, fps_value: float, state: str) -> None:
    setup()
    if config.DEBUG:
        _LOGGER.info(f"FPS mode={mode} fps={fps_value:.2f} state={state}")


def exception(prefix: str, exc: BaseException) -> None:
    setup()
    if config.DEBUG:
        _LOGGER.exception(prefix)
    else:
        _LOGGER.error(f"{prefix}: {exc}")

def save_fight_entry_screenshot(frame_bgr: np.ndarray) -> None:
    if not config.DEBUG:
        return

    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)

    out_path = (
        config.AUDIT_DIR
        / f"fight_entry_{timestamp}.png"
    )

    ok = cv2.imwrite(str(out_path), frame_bgr)

    if ok:
        debug(f"saved fight entry screenshot: {out_path}")
    else:
        warn(f"failed to save fight entry screenshot: {out_path}")