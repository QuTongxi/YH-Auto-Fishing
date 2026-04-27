from pathlib import Path
from dataclasses import dataclass
from typing import Literal

SCREEN_MODE: Literal["16:9", "8:5"] = "16:9"

KEY_START_OR_CATCH = "f"
KEY_MOVE_LEFT = "d"
KEY_MOVE_RIGHT = "a"

DEBUG = False

WORK_DIR = Path(__file__).parent.parent
AUDIT_DIR = WORK_DIR / "runs" / "audit"


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


ROI = {
    # 3840*2160, 2560*1440, 1920*1080
    "16:9": {
        "SKILL_GROUP": PercentRect(0.690, 0.860, 0.975, 0.965),
        "WAIT_BANNER": PercentRect(0.300, 0.210, 0.700, 0.280),
        "FIGHT_BAR": PercentRect(0.300, 0.045, 0.700, 0.090),
    },
    # 2560*1600
    "8:5": {
        "SKILL_GROUP": PercentRect(0.690, 0.860, 0.975, 0.965),
        "WAIT_BANNER": PercentRect(0.300, 0.190, 0.700, 0.260),
        "FIGHT_BAR": PercentRect(0.300, 0.045, 0.700, 0.090),
    },
}

MOBILE_NET_MODEL_PATH = WORK_DIR / "runs" / "model" / "best.pt"

# =========================
# Runtime
# =========================
WINDOW_TITLE_KEYWORD = "异环"
WINDOW_CLASS_KEYWORD = "Unreal"

NORMAL_TARGET_FPS = 10.0
FIGHT_TARGET_FPS = 15.0

POST_FIGHT_VERIFY_TIMEOUT_SEC = 3.0
ERROR_RECOVERY_TIMEOUT_SEC = 10.0

MODEL_INPUT_SIZE = 320
MODEL_DEVICE: Literal["auto", "cpu", "cuda"] = "cpu"
TORCH_NUM_THREADS = 2

MIN_MODEL_CONFIDENCE = 0.70

TRANSITION_GRACE_SEC = 0.8
START_RETRY_SEC = 1.2


# =========================
# Labels
# =========================

LABEL_WAITING_FOR_START = "0_waiting_for_start"
LABEL_WAITING_FOR_FISH = "1_waiting_for_fish"
LABEL_CATCHING_FISH = "2_catching_fish"
LABEL_FIGHTING_WITH_FISH = "3_fighting_with_fish"
LABEL_AFTER_FIGHTING = "4_after_fighting"
LABEL_NONSENSE = "5_nonsense"


# =========================
# Input
# =========================

KEY_PRESS_SEC = 0.100

AFTER_FIGHT_CLICK_DELAY_SEC = 0.05
AFTER_FIGHT_CLICK_COUNT = 1


# =========================
# Fight detector
# OpenCV HSV uses H: 0-179
# =========================

GREEN_HSV_LOWER = (45, 70, 80)
GREEN_HSV_UPPER = (95, 255, 255)

# Yellow detection:
# color pre-filter + geometry constraints + track-band constraints.
YELLOW_HSV_LOWER = (20, 110, 190)
YELLOW_HSV_UPPER = (38, 255, 255)
YELLOW_BAND_PAD = 8
YELLOW_MAX_GROUP_WIDTH = 12
YELLOW_COL_MIN_PIXELS = 2
YELLOW_COLUMN_PAD = 1
YELLOW_VERTICAL_CLOSE_KERNEL = 7
YELLOW_BOX_MAX_WIDTH = 12
YELLOW_BOX_MIN_HEIGHT_BASE = 8
YELLOW_BOX_MIN_HEIGHT_RATIO = 0.55
YELLOW_BOX_MIN_ASPECT_RATIO = 2.0
YELLOW_BOX_MIN_AREA = 6
YELLOW_OVERLAP_RATIO_MIN = 0.35
YELLOW_CENTER_Y_TOL_MIN = 10.0
YELLOW_CENTER_Y_TOL_RATIO = 0.75

FIGHT_MISSING_DETECTION_LIMIT = 10

FIGHT_SAFE_LEFT_RATIO = 0.25
FIGHT_SAFE_RIGHT_RATIO = 0.75

FIGHT_HOLD_MIN_SEC = 0.010
FIGHT_HOLD_MAX_SEC = 0.200
FIGHT_HOLD_DT_FACTOR = 1.50

FIGHT_INPUT_LOOP_SLEEP_SEC = 0.003


# =========================
# Audit
# =========================

FPS_LOG_INTERVAL_SEC = 1.0
SAVE_DEBUG_FRAMES = False