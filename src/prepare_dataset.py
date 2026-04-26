from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


# =========================
# Config
# =========================

INPUT_DIR = Path("./unique_frames/")
OUTPUT_DIR = Path("./training_dataset/")

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "tools" / "prepare_training.py"

SIZE = 320


@dataclass(frozen=True)
class PercentRect:
    left: float
    top: float
    right: float
    bottom: float

    def to_arg(self) -> str:
        return f"{self.left},{self.top},{self.right},{self.bottom}"


FIGHT_BAR = PercentRect(0.300, 0.045, 0.700, 0.090)
WAIT_BANNER = PercentRect(0.300, 0.210, 0.700, 0.280)
SKILL_GROUP = PercentRect(0.690, 0.860, 0.975, 0.965)


# =========================
# Core logic
# =========================

def collect_png_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"INPUT_DIR is not a directory: {input_dir}")

    return sorted(input_dir.glob("*.png"))


def convert_one_image(image_path: Path) -> tuple[Path, bool, str]:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "-i",
        str(image_path),
        "-o",
        str(OUTPUT_DIR),
        "--size",
        str(SIZE),
        "--fight-bar",
        FIGHT_BAR.to_arg(),
        "--wait-banner",
        WAIT_BANNER.to_arg(),
        "--skill-group",
        SKILL_GROUP.to_arg(),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        return image_path, False, msg

    return image_path, True, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--max-thread",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Maximum number of parallel subprocesses.",
    )
    args = parser.parse_args()

    if args.max_thread <= 0:
        raise ValueError("--max-thread must be positive")

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Cannot find script: {SCRIPT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_files = collect_png_files(INPUT_DIR)

    if not png_files:
        print(f"No png files found in {INPUT_DIR}")
        return

    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Image count: {len(png_files)}")
    print(f"Max thread: {args.max_thread}")
    print()

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.max_thread) as executor:
        futures = [executor.submit(convert_one_image, path) for path in png_files]

        for future in as_completed(futures):
            image_path, ok, msg = future.result()

            if ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"[FAILED] {image_path}")
                print(msg)
                print()

    print("Done.")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()