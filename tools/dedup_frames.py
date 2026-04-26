from pathlib import Path
import argparse
import shutil

from PIL import Image
import imagehash
from tqdm import tqdm

WORK_DIR = Path(__file__).parent.parent

INPUT_DIR = WORK_DIR / "frames"
OUTPUT_DIR = WORK_DIR / "unique_frames"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def compute_phash(path: Path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return imagehash.phash(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="pHash 距离阈值，越大去重越激进。推荐从 4 或 5 开始。"
    )
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = list_images(INPUT_DIR)
    if not images:
        print(f"没有在 {INPUT_DIR} 中找到图片")
        return

    representatives: list[tuple[Path, imagehash.ImageHash]] = []

    keep_count = 0
    duplicate_count = 0
    error_count = 0

    for img_path in tqdm(images, desc="Deduplicating"):
        try:
            current_hash = compute_phash(img_path)
        except Exception as e:
            error_count += 1
            print(f"[ERROR] 无法读取图片: {img_path}, error={e}")
            continue

        is_duplicate = False

        for rep_path, rep_hash in representatives:
            distance = current_hash - rep_hash

            if distance <= args.threshold:
                is_duplicate = True
                duplicate_count += 1
                break

        if not is_duplicate:
            representatives.append((img_path, current_hash))
            shutil.copy2(img_path, OUTPUT_DIR / img_path.name)
            keep_count += 1

    print()
    print(f"输入图片数: {len(images)}")
    print(f"保留图片数: {keep_count}")
    print(f"重复图片数: {duplicate_count}")
    print(f"错误图片数: {error_count}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()