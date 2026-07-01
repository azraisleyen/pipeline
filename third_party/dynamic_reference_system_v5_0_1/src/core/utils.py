
from pathlib import Path
import csv
import json
import shutil
from typing import Iterable, List

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        ensure_dir(path)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def list_files_by_extensions(directory: Path, extensions: Iterable[str]) -> List[Path]:
    if not directory.exists():
        return []

    ext_set = {ext.lower() for ext in extensions}
    files = []

    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in ext_set:
            files.append(item)

    return sorted(files)


def load_image_bgr(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return image


def clip_bbox_xyxy(box, image_w: int, image_h: int):
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(0, min(x2, image_w))
    y2 = max(0, min(y2, image_h))

    return [x1, y1, x2, y2]


def expand_bbox_xyxy(box, image_w: int, image_h: int, expand_ratio: float):
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    dx = w * float(expand_ratio)
    dy = h * float(expand_ratio)

    expanded = [x1 - dx, y1 - dy, x2 + dx, y2 + dy]
    return clip_bbox_xyxy(expanded, image_w, image_h)


def bbox_area(box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_aspect_ratio(box) -> float:
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return float(w) / float(h)


def crop_xyxy(image_bgr, box):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clip_bbox_xyxy(box, w, h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    return crop


def compute_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area) / float(union)


def resize_keep_aspect(image_bgr, max_size: int):
    h, w = image_bgr.shape[:2]
    longest = max(h, w)

    if longest <= max_size:
        return image_bgr

    scale = max_size / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def adjust_brightness_contrast(image_bgr, brightness_factor=1.0, contrast_factor=1.0):
    img = image_bgr.astype(np.float32)

    img = img * float(brightness_factor)

    if abs(float(contrast_factor) - 1.0) > 1e-6:
        mean = 127.5
        img = (img - mean) * float(contrast_factor) + mean

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def make_padded_template(image_bgr, pad_ratio=0.08):
    h, w = image_bgr.shape[:2]
    pad = max(2, int(round(max(h, w) * float(pad_ratio))))

    padded = cv2.copyMakeBorder(
        image_bgr,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_REPLICATE,
    )

    padded = cv2.resize(padded, (w, h), interpolation=cv2.INTER_AREA)
    return padded


def save_csv_rows(rows, path: Path, fieldnames=None) -> None:
    ensure_dir(path.parent)

    if fieldnames is None:
        key_set = set()
        for row in rows:
            key_set.update(row.keys())
        fieldnames = sorted(list(key_set))

    with path.open("w", newline="", encoding="utf-8") as f:
        if not fieldnames:
            f.write("")
            return

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def draw_objects_on_image(image_bgr, objects):
    vis = image_bgr.copy()

    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        ref_id = obj.get("reference_id", "unknown")
        score = float(obj.get("score", 0.0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{ref_id} | {score:.3f}"
        cv2.putText(
            vis,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis
