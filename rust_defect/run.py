"""
run.py

This is the only entry point I run.
It reads images/masks from the dataset folder, runs the pipeline,
and saves CSV + TXT report under outputs/.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pipeline import RustParams, RustDetector, visualize_result
from eval_utils import (
    compute_area_ratio,
    degradation_score,
    evaluate,
    to_dataframe,
    save_csv,
    generate_report,
)

# ----------------------------
# Dataset path (fixed as requested)
# ----------------------------
DATASET_ROOT = Path(r"C:\Users\Medine\Desktop\rust-defect-segmentation\dataset")
IMAGES_DIR = DATASET_ROOT / "images"
MASKS_DIR = DATASET_ROOT / "masks"

# ----------------------------
# Outputs
# ----------------------------
OUTPUTS_DIR = Path("outputs")
CSV_PATH = OUTPUTS_DIR / "rust_detection_results.csv"
REPORT_PATH = OUTPUTS_DIR / "rust_detection_report.txt"
VIZ_DIR = OUTPUTS_DIR / "viz"


def list_images(images_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    files = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return files


def match_mask(image_path: Path) -> Optional[Path]:
    """
    I keep mask matching flexible:
    - same name in masks/
    - or stem + _mask.png
    """
    cand1 = MASKS_DIR / image_path.name
    cand2 = MASKS_DIR / f"{image_path.stem}_mask.png"
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return img


def read_mask_255(path: Path) -> Optional[np.ndarray]:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.uint8) * 255


def main() -> None:
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

    # Parameters (edit here if I want to tune)
    params = RustParams(
        cr_threshold=None,       # None -> percentile threshold
        cr_percentile=85.0,
        y_threshold=75,
        min_object_size=100,
        closing_kernel_size=7,
    )

    detector = RustDetector(params)

    image_files = list_images(IMAGES_DIR)

    print("=" * 70)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Images found: {len(image_files)}")
    print("=" * 70)

    rows = []
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # If you want visuals for a few samples, set this to True
    SAVE_SOME_VIS = False
    VIS_EVERY_N = 50  # save one visualization per N images (if SAVE_SOME_VIS is True)

    for idx, img_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")

        bgr = read_bgr(img_path)
        result = detector.process_image(bgr)

        final_mask = result["final_mask"]
        rust_ratio = compute_area_ratio(final_mask)
        score = degradation_score(rust_ratio)

        # Evaluate if GT exists
        gt_path = match_mask(img_path)
        metrics = None
        if gt_path is not None:
            gt_mask = read_mask_255(gt_path)
            if gt_mask is not None:
                metrics = evaluate(final_mask, gt_mask)

        row = {
            "image_name": img_path.name,
            "rust_ratio": rust_ratio,
            "degradation_score": score,
            "iou": metrics["iou"] if metrics else None,
            "dice": metrics["dice"] if metrics else None,
            "precision": metrics["precision"] if metrics else None,
            "recall": metrics["recall"] if metrics else None,
        }
        rows.append(row)

        if metrics:
            print(f"   rust={rust_ratio:.2f}% | IoU={metrics['iou']:.4f} | P={metrics['precision']:.4f} | R={metrics['recall']:.4f}")
        else:
            print(f"   rust={rust_ratio:.2f}% (no GT found)")

        # Optional: save a visualization sometimes
        if SAVE_SOME_VIS and (idx % VIS_EVERY_N == 0):
            save_path = str(VIZ_DIR / f"{img_path.stem}_viz.png")
            visualize_result(result, y_threshold=params.y_threshold, metrics=metrics, save_path=save_path)

    # Save outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results_obj = to_dataframe(rows)
    save_csv(results_obj, CSV_PATH)
    generate_report(results_obj, REPORT_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
