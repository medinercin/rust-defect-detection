"""
pipeline.py

This file contains the main rust segmentation logic.
I keep the core image processing here so it is easy to follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, binary_closing, disk


@dataclass
class RustParams:
    """
    Small parameter container (not over-engineered).
    I only store the knobs I actually tune in experiments.
    """
    cr_threshold: Optional[int] = None     # If None, I use percentile threshold
    cr_percentile: float = 85.0
    y_threshold: int = 75

    min_object_size: int = 100
    closing_kernel_size: int = 7


class RustDetector:
    """
    Rust = (Cr is high) AND (Y is low) in YCrCb space.
    Then I clean the mask with morphology.
    """

    def __init__(self, params: RustParams) -> None:
        self.p = params
        self._kernel = disk(self.p.closing_kernel_size)

    # ---------- core channels ----------
    @staticmethod
    def extract_y_cr_channels(bgr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        I convert BGR -> YCrCb and take:
        - Y: brightness
        - Cr: red chroma channel (rust tends to push this higher)
        """
        ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCR_CB)
        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        return cr, y

    # ---------- raw mask ----------
    def build_raw_mask(self, cr_channel: np.ndarray, y_channel: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Rule-based segmentation:
        - Cr > threshold  (red-ish)
        - Y  < y_threshold (darker)
        """
        if self.p.cr_threshold is None:
            actual_cr_thr = float(np.percentile(cr_channel, self.p.cr_percentile))
        else:
            actual_cr_thr = float(self.p.cr_threshold)

        cr_mask = (cr_channel > actual_cr_thr).astype(np.uint8) * 255
        y_mask = (y_channel < self.p.y_threshold).astype(np.uint8) * 255

        raw = cv2.bitwise_and(cr_mask, y_mask)
        return raw, actual_cr_thr

    # ---------- refinement ----------
    def refine_mask(self, mask_255: np.ndarray) -> np.ndarray:
        """
        I remove small noise and fill tiny holes.
        This improves stability in batch evaluation.
        """
        mask_bool = mask_255 > 0
        closed = binary_closing(mask_bool, self._kernel)
        cleaned = remove_small_objects(closed, min_size=self.p.min_object_size)
        return (cleaned.astype(np.uint8) * 255)

    # ---------- one-image processing ----------
    def process_image(self, bgr: np.ndarray) -> Dict:
        """
        I return a dict so that the runner can decide
        whether to evaluate, visualize, or save stuff.
        """
        cr, y = self.extract_y_cr_channels(bgr)
        raw_mask, actual_cr_thr = self.build_raw_mask(cr, y)
        final_mask = self.refine_mask(raw_mask)

        return {
            "image": bgr,
            "cr_channel": cr,
            "y_channel": y,
            "raw_mask": raw_mask,
            "final_mask": final_mask,
            "actual_cr_threshold": actual_cr_thr,
        }


def visualize_result(result: Dict, y_threshold: int, metrics: Optional[Dict] = None, save_path: Optional[str] = None) -> None:
    """
    Quick visualization for sanity-checking.
    I keep it simple: original + Cr + Y + overlay.
    """
    bgr = result["image"]
    cr = result["cr_channel"]
    y = result["y_channel"]
    mask = result["final_mask"]
    cr_thr = result["actual_cr_threshold"]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original", fontweight="bold")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(cr, cmap="hot")
    axes[0, 1].set_title(f"Cr channel\nCr > {cr_thr:.1f}", fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(y, cmap="gray")
    axes[1, 0].set_title(f"Y channel\nY < {y_threshold}", fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0])

    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[mask > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 0.7, red, 0.3, 0)

    lines = [
        f"Cr > {cr_thr:.1f}",
        f"Y  < {y_threshold}",
    ]
    if metrics:
        lines += [
            f"IoU: {metrics['iou']:.3f}",
            f"Dice: {metrics['dice']:.3f}",
            f"Precision: {metrics['precision']:.3f}",
            f"Recall: {metrics['recall']:.3f}",
        ]

    y0 = 30
    for i, t in enumerate(lines):
        cv2.putText(overlay, t, (10, y0 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Final mask overlay", fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved visualization: {save_path}")

    plt.show()
