"""
eval_utils.py

This file has evaluation + reporting helpers.
I separate it so the processing code stays clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


# ---------- basic scores ----------
def compute_area_ratio(mask_255: np.ndarray) -> float:
    """Percent of pixels detected as rust."""
    return float((np.sum(mask_255 > 0) / mask_255.size) * 100.0)

def degradation_score(rust_ratio: float) -> float:
    """
    Simple score: map rust ratio to 0..100.
    (I keep it simple for a course project.)
    """
    rust_ratio_norm = min(rust_ratio / 100.0, 1.0)
    return float(np.clip(rust_ratio_norm * 100.0, 0.0, 100.0))

def evaluate(pred_mask_255: np.ndarray, gt_mask_255: np.ndarray) -> Dict[str, float]:
    """
    Standard binary segmentation metrics.
    """
    pred = (pred_mask_255 > 127).astype(np.uint8)
    gt = (gt_mask_255 > 127).astype(np.uint8)

    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    iou = (intersection / union) if union > 0 else 0.0
    dice = (2.0 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else 0.0

    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    tp = intersection

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


# ---------- saving ----------
def to_dataframe(rows: List[Dict]) -> Union[List[Dict], "object"]:
    if PANDAS_AVAILABLE:
        return pd.DataFrame(rows)
    return rows

def save_csv(results: Union[List[Dict], "object"], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(results, "to_csv"):
        results.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[OK] CSV saved: {csv_path}")
    else:
        # If pandas not installed, I skip CSV to avoid messy custom writer here.
        print("[WARN] pandas not installed -> CSV export skipped.")
        print("       Install with: pip install pandas")

def generate_report(results: Union[List[Dict], "object"], txt_path: Path) -> None:
    """
    I write a readable TXT report for my instructor.
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to list[dict]
    if hasattr(results, "to_dict"):
        rows = results.to_dict("records")
    else:
        rows = results if isinstance(results, list) else []

    total = len(rows)
    avg_rust = float(sum(r["rust_ratio"] for r in rows) / total) if total else 0.0
    avg_score = float(sum(r["degradation_score"] for r in rows) / total) if total else 0.0

    metric_rows = [r for r in rows if r.get("iou") is not None]
    has_metrics = len(metric_rows) > 0

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in metric_rows if r.get(key) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    lines: List[str] = []
    lines += [
        "=" * 80,
        "RUST DEFECT SEGMENTATION REPORT",
        "=" * 80,
        "",
        f"Total images: {total}",
        f"Average Rust Ratio: {avg_rust:.2f}%",
        f"Average Degradation Score: {avg_score:.2f}/100",
        "",
    ]

    if has_metrics:
        lines += [
            "-" * 80,
            "METRICS (images with ground truth)",
            "-" * 80,
            f"Average IoU: {_avg('iou'):.4f}",
            f"Average Dice: {_avg('dice'):.4f}",
            f"Average Precision: {_avg('precision'):.4f}",
            f"Average Recall: {_avg('recall'):.4f}",
            "",
        ]

    lines += [
        "-" * 80,
        "PER-IMAGE RESULTS",
        "-" * 80,
        "",
    ]

    for r in rows:
        lines.append(f"Image: {r['image_name']}")
        lines.append(f"  Rust Ratio: {r['rust_ratio']:.2f}%")
        lines.append(f"  Degradation Score: {r['degradation_score']:.2f}/100")
        if r.get("iou") is not None:
            lines.append(f"  IoU: {r['iou']:.4f}")
            lines.append(f"  Dice: {r['dice']:.4f}")
            lines.append(f"  Precision: {r['precision']:.4f}")
            lines.append(f"  Recall: {r['recall']:.4f}")
        lines.append("")

    lines.append("=" * 80)

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Report saved: {txt_path}")
