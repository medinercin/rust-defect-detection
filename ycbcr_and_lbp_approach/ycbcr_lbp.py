"""
Rust Segmentation (IoU + Recall Oriented)
- Base: YCbCr (Cr high, Y low-ish) => high recall candidate
- Stabilize IoU: line/crack suppression + morphology cleanup
- LBP: optional SOFT refine (never hard AND). Auto-bypass if it shrinks too much.

Bu sürümde amacım pası kaçırmamak (recall) ve aynı zamanda IoU'yu stabil artırmak.
LBP'yi maske üretiminde keskin bir filtre olarak kullanmadım; çünkü bazı görüntülerde pası da siliyor.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# -----------------------------
# Config
# -----------------------------
@dataclass
class RustConfig:
    # ---- YCbCr thresholds (Recall-oriented) ----
    # cr_thr: use percentile on Cr channel (lower percentile -> more recall)
    cr_percentile: float = 83.0       # 80-88 arası denenebilir
    y_max: int = 95                   # Y < y_max => "darker" areas included (recall)
    cb_min: int = 80                  # optional: remove blue-ish highlights (tune if needed)
    use_cb_gate: bool = False         # False by default (recall first)

    # ---- post-processing ----
    min_area: int = 120               # remove small speckles
    close_ksize: int = 5              # fill small holes
    open_ksize: int = 3               # remove tiny noise

    # ---- line / crack suppression ----
    use_line_suppression: bool = True
    line_len: int = 19                # 15-25 typical for 1024px
    line_iter: int = 1                # 1-2

    # ---- LBP soft refine ----
    use_lbp_refine: bool = True
    lbp_radius: int = 1               # 1 is fine
    lbp_points: int = 8               # 8 for r=1
    lbp_percentile_keep: float = 55.0 # keep top texture regions (higher => more conservative)
    lbp_shrink_bypass_ratio: float = 0.40
    # if (refined_area / base_area) < 0.40 => bypass refine to protect recall

# -----------------------------
# Utilities
# -----------------------------
def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def _morph_cleanup(mask: np.ndarray, open_ksize: int, close_ksize: int) -> np.ndarray:
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _line_suppression(mask: np.ndarray, line_len: int, iters: int) -> np.ndarray:
    """
    Suppress thin long line-like components (cracks/derz).
    We apply directional OPEN with line kernels; then subtract found lines.
    """
    m = mask.copy()

    # Directional line kernels
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_len))

    # diagonal kernels (manual)
    k_d1 = np.eye(line_len, dtype=np.uint8)  # \
    k_d2 = np.fliplr(k_d1)                   # /

    for _ in range(max(1, iters)):
        lines_h = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_h)
        lines_v = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_v)
        lines_d1 = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_d1)
        lines_d2 = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_d2)

        lines = cv2.max(lines_h, lines_v)
        lines = cv2.max(lines, lines_d1)
        lines = cv2.max(lines, lines_d2)

        # subtract line candidates (keep blobs)
        m = cv2.subtract(m, lines)

    return m

# -----------------------------
# LBP (simple implementation)
# -----------------------------
def _lbp_8u(gray: np.ndarray) -> np.ndarray:
    """
    Very small, dependency-free LBP for radius=1, points=8.
    Returns uint8 LBP code image.
    """
    g = gray.astype(np.uint8)
    h, w = g.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # neighbors offsets (clockwise)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               ( 0, 1), ( 1, 1), ( 1, 0),
               ( 1,-1), ( 0,-1)]

    center = g[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)

    for i, (dy, dx) in enumerate(offsets):
        neigh = g[1+dy:h-1+dy, 1+dx:w-1+dx]
        code |= ((neigh >= center).astype(np.uint8) << (7 - i))

    lbp[1:-1, 1:-1] = code
    return lbp

def _lbp_soft_refine(bgr: np.ndarray, base_mask: np.ndarray, cfg: RustConfig) -> np.ndarray:
    """
    Soft refine: we DO NOT AND hard with texture. We create a texture-prior mask
    and blend it by intersection only where it doesn't kill too much area.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = _lbp_8u(gray)

    # Use local texture score: abs(lbp - median_blur(lbp))
    med = cv2.medianBlur(lbp, 5)
    tex = cv2.absdiff(lbp, med)

    # Only consider texture inside base candidate
    tex_inside = tex.copy()
    tex_inside[base_mask == 0] = 0

    # threshold by percentile of non-zero texture values
    vals = tex_inside[tex_inside > 0]
    if vals.size < 50:
        # too little info -> return base
        return base_mask

    thr = np.percentile(vals, cfg.lbp_percentile_keep)
    tex_mask = (tex_inside >= thr).astype(np.uint8) * 255

    # small cleanup on texture mask
    tex_mask = _morph_cleanup(tex_mask, open_ksize=3, close_ksize=3)

    base_area = int(np.count_nonzero(base_mask))
    if base_area == 0:
        return base_mask

    # intersection as a refine candidate
    refined = cv2.bitwise_and(base_mask, tex_mask)
    ref_area = int(np.count_nonzero(refined))

    # bypass if it shrinks too much => protect recall + IoU stability
    if (ref_area / base_area) < cfg.lbp_shrink_bypass_ratio:
        return base_mask

    # else: combine refined and base with a union to keep recall
    # (this "soft" keeps most base, but texture helps remove some FP islands)
    # We do: base - (base - refined)?? Actually refined is subset, so we can
    # slightly penalize low-texture pixels by eroding base a bit then OR with refined.
    er_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    base_er = cv2.erode(base_mask, er_k, iterations=1)
    out = cv2.bitwise_or(base_er, refined)
    return out

# -----------------------------
# Main detector
# -----------------------------
class RustDetectorIoURecall:
    def __init__(self, cfg: Optional[RustConfig] = None):
        self.cfg = cfg or RustConfig()

    def predict_mask(self, bgr: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.cfg

        # 1) YCbCr candidate (recall-first)
        ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycbcr[:, :, 0]
        Cr = ycbcr[:, :, 1]
        Cb = ycbcr[:, :, 2]

        cr_thr = np.percentile(Cr, cfg.cr_percentile)
        cand = (Cr >= cr_thr) & (Y <= cfg.y_max)

        if cfg.use_cb_gate:
            cand = cand & (Cb >= cfg.cb_min)

        base_mask = (cand.astype(np.uint8) * 255)

        # 2) morphology cleanup
        base_mask = _morph_cleanup(base_mask, cfg.open_ksize, cfg.close_ksize)
        base_mask = _remove_small_components(base_mask, cfg.min_area)

        # 3) line suppression (helps IoU by removing crack-like FP)
        if cfg.use_line_suppression:
            base_mask = _line_suppression(base_mask, cfg.line_len, cfg.line_iter)
            base_mask = _morph_cleanup(base_mask, 3, 5)
            base_mask = _remove_small_components(base_mask, cfg.min_area)

        # 4) optional soft LBP refine (never hard eliminate)
        final_mask = base_mask
        if cfg.use_lbp_refine:
            final_mask = _lbp_soft_refine(bgr, base_mask, cfg)
            final_mask = _morph_cleanup(final_mask, 3, 5)
            final_mask = _remove_small_components(final_mask, cfg.min_area)

        info = {
            "cr_percentile": cfg.cr_percentile,
            "cr_thr_value": float(cr_thr),
            "y_max": int(cfg.y_max),
            "used_cb_gate": bool(cfg.use_cb_gate),
            "used_line_supp": bool(cfg.use_line_suppression),
            "used_lbp_refine": bool(cfg.use_lbp_refine),
        }
        return final_mask, info

if __name__ == "__main__":
    import cv2
    from pathlib import Path

    img_path = Path("data/images/040.png")  # test resmi
    img = cv2.imread(str(img_path))

    detector = RustDetectorIoURecall(RustConfig())
    mask, info = detector.predict_mask(img)

    cv2.imwrite("040_pred.png", mask)
    print("Saved 040_pred.png")
    print(info)

