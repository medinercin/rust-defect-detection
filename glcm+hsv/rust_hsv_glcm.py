import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import apply_hysteresis_threshold
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List


@dataclass
class RustConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    median_kernel_size: int = 5
    saturation_threshold: int = 25
    morph_kernel_size: int = 7
    tophat_kernel_size: int = 15
    hysteresis_high_k: float = 2.0
    hysteresis_low_k: float = 0.5
    min_rust_area: int = 50


class RustDetector:
    def __init__(self, cfg: RustConfig = RustConfig()):
        self.cfg = cfg
        self.clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=cfg.clahe_tile_size
        )
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.morph_kernel_size, cfg.morph_kernel_size)
        )
        self.tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.tophat_kernel_size, cfg.tophat_kernel_size)
        )

    def preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        b = lab[:, :, 2]
        tophat = cv2.morphologyEx(b, cv2.MORPH_TOPHAT, self.tophat_kernel)
        b_enh = self.clahe.apply(tophat)
        b_enh = cv2.medianBlur(b_enh, self.cfg.median_kernel_size)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return b_enh, gray

    def segment_rust(self, image_bgr: np.ndarray, b_channel: np.ndarray) -> np.ndarray:
        mean_val = float(np.mean(b_channel))
        std_val = float(np.std(b_channel))
        high_t = mean_val + (self.cfg.hysteresis_high_k * std_val)
        low_t = mean_val + (self.cfg.hysteresis_low_k * std_val)

        mask_color = apply_hysteresis_threshold(b_channel, low_t, high_t)
        mask_color = (mask_color.astype(np.uint8) * 255)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        mask_sat = (s > self.cfg.saturation_threshold).astype(np.uint8) * 255

        combined = cv2.bitwise_and(mask_color, mask_sat)

        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self.morph_kernel, iterations=3)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        filtered = np.zeros_like(opened)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= self.cfg.min_rust_area:
                filtered[labels == label_id] = 255

        return filtered

    def compute_texture_features(self, gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        if np.sum(mask > 0) == 0:
            return {
                "contrast": 0.0,
                "dissimilarity": 0.0,
                "homogeneity": 1.0,
                "energy": 0.0,
                "correlation": 0.0
            }

        masked = cv2.bitwise_and(gray, mask)
        masked_u8 = img_as_ubyte(masked)

        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm = graycomatrix(
            masked_u8,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        return {
            "contrast": float(np.mean(graycoprops(glcm, "contrast"))),
            "dissimilarity": float(np.mean(graycoprops(glcm, "dissimilarity"))),
            "homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
            "energy": float(np.mean(graycoprops(glcm, "energy"))),
            "correlation": float(np.mean(graycoprops(glcm, "correlation"))),
        }

    def compute_degradation_score(self, texture: Dict[str, float], rust_ratio: float) -> float:
        contrast_norm = min(texture["contrast"] / 1000.0, 1.0)
        rust_norm = min(rust_ratio / 100.0, 1.0)
        hom_inv = max(0.0, 1.0 - float(texture["homogeneity"]))

        score = (0.40 * contrast_norm) + (0.40 * rust_norm) + (0.20 * hom_inv)
        return float(np.clip(score * 100.0, 0.0, 100.0))

    def evaluate(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        pred = (pred_mask > 127).astype(np.uint8)
        gt = (gt_mask > 127).astype(np.uint8)

        inter = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        iou = (inter / union) if union > 0 else 0.0

        denom_dice = int(pred.sum() + gt.sum())
        dice = (2.0 * inter / denom_dice) if denom_dice > 0 else 0.0

        fp = int(np.logical_and(pred, 1 - gt).sum())
        fn = int(np.logical_and(1 - pred, gt).sum())

        precision = (inter / (inter + fp)) if (inter + fp) > 0 else 0.0
        recall = (inter / (inter + fn)) if (inter + fn) > 0 else 0.0

        return {
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall)
        }

    def process_image(self, image_path: str, gt_mask_path: Optional[str] = None) -> Dict:
        print("Pipeline: load -> preprocess(LAB+TopHat+CLAHE) -> segment(hysteresis+HSV-S) -> filter(morph+CC) -> texture(GLCM) -> score -> eval(optional)")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        b_enh, gray = self.preprocess(image)
        pred = self.segment_rust(image, b_enh)

        rust_ratio = (float(np.sum(pred > 0)) / float(pred.size)) * 100.0
        texture = self.compute_texture_features(gray, pred)
        score = self.compute_degradation_score(texture, rust_ratio)

        gt = None
        eval_m = None
        if gt_mask_path and os.path.exists(gt_mask_path):
            gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                eval_m = self.evaluate(pred, gt)

        return {
            "image": image,
            "b_channel": b_enh,
            "grayscale": gray,
            "rust_mask": pred,
            "texture_features": texture,
            "degradation_score": score,
            "rust_ratio": rust_ratio,
            "evaluation_metrics": eval_m,
            "gt_mask": gt
        }

    def visualize(self, results: Dict, save_path: Optional[str] = None):
        img = results["image"]
        b = results["b_channel"]
        pred = results["rust_mask"]
        gt = results.get("gt_mask")
        score = results["degradation_score"]
        ratio = results["rust_ratio"]
        ev = results.get("evaluation_metrics")

        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        view = img.copy()
        cv2.drawContours(view, contours, -1, (0, 0, 255), 2)

        lines = [f"Score: {score:.2f}", f"Rust ratio: {ratio:.2f}%"]
        if ev:
            lines += [
                f"IoU: {ev['iou']:.3f}",
                f"Dice: {ev['dice']:.3f}",
                f"Prec: {ev['precision']:.3f}",
                f"Rec: {ev['recall']:.3f}"
            ]

        y = 30
        for t in lines:
            cv2.putText(view, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y += 25

        view_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 2, figsize=(14, 14))
        axes[0, 0].imshow(view_rgb)
        axes[0, 0].set_title("Overlay (contours + metrics)")
        axes[0, 0].axis("off")

        im1 = axes[0, 1].imshow(b, cmap="hot")
        axes[0, 1].set_title("Enhanced LAB-b (TopHat + CLAHE)")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1])

        axes[1, 0].imshow(pred, cmap="gray")
        axes[1, 0].set_title("Predicted mask")
        axes[1, 0].axis("off")

        if gt is not None:
            axes[1, 1].imshow(gt, cmap="gray")
            axes[1, 1].set_title("Ground truth")
            axes[1, 1].axis("off")
        else:
            axes[1, 1].axis("off")
            txt = "GLCM features\n\n" + "\n".join([f"{k}: {v:.4f}" for k, v in results["texture_features"].items()])
            axes[1, 1].text(0.05, 0.5, txt, fontsize=11, family="monospace", va="center")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        plt.show()


def process_dataset(images_dir: str, masks_dir: Optional[str] = None, output_dir: Optional[str] = None):
    detector = RustDetector()

    img_paths = sorted(list(Path(images_dir).glob("*.png")))
    if not img_paths:
        raise RuntimeError(f"No .png images found in: {images_dir}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metrics_list: List[Dict[str, float]] = []

    print("Dataset run: iterate images -> pair mask -> run pipeline -> collect metrics -> summarize")

    for p in img_paths:
        gt_path = None
        if masks_dir:
            m1 = Path(masks_dir) / p.name
            m2 = Path(masks_dir) / f"{p.stem}_mask.png"
            if m1.exists():
                gt_path = str(m1)
            elif m2.exists():
                gt_path = str(m2)

        try:
            res = detector.process_image(str(p), gt_path)
            print(f"\n{p.name}")
            print(f"  Rust ratio: {res['rust_ratio']:.2f}%")
            print(f"  Degradation score: {res['degradation_score']:.2f}")

            if res["evaluation_metrics"]:
                ev = res["evaluation_metrics"]
                print(f"  IoU: {ev['iou']:.3f} | Dice: {ev['dice']:.3f} | Prec: {ev['precision']:.3f} | Rec: {ev['recall']:.3f}")
                metrics_list.append(ev)

            if output_dir:
                out_path = str(Path(output_dir) / f"{p.stem}_result.png")
                detector.visualize(res, out_path)

        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    if metrics_list:
        avg_iou = float(np.mean([m["iou"] for m in metrics_list]))
        avg_dice = float(np.mean([m["dice"] for m in metrics_list]))
        avg_prec = float(np.mean([m["precision"] for m in metrics_list]))
        avg_rec = float(np.mean([m["recall"] for m in metrics_list]))

        print("\nSUMMARY")
        print(f"Average IoU: {avg_iou:.3f}")
        print(f"Average Dice: {avg_dice:.3f}")
        print(f"Average Precision: {avg_prec:.3f}")
        print(f"Average Recall: {avg_rec:.3f}")


if __name__ == "__main__":
    images_dir = r"data/images"
    masks_dir = r"data/masks"
    output_dir = r"results"
    process_dataset(images_dir, masks_dir, output_dir)
