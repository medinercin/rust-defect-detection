# Rust Defect Segmentation (YCbCr-Based Image Processing)

This project aims to detect and segment rust defects on surface images using classical
image processing techniques. The approach is fully explainable and does not rely on
deep learning methods.

The main observation behind the method is:
- Rust regions tend to appear **more reddish** → higher **Cr** values
- Rust regions are usually **darker** → lower **Y (luminance)** values

Based on this observation, a rule-based segmentation is applied in the **YCrCb color space**
and refined using morphological operations.

---

## Dataset

The dataset used in this project is derived from a publicly available surface defect dataset.

**Original dataset source:**
https://github.com/ben-z-original/s2ds

Only rust-related surface images and their corresponding masks are used in this study.

### Generated results
- `outputs/rust_detection_results.csv`  
  Contains per-image rust ratio, degradation score, and evaluation metrics.

- `outputs/rust_detection_report.txt`  
  A human-readable summary report suitable for academic evaluation.

- `outputs/images/`  
  Segmentation result images showing rust regions overlaid on the original images.

## How to Run

### 1. Create a virtual environment (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python rust_defect\run.py



## Notes
This project intentionally avoids deep learning approaches.

All processing steps are explainable and suitable for academic purposes.

The code structure is kept simple and readable to reflect a student-level implementation.