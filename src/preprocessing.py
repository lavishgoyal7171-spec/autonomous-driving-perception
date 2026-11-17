"""Stage 2 - Image preprocessing.

Resizing, contrast enhancement, colour-space conversion, noise filtering and
thresholding. Every intermediate image is returned so it can be shown on the
dashboard and in the report.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class Preprocessed:
    frame: np.ndarray       # resized BGR frame (used for drawing)
    enhanced: np.ndarray    # contrast-enhanced BGR frame
    gray: np.ndarray        # grayscale of enhanced frame
    blurred: np.ndarray     # denoised grayscale
    hsv: np.ndarray         # HSV of the *original* resized frame (colours preserved)
    hls: np.ndarray         # HLS of the original resized frame (lane colour masks)
    binary: np.ndarray      # Otsu-thresholded grayscale (bright regions)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize keeping the aspect ratio."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, int(round(h * scale))), interpolation=interp)


def enhance_contrast(frame: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE on the lightness channel of LAB - boosts faded lane paint and
    dark/shadowed regions without shifting colours."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise(gray: np.ndarray, method: str = "gaussian", k: int = 5) -> np.ndarray:
    k = k if k % 2 == 1 else k + 1
    if method == "median":
        return cv2.medianBlur(gray, k)
    return cv2.GaussianBlur(gray, (k, k), 0)


def preprocess(frame: np.ndarray, cfg: Config) -> Preprocessed:
    frame = resize_frame(frame, cfg.frame_width)
    enhanced = enhance_contrast(frame, cfg.clahe_clip, cfg.clahe_grid) if cfg.use_clahe else frame
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = denoise(gray, cfg.blur_method, cfg.blur_kernel)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Preprocessed(frame, enhanced, gray, blurred, hsv, hls, binary)
