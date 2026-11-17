"""Save a Matplotlib figure showing every image-processing stage for one frame.

    python scripts/visualize_stages.py --input data/project_video.mp4 --frame 600
    python scripts/visualize_stages.py --input data/synthetic_road.mp4 --frame 350 --signal-mode both

Useful for the report / viva: it shows resizing, CLAHE enhancement, grayscale,
Gaussian blur, Otsu thresholding, HSV conversion, white/yellow colour masks,
Canny edges, region of interest, Hough lines, HSV traffic-light mask and the
final annotated output.
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config  # noqa: E402
from src.pipeline import PerceptionPipeline  # noqa: E402
from src.traffic_light import color_masks  # noqa: E402


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--signal-mode", default="auto")
    a = ap.parse_args()

    cfg = Config()
    cfg.signal_mode = a.signal_mode
    pipe = PerceptionPipeline(cfg, use_yolo=not a.no_yolo)
    cap = cv2.VideoCapture(a.input)
    res, raw = None, None
    # run the pipeline up to the requested frame so temporal smoothing is warmed up
    for i in range(a.frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        raw = frame
        res = pipe.process_frame(frame, i, render=(i == a.frame))
    cap.release()
    if res is None or res.dashboard is None:
        raise SystemExit("could not read the requested frame")

    pre, lane = res.pre, res.lane
    h, w = pre.frame.shape[:2]
    roi_vis = pre.frame.copy()
    cv2.polylines(roi_vis, [np.array(cfg.roi_vertices(w, h), np.int32)], True, (0, 255, 255), 3)
    roi_vis = cv2.addWeighted(roi_vis, 0.75, cv2.cvtColor(lane.debug["roi"], cv2.COLOR_GRAY2BGR), 0.25, 0)
    hough = cv2.addWeighted(pre.frame, 0.5, lane.debug["hough"], 1.0, 0)
    sig = np.zeros((h, w), np.uint8)
    for m in color_masks(pre.hsv).values():
        sig |= m
    annotated = res.dashboard[:h, :w]

    panels = [
        ("1. Input frame (original size %dx%d)" % (raw.shape[1], raw.shape[0]), rgb(raw)),
        ("2. Resized + CLAHE contrast enhancement", rgb(pre.enhanced)),
        ("3. Grayscale", pre.gray),
        ("4. Gaussian blur (noise reduction)", pre.blurred),
        ("5. Otsu threshold", pre.binary),
        ("6. HSV - hue channel", pre.hsv[:, :, 0]),
        ("7. White/yellow mask + morphology (ROI)", lane.debug["color_mask"]),
        ("8. Canny edge detection", lane.debug["edges"]),
        ("9. Trapezoidal region of interest", rgb(roi_vis)),
        ("10. Edges inside ROI (lane evidence)", lane.debug["masked_edges"]),
        ("11. Hough segments: left (blue) / right (red)", rgb(hough)),
        ("12. HSV red/yellow/green masks (before shape filter)", sig),
        ("13. Final annotated output", rgb(annotated)),
    ]
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 3.2))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
    name = os.path.splitext(os.path.basename(a.input))[0]
    fig.suptitle(f"Processing stages - {name}, frame {a.frame}  |  lane {lane.status}, "
                 f"signal {res.signal.state}, decision {res.decision.action}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = a.out or os.path.join("outputs", f"stages_{name}_{a.frame}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
