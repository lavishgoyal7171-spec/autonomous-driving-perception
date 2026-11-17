"""Stage 4 - Traffic-signal analysis using HSV colour segmentation.

Two complementary ways of locating traffic lights:
  * YOLO "traffic light" boxes  -> the crop is classified with HSV masks
  * pure image processing      -> bright, saturated, circular blobs surrounded by
                                  a dark housing in the upper part of the frame
The per-frame state is smoothed by majority voting over the last few frames.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Config

STATES = ("RED", "YELLOW", "GREEN")

# HSV ranges (OpenCV hue is 0..179) for *lit* lamps: high saturation and value
HSV_RANGES = {
    "RED": [((0, 90, 140), (10, 255, 255)), ((160, 90, 140), (179, 255, 255))],
    "YELLOW": [((11, 90, 140), (35, 255, 255))],
    "GREEN": [((40, 90, 140), (92, 255, 255))],
}


@dataclass
class LightObservation:
    box: Tuple[int, int, int, int]
    state: str
    score: float
    source: str  # "yolo" or "hsv"


@dataclass
class SignalResult:
    state: str = "NONE"          # smoothed state passed to the decision module
    raw_state: str = "NONE"      # this frame only
    confidence: float = 0.0
    lights: List[LightObservation] = field(default_factory=list)
    debug: Dict[str, np.ndarray] = field(default_factory=dict)


def color_masks(hsv: np.ndarray) -> Dict[str, np.ndarray]:
    masks = {}
    for name, ranges in HSV_RANGES.items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        masks[name] = m
    return masks


def classify_crop(crop: np.ndarray, min_ratio: float = 0.04) -> Tuple[str, Dict[str, float]]:
    """Classify a traffic-light crop as RED / YELLOW / GREEN / UNKNOWN."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
        return "UNKNOWN", {}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    area = float(crop.shape[0] * crop.shape[1])
    scores = {k: cv2.countNonZero(m) / area for k, m in color_masks(hsv).items()}
    best = max(scores, key=scores.get)
    if scores[best] >= min_ratio:
        return best, scores

    # Fallback: position of the brightest lamp in a vertical housing
    h, w = crop.shape[:2]
    if h > 1.6 * w:
        v = hsv[:, :, 2].astype(np.float32)
        thirds = [v[i * h // 3:(i + 1) * h // 3].mean() for i in range(3)]
        if max(thirds) - min(thirds) > 40:
            return STATES[int(np.argmax(thirds))], scores
    return "UNKNOWN", scores


def _inside(pt, box) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class TrafficLightAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history: deque = deque(maxlen=cfg.signal_history)

    # ------------------------------------------------------- HSV blob search
    def find_hsv_lights(self, frame: np.ndarray, hsv: np.ndarray,
                        exclude: Sequence[Tuple[int, int, int, int]] = ()) -> Tuple[List[LightObservation], np.ndarray]:
        h, w = frame.shape[:2]
        top, bottom = int(self.cfg.signal_search_top * h), int(self.cfg.signal_search_bottom * h)
        band = hsv[top:bottom]
        value = band[:, :, 2]
        dark = (value < 90).astype(np.uint8)
        debug = np.zeros((h, w), np.uint8)
        found: List[LightObservation] = []
        kernel = np.ones((3, 3), np.uint8)
        for state, mask in color_masks(band).items():
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            debug[top:bottom] |= mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 12 or area > 0.003 * w * h:
                    continue
                perim = cv2.arcLength(c, True)
                circularity = 4 * np.pi * area / (perim * perim + 1e-6)
                x, y, bw, bh = cv2.boundingRect(c)
                if circularity < 0.65 or not (0.6 <= bw / float(bh) <= 1.6):
                    continue
                # lamps are round: reject diamonds / squares (e.g. yellow warning signs)
                (_, _), radius = cv2.minEnclosingCircle(c)
                if area / (np.pi * radius * radius + 1e-6) < 0.70:
                    continue
                # the lamp must sit in a dark housing: check a ring around it
                r = max(bw, bh)
                x1, y1 = max(0, x - r), max(0, y - r)
                x2, y2 = min(w, x + bw + r), min(band.shape[0], y + bh + r)
                ring = dark[y1:y2, x1:x2].copy()
                ring[y - y1:y - y1 + bh, x - x1:x - x1 + bw] = 0
                ring_area = (y2 - y1) * (x2 - x1) - bw * bh
                darkness = ring.sum() / float(max(ring_area, 1))
                if darkness < 0.25:
                    continue
                # ...and the housing continues past the lamp (other unlit lamps):
                # a dark block of ~2 lamp sizes above/below (or left/right) of it
                def dark_frac(xa, ya, xb, yb):
                    xa, ya, xb, yb = max(0, xa), max(0, ya), min(w, xb), min(band.shape[0], yb)
                    return dark[ya:yb, xa:xb].mean() if xb > xa and yb > ya else 0.0
                housing = max(dark_frac(x, y - 2 * bh, x + bw, y),            # above
                              dark_frac(x, y + bh, x + bw, y + 3 * bh),        # below
                              dark_frac(x - 2 * bw, y, x, y + bh),             # left
                              dark_frac(x + bw, y, x + 3 * bw, y + bh))        # right
                if housing < 0.6:
                    continue
                box = (x1, y1 + top, x2, y2 + top)
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if any(_inside(center, e) for e in exclude):
                    continue  # e.g. brake lights inside a detected vehicle
                found.append(LightObservation(box, state, float(area * darkness), "hsv"))
        return found, debug

    # ------------------------------------------------------------- analyse
    def analyze(self, frame: np.ndarray, hsv: np.ndarray, yolo_lights: Sequence = (),
                vehicle_boxes: Sequence = (), use_hsv_search: bool = False) -> SignalResult:
        res = SignalResult()
        for det in yolo_lights:
            x1, y1, x2, y2 = det.box
            state, scores = classify_crop(frame[max(0, y1):y2, max(0, x1):x2], self.cfg.signal_min_ratio)
            res.lights.append(LightObservation(det.box, state, det.conf, "yolo"))

        if use_hsv_search:
            hsv_lights, dbg = self.find_hsv_lights(frame, hsv, vehicle_boxes)
            res.debug["signal_mask"] = dbg
            for obs in hsv_lights:
                if not any(_iou(obs.box, l.box) > 0.1 or _inside(
                        ((obs.box[0] + obs.box[2]) / 2, (obs.box[1] + obs.box[3]) / 2), l.box)
                        for l in res.lights):
                    res.lights.append(obs)

        known = [l for l in res.lights if l.state in STATES]
        if known:
            # most relevant light = the largest (closest) one
            main = max(known, key=lambda l: (l.box[2] - l.box[0]) * (l.box[3] - l.box[1]))
            res.raw_state = main.state
        elif res.lights:
            res.raw_state = "UNKNOWN"

        self.history.append(res.raw_state)
        votes = Counter(s for s in self.history if s in STATES)
        if votes:
            state, n = votes.most_common(1)[0]
            if n >= min(2, len(self.history)):
                res.state, res.confidence = state, n / float(len(self.history))
        if res.state == "NONE" and res.raw_state == "UNKNOWN":
            res.state = "UNKNOWN"
        return res
