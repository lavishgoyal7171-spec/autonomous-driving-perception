"""Stage 6 - Multi-object tracking and relative proximity.

A lightweight IoU tracker (greedy association with a centroid-distance
fallback) keeps a persistent ID for every road object. From each track's
box history we derive:
  * proximity score  - how low the object's base sits in the image
                       (flat-road assumption) combined with its apparent size
  * growth rate      - relative change in box height per frame
                       (positive => getting closer / approaching)
These are *relative* cues, not metric distances.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .object_detection import VEHICLE_LABELS, Detection


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def same_group(a: str, b: str) -> bool:
    """Car/truck/bus are often confused frame-to-frame - treat them as compatible."""
    return a == b or (a in VEHICLE_LABELS and b in VEHICLE_LABELS)


@dataclass
class Track:
    track_id: int
    box: Tuple[int, int, int, int]
    label: str
    conf: float
    first_frame: int
    history: Deque = field(default_factory=lambda: deque(maxlen=30))
    hits: int = 1
    missed: int = 0
    # filled in by estimate_proximity()
    proximity: float = 0.0
    proximity_level: str = "FAR"
    growth: float = 0.0
    approaching: bool = False
    in_path: bool = False

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def height(self):
        return self.box[3] - self.box[1]


class ObjectTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self.next_id = 1
        self.finished_lengths: List[int] = []  # for tracking-consistency statistics

    def update(self, detections: Sequence[Detection], frame_idx: int) -> List[Track]:
        dets = [d for d in detections if d.label != "traffic light"]
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                if not same_group(t.label, d.label):
                    continue
                score = iou(t.box, d.box)
                if score < self.cfg.track_iou_threshold:
                    # centroid fallback for small / fast-moving objects
                    dist = np.hypot(t.center[0] - d.center[0], t.center[1] - d.center[1])
                    size = max(t.box[2] - t.box[0], t.height, 1)
                    if dist < 0.6 * size:
                        score = 0.01 + 0.2 * (1 - dist / (0.6 * size))
                    else:
                        continue
                pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            t.box, t.conf, t.label = d.box, d.conf, d.label
            t.hits += 1
            t.missed = 0
            t.history.append((frame_idx, d.box))

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(self.next_id, d.box, d.label, d.conf, frame_idx)
                t.history.append((frame_idx, d.box))
                self.tracks.append(t)
                self.next_id += 1

        alive = []
        for t in self.tracks:
            if t.missed > self.cfg.track_max_missed:
                self.finished_lengths.append(t.hits)
            else:
                alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.hits >= self.cfg.track_min_hits and t.missed == 0]

    def estimate_proximity(self, tracks: Sequence[Track], frame_shape, lane=None) -> None:
        h, w = frame_shape[:2]
        horizon = self.cfg.roi_top_y * h
        for t in tracks:
            x1, y1, x2, y2 = t.box
            ground = np.clip((y2 - horizon) / max(h - horizon, 1), 0, 1)
            size = np.clip(t.height / (0.6 * h), 0, 1)
            t.proximity = float(max(ground, size))
            if t.proximity >= self.cfg.prox_near:
                t.proximity_level = "NEAR"
            elif t.proximity >= self.cfg.prox_medium:
                t.proximity_level = "MEDIUM"
            else:
                t.proximity_level = "FAR"

            # growth: compare heights over the last few observations
            hist = list(t.history)[-6:]
            if len(hist) >= 3:
                (f0, b0), (f1, b1) = hist[0], hist[-1]
                h0, h1 = max(b0[3] - b0[1], 1), max(b1[3] - b1[1], 1)
                t.growth = (h1 - h0) / float(h0) / max(f1 - f0, 1)
            else:
                t.growth = 0.0
            t.approaching = t.growth > self.cfg.approach_growth

            # is the object inside our driving corridor?
            cx, base = (x1 + x2) / 2.0, float(y2)
            corridor: Optional[Tuple[float, float]] = lane.corridor_at(base) if lane is not None else None
            if corridor is None:
                # no lane information: assume a corridor around the image centre
                half = (0.12 + 0.25 * ground) * w
                corridor = (w / 2 - half, w / 2 + half)
            overlap = min(x2, corridor[1]) - max(x1, corridor[0])
            t.in_path = (corridor[0] <= cx <= corridor[1]) or overlap > 0.5 * (x2 - x1)
