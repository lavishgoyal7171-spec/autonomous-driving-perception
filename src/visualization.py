"""Annotated output frame + dashboard (side panel and processing-stage strip)."""
from collections import Counter
from typing import Dict, List, Sequence

import cv2
import numpy as np

from .config import Config
from .decision import Decision
from .lane_detection import LaneResult
from .traffic_light import SignalResult
from .tracking import Track

FONT = cv2.FONT_HERSHEY_SIMPLEX

ACTION_COLORS = {
    "MAINTAIN": (60, 180, 60),
    "STEER LEFT": (200, 140, 30),
    "STEER RIGHT": (200, 140, 30),
    "SLOW DOWN": (0, 170, 255),
    "STOP": (40, 40, 220),
}
SIGNAL_COLORS = {"RED": (40, 40, 230), "YELLOW": (0, 210, 255), "GREEN": (60, 200, 60),
                 "UNKNOWN": (160, 160, 160), "NONE": (110, 110, 110)}
CLASS_COLORS = {"car": (255, 180, 0), "truck": (255, 120, 60), "bus": (200, 90, 200),
                "motorcycle": (0, 220, 220), "bicycle": (0, 200, 140), "person": (180, 60, 255),
                "stop sign": (60, 60, 255)}


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1, bg=None):
    if bg is not None:
        (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_lane(img: np.ndarray, lane: LaneResult, cfg: Config) -> None:
    h, w = img.shape[:2]
    cv2.polylines(img, [np.array(cfg.roi_vertices(w, h), np.int32)], True, (90, 90, 90), 1, cv2.LINE_AA)
    if not lane.detected:
        return
    yb, yt = lane.y_bottom, int(lane.draw_top_y)
    lb, rb = lane.left[0], lane.right[0]
    lt, rt = lane.x_at(lane.left, yt), lane.x_at(lane.right, yt)
    poly = np.array([(lb, yb), (lt, yt), (rt, yt), (rb, yb)], np.int32)
    fresh = lane.status == "BOTH"
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], (60, 200, 60) if fresh else (0, 160, 230))
    cv2.addWeighted(overlay, 0.28, img, 0.72, 0, dst=img)
    for (xb, xt), ok in (((lb, lt), lane.left_fresh), ((rb, rt), lane.right_fresh)):
        cv2.line(img, (int(xb), yb), (int(xt), yt), (0, 255, 255) if ok else (0, 140, 255), 6, cv2.LINE_AA)
    # lane centre vs. vehicle centre
    cx, vx = int(lane.lane_center_x), int(lane.vehicle_center_x)
    ct = int((lt + rt) / 2)
    cv2.line(img, (cx, yb), (ct, yt), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, yb - 18), 7, (255, 255, 255), -1)
    cv2.line(img, (vx, yb), (vx, yb - 40), (0, 0, 255), 3)
    cv2.arrowedLine(img, (vx, yb - 18), (cx, yb - 18), (0, 0, 255), 2, tipLength=0.3)


def draw_tracks(img, tracks: Sequence[Track], critical_id=None):
    for t in tracks:
        x1, y1, x2, y2 = t.box
        color = CLASS_COLORS.get(t.label, (200, 200, 200))
        thick = 2
        if t.in_path and t.proximity_level in ("NEAR", "MEDIUM"):
            color = (0, 0, 255) if t.proximity_level == "NEAR" else (0, 140, 255)
            thick = 3
        if t.track_id == critical_id:
            thick = 4
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        tag = f"#{t.track_id} {t.label} {t.conf:.2f} {t.proximity_level}"
        if t.approaching:
            tag += " APPR"
        _text(img, tag, (x1, max(14, y1 - 4)), 0.45, (0, 0, 0), 1, bg=color)


def draw_lights(img, signal: SignalResult):
    for l in signal.lights:
        x1, y1, x2, y2 = l.box
        c = SIGNAL_COLORS.get(l.state, (160, 160, 160))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        _text(img, f"TL {l.state} ({l.source})", (x1, max(14, y1 - 4)), 0.42, (0, 0, 0), 1, bg=c)


def draw_banner(img, decision: Decision):
    w = img.shape[1]
    color = ACTION_COLORS.get(decision.action, (128, 128, 128))
    label = decision.action
    (tw, th), _ = cv2.getTextSize(label, FONT, 1.1, 3)
    x = (w - tw) // 2
    cv2.rectangle(img, (x - 18, 8), (x + tw + 18, 20 + th + 12), color, -1)
    cv2.rectangle(img, (x - 18, 8), (x + tw + 18, 20 + th + 12), (255, 255, 255), 2)
    _text(img, label, (x, 18 + th), 1.1, (255, 255, 255), 3)


def _thumb(img, size, title):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    _text(t, title, (6, 18), 0.48, (255, 255, 255), 1, bg=(0, 0, 0))
    cv2.rectangle(t, (0, 0), (size[0] - 1, size[1] - 1), (70, 70, 70), 1)
    return t


def build_panel(h, width, lane: LaneResult, signal: SignalResult, tracks: Sequence[Track],
                decision: Decision, frame_idx: int, fps: float, det_enabled: bool):
    p = np.full((h, width, 3), 28, np.uint8)
    y = 30
    _text(p, "PERCEPTION DASHBOARD", (12, y), 0.62, (255, 255, 255), 2)
    y += 26
    _text(p, f"Frame {frame_idx}   {fps:5.1f} FPS", (12, y), 0.5, (190, 190, 190))

    def section(title):
        nonlocal y
        y += 30
        cv2.line(p, (10, y - 18), (width - 10, y - 18), (70, 70, 70), 1)
        _text(p, title, (12, y), 0.52, (0, 200, 255), 1)

    def line(s, color=(230, 230, 230)):
        nonlocal y
        y += 22
        _text(p, s, (20, y), 0.48, color)

    section("LANE")
    line(f"Status : {lane.status}", (60, 220, 60) if lane.status == "BOTH" else (0, 170, 255))
    if lane.offset_m is not None:
        line(f"Offset : {lane.offset_m:+.2f} m  ({lane.offset_norm * 100:+.0f}%)")
        line(f"Heading: {lane.heading_deg:+.1f} deg")
        # offset gauge
        y += 16
        gx0, gx1 = 30, width - 30
        gm = (gx0 + gx1) // 2
        cv2.line(p, (gx0, y), (gx1, y), (120, 120, 120), 2)
        cv2.line(p, (gm, y - 8), (gm, y + 8), (255, 255, 255), 1)
        pos = int(gm + np.clip(lane.offset_norm, -0.5, 0.5) * (gx1 - gx0))
        cv2.circle(p, (pos, y), 7, (0, 0, 255), -1)
        y += 4
    else:
        line("Offset : --")

    section("TRAFFIC SIGNAL")
    c = SIGNAL_COLORS.get(signal.state, (120, 120, 120))
    y += 24
    cv2.circle(p, (32, y - 6), 10, c, -1)
    _text(p, f"{signal.state}  (raw {signal.raw_state}, {len(signal.lights)} lights)", (50, y), 0.48)

    section("OBJECTS")
    if not det_enabled:
        line("YOLO disabled", (150, 150, 150))
    counts = Counter(t.label for t in tracks)
    line(", ".join(f"{k}:{v}" for k, v in counts.most_common()) or "none")
    in_path = sorted([t for t in tracks if t.in_path], key=lambda t: -t.proximity)
    if in_path:
        n = in_path[0]
        line(f"Lead: #{n.track_id} {n.label} {n.proximity_level}", (0, 170, 255))
        line(f"prox {n.proximity:.2f}  growth {n.growth * 100:+.1f}%/f")
    else:
        line("Lead: path clear", (60, 220, 60))

    section("DECISION")
    y += 34
    col = ACTION_COLORS.get(decision.action, (128, 128, 128))
    cv2.rectangle(p, (14, y - 28), (width - 14, y + 10), col, -1)
    _text(p, decision.action, (24, y), 0.8, (255, 255, 255), 2)
    line(decision.reason[:38])
    if decision.raw_action != decision.action:
        line(f"(raw: {decision.raw_action})", (150, 150, 150))
    _text(p, "Simulation only - no vehicle control", (12, h - 12), 0.4, (130, 130, 130))
    return p


def compose_dashboard(annotated, stages: Dict[str, np.ndarray], panel, cfg: Config):
    h, w = annotated.shape[:2]
    top = np.hstack([annotated, panel])
    total_w = top.shape[1]
    n = len(stages)
    tw = total_w // n
    th = int(tw * h / w)
    thumbs = [_thumb(img, (tw, th), title) for title, img in stages.items()]
    strip = np.hstack(thumbs)
    if strip.shape[1] < total_w:
        strip = np.hstack([strip, np.zeros((th, total_w - strip.shape[1], 3), np.uint8)])
    return np.vstack([top, strip])
