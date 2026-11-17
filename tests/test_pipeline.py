"""Unit tests for every stage of the pipeline (run with:  python -m pytest -q)."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.decision import DecisionModule
from src.lane_detection import LaneDetector, LaneResult
from src.object_detection import Detection
from src.pipeline import PerceptionPipeline
from src.preprocessing import preprocess, resize_frame
from src.traffic_light import SignalResult, TrafficLightAnalyzer, classify_crop
from src.tracking import ObjectTracker, Track, iou


# ----------------------------------------------------------------- helpers
def road_frame(offset_px=0, w=960, h=540):
    """Simple synthetic road: grey asphalt, yellow left line, white right line."""
    img = np.full((h, w, 3), (90, 140, 90), np.uint8)
    vp = (w // 2, int(0.55 * h))
    cv2.fillPoly(img, [np.array([(0, h), (vp[0] - 5, vp[1]), (vp[0] + 5, vp[1]), (w, h)])], (85, 85, 85))
    cx = w // 2 - offset_px
    cv2.line(img, (cx - 300, h), vp, (0, 200, 240), 12)
    cv2.line(img, (cx + 300, h), vp, (235, 235, 235), 12)
    return img


def traffic_light(state, size=(40, 100)):
    w, h = size
    img = np.full((h, w, 3), 25, np.uint8)
    colors = {"RED": (0, 0, 255), "YELLOW": (0, 200, 255), "GREEN": (120, 255, 0)}
    for i, s in enumerate(("RED", "YELLOW", "GREEN")):
        cv2.circle(img, (w // 2, int((i + 0.5) * h / 3)), 12, colors[s] if s == state else (55, 55, 55), -1)
    return img


def lane_result(offset_norm=0.0, detected=True):
    r = LaneResult(y_bottom=539, y_top=324, vehicle_center_x=480)
    if detected:
        r.left, r.right = (180.0, 470.0), (780.0, 490.0)
        r.offset_norm, r.offset_m = offset_norm, offset_norm * 3.7
        r.status = "BOTH"
    return r


def track(label="car", prox=0.2, in_path=True, approaching=False, tid=1):
    t = Track(tid, (400, 300, 560, 420), label, 0.9, 0)
    t.proximity, t.in_path, t.approaching = prox, in_path, approaching
    return t


# ----------------------------------------------------------------- preprocessing
def test_resize_keeps_aspect():
    out = resize_frame(np.zeros((720, 1280, 3), np.uint8), 960)
    assert out.shape[:2] == (540, 960)


def test_preprocess_outputs():
    pre = preprocess(road_frame(), Config())
    assert pre.gray.ndim == 2 and pre.blurred.shape == pre.gray.shape
    assert pre.hsv.shape == pre.frame.shape
    assert set(np.unique(pre.binary)) <= {0, 255}


# ----------------------------------------------------------------- lanes
def test_lane_centered():
    cfg = Config()
    det = LaneDetector(cfg)
    res = det.detect(preprocess(road_frame(0), cfg))
    assert res.detected and res.status == "BOTH"
    assert abs(res.offset_norm) < 0.05


@pytest.mark.parametrize("shift,sign", [(150, 1), (-150, -1)])
def test_lane_offset_sign(shift, sign):
    cfg = Config()
    res = LaneDetector(cfg).detect(preprocess(road_frame(shift), cfg))
    assert res.detected
    assert np.sign(res.offset_norm) == sign
    assert abs(res.offset_norm) > 0.15


def test_lane_memory_when_lost():
    cfg = Config()
    det = LaneDetector(cfg)
    det.detect(preprocess(road_frame(), cfg))
    blank = np.full((540, 960, 3), 90, np.uint8)
    res = det.detect(preprocess(blank, cfg))
    assert res.detected and res.status == "MEMORY"
    for _ in range(cfg.lane_memory_frames + 1):
        res = det.detect(preprocess(blank, cfg))
    assert not res.detected


# ----------------------------------------------------------------- traffic lights
@pytest.mark.parametrize("state", ["RED", "YELLOW", "GREEN"])
def test_classify_crop(state):
    assert classify_crop(traffic_light(state))[0] == state


def test_hsv_light_search_finds_light():
    cfg = Config()
    img = np.full((540, 960, 3), (200, 170, 120), np.uint8)  # sky-like background
    img[50:150, 700:740] = traffic_light("RED")
    an = TrafficLightAnalyzer(cfg)
    res = an.analyze(img, cv2.cvtColor(img, cv2.COLOR_BGR2HSV), use_hsv_search=True)
    assert res.raw_state == "RED"


def test_signal_majority_vote_suppresses_flicker():
    cfg = Config()
    an = TrafficLightAnalyzer(cfg)
    crop_g = traffic_light("GREEN")
    img = np.full((540, 960, 3), 120, np.uint8)
    img[50:150, 700:740] = crop_g
    det = Detection((700, 50, 740, 150), "traffic light", 0.9)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for _ in range(4):
        an.analyze(img, hsv, yolo_lights=[det])
    img2 = img.copy()
    img2[50:150, 700:740] = traffic_light("RED")
    res = an.analyze(img2, cv2.cvtColor(img2, cv2.COLOR_BGR2HSV), yolo_lights=[det])
    assert res.raw_state == "RED" and res.state == "GREEN"  # one noisy frame doesn't flip it


# ----------------------------------------------------------------- tracking
def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_tracker_keeps_id_and_detects_approach():
    cfg = Config()
    tr = ObjectTracker(cfg)
    ids = set()
    for f in range(8):
        g = 4 * f  # box grows -> object approaching
        dets = [Detection((400 - g, 300 - g, 500 + g, 380 + g), "car", 0.9)]
        tracks = tr.update(dets, f)
        tr.estimate_proximity(tracks, (540, 960), None)
        ids |= {t.track_id for t in tracks}
    assert ids == {1}
    assert tracks[0].approaching and tracks[0].in_path


def test_tracker_drops_lost_tracks():
    cfg = Config()
    tr = ObjectTracker(cfg)
    tr.update([Detection((10, 10, 50, 50), "person", 0.8)], 0)
    for f in range(1, cfg.track_max_missed + 3):
        tr.update([], f)
    assert tr.tracks == []


# ----------------------------------------------------------------- decision
def decide(lane=None, signal="NONE", tracks=()):
    dm = DecisionModule(Config())
    return dm.rule_based(lane or lane_result(), SignalResult(state=signal), list(tracks))[0]


def test_decision_rules():
    assert decide() == "MAINTAIN"
    assert decide(signal="RED") == "STOP"
    assert decide(signal="YELLOW") == "SLOW DOWN"
    assert decide(signal="GREEN") == "MAINTAIN"
    assert decide(lane_result(0.3)) == "STEER LEFT"
    assert decide(lane_result(-0.3)) == "STEER RIGHT"
    assert decide(tracks=[track(prox=0.9)]) == "STOP"
    assert decide(tracks=[track(prox=0.6)]) == "SLOW DOWN"
    assert decide(tracks=[track(prox=0.4, approaching=True)]) == "SLOW DOWN"
    assert decide(tracks=[track(prox=0.9, in_path=False)]) == "MAINTAIN"
    assert decide(tracks=[track("person", prox=0.35)]) == "STOP"


def test_red_light_has_priority_over_steering():
    assert decide(lane_result(0.4), signal="RED") == "STOP"


def test_decision_hysteresis():
    cfg = Config()
    dm = DecisionModule(cfg)
    lane = lane_result()
    assert dm.decide(lane, SignalResult(state="RED"), []).action == "STOP"     # escalates at once
    assert dm.decide(lane, SignalResult(state="GREEN"), []).action == "STOP"   # holds
    for _ in range(cfg.decision_hold_frames):
        out = dm.decide(lane, SignalResult(state="GREEN"), [])
    assert out.action == "MAINTAIN"


# ----------------------------------------------------------------- end-to-end
def test_pipeline_without_yolo_renders_dashboard():
    cfg = Config()
    pipe = PerceptionPipeline(cfg, use_yolo=False)
    res = pipe.process_frame(road_frame(), 0)
    assert res.lane.detected
    assert res.dashboard.shape[1] == cfg.frame_width + cfg.panel_width
    row = pipe.log_row(res)
    assert row["decision"] == "MAINTAIN"
