"""End-to-end perception pipeline tying all stages together.

ROAD VIDEO -> FRAME EXTRACTION -> PREPROCESSING -> LANE DETECTION + OBJECT DETECTION
+ TRAFFIC-SIGNAL ANALYSIS -> TRACKING / PROXIMITY -> DECISION -> ANNOTATED VIDEO
"""
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import Config
from .decision import Decision, DecisionModule
from .lane_detection import LaneDetector, LaneResult
from .object_detection import VEHICLE_LABELS, Detection
from .preprocessing import Preprocessed, preprocess
from .traffic_light import SignalResult, TrafficLightAnalyzer
from .tracking import ObjectTracker, Track
from . import visualization as viz

LOG_FIELDS = ["frame", "proc_ms", "fps", "lane_status", "lane_detected", "left_fresh", "right_fresh",
              "offset_m", "offset_norm", "heading_deg", "hough_segments", "signal_raw", "signal_state",
              "num_lights", "num_detections", "num_tracks", "num_vehicles", "num_persons", "mean_conf",
              "track_ids", "lead_id", "lead_label", "lead_proximity", "lead_approaching",
              "raw_decision", "decision", "reason"]


@dataclass
class FrameResult:
    index: int
    pre: Preprocessed
    lane: LaneResult
    detections: List[Detection]
    tracks: List[Track]
    signal: SignalResult
    decision: Decision
    proc_ms: float
    dashboard: Optional[np.ndarray] = None


class PerceptionPipeline:
    def __init__(self, cfg: Config, use_yolo: bool = True):
        self.cfg = cfg
        self.lane_detector = LaneDetector(cfg)
        self.signal_analyzer = TrafficLightAnalyzer(cfg)
        self.tracker = ObjectTracker(cfg)
        self.decider = DecisionModule(cfg)
        self.detector = None
        if use_yolo:
            try:
                from .object_detection import ObjectDetector
                self.detector = ObjectDetector(cfg)
                print(f"[pipeline] YOLO model '{cfg.yolo_model}' loaded on {self.detector.device}")
            except Exception as exc:  # pragma: no cover - depends on environment
                print(f"[pipeline] YOLO unavailable ({exc}); continuing without object detection")
        mode = cfg.signal_mode
        if mode == "auto":
            mode = "yolo" if self.detector is not None else "hsv"
        self.signal_mode = mode
        self._fps_ema = None

    def process_frame(self, frame: np.ndarray, index: int, render: bool = True) -> FrameResult:
        t0 = time.perf_counter()
        pre = preprocess(frame, self.cfg)                         # stage 2
        lane = self.lane_detector.detect(pre)                     # stage 3
        dets = self.detector.detect(pre.frame) if self.detector else []   # stage 5
        lights = [d for d in dets if d.label == "traffic light"]
        vehicles = [d.box for d in dets if d.label in VEHICLE_LABELS]
        signal = self.signal_analyzer.analyze(                    # stage 4
            pre.frame, pre.hsv,
            yolo_lights=lights if self.signal_mode in ("yolo", "both") else [],
            vehicle_boxes=vehicles,
            use_hsv_search=self.signal_mode in ("hsv", "both"))
        tracks = self.tracker.update(dets, index)                 # stage 6
        self.tracker.estimate_proximity(tracks, pre.frame.shape, lane)
        decision = self.decider.decide(lane, signal, tracks)      # stage 7
        proc_ms = (time.perf_counter() - t0) * 1000.0

        fps_now = 1000.0 / max(proc_ms, 1e-3)
        self._fps_ema = fps_now if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 * fps_now
        res = FrameResult(index, pre, lane, dets, tracks, signal, decision, proc_ms)
        if render:
            res.dashboard = self.render(res)
        return res

    def render(self, res: FrameResult) -> np.ndarray:
        cfg = self.cfg
        img = res.pre.frame.copy()
        viz.draw_lane(img, res.lane, cfg)
        viz.draw_tracks(img, res.tracks, res.decision.critical_track)
        viz.draw_lights(img, res.signal)
        viz.draw_banner(img, res.decision)
        panel = viz.build_panel(img.shape[0], cfg.panel_width, res.lane, res.signal, res.tracks,
                                res.decision, res.index, self._fps_ema or 0.0, self.detector is not None)
        roi_edges = cv2.cvtColor(res.lane.debug["masked_edges"], cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        cv2.polylines(roi_edges, [np.array(cfg.roi_vertices(w, h), np.int32)], True, (0, 200, 255), 2)
        hough = cv2.addWeighted(res.pre.frame, 0.45, res.lane.debug["hough"], 1.0, 0)
        stages = {
            "1. CLAHE + gray + blur": res.pre.blurred,
            "2. White/yellow threshold": res.lane.debug["color_mask"],
            "3. Canny edges in ROI": roi_edges,
            "4. Hough segments (L/R)": hough,
        }
        return viz.compose_dashboard(img, stages, panel, cfg)

    def log_row(self, res: FrameResult) -> Dict:
        lane, sig, dec = res.lane, res.signal, res.decision
        in_path = sorted([t for t in res.tracks if t.in_path], key=lambda t: -t.proximity)
        lead = in_path[0] if in_path else None
        confs = [d.conf for d in res.detections if d.label != "traffic light"]
        r = lambda v, n=4: "" if v is None else round(float(v), n)
        return {
            "frame": res.index, "proc_ms": round(res.proc_ms, 2),
            "fps": round(1000.0 / max(res.proc_ms, 1e-3), 2),
            "lane_status": lane.status, "lane_detected": int(lane.detected),
            "left_fresh": int(lane.left_fresh), "right_fresh": int(lane.right_fresh),
            "offset_m": r(lane.offset_m), "offset_norm": r(lane.offset_norm),
            "heading_deg": r(lane.heading_deg, 2), "hough_segments": lane.num_segments,
            "signal_raw": sig.raw_state, "signal_state": sig.state, "num_lights": len(sig.lights),
            "num_detections": len(confs), "num_tracks": len(res.tracks),
            "num_vehicles": sum(t.label in VEHICLE_LABELS for t in res.tracks),
            "num_persons": sum(t.label == "person" for t in res.tracks),
            "mean_conf": r(np.mean(confs)) if confs else "",
            "track_ids": " ".join(str(t.track_id) for t in res.tracks),
            "lead_id": lead.track_id if lead else "", "lead_label": lead.label if lead else "",
            "lead_proximity": r(lead.proximity, 3) if lead else "",
            "lead_approaching": int(lead.approaching) if lead else "",
            "raw_decision": dec.raw_action, "decision": dec.action, "reason": dec.reason,
        }


def open_source(src: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {src}")
    return cap


def make_writer(path: str, fps: float, size) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not open a video writer for {path}")


def run_video(pipeline: PerceptionPipeline, source: str, output: Optional[str] = None,
              log_path: Optional[str] = None, show: bool = False, max_frames: int = 0,
              snapshot_dir: Optional[str] = None, snapshot_every: int = 0) -> Dict:
    cap = open_source(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if src_fps <= 1 or src_fps > 120:
        src_fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    writer = None
    log_file = csv_writer = None
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        log_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        csv_writer.writeheader()
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    idx, t_start, paused = 0, time.perf_counter(), False
    render = bool(output or show or snapshot_dir)
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                res = pipeline.process_frame(frame, idx, render=render)
                if csv_writer:
                    csv_writer.writerow(pipeline.log_row(res))
                if output:
                    if writer is None:
                        hh, ww = res.dashboard.shape[:2]
                        writer = make_writer(output, src_fps, (ww, hh))
                    writer.write(res.dashboard)
                if snapshot_dir and snapshot_every and idx % snapshot_every == 0:
                    cv2.imwrite(os.path.join(snapshot_dir, f"frame_{idx:05d}.jpg"), res.dashboard)
                if idx % 50 == 0:
                    prog = f"{idx}/{total}" if total else str(idx)
                    print(f"[run] frame {prog}  {res.proc_ms:6.1f} ms  lane={res.lane.status:<10} "
                          f"signal={res.signal.state:<7} objects={len(res.tracks):2d}  -> {res.decision.action}")
                idx += 1
                if max_frames and idx >= max_frames:
                    break
            if show:
                cv2.imshow("Autonomous Driving Perception (q: quit, space: pause)", res.dashboard)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = not paused
    finally:
        cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if show:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    stats = {"frames": idx, "seconds": round(elapsed, 2), "avg_fps": round(idx / max(elapsed, 1e-6), 2),
             "output": output, "log": log_path}
    print(f"[run] processed {idx} frames in {elapsed:.1f}s ({stats['avg_fps']} FPS overall)")
    return stats
