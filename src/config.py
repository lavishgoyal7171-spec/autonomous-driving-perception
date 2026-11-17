"""Central configuration for the perception pipeline.

All tunable thresholds live here so experiments can be run by editing one file
(or by overriding fields from the command line in ``main.py``).
"""
from dataclasses import dataclass, field
import math


@dataclass
class Config:
    # ------------------------------------------------------------------ input
    frame_width: int = 960            # frames are resized to this width (aspect kept)

    # ---------------------------------------------------------- preprocessing
    use_clahe: bool = True            # contrast enhancement (CLAHE on LAB L-channel)
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    blur_method: str = "gaussian"     # "gaussian" or "median"
    blur_kernel: int = 5

    # --------------------------------------------------------- lane detection
    canny_low: int = 50
    canny_high: int = 150
    # Region of interest (trapezoid) as fractions of frame width / height
    roi_top_y: float = 0.60
    roi_top_left_x: float = 0.40
    roi_top_right_x: float = 0.60
    roi_bottom_left_x: float = 0.02
    roi_bottom_right_x: float = 0.98
    roi_bottom_y: float = 0.95        # ignore the very bottom (car bonnet)
    # Probabilistic Hough transform
    hough_rho: float = 2
    hough_theta: float = math.pi / 180
    hough_threshold: int = 30
    hough_min_line_len: int = 25
    hough_max_line_gap: int = 100
    min_abs_slope: float = 0.45       # reject near-horizontal segments
    max_abs_slope: float = 4.0        # reject near-vertical noise
    lane_smoothing: float = 0.75      # EMA weight of the previous frame (0 = no smoothing)
    lane_memory_frames: int = 12      # keep last good lane this many frames when lost
    lane_width_m: float = 3.7         # standard lane width, used for metric offset
    min_lane_width_frac: float = 0.25 # sanity check on lane width at the bottom row
    max_lane_width_frac: float = 1.30

    # ------------------------------------------------------- object detection
    yolo_model: str = "models/yolov8n.pt"
    det_conf: float = 0.35
    det_imgsz: int = 640
    device: str = "auto"              # "auto", "cpu", "mps", "cuda"

    # --------------------------------------------------- traffic-signal (HSV)
    signal_mode: str = "auto"         # "yolo", "hsv", "both", "auto", "off"
    signal_search_top: float = 0.0    # HSV blob search band (fractions of height)
    signal_search_bottom: float = 0.48
    signal_min_ratio: float = 0.04    # min lit-pixel ratio inside a light crop
    signal_history: int = 5           # frames used for majority voting

    # --------------------------------------------------- tracking / proximity
    track_iou_threshold: float = 0.25
    track_max_missed: int = 8
    track_min_hits: int = 2
    approach_growth: float = 0.015    # per-frame relative height growth => approaching
    prox_near: float = 0.55           # proximity score thresholds (0 = horizon, 1 = bumper)
    prox_medium: float = 0.30
    prox_stop: float = 0.80

    # ----------------------------------------------------------------- decision
    steer_threshold: float = 0.15     # |offset| as a fraction of lane width
    decision_hold_frames: int = 4     # frames a calmer action must persist before switching
    lane_lost_slowdown_frames: int = 30

    # ------------------------------------------------------------ visualisation
    panel_width: int = 330
    thumb_height: int = 180

    extra: dict = field(default_factory=dict)

    def roi_vertices(self, w: int, h: int):
        """Trapezoid ROI in pixel coordinates (bottom-left, top-left, top-right, bottom-right)."""
        return [
            (int(self.roi_bottom_left_x * w), int(self.roi_bottom_y * h)),
            (int(self.roi_top_left_x * w), int(self.roi_top_y * h)),
            (int(self.roi_top_right_x * w), int(self.roi_top_y * h)),
            (int(self.roi_bottom_right_x * w), int(self.roi_bottom_y * h)),
        ]
