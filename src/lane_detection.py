"""Stage 3 - Lane detection.

Pipeline:
    colour thresholding (white / yellow lane paint, HLS + HSV)
    -> morphological opening / closing
    -> Canny edge detection
    -> trapezoidal region of interest
    -> probabilistic Hough transform
    -> slope-based left / right classification + weighted line fit
    -> temporal smoothing (EMA) and short-term memory
    -> lane centre, lateral offset (px, metres, fraction of lane width)
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import Config
from .preprocessing import Preprocessed

Line = Tuple[float, float]  # (x_at_bottom, x_at_top) - a lane line in a fixed y-band


@dataclass
class LaneResult:
    left: Optional[Line] = None
    right: Optional[Line] = None
    y_bottom: int = 0
    y_top: int = 0
    left_fresh: bool = False          # detected in *this* frame (vs. remembered/estimated)
    right_fresh: bool = False
    lane_center_x: Optional[float] = None
    vehicle_center_x: float = 0.0
    offset_px: Optional[float] = None    # + => vehicle is right of lane centre
    offset_m: Optional[float] = None
    offset_norm: Optional[float] = None  # offset / lane width
    heading_deg: Optional[float] = None  # lane direction relative to straight-ahead
    status: str = "NONE"                 # BOTH / LEFT_ONLY / RIGHT_ONLY / MEMORY / NONE
    num_segments: int = 0
    debug: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        return self.left is not None and self.right is not None

    def x_at(self, line: Line, y: float) -> float:
        xb, xt = line
        t = (y - self.y_bottom) / float(self.y_top - self.y_bottom)
        return xb + t * (xt - xb)

    @property
    def draw_top_y(self) -> float:
        """Top row for drawing: stop just below the point where the two lines meet
        (the vanishing point), which may lie below the ROI top on pitched cameras."""
        if not self.detected:
            return float(self.y_top)
        wb = self.right[0] - self.left[0]
        wt = self.right[1] - self.left[1]
        t = 1.0
        if wb - wt > 1e-6:
            t = min(1.0, 0.92 * wb / (wb - wt))
        return self.y_bottom + t * (self.y_top - self.y_bottom)

    def corridor_at(self, y: float) -> Optional[Tuple[float, float]]:
        """Horizontal extent of the ego lane at image row ``y``."""
        if not self.detected:
            return None
        return self.x_at(self.left, y), self.x_at(self.right, y)


class LaneDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev_left: Optional[Line] = None
        self.prev_right: Optional[Line] = None
        self.left_age = 0     # frames since the side was last freshly detected
        self.right_age = 0
        self.lane_width: Optional[Line] = None  # (width_bottom, width_top) EMA

    # ------------------------------------------------------------ masks
    def roi_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.cfg.roi_vertices(w, h), dtype=np.int32)], 255)
        return mask

    @staticmethod
    def color_mask(pre: Preprocessed, roi: np.ndarray) -> np.ndarray:
        """Binary mask of lane-paint coloured pixels (white + yellow)."""
        l_chan = pre.hls[:, :, 1]
        roi_l = l_chan[roi > 0]
        # adaptive white threshold: brightest pixels of the road region
        thr = max(165, int(np.percentile(roi_l, 97))) if roi_l.size else 200
        white = cv2.inRange(pre.hls, (0, thr, 0), (180, 255, 255))
        yellow = cv2.inRange(pre.hsv, (15, 100, 120), (35, 255, 255))
        mask = cv2.bitwise_and(cv2.bitwise_or(white, yellow), roi)
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)   # remove speckles
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)  # fill gaps in paint
        return mask

    # ------------------------------------------------------------ main
    def detect(self, pre: Preprocessed) -> LaneResult:
        cfg = self.cfg
        h, w = pre.gray.shape[:2]
        roi = self.roi_mask(h, w)

        edges = cv2.Canny(pre.blurred, cfg.canny_low, cfg.canny_high)
        cmask = self.color_mask(pre, roi)
        color_edges = cv2.Canny(cmask, 50, 150)
        near_paint = cv2.dilate(cmask, np.ones((7, 7), np.uint8))
        combined = cv2.bitwise_or(color_edges, cv2.bitwise_and(edges, near_paint))
        # fall back to plain Canny if the colour mask found (almost) nothing
        if cv2.countNonZero(cv2.bitwise_and(cmask, roi)) < 0.002 * cv2.countNonZero(roi):
            combined = edges
        masked = cv2.bitwise_and(combined, roi)

        segments = cv2.HoughLinesP(masked, cfg.hough_rho, cfg.hough_theta, cfg.hough_threshold,
                                   minLineLength=cfg.hough_min_line_len,
                                   maxLineGap=cfg.hough_max_line_gap)
        segments = segments.reshape(-1, 4) if segments is not None else np.empty((0, 4))

        y_bottom, y_top = h - 1, int(cfg.roi_top_y * h)
        left, right, hough_img = self._fit_sides(segments, w, h, y_bottom, y_top)

        res = LaneResult(y_bottom=y_bottom, y_top=y_top, vehicle_center_x=w / 2.0,
                         num_segments=len(segments))
        left, right = self._sanity_check(left, right, w)
        res.left_fresh, res.right_fresh = left is not None, right is not None
        left, right = self._smooth_and_remember(left, right)
        res.left, res.right = left, right

        if res.left_fresh and res.right_fresh:
            res.status = "BOTH"
        elif res.left_fresh:
            res.status = "LEFT_ONLY"
        elif res.right_fresh:
            res.status = "RIGHT_ONLY"
        elif res.detected:
            res.status = "MEMORY"

        if res.detected:
            self._compute_geometry(res)

        res.debug = {"edges": edges, "color_mask": cmask, "masked_edges": masked,
                     "hough": hough_img, "roi": roi}
        return res

    # ------------------------------------------------------------ helpers
    def _fit_sides(self, segments, w, h, y_bottom, y_top):
        cfg = self.cfg
        hough_img = np.zeros((h, w, 3), dtype=np.uint8)
        sides = {"L": ([], [], []), "R": ([], [], [])}
        for x1, y1, x2, y2 in segments:
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if not (cfg.min_abs_slope <= abs(slope) <= cfg.max_abs_slope):
                cv2.line(hough_img, (int(x1), int(y1)), (int(x2), int(y2)), (90, 90, 90), 2)
                continue
            mid_x = (x1 + x2) / 2.0
            length = float(np.hypot(x2 - x1, y2 - y1))
            if slope < 0 and mid_x < 0.62 * w:
                key, color = "L", (255, 80, 80)
            elif slope > 0 and mid_x > 0.38 * w:
                key, color = "R", (80, 80, 255)
            else:
                cv2.line(hough_img, (int(x1), int(y1)), (int(x2), int(y2)), (90, 90, 90), 2)
                continue
            xs, ys, ws = sides[key]
            xs += [x1, x2]
            ys += [y1, y2]
            ws += [length, length]
            cv2.line(hough_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        def fit(key) -> Optional[Line]:
            xs, ys, ws = sides[key]
            if len(xs) < 2 or np.ptp(ys) < 5:
                return None
            a, b = np.polyfit(np.array(ys, float), np.array(xs, float), 1, w=np.array(ws, float))
            return (a * y_bottom + b, a * y_top + b)

        return fit("L"), fit("R"), hough_img

    def _sanity_check(self, left, right, w):
        cfg = self.cfg
        # reject sudden jumps relative to the remembered line
        def jump(new, prev, age):
            if new is None or prev is None or age > cfg.lane_memory_frames:
                return new
            if abs(new[0] - prev[0]) > 0.18 * w and abs(new[1] - prev[1]) > 0.08 * w:
                return None
            return new
        left = jump(left, self.prev_left, self.left_age)
        right = jump(right, self.prev_right, self.right_age)
        if left is not None and right is not None:
            wb, wt = right[0] - left[0], right[1] - left[1]
            if not (cfg.min_lane_width_frac * w <= wb <= cfg.max_lane_width_frac * w):
                return None, None
            # the two lines must converge towards a vanishing point that is not
            # too close to the bottom of the image (wb / (wb - wt) = relative VP height)
            if wb - wt <= 0 or wb / (wb - wt) < 0.5:
                return None, None
        return left, right

    def _smooth_and_remember(self, left, right):
        a = self.cfg.lane_smoothing

        def ema(new, prev, age):
            if prev is None or age > self.cfg.lane_memory_frames:
                return new
            return (a * prev[0] + (1 - a) * new[0], a * prev[1] + (1 - a) * new[1])

        if left is not None:
            self.prev_left, self.left_age = ema(left, self.prev_left, self.left_age), 0
        else:
            self.left_age += 1
        if right is not None:
            self.prev_right, self.right_age = ema(right, self.prev_right, self.right_age), 0
        else:
            self.right_age += 1

        mem = self.cfg.lane_memory_frames
        out_l = self.prev_left if self.left_age <= mem else None
        out_r = self.prev_right if self.right_age <= mem else None

        # learn the lane width when both sides are fresh, use it to infer a missing side
        if left is not None and right is not None:
            wnow = (self.prev_right[0] - self.prev_left[0], self.prev_right[1] - self.prev_left[1])
            self.lane_width = wnow if self.lane_width is None else (
                0.9 * self.lane_width[0] + 0.1 * wnow[0], 0.9 * self.lane_width[1] + 0.1 * wnow[1])
        elif self.lane_width is not None:
            if left is not None and out_r is None:
                out_r = (out_l[0] + self.lane_width[0], out_l[1] + self.lane_width[1])
            elif right is not None and out_l is None:
                out_l = (out_r[0] - self.lane_width[0], out_r[1] - self.lane_width[1])
        return out_l, out_r

    def _compute_geometry(self, res: LaneResult):
        lb, lt = res.left
        rb, rt = res.right
        center_b, center_t = (lb + rb) / 2.0, (lt + rt) / 2.0
        width_px = max(rb - lb, 1.0)
        res.lane_center_x = center_b
        res.offset_px = res.vehicle_center_x - center_b
        res.offset_norm = res.offset_px / width_px
        res.offset_m = res.offset_norm * self.cfg.lane_width_m
        res.heading_deg = float(np.degrees(np.arctan2(center_t - center_b, res.y_bottom - res.y_top)))
