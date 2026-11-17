"""Stage 7 - Rule-based driving decision module (simulation only).

Rules, in priority order (first match wins):
  1. STOP       - red traffic signal
  2. STOP       - obstacle in our lane extremely close, or a pedestrian close ahead
  3. SLOW DOWN  - yellow traffic signal
  4. SLOW DOWN  - obstacle in our lane that is near, or approaching at medium range
  5. SLOW DOWN  - lane markings lost for a long time
  6. STEER L/R  - lateral offset from the lane centre above a threshold
  7. MAINTAIN   - otherwise

Temporal hysteresis: a more severe action is applied immediately (safety first),
while relaxing to a calmer action requires it to persist for a few frames.
This removes flicker caused by single-frame detection noise.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import Config
from .lane_detection import LaneResult
from .traffic_light import SignalResult
from .tracking import Track

SEVERITY = {"MAINTAIN": 0, "STEER LEFT": 1, "STEER RIGHT": 1, "SLOW DOWN": 2, "STOP": 3}


@dataclass
class Decision:
    action: str
    reason: str
    raw_action: str
    raw_reason: str
    critical_track: Optional[int] = None


class DecisionModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current = "MAINTAIN"
        self.current_reason = "Initialising"
        self.pending: Optional[str] = None
        self.pending_count = 0
        self.lane_missing = 0

    def rule_based(self, lane: LaneResult, signal: SignalResult, tracks: Sequence[Track]):
        cfg = self.cfg
        self.lane_missing = 0 if lane.detected else self.lane_missing + 1
        in_path = sorted([t for t in tracks if t.in_path], key=lambda t: -t.proximity)
        nearest = in_path[0] if in_path else None

        if signal.state == "RED":
            return "STOP", "Red traffic signal", None
        if nearest is not None and nearest.proximity >= cfg.prox_stop:
            return "STOP", f"{nearest.label} #{nearest.track_id} very close ahead", nearest.track_id
        peds = [t for t in in_path if t.label == "person" and t.proximity >= cfg.prox_medium]
        if peds:
            return "STOP", f"Pedestrian #{peds[0].track_id} in path", peds[0].track_id
        if signal.state == "YELLOW":
            return "SLOW DOWN", "Yellow traffic signal", None
        if nearest is not None and (nearest.proximity >= cfg.prox_near or
                                    (nearest.approaching and nearest.proximity >= cfg.prox_medium)):
            why = "approaching" if nearest.approaching else "near"
            return "SLOW DOWN", f"{nearest.label} #{nearest.track_id} {why} in lane", nearest.track_id
        if self.lane_missing >= cfg.lane_lost_slowdown_frames:
            return "SLOW DOWN", "Lane markings not visible", None
        if lane.detected and lane.offset_norm is not None and abs(lane.offset_norm) > cfg.steer_threshold:
            # offset > 0: vehicle is right of the lane centre -> steer left
            side = "LEFT" if lane.offset_norm > 0 else "RIGHT"
            return f"STEER {side}", f"Lane offset {lane.offset_m:+.2f} m", None
        return "MAINTAIN", "Lane centred, path clear", None

    def decide(self, lane: LaneResult, signal: SignalResult, tracks: Sequence[Track]) -> Decision:
        raw, reason, tid = self.rule_based(lane, signal, tracks)
        if raw == self.current:
            self.current_reason = reason
            self.pending, self.pending_count = None, 0
        elif SEVERITY[raw] > SEVERITY[self.current]:
            self.current, self.current_reason = raw, reason
            self.pending, self.pending_count = None, 0
        else:
            if raw == self.pending:
                self.pending_count += 1
            else:
                self.pending, self.pending_count = raw, 1
            if self.pending_count >= self.cfg.decision_hold_frames:
                self.current, self.current_reason = raw, reason
                self.pending, self.pending_count = None, 0
        return Decision(self.current, self.current_reason, raw, reason, tid)
