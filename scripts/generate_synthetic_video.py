"""Generate a synthetic road video with known ground truth.

The scene contains a perspective road with a solid yellow left line and a dashed
white right line, moving dashes, shadow bands, sensor noise, lateral drift of the
ego-vehicle and a traffic light that cycles GREEN -> YELLOW -> RED -> GREEN.

Ground truth (signal state, lateral offset, expected decision) is written to a CSV
so that scripts/evaluate.py can compute signal-classification and decision accuracy.

    python scripts/generate_synthetic_video.py --out data/synthetic_road.mp4
"""
import argparse
import csv
import os

import cv2
import numpy as np

W, H = 960, 540
VPX, VPY = 480, 300          # vanishing point
LANE_HALF = 300              # half lane width in pixels at the bottom row
STEER_THR = 0.15             # must match Config.steer_threshold


def signal_at(f):
    if f < 230:
        return "GREEN"
    if f < 290:
        return "YELLOW"
    if f < 440:
        return "RED"
    return "GREEN"


def offset_at(f):
    """Vehicle offset from lane centre as a fraction of lane width (+ = right of centre)."""
    if 40 <= f < 200:
        return 0.30 * np.sin(np.pi * (f - 40) / 160.0)
    if 460 <= f < 620:
        return -0.30 * np.sin(np.pi * (f - 460) / 160.0)
    return 0.0


def expected_decision(sig, off):
    if sig == "RED":
        return "STOP"
    if sig == "YELLOW":
        return "SLOW DOWN"
    if abs(off) > STEER_THR:
        return "STEER LEFT" if off > 0 else "STEER RIGHT"
    return "MAINTAIN"


def x_on_line(x_bottom, y):
    return VPX + (x_bottom - VPX) * (y - VPY) / float(H - VPY)


def draw_frame(f, rng):
    img = np.zeros((H, W, 3), np.uint8)
    # sky gradient (blue) and grass
    for y in range(VPY):
        t = y / VPY
        img[y] = (int(235 - 40 * t), int(180 - 20 * t), int(110 - 10 * t))
    img[VPY:] = (60, 120, 70)
    off = offset_at(f)
    cx = W / 2 - off * 2 * LANE_HALF          # lane centre at the bottom row
    left_b, right_b = cx - LANE_HALF, cx + LANE_HALF
    # asphalt (wider than the lane)
    road = np.array([(x_on_line(left_b - 260, H), H), (VPX - 4, VPY), (VPX + 4, VPY),
                     (x_on_line(right_b + 260, H), H)], np.int32)
    cv2.fillPoly(img, [road], (85, 85, 88))
    # moving shadow bands across the road
    for k in range(2):
        ys = VPY + 20 + ((f * 3 + k * 140) % (H - VPY))
        band = np.array([(0, ys), (W, ys), (W, ys + 25 + (ys - VPY) // 6), (0, ys + 25 + (ys - VPY) // 6)])
        shadow = img.copy()
        cv2.fillPoly(shadow, [band.astype(np.int32)], (0, 0, 0))
        img = cv2.addWeighted(shadow, 0.35, img, 0.65, 0)
    # lane lines
    for y in range(VPY + 8, H, 2):
        th = max(1, int(1 + 14 * (y - VPY) / (H - VPY)))
        xl = int(x_on_line(left_b, y))
        cv2.line(img, (xl - th // 2, y), (xl + th // 2, y), (0, 200, 240), 2)   # yellow solid
        depth = 4000.0 / (y - VPY + 1)
        if int((depth + f * 0.9) / 6) % 2 == 0:                                # white dashed
            xr = int(x_on_line(right_b, y))
            cv2.line(img, (xr - th // 2, y), (xr + th // 2, y), (235, 235, 235), 2)
    # traffic light on a pole (slowly getting bigger as we approach)
    scale = 1.0 + 0.4 * min(f, 440) / 440.0
    hw, hh = int(20 * scale), int(48 * scale)
    tx, ty = 690 + int(60 * min(f, 440) / 440.0), 150
    cv2.line(img, (tx, ty), (tx, VPY + 60), (50, 50, 50), max(2, int(5 * scale)))
    cv2.rectangle(img, (tx - hw, ty - hh), (tx + hw, ty + hh), (25, 25, 25), -1)
    state = signal_at(f)
    r = int(12 * scale)
    lamps = {"RED": (0, 0, 255), "YELLOW": (0, 200, 255), "GREEN": (120, 255, 0)}
    for i, name in enumerate(("RED", "YELLOW", "GREEN")):
        cy = ty - hh + int((i + 0.5) * 2 * hh / 3)
        cv2.circle(img, (tx, cy), r, lamps[name] if name == state else (55, 55, 55), -1, cv2.LINE_AA)
    # sensor noise + slight blur
    noise = rng.normal(0, 6, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, state, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_road.mp4")
    ap.add_argument("--frames", type=int, default=700)
    ap.add_argument("--fps", type=float, default=25)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    gt_path = os.path.splitext(a.out)[0] + "_gt.csv"
    rng = np.random.default_rng(0)
    with open(gt_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "signal", "offset_norm", "expected_decision"])
        for f in range(a.frames):
            img, state, off = draw_frame(f, rng)
            writer.write(img)
            wr.writerow([f, state, round(off, 4), expected_decision(state, off)])
    writer.release()
    print(f"video -> {a.out}\nground truth -> {gt_path}")


if __name__ == "__main__":
    main()
