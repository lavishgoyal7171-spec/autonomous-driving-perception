"""Run the perception pipeline on a road video.

Examples
--------
    python main.py --input data/project_video.mp4
    python main.py --input data/solidWhiteRight.mp4 --show
    python main.py --input data/synthetic_road.mp4 --signal-mode both
    python main.py --input 0 --show            # webcam
"""
import argparse
import json
import os

from src.config import Config
from src.pipeline import PerceptionPipeline, run_video


def parse_args():
    p = argparse.ArgumentParser(description="Vision-Based Autonomous Driving Perception and Lane Guidance")
    p.add_argument("--input", "-i", required=True, help="video file path or camera index")
    p.add_argument("--output", "-o", help="annotated output video (default: outputs/<name>_annotated.mp4)")
    p.add_argument("--log", help="per-frame CSV log (default: outputs/<name>_log.csv)")
    p.add_argument("--show", action="store_true", help="display a live window (q quits, space pauses)")
    p.add_argument("--no-yolo", action="store_true", help="disable YOLO object detection")
    p.add_argument("--model", default=None, help="YOLO weights, e.g. yolov8n.pt / yolov8s.pt")
    p.add_argument("--device", default=None, help="auto | cpu | mps | cuda")
    p.add_argument("--conf", type=float, default=None, help="detection confidence threshold")
    p.add_argument("--signal-mode", choices=["auto", "yolo", "hsv", "both", "off"], default=None,
                   help="how traffic lights are located (classification is always HSV)")
    p.add_argument("--width", type=int, default=None, help="processing width in pixels")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--snapshots", type=int, default=0, help="save a dashboard JPEG every N frames")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    if args.model:
        cfg.yolo_model = args.model
    if args.device:
        cfg.device = args.device
    if args.conf is not None:
        cfg.det_conf = args.conf
    if args.signal_mode:
        cfg.signal_mode = args.signal_mode
    if args.width:
        cfg.frame_width = args.width

    name = "camera" if args.input.isdigit() else os.path.splitext(os.path.basename(args.input))[0]
    output = args.output or os.path.join("outputs", f"{name}_annotated.mp4")
    log = args.log or os.path.join("outputs", f"{name}_log.csv")
    snap_dir = os.path.join("outputs", "snapshots", name) if args.snapshots else None

    pipeline = PerceptionPipeline(cfg, use_yolo=not args.no_yolo)
    stats = run_video(pipeline, args.input, output=output, log_path=log, show=args.show,
                      max_frames=args.max_frames, snapshot_dir=snap_dir, snapshot_every=args.snapshots)
    print(json.dumps(stats, indent=2))
    print(f"\nEvaluate with:  python scripts/evaluate.py --log {log}")


if __name__ == "__main__":
    main()
