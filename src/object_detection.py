"""Stage 5 - Object detection with a pretrained YOLO model (COCO classes)."""
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Config

# COCO class ids relevant to driving
ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck", "bicycle"}
VULNERABLE_LABELS = {"person", "bicycle", "motorcycle"}

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class Detection:
    box: Box
    label: str
    conf: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class ObjectDetector:
    def __init__(self, cfg: Config):
        from ultralytics import YOLO  # imported lazily so the rest works without it

        self.cfg = cfg
        self.device = pick_device(cfg.device)
        weights = cfg.yolo_model
        if not os.path.isabs(weights) and os.path.dirname(weights):
            # resolve "models/yolov8n.pt" relative to the project root, not the cwd
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights = os.path.join(root, weights)
            os.makedirs(os.path.dirname(weights), exist_ok=True)
        self.model = YOLO(weights)  # downloads the weights on first use
        self.classes = list(ROAD_CLASSES.keys())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self.model.predict(frame, conf=self.cfg.det_conf, imgsz=self.cfg.det_imgsz,
                                    classes=self.classes, device=self.device, verbose=False)[0]
        dets: List[Detection] = []
        if result.boxes is None:
            return dets
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(boxes, confs, clss):
            dets.append(Detection(tuple(int(v) for v in b), ROAD_CLASSES.get(int(k), str(k)), float(c)))
        return dets
