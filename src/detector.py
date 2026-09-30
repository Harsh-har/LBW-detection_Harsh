"""YOLO detection of `ball` and `stump` for every frame of a video."""
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2


@dataclass
class FrameDet:
    balls: List[Tuple[float, float, float]] = field(default_factory=list)   # (cx, cy, conf), best first
    stumps: List[Tuple[Tuple[float, float, float, float], float]] = field(default_factory=list)  # (xyxy, conf)

    @property
    def ball(self):
        return self.balls[0] if self.balls else None


def resolve_device(device):
    if device != "auto":
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def drop_static(dets, H, cfg):
    """Remove ball 'detections' that keep appearing at the same pixel (logos, bright spots, cones...)."""
    cell = max(8.0, cfg.static_cell_frac * H)
    cnt = Counter()
    for d in dets:
        for x, y, _ in d.balls:
            cnt[(int(x // cell), int(y // cell))] += 1
    limit = max(cfg.static_min_count, cfg.static_frac * len(dets))
    removed = 0
    for d in dets:
        keep = [b for b in d.balls if cnt[(int(b[0] // cell), int(b[1] // cell))] <= limit]
        removed += len(d.balls) - len(keep)
        d.balls = keep
    return removed


def detect_video(video_path, cfg):
    """Returns (dets, (W, H), fps). Frames are NOT kept in memory."""
    from ultralytics import YOLO  # imported lazily so unit tests need no torch

    model = YOLO(cfg.weights)
    names = model.names
    device = resolve_device(cfg.device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1 or fps > 500:
        fps = 30.0
    dets = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        r = model.predict(frame, conf=cfg.conf, imgsz=cfg.imgsz, verbose=False, device=device)[0]
        d = FrameDet()
        balls = []
        for box, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            if names[int(k)] == "ball":
                balls.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, c))
            else:
                d.stumps.append((tuple(box), c))
        balls.sort(key=lambda b: -b[2])
        d.balls = balls[: cfg.max_ball_cands]
        dets.append(d)
    cap.release()
    if dets:
        drop_static(dets, H, cfg)
    return dets, (W, H), fps
