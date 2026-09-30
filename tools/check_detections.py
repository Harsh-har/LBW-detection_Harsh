"""Quick health-check of the model on a new video (run from project root):
    python tools/check_detections.py videos/input/x.mp4"""
import sys

sys.path.insert(0, ".")
from src.config import Config
from src.detector import detect_video

cfg = Config()
dets, (W, H), fps = detect_video(sys.argv[1], cfg)
n = sum(bool(d.balls) for d in dets)
print(f"{W}x{H} @ {fps:.0f}fps, {len(dets)} frames (static false positives already removed)")
print(f"ball frames: {n} | stump frames: {sum(bool(d.stumps) for d in dets)}")
if n < 10:
    print("WARNING: ball seen in < 10 frames -> tracking will be unreliable on this video")
