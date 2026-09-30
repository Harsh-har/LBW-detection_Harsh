"""Usage:  python main.py videos/input/delivery.mp4 [-o videos/output/out.mp4]
Writes the annotated video and a .json next to it (ball track, bounce, impact, pitching, warnings)."""
import argparse
import os

from src.config import Config
from src.pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Cricket ball tracking / LBW overlay")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--weights", default=Config.weights)
    ap.add_argument("--conf", type=float, default=Config.conf)
    ap.add_argument("--imgsz", type=int, default=Config.imgsz)
    ap.add_argument("--device", default=Config.device, help="auto, cpu or 0 (GPU)")
    a = ap.parse_args()

    if not os.path.isfile(a.video):
        raise SystemExit(f"video not found: {a.video}")
    cfg = Config(weights=a.weights, conf=a.conf, imgsz=a.imgsz, device=a.device)
    out = a.out or os.path.join("videos", "output", os.path.splitext(os.path.basename(a.video))[0] + "_out.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    res = run(a.video, out, cfg)
    print(f"status   : {res['status']}")
    print(f"pitching : {res['pitching']['label']}")
    print(f"bounce   : {res['bounce']}")
    print(f"impact   : {res['impact']}")
    if res["warnings"]:
        print("warnings :", ", ".join(res["warnings"]))
    print("video    :", out)
    print("json     :", res["json"])


if __name__ == "__main__":
    main()
