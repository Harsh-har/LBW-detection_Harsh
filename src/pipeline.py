"""End-to-end: video in -> annotated video + JSON out.

Playback is clean (no overlays). Overlays (stump boxes, lane, ball arc, bounce, impact, pitching panel)
appear only on the final frame, which is held for a couple of seconds.
"""
import json
import os
import shutil
import subprocess

import cv2

from . import overlay
from .calibration import lane_polygon, stable_batter_stumps
from .detector import detect_video
from .lbw import pitching_status
from .tracker import build_track


def analyse(video_path, cfg):
    """Runs detection + tracking + pitching. Returns a plain dict (JSON-serialisable except `_internal`)."""
    dets, (W, H), fps = detect_video(video_path, cfg)
    n = len(dets)
    warnings = []
    if n == 0:
        raise RuntimeError("video has no readable frames")
    if n < 8:
        warnings.append("clip_too_short")
    if n / fps > cfg.max_clip_seconds:
        warnings.append(f"clip_longer_than_{cfg.max_clip_seconds:g}s_expect_worse_results")
    raw = {i: d.balls for i, d in enumerate(dets) if d.balls}
    track = build_track(raw, H, cfg, fps)
    stump = stable_batter_stumps(dets, H)
    poly = lane_polygon(stump, H, cfg.lane_k) if stump is not None else None
    if stump is None:
        warnings.append("batter_stumps_not_found")

    res = {"video": os.path.basename(video_path), "width": W, "height": H, "fps": round(fps, 3),
           "frames": n, "frame_numbering": "0-based", "ball_detections": sum(bool(d.balls) for d in dets),
           "batter_stumps": [round(v, 1) for v in stump] if stump is not None else None,
           "lane_polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in poly] if poly is not None else None}
    if track is None:
        res.update(status="ball_not_found", ball_track=[], bounce=None, impact=None,
                   pitching={"label": "N/A", "kind": "na"}, warnings=warnings + ["too_few_ball_detections"])
    else:
        tw = list(track.warnings)
        if "track_ends_at_possible_bounce" in tw and stump is not None:
            sx, sy, sx2, sy2 = stump
            sh, sw = sy2 - sy, sx2 - sx
            ix, iy = track.xs[-1], track.ys[-1]
            if sy - 0.5 * sh <= iy <= sy2 + 1.5 * sh and abs(ix - (sx + sx2) / 2) <= 4 * sw:
                tw.remove("track_ends_at_possible_bounce")     # ball vanished next to the batter: that is the impact
        track.warnings = tw
        warnings += tw
        det_set = set(track.detected)
        res["ball_track"] = [{"frame": int(track.f0 + i), "x": round(float(x), 1), "y": round(float(y), 1),
                              "source": "detected" if (track.f0 + i) in det_set else "interpolated"}
                             for i, (x, y) in enumerate(zip(track.xs, track.ys))]
        res["bounce"] = {"frame": track.bounce[0], "x": round(track.bounce[1], 1), "y": round(track.bounce[2], 1)} if track.bounce else None
        res["impact"] = {"frame": track.impact[0], "x": round(track.impact[1], 1), "y": round(track.impact[2], 1)}
        label, kind = pitching_status(track.bounce[1:] if track.bounce else None, poly,
                                      unclear="track_ends_at_possible_bounce" in track.warnings)
        res["pitching"] = {"label": label, "kind": kind}
        span = track.f1 - track.f0 + 1
        if track.kept < 10:
            warnings.append("few_ball_detections_in_track")
        if track.kept / span < 0.5:
            warnings.append("more_than_half_of_track_is_interpolated")
        res["track_points_detected"] = track.kept
        res["track_points_interpolated"] = track.gaps
        res["mean_ball_confidence"] = round(track.mean_conf, 3)
        res["status"] = "ok"
    hard = {"batter_stumps_not_found", "few_ball_detections_in_track", "more_than_half_of_track_is_interpolated",
            "no_confident_seed", "track_ends_at_possible_bounce", "clip_too_short"}
    if res["status"] == "ok" and any(w in hard for w in warnings):
        res["status"] = "low_confidence"
    res["warnings"] = warnings
    res["_internal"] = (dets, track, stump, poly)
    return res


def compose_final(frame, res):
    dets, track, stump, poly = res["_internal"]
    H = frame.shape[0]
    img = frame.copy()
    if stump is not None:
        overlay.draw_stump_boxes(img, [stump])
        others = next(([bx for bx, _ in d.stumps] for d in reversed(dets) if d.stumps), [])
        img = overlay.draw_lane(img, poly, keep_boxes=[stump] + others)
    if track is not None:
        overlay.draw_arc(img, track.xs, track.ys)
        if track.bounce:
            overlay.draw_bounce(img, track.bounce[1], track.bounce[2])
        overlay.draw_impact(img, track.xs[-1], track.ys[-1])
        overlay.banner(img, "Ball tracking (until bat/pad impact)" + ("  [LOW CONFIDENCE]" if res["status"] == "low_confidence" else ""))
        overlay.draw_pitching_panel(img, res["pitching"]["label"], res["pitching"]["kind"])
    else:
        overlay.banner(img, "Ball not tracked (too few detections)")
    return img


def _encode(tmp, out_path):
    if shutil.which("ffmpeg"):                 # re-encode to h264 (even dimensions) so it plays everywhere
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", out_path], check=True)
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)


def run(video_path, out_path, cfg, json_path=None):
    res = analyse(video_path, cfg)
    W, H = res["width"], res["height"]
    tmp = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), cfg.out_fps, (W, H))
    cap = cv2.VideoCapture(video_path)
    last = None
    track = res["_internal"][1]
    frame_index = 0
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if track is not None and track.f0 <= frame_index <= track.f1:
            overlay.draw_track_progress(fr, track.xs, track.ys, frame_index - track.f0)
        writer.write(fr)
        frame_index += 1
        last = fr
    cap.release()
    if last is not None:
        final = compose_final(last, res)
        for _ in range(cfg.hold_frames):
            writer.write(final)
    writer.release()
    _encode(tmp, out_path)
    res.pop("_internal")
    json_path = json_path or os.path.splitext(out_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    res["json"] = json_path
    return res
