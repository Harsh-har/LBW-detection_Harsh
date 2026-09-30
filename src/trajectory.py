"""Bounce / impact events on the tracked ball points (image coordinates).

A *bounce* and a *bat/pad hit* both reverse the ball's image-y direction, so they cannot be told
apart by the direction change alone. The difference: after a real bounce the ball keeps going away
from the camera (image-y keeps decreasing for several points); after a pad/bat hit it comes back.
Events are therefore classified in time order: bounce -> remember it and go on; impact -> stop.
"""
import numpy as np


def _turn_angle(p_prev, p, p_next):
    v1 = np.asarray(p, float) - np.asarray(p_prev, float)
    v2 = np.asarray(p_next, float) - np.asarray(p, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1))))


def is_bounce(ks, ys, b, H, cfg):
    """True if point b is the lowest image point of a fall-then-rise that continues afterwards.
    Uses 1- and 2-step slopes and a local-maximum test so a flat apex (bounce between two frames) still counts."""
    n = len(ks)
    if b < 1 or b > n - 2:
        return False
    thr = cfg.bounce_speed_frac * H
    lo, hi = max(0, b - 2), min(n - 1, b + 2)
    if ys[b] < max(ys[lo: hi + 1]):
        return False                                   # not the lowest image point of its neighbourhood
    v_in = max((ys[b] - ys[b - 1]) / (ks[b] - ks[b - 1]), (ys[b] - ys[lo]) / (ks[b] - ks[lo]))
    v_out = min((ys[b + 1] - ys[b]) / (ks[b + 1] - ks[b]), (ys[hi] - ys[b]) / (ks[hi] - ks[b]))
    if not (v_in > thr and v_out < -thr):
        return False
    fall = ys[b] - min(ys[max(0, b - 6): b + 1])
    if fall < cfg.bounce_fall_frac * H:
        return False
    post = ys[b + 1: b + 4]
    rise_min = cfg.bounce_rise_frac * H
    # ball must climb away from the bounce and still be above it at the end of the window
    # (after a pad/bat hit it comes back down, so the last point ends up below the hit point)
    return min(post) <= ys[b] - rise_min and post[-1] <= ys[b] - 0.5 * rise_min


def find_events(ks, xs, ys, H, cfg):
    """ks: sorted frame numbers of accepted detections, xs/ys their coordinates.
    Returns (bounce_index or None, impact_index, warnings) - indices into ks."""
    n = len(ks)
    bounce, impact, warnings, found = None, n - 1, [], False
    lo = 0                                   # do not measure turn angles across the bounce corner
    for b in range(1, n - 1):
        if is_bounce(ks, ys, b, H, cfg):
            if bounce is None:
                bounce, lo = b, b
            continue
        if bounce is not None and b <= bounce + 1:
            continue                         # still on the bounce corner
        if any(is_bounce(ks, ys, j, H, cfg) for j in (b + 1, b + 2) if j <= n - 2):
            continue                         # corner just before a bounce belongs to the bounce
        l1 = min(2, b - lo)
        l2 = min(2, n - 1 - b)
        if l1 < 1 or l2 < 1:
            continue
        ang = _turn_angle((xs[b - l1], ys[b - l1]), (xs[b], ys[b]), (xs[b + l2], ys[b + l2]))
        sp1 = np.hypot(xs[b] - xs[b - l1], ys[b] - ys[b - l1]) / (ks[b] - ks[b - l1])
        sp2 = np.hypot(xs[b + l2] - xs[b], ys[b + l2] - ys[b]) / (ks[b + l2] - ks[b])
        step_ok = min(sp1, sp2) > 0.004 * H          # both sides must be real motion, not jitter
        if ang > cfg.impact_angle_deg and step_ok:
            impact, found = b, True
            break
    if bounce is None and not found and n >= 3:   # track ends exactly at a possible bounce with nothing after
        b = n - 1
        fall = ys[b] - min(ys[max(0, b - 6): b + 1])
        if ys[b] - ys[b - 1] > cfg.bounce_speed_frac * H and fall >= 2 * cfg.bounce_rise_frac * H:
            warnings.append("track_ends_at_possible_bounce")
    return bounce, impact, warnings


def detect_bounce(xs, ys, min_speed=1.0):
    """Legacy helper (kept for compatibility): first fall->rise turn of a dense path."""
    vy = np.diff(ys)
    for i in range(len(vy) - 1):
        if vy[i] > min_speed and vy[i + 1] < -min_speed:
            return float(xs[i + 1]), float(ys[i + 1])
    return None
