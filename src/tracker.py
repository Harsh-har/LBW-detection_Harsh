"""Turn noisy per-frame ball candidates into one clean track that ends at bat/pad impact."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .trajectory import find_events


@dataclass
class Track:
    f0: int                               # first frame index (0-based)
    f1: int                               # impact frame index (0-based)
    xs: np.ndarray                        # smoothed x per frame f0..f1
    ys: np.ndarray                        # smoothed y per frame f0..f1
    kept: int                             # real detections used
    gaps: int                             # frames filled by interpolation
    detected: List[int] = field(default_factory=list)            # frame numbers that are real detections
    bounce: Optional[Tuple[int, float, float]] = None            # (frame, x, y) or None
    impact: Optional[Tuple[int, float, float]] = None            # (frame, x, y)
    warnings: List[str] = field(default_factory=list)
    mean_conf: float = 0.0


def _norm(raw):
    """Accept {frame: (x,y,c)} or {frame: [(x,y,c), ...]}."""
    out = {}
    for i, v in raw.items():
        if v is None:
            continue
        if len(v) and not isinstance(v[0], (tuple, list)):
            v = [v]
        if v:
            out[i] = [tuple(map(float, c)) for c in v]
    return out


def _extend(cands, start, step, max_step, min_conf, max_gap, last_frame):
    """Grow a chain from `start` (frame, x, y) in direction `step` (+1 / -1)."""
    chain, (lf, lx, ly), vel, miss = [], start, None, 0
    j = lf
    while True:
        j += step
        if j < 0 or j > last_frame:
            break
        dt = j - lf
        best, best_d = None, None
        px, py = (lx + vel[0] * dt, ly + vel[1] * dt) if vel else (lx, ly)
        for x, y, c in cands.get(j, []):
            if c < min_conf or np.hypot(x - lx, y - ly) > max_step * (1 + 0.5 * (abs(dt) - 1)):
                continue
            d = np.hypot(x - px, y - py)
            if best is None or d < best_d:
                best, best_d = (x, y, c), d
        if best is None:
            miss += 1
            if miss > max_gap:
                break
            continue
        miss = 0
        vel = ((best[0] - lx) / dt, (best[1] - ly) / dt)
        lf, lx, ly = j, best[0], best[1]
        chain.append((j, best[0], best[1], best[2]))
    return chain


def best_chain(cands, H, fps, cfg):
    """Longest physically consistent chain of ball detections -> {frame: (x, y, conf)}."""
    if not cands:
        return {}, []
    scale = float(np.clip(30.0 / max(fps, 1.0), 1.0, 1.5))     # slow clips: ball moves more per frame
    max_step = cfg.max_step_frac * H * scale
    last_frame = max(cands)
    warnings = []
    seeds = [(i, x, y, c) for i, cs in cands.items() for x, y, c in cs if c >= cfg.seed_conf]
    if not seeds:
        seeds = [(i, x, y, c) for i, cs in cands.items() for x, y, c in cs if c >= cfg.min_conf]
        warnings.append("no_confident_seed")
    best, best_score = None, -1.0
    for i, x, y, c in seeds:
        fwd = _extend(cands, (i, x, y), +1, max_step, cfg.min_conf, cfg.max_gap, last_frame)
        bwd = _extend(cands, (i, x, y), -1, max_step, cfg.min_conf, cfg.max_gap, last_frame)
        chain = sorted(bwd + [(i, x, y, c)] + fwd)
        score = len(chain) + 0.01 * sum(p[3] for p in chain)
        if score > best_score:
            best, best_score = chain, score
    return {f: (x, y, c) for f, x, y, c in best}, warnings


def filter_outliers(raw, max_step, seed_conf, min_conf):
    """Legacy greedy filter (kept for compatibility / tests): {frame: (x, y, conf)} -> {frame: (x, y)}."""
    acc, last, lasti = {}, None, None
    for i in sorted(raw):
        x, y, c = raw[i]
        if last is None:
            if c >= seed_conf:
                acc[i] = (x, y)
                last, lasti = (x, y), i
            continue
        if np.hypot(x - last[0], y - last[1]) <= max_step * (i - lasti) and c >= min_conf:
            acc[i] = (x, y)
            last, lasti = (x, y), i
    return acc


def _smooth(a, window):
    n = len(a)
    w = min(window if window % 2 else window + 1, n if n % 2 else n - 1)
    return savgol_filter(a, w, 2) if w >= 3 and w > 2 else a


def build_track(raw, frame_h, cfg, fps=30.0):
    cands = _norm(raw)
    acc, warnings = best_chain(cands, frame_h, fps, cfg)
    if len(acc) < cfg.min_track_points:
        return None
    ks = sorted(acc)
    xs = [acc[k][0] for k in ks]
    ys = [acc[k][1] for k in ks]
    bi, ii, w2 = find_events(ks, xs, ys, frame_h, cfg)
    warnings += w2
    ks, xs, ys = ks[: ii + 1], xs[: ii + 1], ys[: ii + 1]
    if len(ks) < 4:
        return None
    frames = np.arange(ks[0], ks[-1] + 1)
    fx = np.interp(frames, ks, xs)
    fy = np.interp(frames, ks, ys)
    if bi is not None:                                   # smooth each side of the bounce separately
        cut = ks[bi] - ks[0]
        fx = np.concatenate([_smooth(fx[: cut + 1], cfg.smooth_window), _smooth(fx[cut:], cfg.smooth_window)[1:]])
        fy = np.concatenate([_smooth(fy[: cut + 1], cfg.smooth_window), _smooth(fy[cut:], cfg.smooth_window)[1:]])
        fx[cut], fy[cut] = xs[bi], ys[bi]
    else:
        fx, fy = _smooth(fx, cfg.smooth_window), _smooth(fy, cfg.smooth_window)
    bounce = (ks[bi], float(xs[bi]), float(ys[bi])) if bi is not None else None
    impact = (ks[-1], float(fx[-1]), float(fy[-1]))
    confs = [acc[k][2] for k in ks]
    return Track(ks[0], ks[-1], fx, fy, len(ks), (ks[-1] - ks[0] + 1) - len(ks), list(ks),
                 bounce, impact, warnings, float(np.mean(confs)))
