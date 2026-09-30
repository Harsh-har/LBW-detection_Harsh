from dataclasses import dataclass


@dataclass
class Config:
    # --- detection ---
    weights: str = "weights/best.pt"
    conf: float = 0.10            # low on purpose: ball is tiny, tracker removes false hits
    imgsz: int = 960
    device: str = "auto"          # "auto" (GPU if available), "cpu", or "0"
    max_ball_cands: int = 3       # ball candidates kept per frame (tracker picks the right one)
    static_cell_frac: float = 0.02    # static-false-positive grid cell = this * frame height
    static_frac: float = 0.15     # a spot that "detects" a ball in > this share of frames is not a ball
    static_min_count: int = 5

    # --- tracking ---
    seed_conf: float = 0.60       # chain may start from a detection at least this confident
    min_conf: float = 0.25        # other chain points must be at least this confident
    max_step_frac: float = 0.07   # max ball jump per frame, as a fraction of frame height
    max_gap: int = 6              # frames the ball may be missing inside a chain
    impact_angle_deg: float = 50.0    # direction change that counts as bat/pad hit
    smooth_window: int = 5        # Savitzky-Golay window (odd)
    min_track_points: int = 6     # fewer real detections than this -> "not tracked"

    # --- bounce detection (image y falls then rises) ---
    bounce_fall_frac: float = 0.012   # ball must have dropped at least this (x frame height) into the bounce
    bounce_rise_frac: float = 0.010   # ... and climbed at least this after it
    bounce_speed_frac: float = 0.003  # min vertical speed (per frame) before / after the bounce

    # --- overlay / output ---
    lane_k: float = 0.40          # lane half-width = k * stump box width
    out_fps: int = 10             # slow-motion playback
    hold_frames: int = 20         # how long the final frame is held
    max_clip_seconds: float = 8.0 # warn above this (model/logic is meant for single short deliveries)
