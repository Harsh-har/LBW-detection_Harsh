"""LBW checks. Only PITCHING is implemented so far (see README)."""
import cv2
import numpy as np


def pitching_status(bounce_xy, lane_poly, unclear=False):
    """Returns (label, kind) with kind in {'in', 'out', 'na'}."""
    if lane_poly is None:
        return "N/A (no stumps)", "na"
    if bounce_xy is None:
        return ("N/A (unclear)", "na") if unclear else ("N/A (full toss)", "na")
    pt = (float(bounce_xy[0]), float(bounce_xy[1]))
    inside = cv2.pointPolygonTest(np.asarray(lane_poly, np.float32), pt, False) >= 0
    return ("In Line", "in") if inside else ("Outside", "out")
