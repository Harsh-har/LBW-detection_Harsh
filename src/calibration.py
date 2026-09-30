"""Batter-end stumps and the in-line lane. The bowler/umpire-end stumps are NOT needed."""
import numpy as np


def batter_stumps(boxes, frame_h):
    """Pick the batter's stumps from one frame's stump boxes (or None).
    - several boxes -> the far one (smallest bottom-y)
    - one box       -> assumed to be the batter's, unless it sits in the lower part of the frame
                       (then it is probably the near/umpire-end stump and is ignored)."""
    if not boxes:
        return None
    if len(boxes) == 1:
        b = boxes[0]
        return b if (b[1] + b[3]) / 2 < 0.70 * frame_h else None
    return min(boxes, key=lambda b: b[3])


def stable_batter_stumps(dets, frame_h):
    """Median batter-stump box over all frames where it was found (robust to missed frames)."""
    found = []
    for d in dets:
        b = batter_stumps([bx for bx, _ in d.stumps], frame_h)
        if b is not None:
            found.append(b)
    if not found:
        return None
    return tuple(float(v) for v in np.median(np.array(found), axis=0))


def lane_polygon(stump, frame_h, k=0.40, widen=1.6):
    """Blue in-line lane: from the stump base down to the bottom of the frame.
    Width at the stumps = k * stump-box width; it widens towards the camera (perspective)."""
    cx, w = (stump[0] + stump[2]) / 2, stump[2] - stump[0]
    top, bot = k * w, k * w * widen
    y_top, y_bot = stump[3], frame_h - 1
    return np.array([(cx - bot, y_bot), (cx + bot, y_bot), (cx + top, y_top), (cx - top, y_top)], np.float32)
