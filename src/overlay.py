"""All drawing code (BGR colours)."""
import cv2
import numpy as np


def _s(img):
    return max(img.shape[1] / 720.0, 1.0)      # scale text/lines with resolution


def banner(img, text):
    s = _s(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], int(60 * s)), (0, 0, 0), -1)
    cv2.putText(img, text, (int(12 * s), int(38 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, (255, 255, 255), max(1, int(2 * s)))


def draw_stump_boxes(img, boxes):
    for b in boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 0), max(1, int(2 * _s(img))))


def draw_lane(img, poly, keep_boxes=()):
    """Translucent blue lane; the stump boxes are pasted back on top so stumps stay un-tinted."""
    orig = img.copy()
    p = poly.astype(np.int32)
    ov = img.copy()
    cv2.fillPoly(ov, [p], (255, 120, 60))
    out = cv2.addWeighted(ov, 0.40, img, 0.60, 0)
    cv2.polylines(out, [p], True, (255, 160, 90), max(1, int(2 * _s(img))), cv2.LINE_AA)
    for b in keep_boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        out[y1:y2, x1:x2] = orig[y1:y2, x1:x2]
    return out


def draw_arc(img, xs, ys):
    pts = np.stack([xs, ys], 1).astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (0, 0, 255), max(2, int(5 * _s(img))), cv2.LINE_AA)


def draw_track_progress(img, xs, ys, index):
    index = min(max(int(index), 0), len(xs) - 1)
    draw_arc(img, xs[:index + 1], ys[:index + 1])
    s = _s(img)
    point = (int(xs[index]), int(ys[index]))
    cv2.circle(img, point, int(9 * s), (0, 0, 0), max(2, int(2 * s)), cv2.LINE_AA)
    cv2.circle(img, point, int(6 * s), (0, 255, 255), -1, cv2.LINE_AA)


def draw_impact(img, x, y):
    s = _s(img)
    cv2.circle(img, (int(x), int(y)), int(16 * s), (0, 165, 255), max(2, int(3 * s)))
    cv2.putText(img, "IMPACT", (int(x + 20 * s), int(y + 5 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 165, 255), max(1, int(2 * s)))


def draw_pitching_panel(img, label, kind):
    s = _s(img)
    color = {"in": (40, 40, 200), "out": (0, 140, 255), "na": (90, 90, 90)}[kind]
    x0, y0, w, h = int(20 * s), int(80 * s), int(240 * s), int(38 * s)
    t = max(1, int(2 * s))
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
    cv2.putText(img, "Pitching", (x0 + int(75 * s), y0 + int(26 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 255), t)
    cv2.rectangle(img, (x0, y0 + h), (x0 + w, y0 + 2 * h), color, -1)
    cv2.putText(img, label, (x0 + int(12 * s), y0 + h + int(27 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 255), t)


def draw_bounce(img, x, y):
    s = _s(img)
    cv2.circle(img, (int(x), int(y)), int(10 * s), (255, 255, 0), max(2, int(3 * s)))
    cv2.putText(img, "BOUNCE", (int(x + 14 * s), int(y - 8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 0), max(1, int(2 * s)))
