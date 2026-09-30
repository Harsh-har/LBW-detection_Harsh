"""Tests for the pure logic (no video / model needed):  python -m unittest"""
import unittest

import numpy as np

from src.calibration import batter_stumps, lane_polygon
from src.config import Config
from src.detector import FrameDet, drop_static
from src.lbw import pitching_status
from src.overlay import draw_track_progress
from src.tracker import build_track, filter_outliers

H = 850


def make_delivery(rng, bounce=True, noise=1.5, dropout=0.25, fps=30.0, impact=True):
    """Synthetic image-space delivery: fall -> (bounce) -> rise -> pad hit that sends the ball back.
    Returns (raw {frame: [(x,y,c),...]}, truth dict)."""
    raw, truth, f = {}, {}, int(rng.integers(3, 12))
    x, y = float(rng.uniform(120, 360)), float(rng.uniform(150, 250))
    vx = float(rng.uniform(-4, 4))
    n1 = int(rng.integers(7, 14))
    vy = float(rng.uniform(5, 11))
    pts = []
    for _ in range(n1):                       # falling in the image
        pts.append((x, y)); x += vx; y += vy
    if bounce:
        truth["bounce"] = f + len(pts) - 1
        vr = float(rng.uniform(5, 11))
        for _ in range(int(rng.integers(4, 9))):   # rising after the bounce
            x += vx; y -= vr; pts.append((x, y))
    if impact:
        truth["impact"] = f + len(pts) - 1
        for k in range(1, 5):                 # comes back after hitting the pad
            pts.append((x - 15 * k * np.sign(vx or 1), y + 14 * k))
    for i, (px, py) in enumerate(pts):
        if rng.random() < dropout and i not in (truth.get("bounce", -1) - f, truth.get("impact", -1) - f):
            continue
        raw[f + i] = [(px + rng.normal(0, noise), py + rng.normal(0, noise), float(rng.uniform(0.55, 0.9)))]
    for _ in range(int(rng.integers(0, 4))):  # false positives
        ff = int(rng.integers(0, f + len(pts)))
        raw.setdefault(ff, []).append((float(rng.uniform(0, 478)), float(rng.uniform(0, H)), float(rng.uniform(0.15, 0.5))))
    return raw, truth


class TestTracker(unittest.TestCase):
    def test_bounce_is_found_and_track_continues_to_impact(self):
        raw, truth = make_delivery(np.random.default_rng(1), bounce=True, dropout=0.0)
        t = build_track(raw, H, Config())
        self.assertIsNotNone(t)
        self.assertIsNotNone(t.bounce)
        self.assertLessEqual(abs(t.bounce[0] - truth["bounce"]), 1)
        self.assertLessEqual(abs(t.f1 - truth["impact"]), 1)

    def test_full_toss_has_no_bounce(self):
        raw, truth = make_delivery(np.random.default_rng(2), bounce=False, dropout=0.0)
        t = build_track(raw, H, Config())
        self.assertIsNotNone(t)
        self.assertIsNone(t.bounce)
        self.assertLessEqual(abs(t.f1 - truth["impact"]), 1)

    def test_real_pad_hit_is_not_a_bounce(self):
        # detections from a real full-toss clip (frame: x, y, conf) incl. false positives
        real = {15: (47.4, 174.8, .8), 16: (61.5, 161.1, .8), 17: (75.1, 150, .8), 18: (87.6, 142.6, .8), 19: (99.5, 137.2, .8),
                20: (111, 135.4, .8), 21: (121.5, 135.9, .8), 22: (131.2, 138, .8), 23: (140.9, 142.6, .8), 24: (150.2, 148.6, .8),
                25: (159.6, 155.2, .8), 26: (167.5, 164.3, .8), 27: (175.4, 173.8, .8), 28: (183.4, 184.9, .8), 29: (190.6, 195.9, .8),
                30: (198.1, 208.4, .8), 31: (204.7, 222.3, .8), 32: (210.8, 237.4, .7), 35: (230, 284.2, .8), 36: (236, 301.6, .8),
                37: (238.4, 287.7, .6), 39: (207.7, 303.7, .8), 40: (184.2, 322.5, .8), 41: (162.2, 324.9, .7), 42: (138.4, 330.4, .8),
                7: (10.3, 682.5, .1), 9: (3.9, 687.6, .4), 13: (54.4, 442.7, .2), 43: (26.2, 468.4, .5), 45: (57.7, 359.7, .3)}
        t = build_track(real, H, Config(), fps=10.0)
        self.assertEqual((t.f0, t.f1), (15, 36))
        self.assertIsNone(t.bounce)

    def test_false_positives_before_ball_are_ignored(self):
        raw, _ = make_delivery(np.random.default_rng(3), bounce=True, dropout=0.0)
        raw[0] = [(5.0, 900.0, 0.45)]
        t = build_track(raw, H, Config())
        self.assertGreater(t.f0, 0)

    def test_too_few_points(self):
        self.assertIsNone(build_track({1: (1, 1, 0.9)}, H, Config()))

    def test_fuzz_bounce_detection_rate(self):
        rng, ok_b, ok_f, n = np.random.default_rng(7), 0, 0, 150
        for k in range(n):
            bounce = k % 2 == 0
            raw, truth = make_delivery(rng, bounce=bounce)
            t = build_track(raw, H, Config())
            if t is None:
                continue
            if bounce:
                ok_b += t.bounce is not None and abs(t.bounce[0] - truth["bounce"]) <= 2
            else:
                ok_f += t.bounce is None
        self.assertGreaterEqual(ok_b / (n // 2), 0.85)
        self.assertGreaterEqual(ok_f / (n // 2), 0.85)

    def test_legacy_outlier_filter(self):
        acc = filter_outliers({10: (0, 0, .9), 11: (5, 5, .9), 12: (900, 900, .9)}, 90, 0.7, 0.3)
        self.assertNotIn(12, acc)


class TestDetectorFilter(unittest.TestCase):
    def test_static_spot_removed_but_moving_ball_kept(self):
        dets = []
        for i in range(40):
            d = FrameDet(balls=[(100.0, 100.0, 0.5), (50.0 + 8 * i, 300.0 + 3 * i, 0.8)])
            dets.append(d)
        drop_static(dets, H, Config())
        self.assertTrue(all(len(d.balls) == 1 for d in dets[5:-5]))
        self.assertTrue(all(abs(d.balls[0][0] - 100) > 1 for d in dets[5:-5]))


class TestBounceAndLbw(unittest.TestCase):
    def test_pitching(self):
        poly = lane_polygon((190, 100, 230, 250), 1280)
        self.assertEqual(pitching_status(None, poly)[1], "na")
        self.assertEqual(pitching_status((210, 600), poly)[1], "in")
        self.assertEqual(pitching_status((600, 600), poly)[1], "out")
        self.assertEqual(pitching_status((210, 600), None)[0], "N/A (no stumps)")
        self.assertEqual(pitching_status(None, poly, unclear=True)[0], "N/A (unclear)")

    def test_batter_stumps_single_and_double(self):
        far, near = (190, 100, 230, 250), (200, 1100, 280, 1200)
        self.assertEqual(batter_stumps([far], 1280), far)
        self.assertEqual(batter_stumps([far, near], 1280), far)
        self.assertIsNone(batter_stumps([near], 1280))
        self.assertIsNone(batter_stumps([], 1280))


class TestOverlay(unittest.TestCase):
    def test_track_progress_draws_only_points_up_to_current_frame(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        xs, ys = np.array([10, 20, 30]), np.array([10, 20, 30])

        draw_track_progress(img, xs, ys, 1)

        np.testing.assert_array_equal(img[20, 20], (0, 255, 255))
        np.testing.assert_array_equal(img[30, 30], (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
