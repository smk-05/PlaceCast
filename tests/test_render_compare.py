"""Tests for perception/render_compare.py using trimesh boxes and hand-drawn masks (no real meshes).

Run: pytest tests/test_render_compare.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

from perception import render_compare as rc  # noqa: E402

UP_Y, DOWN_Y = (0, 1, 0), (0, -1, 0)

# glTF frame (+Y up, front on +Z). Seen from +Z (azimuth 0) or -Z (180) it is 20 wide x 40 tall
# (aspect 0.5); seen from +X or -X (azimuth 90 / 270) it is 30 wide x 40 tall (aspect 0.75).
BOX = trimesh.creation.box(extents=(20, 40, 30))


def rect_mask(shape, y0, x0, h, w):
    m = np.zeros(shape, bool)
    m[y0 : y0 + h, x0 : x0 + w] = True
    return m


def stepped_building():
    """A 40x15x30 base with a 15x30x15 tower over its -X end: an L-shaped silhouette from +Z."""
    base = trimesh.creation.box(extents=(40, 15, 30)).apply_translation((0, 7.5, 0))
    tower = trimesh.creation.box(extents=(15, 30, 15)).apply_translation((-12.5, 30, 0))
    return trimesh.util.concatenate([base, tower])


def stepped_mask_seen_from_front():
    """Independently drawn (not rendered): the +Z view, so the tower (at -X) is on the image LEFT."""
    to_px = lambda x, y: [(x + 20) * 8 + 20, (45 - y) * 8 + 20]
    poly = np.array([to_px(x, y) for x, y in [(-20, 0), (20, 0), (20, 15), (-5, 15), (-5, 45), (-20, 45)]], np.int32)
    canvas = np.zeros((400, 360), np.uint8)
    cv2.fillPoly(canvas, [poly], 1)
    return canvas.astype(bool)


class SilhouetteTests(unittest.TestCase):
    def test_overlapping_front_and_back_faces_do_not_cancel(self):
        # Regression: cv2.fillPoly is even-odd, so a closed box seen face-on rendered as an outline.
        sil = rc.render_silhouette(BOX, UP_Y, 0.0)
        ys, xs = np.nonzero(sil)
        bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        self.assertGreater(sil.sum(), 0.98 * bbox_area)

    def test_silhouette_aspect_matches_the_view(self):
        for az, aspect in ((0, 0.5), (np.pi / 2, 0.75)):
            ys, xs = np.nonzero(rc.render_silhouette(BOX, UP_Y, az))
            self.assertAlmostEqual((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1), aspect, delta=0.03)

    def test_camera_is_right_handed_with_plus_x_on_the_right(self):
        right, up, toward = rc.camera_basis(UP_Y, 0.0)
        np.testing.assert_allclose([right, up, toward], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-12)
        right, _, toward = rc.camera_basis(UP_Y, np.pi / 2)  # camera swings CCW seen from above, to +X
        np.testing.assert_allclose([right, toward], [[0, 0, -1], [1, 0, 0]], atol=1e-12)


class BoxScoringTests(unittest.TestCase):
    def test_finds_the_matching_faces_regardless_of_mask_position_and_scale(self):
        big = rect_mask((800, 1000), 150, 300, 400, 200)  # aspect 0.5, mid-image
        small = rect_mask((200, 300), 10, 200, 120, 60)  # aspect 0.5, tiny, near a corner
        for mask in (big, small):
            r = rc.render_compare(BOX, mask)
            self.assertEqual(len(r.candidates), 24)
            self.assertGreater(r.best.score, 0.97)
            self.assertIn(r.best.up_axis, (UP_Y, DOWN_Y))
            self.assertIn(r.best.azimuth_deg, (0.0, 180.0))
            side = next(c for c in r.candidates if c.up_axis == UP_Y and c.azimuth_deg == 90.0)
            self.assertAlmostEqual(side.score, 0.5 / 0.75, delta=0.03)  # wrong faces: aspect 0.75 vs 0.5

    def test_a_box_ties_with_its_own_back_view_so_margin_is_zero(self):
        r = rc.render_compare(BOX, rect_mask((800, 1000), 150, 300, 400, 200))
        self.assertAlmostEqual(r.margin, 0.0, places=9)
        self.assertEqual(r.ties_with_best, 3)  # +Y/-Y up x front/back
        self.assertEqual(r.best.up_axis, UP_Y)  # ties resolve to the glTF-canonical orientation
        self.assertEqual(r.best.azimuth_deg, 0.0)

    def test_perspective_keystone_does_not_change_the_answer(self):
        # A tall building photographed looking up: the top is narrower than the base.
        mask = np.zeros((500, 300), np.uint8)
        cv2.fillPoly(mask, [np.array([[50, 450], [250, 450], [225, 50], [75, 50]], np.int32)], 1)
        r = rc.render_compare(BOX, mask.astype(bool))
        self.assertIn(r.best.up_axis, (UP_Y, DOWN_Y))
        self.assertIn(r.best.azimuth_deg, (0.0, 180.0))
        self.assertGreater(r.best.score, 0.85)
        side = next(c for c in r.candidates if c.up_axis == UP_Y and c.azimuth_deg == 90.0)
        self.assertGreater(r.best.score - side.score, 0.1)

    def test_theta0_shifts_every_azimuth(self):
        mask = rect_mask((800, 1000), 150, 300, 400, 200)
        r = rc.render_compare(BOX, mask, theta0=np.pi / 4)
        self.assertEqual({round(c.azimuth_deg) for c in r.candidates}, {45, 135, 225, 315})
        self.assertLess(r.best.score, rc.render_compare(BOX, mask).best.score - 0.05)  # diagonal views fit worse

    def test_up_axes_restricts_the_candidate_set(self):
        r = rc.render_compare(BOX, rect_mask((800, 1000), 150, 300, 400, 200), up_axes=[UP_Y])
        self.assertEqual(len(r.candidates), 4)
        self.assertEqual({c.up_axis for c in r.candidates}, {UP_Y})


class AsymmetricBuildingTests(unittest.TestCase):
    """An L-shaped silhouette has no mirror ties, so it also checks handedness and up-axis handling."""

    def test_unique_best_with_a_healthy_margin(self):
        r = rc.render_compare(stepped_building(), stepped_mask_seen_from_front())
        self.assertEqual((r.best.up_axis, r.best.azimuth_deg), (UP_Y, 0.0))
        self.assertGreater(r.best.score, 0.95)
        self.assertEqual(r.ties_with_best, 0)
        self.assertGreater(r.margin, 0.05)  # SPEC 6.6 review threshold

    def test_mirrored_photo_selects_the_opposite_side(self):
        r = rc.render_compare(stepped_building(), np.fliplr(stepped_mask_seen_from_front()))
        self.assertEqual((r.best.up_axis, r.best.azimuth_deg), (UP_Y, 180.0))
        self.assertGreater(r.best.score, 0.95)

    def test_upside_down_photo_selects_the_flipped_up_axis(self):
        r = rc.render_compare(stepped_building(), np.rot90(stepped_mask_seen_from_front(), 2))
        self.assertEqual((r.best.up_axis, r.best.azimuth_deg), (DOWN_Y, 0.0))
        self.assertGreater(r.best.score, 0.95)


class SymmetryLabelTests(unittest.TestCase):
    """A.4: margin < 0.05 AND footprint aspect < 1.1 -> arbitrary_symmetric, flagged for review."""

    SYM = rc.ARBITRARY_SYMMETRIC

    def test_thresholds_are_strict_on_both_conditions(self):
        self.assertEqual((rc.REVIEW_MARGIN, rc.SYMMETRIC_ASPECT), (0.05, 1.1))
        self.assertEqual(rc.symmetry_label(0.0, 1.0), self.SYM)
        self.assertEqual(rc.symmetry_label(0.049, 1.099), self.SYM)
        self.assertIsNone(rc.symmetry_label(0.05, 1.0))  # margin exactly 0.05: not symmetric
        self.assertIsNone(rc.symmetry_label(0.0, 1.1))  # aspect exactly 1.1: not square
        self.assertIsNone(rc.symmetry_label(0.0, 2.0))  # low margin on an elongated footprint is not symmetry
        self.assertIsNone(rc.symmetry_label(0.2, 1.0))  # square footprint but the photo does decide

    def test_aspect_below_one_is_read_as_its_reciprocal(self):
        self.assertEqual(rc.symmetry_label(0.0, 1 / 1.05), self.SYM)
        self.assertIsNone(rc.symmetry_label(0.0, 1 / 1.5))

    def test_unknown_aspect_never_sets_the_label(self):
        self.assertIsNone(rc.symmetry_label(0.0, None))

    def test_nonsense_aspect_fails_loudly(self):
        for bad in (0, -1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "footprint_aspect"):
                rc.symmetry_label(0.0, bad)

    def test_square_footprint_box_is_arbitrary_symmetric_and_flagged(self):
        square = trimesh.creation.box(extents=(30, 40, 30))  # all four facades identical
        mask = rect_mask((800, 1000), 150, 300, 400, 300)  # 30 x 40 -> matches every azimuth
        r = rc.render_compare(square, mask, footprint_aspect=1.0)
        self.assertAlmostEqual(r.margin, 0.0, places=9)
        self.assertEqual(r.ties_with_best, 7)  # 2 up-axes x 4 azimuths, minus best
        self.assertEqual(r.label, self.SYM)
        self.assertTrue(r.needs_review)
        self.assertEqual(r.to_dict()["label"], self.SYM)

    def test_low_margin_on_an_elongated_footprint_is_flagged_but_not_called_symmetric(self):
        mask = rect_mask((800, 1000), 150, 300, 400, 200)
        for aspect in (1.5, None):
            r = rc.render_compare(BOX, mask, footprint_aspect=aspect)  # front/back tie, margin 0
            self.assertAlmostEqual(r.margin, 0.0, places=9)
            self.assertIsNone(r.label)
            self.assertTrue(r.needs_review)

    def test_decisive_margin_is_neither_labelled_nor_flagged(self):
        r = rc.render_compare(stepped_building(), stepped_mask_seen_from_front(), footprint_aspect=1.0)
        self.assertGreater(r.margin, rc.REVIEW_MARGIN)
        self.assertIsNone(r.label)
        self.assertFalse(r.needs_review)


class InputTests(unittest.TestCase):
    def test_mask_png_path_and_scene_give_the_same_result_as_arrays(self):
        mask = rect_mask((800, 1000), 150, 300, 400, 200)
        expected = rc.render_compare(BOX, mask)
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "mask.png"
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(png)
            for mesh, m in ((BOX, png), (BOX, str(png)), (trimesh.Scene(BOX), mask)):
                r = rc.render_compare(mesh, m)
                self.assertEqual((r.best, r.margin), (expected.best, expected.margin))

    def test_bad_inputs_fail_loudly(self):
        good = rect_mask((100, 100), 10, 10, 50, 30)
        with self.assertRaisesRegex(ValueError, "mask is empty"):
            rc.render_compare(BOX, np.zeros((100, 100), bool))
        with self.assertRaisesRegex(ValueError, "no faces"):
            rc.render_compare(trimesh.Trimesh(), good)
        with self.assertRaisesRegex(ValueError, "up_axes is empty"):
            rc.render_compare(BOX, good, up_axes=[])
        with self.assertRaises(TypeError):
            rc.render_compare("not a mesh", good)
        with self.assertRaisesRegex(ValueError, "2-D"):
            rc.render_compare(BOX, np.ones((10, 10, 3), bool))


if __name__ == "__main__":
    unittest.main()
