"""Tests for confidence/features.py, on the real contracts.py objects.

Run: pytest tests/test_features.py
"""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np

from confidence import features as feat
from contracts import HeightSource, PhotoEvidence
from fixtures.fake_fits import generate_labelled, make_fit, make_footprint

PHOTO = PhotoEvidence(mask=np.zeros((4, 4), bool), silhouette_margin=0.15)
FP = make_footprint(0.93)  # ROOFTOP geocode, strong match


def fit(**kw):
    return replace(make_fit(iou=0.8, hausdorff=2.0, area_ratio=1.0, margin=0.2, aniso=0.0, neighbour=0.01), **kw)


def vec(f=None, p=PHOTO, fp=FP):
    return feat.feature_dict(f or fit(), p, fp)


class TableTests(unittest.TestCase):
    def test_b3_features_and_expected_signs(self):
        self.assertEqual(len(feat.FEATURE_NAMES), 10)
        self.assertEqual(set(feat.EXPECTED_SIGN), set(feat.FEATURE_NAMES))
        self.assertEqual({n for n, s in feat.EXPECTED_SIGN.items() if s < 0},
                         {"hausdorff_m", "abs_area_ratio_log", "abs_anisotropy_log_ratio", "max_neighbor_overlap"})

    def test_reduced_set_is_the_b4_four(self):
        self.assertEqual(set(feat.REDUCED_FEATURES), {"footprint_iou", "hausdorff_m", "rectilinearity", "rotation_margin_footprint"})


class ValueTests(unittest.TestCase):
    def test_each_feature_comes_from_the_right_place(self):
        d = vec(fit(iou=0.71, hausdorff_m=3.5, rotation_margin_footprint=0.33, max_neighbor_overlap=0.04))
        self.assertEqual(d["footprint_iou"], 0.71)
        self.assertEqual(d["hausdorff_m"], 3.5)
        self.assertEqual(d["rotation_margin_footprint"], 0.33)
        self.assertEqual(d["rotation_margin_silhouette"], 0.15)  # photo.silhouette_margin
        self.assertEqual(d["rectilinearity"], 0.93)  # fp.rectilinearity
        self.assertEqual(d["max_neighbor_overlap"], 0.04)

    def test_area_ratio_log_is_symmetric_about_one(self):
        self.assertAlmostEqual(vec(fit(area_ratio=2.0))["abs_area_ratio_log"], math.log(2))
        self.assertAlmostEqual(vec(fit(area_ratio=0.5))["abs_area_ratio_log"], math.log(2))
        self.assertEqual(vec(fit(area_ratio=1.0))["abs_area_ratio_log"], 0.0)

    def test_anisotropy_is_a_magnitude(self):
        self.assertAlmostEqual(vec(fit(anisotropy_log_ratio=-0.3))["abs_anisotropy_log_ratio"], 0.3)
        self.assertAlmostEqual(vec(fit(anisotropy_log_ratio=0.3))["abs_anisotropy_log_ratio"], 0.3)


class BinaryFeatureTests(unittest.TestCase):
    """contracts.py v1.0 puts both binaries on the objects; nothing extra is passed in."""

    def test_geocode_rooftop_comes_from_the_footprint(self):
        for kind, expected in (("ROOFTOP", 1.0), ("RANGE_INTERPOLATED", 0.0), ("GEOMETRIC_CENTER", 0.0), ("APPROXIMATE", 0.0), ("", 0.0)):
            self.assertEqual(vec(fp=make_footprint(0.9, geocode_location_type=kind))["geocode_rooftop"], expected, kind)

    def test_height_source_authoritative_is_osm_tags_only(self):
        # Microsoft is deliberately NOT authoritative (measured 44-67% low on VT buildings; see contracts.HeightSource).
        for source, expected in ((HeightSource.OSM_HEIGHT, 1.0), (HeightSource.OSM_LEVELS, 1.0), (HeightSource.MICROSOFT, 0.0),
                                 (HeightSource.SINGLE_VIEW_METROLOGY, 0.0), (HeightSource.MONOCULAR_DEPTH, 0.0),
                                 (HeightSource.PROPORTIONAL_FALLBACK, 0.0)):
            self.assertEqual(vec(fit(height_source=source))["height_source_authoritative"], expected, source)


class VectorTests(unittest.TestCase):
    def test_vector_follows_the_requested_names_in_order(self):
        full = feat.feature_vector(fit(), PHOTO, FP)
        self.assertEqual(full.shape, (10,))
        reduced = feat.feature_vector(fit(), PHOTO, FP, names=feat.REDUCED_FEATURES)
        d = vec()
        np.testing.assert_allclose(reduced, [d[n] for n in feat.REDUCED_FEATURES])
        np.testing.assert_allclose(full, [d[n] for n in feat.FEATURE_NAMES])

    def test_matrix_over_labelled_rows(self):
        rows = generate_labelled(n_buildings=4, n_configs=2, n_corruptions=3, seed=1)
        X = feat.feature_matrix(rows)
        self.assertEqual(X.shape, (11, 10))
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(feat.feature_matrix(rows, feat.REDUCED_FEATURES).shape, (11, 4))


class FailLoudlyTests(unittest.TestCase):
    def test_nonpositive_or_nonfinite_area_ratio(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=bad):
                vec(fit(area_ratio=bad))

    def test_nonfinite_metrics(self):
        for kw in (dict(iou=float("nan")), dict(hausdorff_m=float("inf")), dict(anisotropy_log_ratio=float("nan")),
                   dict(max_neighbor_overlap=float("nan")), dict(rotation_margin_footprint=float("inf"))):
            with self.assertRaisesRegex(ValueError, "not finite"):
                vec(fit(**kw))
        with self.assertRaisesRegex(ValueError, "not finite"):
            vec(p=replace(PHOTO, silhouette_margin=float("nan")))
        with self.assertRaisesRegex(ValueError, "not finite"):
            vec(fp=replace(FP, rectilinearity=float("nan")))

    def test_the_binaries_are_no_longer_arguments(self):
        with self.assertRaises(TypeError):
            feat.feature_dict(fit(), PHOTO, FP, geocode_rooftop=True, height_source_authoritative=True)

    def test_missing_attribute_and_unknown_feature(self):
        with self.assertRaises(AttributeError):
            feat.feature_dict(NS(iou=0.8), PHOTO, FP)
        with self.assertRaisesRegex(KeyError, "unknown feature"):
            feat.feature_vector(fit(), PHOTO, FP, names=("nope",))


if __name__ == "__main__":
    unittest.main()
