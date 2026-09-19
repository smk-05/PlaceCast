"""Tests for fixtures/fake_fits.py: my labelled generator (generate_labelled) and that their original API survives.

Run: pytest tests/test_fake_fits.py
"""
import math
import unittest

import numpy as np

import contracts
from confidence import features as feat
from contracts import CandidateId, Disambiguator, HeightSource
from fixtures import fake_fits as ff


class ContractTests(unittest.TestCase):
    def test_the_fixture_uses_the_frozen_contract_classes_not_a_mirror(self):
        self.assertIs(ff.FitResult, contracts.FitResult)
        self.assertIs(ff.Footprint, contracts.Footprint)
        self.assertIs(ff.PhotoEvidence, contracts.PhotoEvidence)

    def test_the_original_api_survives_untouched(self):
        # tests/test_smoke.py, test_fit_synthetic.py, test_footprint_selection.py and test_orientation_contract.py use these.
        rows = ff.generate(n=140)
        self.assertGreaterEqual(len(rows), 130)
        fit, fp, label = rows[0]
        self.assertTrue(hasattr(fit, "iou") and hasattr(fp, "rectilinearity") and label in (0, 1))
        self.assertTrue(0.5 < np.mean([lab for *_, lab in rows]) < 0.8)  # "a working solver mostly accepts"
        self.assertEqual(ff.make_fit(iou=0.9, hausdorff=1.0).iou, 0.9)
        self.assertEqual(ff.make_footprint(0.9).rectilinearity, 0.9)


class GenerateLabelledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ff.generate_labelled(seed=0)

    def test_default_layout_follows_b2(self):
        self.assertEqual(len(self.rows), 20 * 4 + 40)
        bench = [r for r in self.rows if r.config_index is not None]
        corrupt = [r for r in self.rows if r.config_index is None]
        self.assertEqual((len(bench), len(corrupt)), (80, 40))
        self.assertEqual({r.config_index for r in bench}, {0, 1, 2, 3})
        self.assertEqual({r.kind for r in corrupt}, set(ff.CORRUPTIONS))
        self.assertTrue(all(r.label == 0 and not r.label_flipped for r in corrupt))

    def test_is_deterministic_per_seed(self):
        a, b, c = ff.generate_labelled(seed=0), ff.generate_labelled(seed=0), ff.generate_labelled(seed=1)
        np.testing.assert_array_equal(feat.feature_matrix(a), feat.feature_matrix(b))
        self.assertFalse(np.array_equal(feat.feature_matrix(a), feat.feature_matrix(c)))
        self.assertEqual([r.label for r in a], [r.label for r in b])

    def test_the_port_onto_contracts_did_not_change_the_random_stream(self):
        # confidence/train.py's numbers were produced before the port; the row count and base rate must not move.
        self.assertEqual(sum(r.label for r in self.rows), 48)

    def test_classes_are_present_but_not_cleanly_separable(self):
        labels = np.array([r.label for r in self.rows])
        self.assertTrue(0.3 < labels.mean() < 0.6)
        self.assertTrue(any(r.label_flipped for r in self.rows))  # annotator disagreement is simulated
        self.assertTrue(any(r.kind == "borderline" for r in self.rows))

    def test_better_solver_configs_are_accepted_more_often(self):
        rate = lambda c: np.mean([r.label for r in self.rows if r.config_index == c])
        self.assertLess(rate(0), rate(3))

    def test_metrics_carry_the_expected_signal(self):
        X, y = feat.feature_matrix(self.rows), np.array([r.label for r in self.rows])
        col = {n: i for i, n in enumerate(feat.FEATURE_NAMES)}
        self.assertGreater(X[y == 1, col["footprint_iou"]].mean() - X[y == 0, col["footprint_iou"]].mean(), 0.15)
        self.assertLess(X[y == 1, col["hausdorff_m"]].mean(), X[y == 0, col["hausdorff_m"]].mean())
        self.assertLess(X[y == 1, col["abs_area_ratio_log"]].mean(), X[y == 0, col["abs_area_ratio_log"]].mean())

    def test_values_are_internally_coherent(self):
        for r in self.rows:
            f, p, fp = r.fit, r.photo, r.footprint
            self.assertAlmostEqual(f.area_ratio, abs(f.scale_x) * f.scale_y, places=9)
            self.assertAlmostEqual(f.anisotropy_log_ratio, math.log(abs(f.scale_x) / f.scale_y), places=9)
            top = sorted(p.silhouette_scores.values(), reverse=True)
            self.assertAlmostEqual(p.silhouette_margin, top[0] - top[1], places=9)
            self.assertEqual(set(p.silhouette_scores), {CandidateId(2, k) for k in range(4)})  # +Y up, azimuth_k 0..3
            self.assertAlmostEqual(p.mask_area_frac, float(p.mask.mean()))
            self.assertGreaterEqual(fp.ombb.a, fp.ombb.b)
            self.assertTrue(0 <= f.iou <= 1 and 0 <= f.rotation_margin_footprint <= 1 and 0 <= f.max_neighbor_overlap <= 1)
            self.assertIsInstance(f.disambiguated_by, Disambiguator)
            self.assertIsInstance(f.height_source, HeightSource)
            self.assertIsNone(f.exif_silhouette_disagree)
            self.assertTrue(p.exif_heading_deg is None or 0 <= p.exif_heading_deg < 360)

    def test_mirrored_corruptions_are_real_reflections_and_nothing_else_is(self):
        for r in self.rows:
            self.assertEqual(r.fit.is_mirrored, r.kind == "corrupt_mirrored", r.kind)

    def test_footprint_properties_the_gate_and_features_read(self):
        self.assertTrue(any(r.footprint.has_holes for r in self.rows))
        self.assertEqual(sum(r.footprint.has_holes for r in self.rows), sum(len(r.footprint.holes_enu) > 0 for r in self.rows))
        self.assertTrue(any(r.footprint.geocode_rooftop for r in self.rows) and not all(r.footprint.geocode_rooftop for r in self.rows))
        self.assertTrue(all(not r.footprint.is_weak_match for r in self.rows))  # strong matches: the gate's weak-match rule is not in play
        self.assertTrue(any(r.fit.height_source_authoritative for r in self.rows) and not all(r.fit.height_source_authoritative for r in self.rows))

    def test_near_square_buildings_have_aspect_under_1_1_and_tiny_margins(self):
        square = [r for r in self.rows if r.footprint.ombb.aspect < 1.1]
        other = [r for r in self.rows if r.footprint.ombb.aspect >= 1.1]
        self.assertTrue(square and other)
        # Near-square footprints: the four OMBB candidates ~tie. (A mirrored-mesh corruption draws its margin
        # from U(0, 0.1) whatever the building, so it is excluded.)
        own = [r for r in square if r.kind != "corrupt_mirrored"]
        self.assertLessEqual(max(r.fit.rotation_margin_footprint for r in own), 0.06)
        self.assertGreater(np.mean([r.fit.rotation_margin_footprint for r in other]), 0.1)

    def test_features_are_finite_for_every_row(self):
        self.assertTrue(np.isfinite(feat.feature_matrix(self.rows)).all())

    def test_bad_label_noise_is_rejected(self):
        with self.assertRaises(ValueError):
            ff.generate_labelled(label_noise=0.6)


if __name__ == "__main__":
    unittest.main()
