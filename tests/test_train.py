"""Tests for confidence/train.py.

Run: pytest tests/test_train.py      (the full 120-row fit takes ~15 s; it runs once)
"""
import contextlib
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from confidence import features as feat  # noqa: E402
from confidence import train  # noqa: E402
from fixtures.fake_fits import generate_labelled as generate  # noqa: E402


class FullModelTests(unittest.TestCase):
    """The default 120-row synthetic run, fitted once."""

    @classmethod
    def setUpClass(cls):
        cls.rows = generate(seed=0)
        cls.result = train.train(cls.rows, seed=0)

    def test_full_feature_set_on_120_rows(self):
        r = self.result
        self.assertEqual((r.n_rows, r.names), (120, feat.FEATURE_NAMES))
        self.assertIsNone(r.class_weight)  # 40% minority is above the 25% B.4 trigger

    def test_loo_predictions_are_out_of_sample_and_beat_the_baseline(self):
        r, m = self.result, self.result.metrics
        self.assertEqual(r.loo_proba.shape, (120,))
        self.assertTrue(((r.loo_proba > 0) & (r.loo_proba < 1)).all())
        self.assertGreater(m["loo_accuracy"], m["majority_class_accuracy"] + 0.1)
        self.assertGreater(m["loo_auc"], 0.85)
        self.assertLess(m["loo_log_loss"], 0.5)

    def test_c_comes_from_the_grid(self):
        self.assertTrue(any(math.isclose(self.result.C, c) for c in train.C_GRID))

    def test_coefficient_table_is_sorted_and_checks_expected_signs(self):
        table = self.result.coefficients
        mags = [abs(c["coef_per_sd"]) for c in table]
        self.assertEqual(mags, sorted(mags, reverse=True))
        self.assertEqual({c["feature"] for c in table}, set(feat.FEATURE_NAMES))
        for c in table:
            self.assertAlmostEqual(c["odds_ratio_per_sd"], math.exp(c["coef_per_sd"]))
            self.assertEqual(c["sign_matches"], np.sign(c["coef_per_sd"]) == feat.EXPECTED_SIGN[c["feature"]])
        binary = {"geocode_rooftop", "height_source_authoritative"}
        self.assertFalse(binary & {c["feature"] for c in table[:3]})  # the metrics dominate, not the flags
        self.assertGreaterEqual(sum(c["sign_matches"] for c in table), 7)

    def test_reliability_uses_all_rows_in_four_bins(self):
        rel = self.result.reliability
        self.assertEqual(len(rel["bins"]), 4)
        self.assertEqual(sum(b["n"] for b in rel["bins"]), 120)
        self.assertLess(rel["ece"], 0.15)

    def test_report_names_the_synthetic_source_and_the_missing_calibration_layer(self):
        text = train.format_report(self.result, "fake")
        self.assertIn("SYNTHETIC DATA", text)
        self.assertIn("No calibration layer", text)
        self.assertNotIn("SYNTHETIC DATA", train.format_report(self.result, "benchmark"))

    def test_artifacts_are_written_and_model_json_reproduces_the_fitted_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = train.write_artifacts(self.result, tmp, "fake", 0)
            for name in ("coefficients.csv", "reliability.csv", "reliability.png", "model.json"):
                self.assertTrue((out / name).is_file(), name)
            self.assertEqual((out / "reliability.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with open(out / "coefficients.csv", newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 10)
            spec = json.loads((out / "model.json").read_text(encoding="utf-8"))
        self.assertTrue(spec["trained_on_synthetic"])  # a fake-trained model announces itself
        self.assertIsNone(spec["calibration_layer"])
        self.assertEqual(spec["feature_names"], list(feat.FEATURE_NAMES))
        # p = sigmoid(intercept + sum coef * (x - mean) / scale) must equal sklearn's own predict_proba.
        X = feat.feature_matrix(self.rows, feat.FEATURE_NAMES)
        z = spec["intercept"] + ((X - spec["scaler_mean"]) / spec["scaler_scale"]) @ np.array(spec["coef_standardised"])
        np.testing.assert_allclose(1 / (1 + np.exp(-z)), self.result.model.predict_proba(X)[:, 1], atol=1e-9)

    def test_no_calibration_layer_in_the_source(self):
        source = Path(train.__file__).read_text(encoding="utf-8")
        for banned in ("CalibratedClassifierCV", "IsotonicRegression", "calibration_curve"):
            self.assertNotIn(banned, source)


class SmallDataTests(unittest.TestCase):
    def test_under_60_rows_falls_back_to_the_four_feature_model(self):
        rows = generate(n_buildings=10, n_configs=4, n_corruptions=10, seed=3)
        self.assertEqual(len(rows), 50)
        r = train.train(rows, seed=0)
        self.assertEqual(r.names, feat.REDUCED_FEATURES)
        self.assertIn("B.4", r.feature_note)
        self.assertEqual(len(r.coefficients), 4)

    def test_training_is_deterministic(self):
        rows = generate(n_buildings=10, n_configs=4, n_corruptions=10, seed=3)
        a, b = train.train(rows, seed=5), train.train(rows, seed=5)
        np.testing.assert_array_equal(a.loo_proba, b.loo_proba)
        self.assertEqual((a.C, a.intercept), (b.C, b.intercept))


class ThresholdTests(unittest.TestCase):
    P = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    def test_thresholds_from_probabilities(self):
        t = train.choose_thresholds(self.Y, self.P, 0.95, 0.95, min_support=3)
        self.assertEqual((t["p_accept"], t["p_reject"]), (0.7, 0.6))  # p>=.6 already includes a negative
        self.assertEqual((t["n_accept"], t["accept_precision"], t["false_accepts"]), (5, 1.0, 0))
        self.assertEqual((t["n_reject"], t["reject_precision"], t["good_placements_rejected"]), (5, 1.0, 0))
        self.assertEqual(t["n_review"], 0)

    def test_min_support_stops_a_single_lucky_row_setting_the_threshold(self):
        t = train.choose_thresholds(self.Y, self.P, 0.95, 0.95, min_support=6)
        self.assertIsNone(t["p_accept"])  # only 5 rows can reach 95% precision
        self.assertIsNone(t["p_reject"])

    def test_unreachable_precision_yields_none_not_a_default(self):
        y, p = np.array([0, 1] * 6), np.linspace(0.1, 0.9, 12)
        t = train.choose_thresholds(y, p, 0.95, 0.95, min_support=3)
        self.assertEqual((t["p_accept"], t["p_reject"]), (None, None))
        self.assertEqual((t["n_accept"], t["n_reject"], t["n_review"]), (0, 0, 12))
        self.assertIsNone(t["accept_precision"])

    def test_reject_threshold_stays_below_accept(self):
        t = train.choose_thresholds(self.Y, self.P, 0.95, 0.5, min_support=3)  # lax rejection target
        self.assertLess(t["p_reject"], t["p_accept"])

    def test_the_report_says_when_nothing_is_auto_accepted(self):
        y, p = np.array([0, 1] * 6), np.linspace(0.1, 0.9, 12)
        thresholds = train.choose_thresholds(y, p, 0.95, 0.95, min_support=3)
        result = type("R", (), dict(
            metrics=dict(base_rate=0.5, majority_class_accuracy=0.5, loo_accuracy=0.5, loo_auc=0.5, loo_log_loss=0.7, loo_brier=0.25),
            n_rows=12, n_positive=6, feature_note="x", class_weight=None, C=1.0, thresholds=thresholds, coefficients=[],
            reliability=train.reliability_bins(y, p)))
        self.assertIn("nothing is auto-accepted", train.format_report(result, "benchmark"))


class ReliabilityTests(unittest.TestCase):
    def test_bin_arithmetic(self):
        p = np.array([0.05, 0.10, 0.30, 0.60, 0.70, 0.90, 0.95, 1.00])
        y = np.array([0, 0, 1, 1, 0, 1, 1, 1])
        bins = train.reliability_bins(y, p, 4)["bins"]
        self.assertEqual([b["n"] for b in bins], [2, 1, 2, 3])  # p=1.0 lands in the last bin
        self.assertAlmostEqual(bins[0]["mean_predicted"], 0.075)
        self.assertEqual([b["observed_frequency"] for b in bins], [0.0, 1.0, 0.5, 1.0])
        self.assertAlmostEqual(train.reliability_bins(y, p, 4)["ece"], (2 * 0.075 + 0.7 + 2 * 0.15 + 3 * abs(1 - 0.95)) / 8 * 1.0, delta=0.05)

    def test_empty_bins_are_kept_and_marked(self):
        bins = train.reliability_bins(np.array([0, 1]), np.array([0.05, 0.95]), 4)["bins"]
        self.assertEqual([b["n"] for b in bins], [1, 0, 0, 1])
        self.assertTrue(math.isnan(bins[1]["observed_frequency"]))

    def test_wilson_interval(self):
        lo, hi = train.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertTrue(all(math.isnan(v) for v in train.wilson(0, 0)))
        self.assertEqual(train.wilson(0, 5)[0], 0.0)
        self.assertEqual(train.wilson(5, 5)[1], 1.0)


class GuardTests(unittest.TestCase):
    def test_class_weight_is_balanced_only_for_a_minority_under_25_percent(self):
        self.assertIsNone(train.choose_class_weight(np.array([0] * 60 + [1] * 40)))
        self.assertEqual(train.choose_class_weight(np.array([0] * 80 + [1] * 20)), "balanced")
        self.assertEqual(train.choose_class_weight(np.array([0] * 20 + [1] * 80)), "balanced")

    def test_label_checks(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            train.check_labels(np.array([1] * 20))
        with self.assertRaisesRegex(ValueError, "at least 3"):
            train.check_labels(np.array([0] * 20 + [1] * 2))
        with self.assertRaisesRegex(ValueError, "0/1"):
            train.check_labels(np.array([0, 1, 2] * 5))
        train.check_labels(np.array([0] * 3 + [1] * 3))

    def test_an_unknown_source_is_rejected_and_the_cli_says_so(self):
        with self.assertRaisesRegex(ValueError, "unknown source"):
            train.load_rows("real")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(train.main(["--source", "real"]), 1)
        self.assertIn("error:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
