"""Tests for confidence/gate.py: the SPEC 9.1 table (standalone, default) and the learned path (flagged).

Run: python confidence/test_gate.py -v
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import gate  # noqa: E402

A, R, X = gate.AUTO_ACCEPT, gate.REVIEW, gate.REJECT

GOOD_FIT = dict(iou=0.85, hausdorff_m=1.2, area_ratio=1.02, rotation_margin_footprint=0.2, anisotropy_log_ratio=0.01,
                max_neighbor_overlap=0.005, disambiguated_by="exif_heading")
PHOTO = NS(silhouette_margin=0.15)
FP = NS(rectilinearity=0.93)


def fit(**kw):
    return NS(**{**GOOD_FIT, **kw})


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in (gate.ENV_FLAG, gate.ENV_MODEL)}
    env.update(extra)
    return env


class TableBandTests(unittest.TestCase):
    """Every SPEC 9.1 edge, on both sides."""

    def band(self, name, **kw):
        return {c.name: c.band for c in gate.table_checks(fit(**kw), NS(rectilinearity=kw.pop("rect", 0.93)))}[name]

    def test_bands_at_every_boundary(self):
        cases = [
            # metric name, fit/fp kwarg, [(value, expected band)]
            ("footprint_iou", "iou", [(0.75, A), (0.7499, R), (0.50, R), (0.4999, X), (0.99, A), (0.0, X)]),
            ("hausdorff_m", "hausdorff_m", [(0.0, A), (2.0, A), (2.0001, R), (5.0, R), (5.0001, X)]),
            ("area_ratio", "area_ratio", [(0.85, A), (1.15, A), (0.8499, R), (1.1501, R), (0.70, R), (1.30, R), (0.6999, X), (1.3001, X), (0.0, X)]),
            ("rotation_margin_footprint", "rotation_margin_footprint", [(0.05, A), (0.0499, R), (0.0, R)]),  # never rejects
            ("anisotropy_log_ratio", "anisotropy_log_ratio", [(0.05, A), (0.0501, R), (0.1499, R), (0.15, X), (-0.05, A), (-0.3, X)]),
            ("max_neighbor_overlap", "max_neighbor_overlap", [(0.02, A), (0.0201, R), (0.10, R), (0.1001, X)]),
        ]
        for name, kwarg, values in cases:
            for value, expected in values:
                with self.subTest(metric=name, value=value):
                    self.assertEqual(self.band(name, **{kwarg: value}), expected)

    def test_rectilinearity_bands(self):
        for value, expected in ((0.75, A), (0.7499, R), (0.0, R), (1.0, A)):
            band = {c.name: c.band for c in gate.table_checks(fit(), NS(rectilinearity=value))}["rectilinearity"]
            self.assertEqual(band, expected, value)

    def test_all_seven_metrics_are_checked_and_carry_their_rule(self):
        checks = gate.table_checks(fit(), FP)
        self.assertEqual(len(checks), 7)
        self.assertTrue(all(c.rule and c.band == A for c in checks))


class TableDecisionTests(unittest.TestCase):
    def test_a_clean_fit_auto_accepts(self):
        r = gate.evaluate_gate(fit(), PHOTO, FP)
        self.assertEqual((r.decision, r.path, r.table_decision, r.probability), (A, "table", A, None))
        self.assertEqual((r.reasons, r.forced_review), ((), ()))

    def test_the_worst_band_wins(self):
        self.assertEqual(gate.evaluate_gate(fit(iou=0.6), PHOTO, FP).decision, R)
        self.assertEqual(gate.evaluate_gate(fit(iou=0.6, max_neighbor_overlap=0.2), PHOTO, FP).decision, X)  # one reject dominates
        self.assertEqual(gate.evaluate_gate(fit(rotation_margin_footprint=0.01), PHOTO, FP).decision, R)

    def test_iou_and_hausdorff_must_both_pass(self):
        # SPEC 9.1: a good main mass (IoU 0.82) with a wing pointing the wrong way is caught by Hausdorff.
        self.assertEqual(gate.evaluate_gate(fit(iou=0.82, hausdorff_m=3.5), PHOTO, FP).decision, R)
        self.assertEqual(gate.evaluate_gate(fit(iou=0.82, hausdorff_m=6.0), PHOTO, FP).decision, X)
        self.assertEqual(gate.evaluate_gate(fit(iou=0.6, hausdorff_m=1.0), PHOTO, FP).decision, R)

    def test_reasons_name_every_metric_that_missed_auto_accept(self):
        r = gate.evaluate_gate(fit(iou=0.6, hausdorff_m=3.0), PHOTO, FP)
        text = " | ".join(r.reasons)
        self.assertIn("footprint_iou=0.6", text)
        self.assertIn("hausdorff_m=3", text)
        self.assertEqual(len(r.reasons), 2)

    def test_photo_is_not_read_by_the_table(self):
        self.assertEqual(gate.evaluate_gate(fit(), object(), FP).decision, A)

    def test_score_confidence_keeps_the_f1_signature_and_has_no_table_probability(self):
        p, decision = gate.score_confidence(fit(), PHOTO, FP)
        self.assertTrue(math.isnan(p))
        self.assertEqual(decision, A)
        p, decision = gate.score_confidence(fit(iou=0.3), PHOTO, FP)
        self.assertEqual(decision, X)

    def test_bad_inputs_fail_loudly(self):
        for kw in (dict(iou=float("nan")), dict(hausdorff_m=float("inf")), dict(area_ratio=float("nan")), dict(area_ratio=-0.1),
                   dict(anisotropy_log_ratio=float("nan")), dict(max_neighbor_overlap=float("nan")), dict(rotation_margin_footprint=float("nan"))):
            with self.assertRaises(ValueError, msg=kw):
                gate.evaluate_gate(fit(**kw), PHOTO, FP)
        with self.assertRaises(ValueError):
            gate.evaluate_gate(fit(), PHOTO, NS(rectilinearity=float("nan")))
        with self.assertRaises(AttributeError):
            gate.evaluate_gate(NS(iou=0.9), PHOTO, FP)


class HardRuleTests(unittest.TestCase):
    def test_a_guessed_orientation_is_reviewed_regardless_of_iou(self):
        for how in ("road_normal", "arbitrary_symmetric", "unresolved"):
            r = gate.evaluate_gate(fit(iou=0.99, disambiguated_by=how), PHOTO, FP)
            self.assertEqual(r.decision, R, how)
            self.assertEqual(r.table_decision, A)  # the table alone would have accepted
            self.assertIn(how, r.forced_review[0])
        for how in ("exif_heading", "silhouette"):
            self.assertEqual(gate.evaluate_gate(fit(disambiguated_by=how), PHOTO, FP).decision, A, how)

    def test_round_buildings_are_forced_to_review(self):
        r = gate.evaluate_gate(fit(), PHOTO, NS(rectilinearity=0.55))
        self.assertEqual(r.decision, R)
        self.assertIn("round", r.forced_review[0])

    def test_hard_rules_never_upgrade_a_reject(self):
        r = gate.evaluate_gate(fit(iou=0.3, disambiguated_by="road_normal"), PHOTO, FP)
        self.assertEqual(r.decision, X)
        self.assertTrue(r.forced_review)


class StandaloneTests(unittest.TestCase):
    def test_the_table_runs_with_no_numpy_no_sklearn_no_features_and_no_model_file(self):
        code = f"""
import sys
for m in ("numpy", "sklearn", "scipy", "matplotlib", "features"):
    sys.modules[m] = None                      # importing any of these now raises ImportError
sys.path.insert(0, {str(HERE)!r})
import gate
from types import SimpleNamespace as NS
fit = NS(**{GOOD_FIT!r})
print(gate.score_confidence(fit, NS(), NS(rectilinearity=0.93)))
print(gate.evaluate_gate(NS(**{{**{GOOD_FIT!r}, "iou": 0.6}}), NS(), NS(rectilinearity=0.93)).decision)
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = clean_env(**{gate.ENV_MODEL: str(Path(tmp) / "no_such_model.json")})
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ["(nan,", "'auto_accept')", "review"])

    def test_flag_off_by_default_never_touches_the_model(self):
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_MODEL: "/definitely/not/here.json"}), clear=True):
            with mock.patch.object(gate, "load_model", side_effect=AssertionError("table path must not load a model")):
                self.assertEqual(gate.evaluate_gate(fit(), PHOTO, FP).path, "table")


def write_model(tmp, names=None, coef_iou=10.0, intercept=-5.0, p_accept=0.9, p_reject=0.2, synthetic=False, **overrides):
    import features

    names = list(names or features.FEATURE_NAMES)
    spec = {
        "feature_names": names, "scaler_mean": [0.0] * len(names), "scaler_scale": [1.0] * len(names),
        "coef_standardised": [coef_iou if n == "footprint_iou" else 0.0 for n in names], "intercept": intercept,
        "thresholds": {"p_accept": p_accept, "p_reject": p_reject}, "trained_on_synthetic": synthetic,
    }
    spec.update(overrides)
    path = Path(tmp) / "model.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


class LearnedPathTests(unittest.TestCase):
    """p = sigmoid(-5 + 10 * iou), thresholds p_accept 0.9 / p_reject 0.2 (iou 0.74 -> 0.917, 0.4 -> 0.269, 0.3 -> 0.119)."""

    BIN = dict(geocode_rooftop=True, height_source_authoritative=False)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.path = write_model(self.tmp)

    def gate(self, path=None, **kw):
        model = gate.load_model(path or self.path)
        return gate.evaluate_gate(kw.pop("fit", fit()), PHOTO, FP, use_learned=True, model=model, **{**self.BIN, **kw})

    def test_probability_and_decision_bands(self):
        for iou, decision in ((0.74, A), (0.4, R), (0.3, X)):
            r = self.gate(fit=fit(iou=iou))
            self.assertAlmostEqual(r.probability, sigmoid(-5 + 10 * iou), places=9)
            self.assertEqual((r.decision, r.path), (decision, "learned"))

    def test_table_decision_is_recorded_so_disagreements_can_be_reported(self):
        r = self.gate(fit=fit(iou=0.74))  # the model accepts what the table sends to review
        self.assertEqual((r.decision, r.table_decision), (A, R))
        self.assertIn("learned p=", r.reasons[0])

    def test_checks_are_still_computed_for_the_review_payload(self):
        self.assertEqual(len(self.gate().checks), 7)

    def test_thresholds_equal_to_p_are_inclusive(self):
        path = write_model(self.tmp, p_accept=sigmoid(-5 + 10 * 0.74), p_reject=sigmoid(-5 + 10 * 0.3))
        self.assertEqual(self.gate(path, fit=fit(iou=0.74)).decision, A)
        self.assertEqual(self.gate(path, fit=fit(iou=0.3)).decision, X)

    def test_a_threshold_training_could_not_set_is_never_applied(self):
        no_accept = write_model(self.tmp, p_accept=None)
        self.assertEqual(self.gate(no_accept, fit=fit(iou=0.99)).decision, R)  # cannot auto-accept
        self.assertIn("cannot reach", self.gate(no_accept).reasons[0])
        no_reject = write_model(self.tmp, p_reject=None)
        self.assertEqual(self.gate(no_reject, fit=fit(iou=0.3)).decision, R)  # cannot reject

    def test_hard_rules_still_downgrade_a_model_accept(self):
        r = self.gate(fit=fit(iou=0.99, disambiguated_by="road_normal"))
        self.assertEqual((r.decision, r.table_decision), (R, A))
        r = gate.evaluate_gate(fit(iou=0.99), PHOTO, NS(rectilinearity=0.5), use_learned=True, model=gate.load_model(self.path), **self.BIN)
        self.assertEqual(r.decision, R)  # round building: the model alone would have accepted

    def test_the_binary_features_are_required_when_the_model_uses_them(self):
        with self.assertRaisesRegex(ValueError, "geocode_rooftop"):
            gate.evaluate_gate(fit(), PHOTO, FP, use_learned=True, model=gate.load_model(self.path))
        reduced = write_model(self.tmp, names=["footprint_iou", "hausdorff_m", "rectilinearity", "rotation_margin_footprint"])
        r = gate.evaluate_gate(fit(), PHOTO, FP, use_learned=True, model=gate.load_model(reduced))  # no binaries needed
        self.assertEqual(r.path, "learned")

    def test_score_confidence_returns_the_model_probability(self):
        p, decision = gate.score_confidence(fit(iou=0.74), PHOTO, FP, use_learned=True, model=gate.load_model(self.path), **self.BIN)
        self.assertAlmostEqual(p, sigmoid(2.4))
        self.assertEqual(decision, A)

    def test_probability_does_not_overflow_at_extreme_scores(self):
        zero = {"footprint_iou": 0.0}
        for intercept, expected in ((-2000.0, 0.0), (2000.0, 1.0), (-500.0, sigmoid(-500)), (500.0, sigmoid(500))):
            m = gate.load_model(write_model(self.tmp, coef_iou=0.0, intercept=intercept))
            p = m.probability({n: zero.get(n, 0.0) for n in m.feature_names})  # math.exp(2000) would raise OverflowError
            self.assertTrue(math.isclose(p, expected, rel_tol=1e-12, abs_tol=0.0), (intercept, p, expected))  # 1 ulp is fine


class FlagTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_env_flag_values(self):
        for raw, expected in (("", False), ("0", False), ("off", False), ("false", False), ("1", True), ("true", True), ("ON", True), (" yes ", True)):
            with mock.patch.dict(os.environ, {gate.ENV_FLAG: raw}):
                self.assertEqual(gate.learned_gate_enabled(), expected, raw)
        with mock.patch.dict(os.environ, clean_env(), clear=True):
            self.assertFalse(gate.learned_gate_enabled())
        with mock.patch.dict(os.environ, {gate.ENV_FLAG: "maybe"}):
            with self.assertRaisesRegex(ValueError, "not a recognised value"):
                gate.learned_gate_enabled()

    def test_env_flag_selects_the_learned_path_and_reads_the_model_path_from_env(self):
        path = write_model(self.tmp)
        env = clean_env(**{gate.ENV_FLAG: "1", gate.ENV_MODEL: str(path)})
        with mock.patch.dict(os.environ, env, clear=True):
            r = gate.evaluate_gate(fit(), PHOTO, FP, geocode_rooftop=True, height_source_authoritative=True)
            self.assertEqual(r.path, "learned")
            self.assertIsNotNone(r.probability)

    def test_explicit_argument_beats_the_environment(self):
        path = write_model(self.tmp)
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_FLAG: "1", gate.ENV_MODEL: str(path)}), clear=True):
            self.assertEqual(gate.evaluate_gate(fit(), PHOTO, FP, use_learned=False).path, "table")
        with mock.patch.dict(os.environ, clean_env(), clear=True):
            r = gate.evaluate_gate(fit(), PHOTO, FP, use_learned=True, model=gate.load_model(path), geocode_rooftop=True, height_source_authoritative=True)
            self.assertEqual(r.path, "learned")

    def test_flag_on_with_no_model_raises_instead_of_silently_using_the_table(self):
        missing = str(Path(self.tmp) / "nope.json")
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_FLAG: "1", gate.ENV_MODEL: missing}), clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "unset PROCEDURA_LEARNED_GATE"):
                gate.evaluate_gate(fit(), PHOTO, FP, geocode_rooftop=True, height_source_authoritative=True)

    def test_a_synthetic_model_is_refused_unless_explicitly_allowed(self):
        path = write_model(self.tmp, synthetic=True)
        with self.assertRaisesRegex(ValueError, "SYNTHETIC"):
            gate.load_model(path)
        self.assertTrue(gate.load_model(path, allow_synthetic=True).trained_on_synthetic)

    def test_malformed_models_are_rejected(self):
        bad = Path(self.tmp) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            gate.load_model(bad)
        bad.write_text(json.dumps({"feature_names": ["footprint_iou"]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing"):
            gate.load_model(bad)
        with self.assertRaisesRegex(ValueError, "lengths disagree"):
            gate.load_model(write_model(self.tmp, scaler_mean=[0.0]))
        with self.assertRaisesRegex(ValueError, "scale is zero"):
            gate.load_model(write_model(self.tmp, names=["footprint_iou"], scaler_scale=[0.0]))


class TrainArtifactTests(unittest.TestCase):
    def test_gate_reproduces_the_model_that_train_py_wrote(self):
        import features
        import train
        from fixtures.fake_fits import generate

        rows = generate(n_buildings=10, n_configs=4, n_corruptions=10, seed=3)  # 50 rows -> the 4-feature model
        result = train.train(rows, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            out = train.write_artifacts(result, tmp, "fake", 0)
            with self.assertRaisesRegex(ValueError, "SYNTHETIC"):
                gate.load_model(out / "model.json")
            model = gate.load_model(out / "model.json", allow_synthetic=True)
        self.assertEqual(model.feature_names, features.REDUCED_FEATURES)
        self.assertEqual((model.p_accept, model.p_reject), (result.thresholds["p_accept"], result.thresholds["p_reject"]))
        X = features.feature_matrix(rows, result.names)
        expected = result.model.predict_proba(X)[:, 1]
        for i in (0, 7, 23, 49):
            r = rows[i]
            g = gate.evaluate_gate(r.fit, r.photo, r.footprint, use_learned=True, model=model)
            self.assertAlmostEqual(g.probability, expected[i], places=9)


if __name__ == "__main__":
    unittest.main()
