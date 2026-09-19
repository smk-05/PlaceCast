"""Tests for confidence/gate.py: the SPEC 9.1 table (standalone, default) and the learned model (flagged).

Run: pytest tests/test_gate.py
"""
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from confidence import features, gate
from contracts import Decision, Disambiguator, HeightSource, PhotoEvidence
from fixtures.fake_fits import generate_labelled, make_fit, make_footprint

ROOT = Path(__file__).resolve().parent.parent
A, R, X = Decision.AUTO_ACCEPT, Decision.REVIEW, Decision.REJECT

PHOTO = PhotoEvidence(mask=np.zeros((4, 4), bool), silhouette_margin=0.15)
BASE_FIT = dict(iou=0.85, hausdorff=1.2, area_ratio=1.02, margin=0.2, aniso=0.01, neighbour=0.005)


def fit(**kw):
    return replace(make_fit(**BASE_FIT), **kw)


def fp(rect=0.93, **kw):
    return replace(make_footprint(rect), **kw)


FP = fp()


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in (gate.ENV_FLAG, gate.ENV_MODEL)}
    env.update(extra)
    return env


class TableBandTests(unittest.TestCase):
    """Every SPEC 9.1 edge, on both sides."""

    def band(self, name, **kw):
        return {c.name: c.band for c in gate.table_checks(fit(**kw), FP)}[name]

    def test_bands_at_every_boundary(self):
        cases = [
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
                    self.assertIs(self.band(name, **{kwarg: value}), expected)

    def test_rectilinearity_bands(self):
        for value, expected in ((0.75, A), (0.7499, R), (0.0, R), (1.0, A)):
            band = {c.name: c.band for c in gate.table_checks(fit(), fp(value))}["rectilinearity"]
            self.assertIs(band, expected, value)

    def test_all_seven_metrics_are_checked_and_carry_their_rule(self):
        checks = gate.table_checks(fit(), FP)
        self.assertEqual(len(checks), 7)
        self.assertTrue(all(c.rule and c.band is A for c in checks))


class TableDecisionTests(unittest.TestCase):
    def test_a_clean_fit_auto_accepts(self):
        r = gate.evaluate_gate(fit(), PHOTO, FP)
        self.assertEqual((r.method, r.requested_method, r.table_decision, r.probability, r.fallback_reason), ("threshold_table", "threshold_table", A, None, None))
        self.assertIs(r.decision, A)
        self.assertEqual((r.reasons, r.forced), ((), ()))

    def test_the_worst_band_wins(self):
        self.assertIs(gate.evaluate_gate(fit(iou=0.6), PHOTO, FP).decision, R)
        self.assertIs(gate.evaluate_gate(fit(iou=0.6, max_neighbor_overlap=0.2), PHOTO, FP).decision, X)  # one reject dominates
        self.assertIs(gate.evaluate_gate(fit(rotation_margin_footprint=0.01), PHOTO, FP).decision, R)

    def test_iou_and_hausdorff_must_both_pass(self):
        # SPEC 9.1: a good main mass (IoU 0.82) with a wing pointing the wrong way is caught by Hausdorff.
        self.assertIs(gate.evaluate_gate(fit(iou=0.82, hausdorff_m=3.5), PHOTO, FP).decision, R)
        self.assertIs(gate.evaluate_gate(fit(iou=0.82, hausdorff_m=6.0), PHOTO, FP).decision, X)
        self.assertIs(gate.evaluate_gate(fit(iou=0.6, hausdorff_m=1.0), PHOTO, FP).decision, R)

    def test_reasons_name_every_metric_that_missed_auto_accept(self):
        r = gate.evaluate_gate(fit(iou=0.6, hausdorff_m=3.0), PHOTO, FP)
        text = " | ".join(r.reasons)
        self.assertIn("footprint_iou=0.6", text)
        self.assertIn("hausdorff_m=3", text)
        self.assertEqual(len(r.reasons), 2)

    def test_photo_is_not_read_by_the_table(self):
        self.assertIs(gate.evaluate_gate(fit(), object(), FP).decision, A)

    def test_bad_inputs_fail_loudly(self):
        for kw in (dict(iou=float("nan")), dict(hausdorff_m=float("inf")), dict(area_ratio=float("nan")), dict(area_ratio=-0.1),
                   dict(anisotropy_log_ratio=float("nan")), dict(max_neighbor_overlap=float("nan")), dict(rotation_margin_footprint=float("nan"))):
            with self.assertRaises(ValueError, msg=kw):
                gate.evaluate_gate(fit(**kw), PHOTO, FP)
        with self.assertRaises(ValueError):
            gate.evaluate_gate(fit(), PHOTO, fp(float("nan")))


class ScoreTests(unittest.TestCase):
    def test_score_confidence_returns_a_float_and_a_decision_enum(self):
        score, decision = gate.score_confidence(fit(), PHOTO, FP)
        self.assertIsInstance(score, float)
        self.assertIs(decision, Decision.AUTO_ACCEPT)  # tests/test_smoke.py checks identity, so an enum, not a string
        self.assertIs(gate.score_confidence(fit(iou=0.3), PHOTO, FP)[1], Decision.REJECT)

    def test_the_table_score_is_finite_json_safe_and_monotone_not_a_probability(self):
        good, poor = gate.score_confidence(fit(), PHOTO, FP)[0], gate.score_confidence(fit(iou=0.55, hausdorff_m=4.0), PHOTO, FP)[0]
        self.assertTrue(0.0 <= poor < good <= 1.0)
        json.dumps({"confidence_p": good}, allow_nan=False)  # record.json is written from this number
        self.assertIsNone(gate.evaluate_gate(fit(), PHOTO, FP).probability)  # the rich result never pretends

    def test_pseudo_score_survives_the_contract_defaults(self):
        worst = replace(fit(), iou=0.0, hausdorff_m=float("inf"), area_ratio=0.0, rotation_margin_footprint=0.0)
        self.assertEqual(gate.pseudo_score(worst, fp(0.0)), 0.0)

    def test_verbose_is_what_the_pipeline_calls(self):
        score, decision, reasons = gate.score_confidence_verbose(fit(iou=0.6), PHOTO, FP, method="threshold_table")
        self.assertIsInstance(score, float)
        self.assertIs(decision, Decision.REVIEW)
        self.assertIsInstance(reasons, list)
        self.assertTrue(all(isinstance(r, str) for r in reasons) and reasons)


class HardRuleTests(unittest.TestCase):
    def _assert_guess_is_reviewed(self, how):
        r = gate.evaluate_gate(fit(iou=0.99, disambiguated_by=how), PHOTO, FP)
        self.assertIs(r.decision, R, how)
        self.assertIs(r.table_decision, A)  # the table alone would have accepted
        self.assertIn(how.value, r.forced[0][1])

    def test_road_normal_is_a_guess(self):
        self._assert_guess_is_reviewed(Disambiguator.ROAD_NORMAL)

    def test_facade_detail_is_a_guess(self):
        self._assert_guess_is_reviewed(Disambiguator.FACADE_DETAIL)

    def test_arbitrary_symmetric_is_a_guess(self):
        self._assert_guess_is_reviewed(Disambiguator.ARBITRARY_SYMMETRIC)

    def test_aspect_ratio_with_a_decisive_margin_is_geometric_evidence(self):
        r = gate.evaluate_gate(fit(iou=0.99, rotation_margin_footprint=0.20,disambiguated_by=Disambiguator.ASPECT_RATIO), PHOTO, FP)
        self.assertIs(r.decision, A)
        self.assertEqual(r.forced, ())

    def test_aspect_ratio_at_the_accept_threshold_is_evidence_and_just_below_is_a_guess(self):
        at = fit(iou=0.99, rotation_margin_footprint=gate.ROTATION_MARGIN_ACCEPT, disambiguated_by=Disambiguator.ASPECT_RATIO)
        self.assertEqual(gate.hard_rules(at, FP), ())  # 9.1 accepts a margin equal to the threshold
        below = replace(at, rotation_margin_footprint=gate.ROTATION_MARGIN_ACCEPT - 1e-9)
        self.assertEqual(len(gate.hard_rules(below, FP)), 1)

    def test_aspect_ratio_with_a_tied_margin_is_a_guess_and_the_reason_names_it(self):
        f = fit(iou=0.99, rotation_margin_footprint=0.01, disambiguated_by=Disambiguator.ASPECT_RATIO)
        r = gate.evaluate_gate(f, PHOTO, FP)
        self.assertIs(r.decision, R)
        self.assertEqual([d for d, _ in gate.hard_rules(f, FP)], [R])  # forced by the rule itself, not only the margin band
        self.assertIn(Disambiguator.ASPECT_RATIO.value, r.forced[0][1])
        self.assertIn("rotation margin", r.forced[0][1])

    def test_the_other_guesses_ignore_the_margin(self):
        for how in gate.GUESSED_ORIENTATION:
            r = gate.evaluate_gate(fit(iou=0.99, rotation_margin_footprint=0.90, disambiguated_by=how), PHOTO, FP)
            self.assertIs(r.decision, R, how)
            self.assertIn(how.value, r.forced[0][1])

    def test_only_exif_and_silhouette_orientation_can_be_accepted(self):
        for how in (Disambiguator.EXIF_HEADING, Disambiguator.SILHOUETTE):
            self.assertIs(gate.evaluate_gate(fit(disambiguated_by=how), PHOTO, FP).decision, A, how)

    def test_every_disambiguator_is_classified_as_evidence_or_guess(self):
        evidence = {Disambiguator.EXIF_HEADING, Disambiguator.SILHOUETTE}
        classified = gate.GUESSED_ORIENTATION | gate.GUESSED_BELOW_MARGIN
        self.assertEqual(classified, set(Disambiguator) - evidence)  # a new value must be decided here
        self.assertFalse(gate.GUESSED_ORIENTATION & gate.GUESSED_BELOW_MARGIN)

    def test_round_buildings_are_forced_to_review(self):
        r = gate.evaluate_gate(fit(), PHOTO, fp(0.55))
        self.assertIs(r.decision, R)
        self.assertIn("round", r.forced[0][1])

    def test_a_mirrored_fit_is_rejected_even_when_every_metric_is_perfect(self):
        r = gate.evaluate_gate(fit(scale_x=-1.0), PHOTO, FP)
        self.assertIs(r.decision, X)
        self.assertIs(r.table_decision, A)
        self.assertIn("mirrored", " ".join(r.reasons))

    def test_a_footprint_matched_on_thin_evidence_is_reviewed(self):
        for quality in ("unnamed_sole_candidate", ""):
            r = gate.evaluate_gate(fit(iou=0.95), PHOTO, fp(match_quality=quality))
            self.assertIs(r.decision, R, quality)
            self.assertIs(r.table_decision, A)
        self.assertIn("sole nearby candidate", gate.evaluate_gate(fit(), PHOTO, fp(match_quality="unnamed_sole_candidate")).reasons[0])
        self.assertIn("unrecorded", gate.evaluate_gate(fit(), PHOTO, fp(match_quality="")).reasons[0])
        for quality in ("contained_and_named", "contained", "name_match"):
            self.assertIs(gate.evaluate_gate(fit(), PHOTO, fp(match_quality=quality)).decision, A, quality)

    def test_a_multipart_footprint_is_reviewed(self):
        r = gate.evaluate_gate(fit(), PHOTO, fp(parts_enu=(np.zeros((4, 2)), np.zeros((4, 2)))))
        self.assertIs(r.decision, R)
        self.assertIn("3 disjoint outer rings", " ".join(r.reasons))

    def test_conflicting_exif_and_silhouette_force_review_but_absence_does_not(self):
        self.assertIs(gate.evaluate_gate(fit(exif_silhouette_disagree=True), PHOTO, FP).decision, R)
        for value in (None, False):
            self.assertIs(gate.evaluate_gate(fit(exif_silhouette_disagree=value), PHOTO, FP).decision, A, value)

    def test_hard_rules_never_lower_a_reject_and_stack_their_reasons(self):
        r = gate.evaluate_gate(fit(iou=0.3, disambiguated_by=Disambiguator.ROAD_NORMAL), PHOTO, FP)
        self.assertIs(r.decision, X)
        self.assertTrue(r.forced)
        many = gate.evaluate_gate(fit(disambiguated_by=Disambiguator.ROAD_NORMAL, exif_silhouette_disagree=True), PHOTO, fp(0.5, match_quality=""))
        self.assertEqual(len(many.forced), 4)


class StandaloneTests(unittest.TestCase):
    def test_the_table_runs_with_no_model_no_sklearn_no_geo_and_no_feature_code(self):
        code = f"""
import sys
for m in ("sklearn", "scipy", "shapely", "matplotlib", "joblib", "geo", "confidence.features"):
    sys.modules[m] = None                      # importing any of these now raises ImportError
sys.path.insert(0, {str(ROOT)!r})
from confidence import gate
from fixtures.fake_fits import make_fit, make_footprint
good = make_fit(iou=0.85, hausdorff=1.2)
print(gate.score_confidence(good, None, make_footprint(0.93))[1].value)
print(gate.evaluate_gate(make_fit(iou=0.6, hausdorff=1.2), None, make_footprint(0.93)).decision.value)
r = gate.evaluate_gate(good, None, make_footprint(0.93), method="learned")          # no model file
print(r.method, r.requested_method, r.decision.value)
"""
        with tempfile.TemporaryDirectory() as tmp:
            env = clean_env(**{gate.ENV_MODEL: str(Path(tmp) / "no_such_model.json")})
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp, env=env, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ["auto_accept", "review", "threshold_table", "learned", "auto_accept"])

    def test_the_default_needs_no_model_and_never_touches_one(self):
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_MODEL: "/definitely/not/here.json"}), clear=True):
            with mock.patch.object(gate, "load_model", side_effect=AssertionError("table path must not load a model")):
                self.assertEqual(gate.evaluate_gate(fit(), PHOTO, FP).method, "threshold_table")


class ParityWithGeoTests(unittest.TestCase):
    """This module duplicates geo.validate's table because it must not import geo/. Prove the copies agree."""

    @classmethod
    def setUpClass(cls):
        try:
            from geo import validate
        except ImportError as exc:  # the geometry stack (shapely) is not installed
            raise unittest.SkipTest(f"geo.validate unavailable: {exc}")
        cls.validate = validate

    def test_the_thresholds_are_the_same_numbers(self):
        t = self.validate.THRESHOLDS
        band = {c.name: gate.MetricCheck for c in ()}  # (documented below; edges checked through the gate)
        edges = [
            ("iou", t["iou"]["accept"], "footprint_iou", 0.0001), ("iou", t["iou"]["reject"], "footprint_iou", 0.0001),
            ("hausdorff_m", t["hausdorff_m"]["accept"], "hausdorff_m", 0.0001), ("hausdorff_m", t["hausdorff_m"]["reject"], "hausdorff_m", 0.0001),
            ("rotation_margin", t["rotation_margin"]["accept"], "rotation_margin_footprint", 0.0001),
            ("rectilinearity", t["rectilinearity"]["accept"], "rectilinearity", 0.0001),
            ("anisotropy", t["anisotropy"]["accept"], "anisotropy_log_ratio", 0.0001), ("anisotropy", t["anisotropy"]["reject"], "anisotropy_log_ratio", 0.0001),
            ("neighbor_overlap", t["neighbor_overlap"]["accept"], "max_neighbor_overlap", 0.0001), ("neighbor_overlap", t["neighbor_overlap"]["reject"], "max_neighbor_overlap", 0.0001),
        ]
        rules = {name: rule for name, _, rule in gate.TABLE}
        for _, value, gate_name, _ in edges:
            self.assertIn(f"{value:g}", rules[gate_name].replace("0.50", "0.5").replace("2.0", "2"), (gate_name, value))
        lo_a, hi_a = t["area_ratio"]["accept"]
        lo_r, hi_r = t["area_ratio"]["reject"]
        self.assertEqual([gate._band_area_ratio(v) for v in (lo_a, hi_a, lo_a - 1e-4, hi_a + 1e-4, lo_r, hi_r, lo_r - 1e-4, hi_r + 1e-4)], [A, A, R, R, R, R, X, X])

    def test_the_decisions_agree_across_every_combination_of_edge_values(self):
        levels = dict(
            iou=(0.4, 0.5, 0.6, 0.75, 0.9), hausdorff=(1.0, 2.0, 3.0, 5.0, 6.0), area_ratio=(0.6, 0.7, 0.8, 0.85, 1.0, 1.15, 1.2, 1.3, 1.4),
            margin=(0.0, 0.049, 0.05, 0.3), aniso=(0.0, 0.05, 0.1, 0.15, 0.3), neighbour=(0.0, 0.02, 0.05, 0.1, 0.2),
        )
        rects = (0.5, 0.74, 0.75, 0.95)
        checked = 0
        for values in itertools.product(*levels.values()):
            kw = dict(zip(levels, values))
            f = make_fit(**kw)  # EXIF-decided, healthy: no hard rule fires except through the footprint below
            for rect in rects:
                p = make_footprint(rect)
                theirs = self.validate.threshold_decision(f, p)[1]
                mine = gate.evaluate_gate(f, PHOTO, p).table_decision  # the pure table, before the hard rules
                # Their table also folds rectilinearity < 0.75 -> review; so does mine. The extra rules (weak match,
                # multipart, mirrored) are hard rules here and not in play for these strong, single-part, unmirrored fits.
                self.assertIs(mine, theirs, (kw, rect))
                self.assertIs(gate.evaluate_gate(f, PHOTO, p).decision, theirs if rect >= 0.6 else worst_of(theirs, R), (kw, rect))
                checked += 1
        self.assertGreater(checked, 20000)

    def test_the_extra_rules_agree_with_theirs_too(self):
        f = make_fit(**BASE_FIT)
        cases = {
            "mirrored": (make_fit(**BASE_FIT, mirrored=True), make_footprint(0.93)),
            "weak match": (f, make_footprint(0.93, match_quality="unnamed_sole_candidate")),
            "unrecorded match": (f, make_footprint(0.93, match_quality="")),
            "multipart": (f, replace(make_footprint(0.93), parts_enu=(np.zeros((4, 2)),))),
        }
        for label, (ff_, pp) in cases.items():
            self.assertIs(gate.evaluate_gate(ff_, PHOTO, pp).decision, self.validate.threshold_decision(ff_, pp)[1], label)


def worst_of(a, b):
    return gate.worst([a, b])


def write_model(tmp, names=None, coef=None, intercept=-5.0, p_accept=0.9, p_reject=0.2, synthetic=False, **overrides):
    names = list(names or features.FEATURE_NAMES)
    coef = coef or {"footprint_iou": 10.0}
    spec = {
        "feature_names": names, "scaler_mean": [0.0] * len(names), "scaler_scale": [1.0] * len(names),
        "coef_standardised": [coef.get(n, 0.0) for n in names], "intercept": intercept,
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

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.path = write_model(self.tmp)

    def gate(self, path=None, f=None, p=FP, **kw):
        return gate.evaluate_gate(f or fit(), PHOTO, p, method="learned", model=gate.load_model(path or self.path), **kw)

    def test_probability_and_decision_bands(self):
        for iou, decision in ((0.74, A), (0.4, R), (0.3, X)):
            r = self.gate(f=fit(iou=iou))
            self.assertAlmostEqual(r.probability, sigmoid(-5 + 10 * iou), places=9)
            self.assertAlmostEqual(r.score, r.probability)  # the learned path returns the model's own probability
            self.assertEqual((r.decision, r.method), (decision, "learned"))

    def test_table_decision_is_recorded_and_a_disagreement_reason_names_both(self):
        r = self.gate(f=fit(iou=0.74))  # the model accepts what the table sends to review
        self.assertEqual((r.decision, r.table_decision), (A, R))
        self.assertTrue(any(x.startswith("DISAGREEMENT: threshold table says review, learned gate says auto_accept") for x in r.reasons))
        self.assertIn("learned p=", r.reasons[0])
        self.assertFalse(any("DISAGREEMENT" in x for x in self.gate(f=fit(iou=0.95)).reasons))

    def test_checks_are_still_computed_for_the_review_payload(self):
        self.assertEqual(len(self.gate().checks), 7)

    def test_thresholds_equal_to_p_are_inclusive(self):
        path = write_model(self.tmp, p_accept=sigmoid(-5 + 10 * 0.74), p_reject=sigmoid(-5 + 10 * 0.3))
        self.assertIs(self.gate(path, f=fit(iou=0.74)).decision, A)
        self.assertIs(self.gate(path, f=fit(iou=0.3)).decision, X)

    def test_a_threshold_training_could_not_set_is_never_applied(self):
        no_accept = write_model(self.tmp, p_accept=None)
        self.assertIs(self.gate(no_accept, f=fit(iou=0.99)).decision, R)  # cannot auto-accept
        self.assertIn("cannot reach", self.gate(no_accept).reasons[0])
        self.assertIs(self.gate(write_model(self.tmp, p_reject=None), f=fit(iou=0.3)).decision, R)  # cannot reject

    def test_hard_rules_still_apply_over_a_model_accept(self):
        self.assertIs(self.gate(f=fit(iou=0.99, disambiguated_by=Disambiguator.ROAD_NORMAL)).decision, R)
        self.assertIs(self.gate(f=fit(iou=0.99), p=fp(0.5)).decision, R)  # round building
        self.assertIs(self.gate(f=fit(iou=0.99), p=fp(match_quality="")).decision, R)  # thin match
        mirrored = self.gate(f=fit(iou=0.99, scale_x=-1.0))
        self.assertEqual((mirrored.decision, mirrored.table_decision), (X, A))  # a reflection is rejected even if the model accepts

    def test_the_binary_features_are_read_from_the_contract_objects(self):
        path = write_model(self.tmp, coef={"geocode_rooftop": 2.0, "height_source_authoritative": 1.0}, intercept=-1.0)
        rooftop = self.gate(path, p=fp(geocode_location_type="ROOFTOP")).probability
        approx = self.gate(path, p=fp(geocode_location_type="APPROXIMATE")).probability
        self.assertAlmostEqual(rooftop, sigmoid(-1.0 + 2.0 + 1.0))
        self.assertAlmostEqual(approx, sigmoid(-1.0 + 0.0 + 1.0))
        microsoft = self.gate(path, f=fit(height_source=HeightSource.MICROSOFT), p=fp(geocode_location_type="ROOFTOP")).probability
        self.assertAlmostEqual(microsoft, sigmoid(-1.0 + 2.0))

    def test_score_confidence_returns_the_model_probability(self):
        score, decision = gate.score_confidence(fit(iou=0.74), PHOTO, FP, method="learned", model=gate.load_model(self.path))
        self.assertAlmostEqual(score, sigmoid(2.4))
        self.assertIs(decision, A)

    def test_probability_does_not_overflow_at_extreme_scores(self):
        zero = {"footprint_iou": 0.0}
        for intercept, expected in ((-2000.0, 0.0), (2000.0, 1.0), (-500.0, sigmoid(-500)), (500.0, sigmoid(500))):
            m = gate.load_model(write_model(self.tmp, coef={}, intercept=intercept))
            p = m.probability({n: zero.get(n, 0.0) for n in m.feature_names})  # math.exp(2000) would raise OverflowError
            self.assertTrue(math.isclose(p, expected, rel_tol=1e-12, abs_tol=0.0), (intercept, p, expected))


class MethodAndFlagTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_env_flag_values(self):
        for raw, expected in (("", False), ("0", False), ("off", False), ("false", False), ("1", True), ("true", True), ("ON", True), (" yes ", True)):
            with mock.patch.dict(os.environ, {gate.ENV_FLAG: raw}):
                self.assertEqual(gate.learned_gate_enabled(), expected, raw)
        with mock.patch.dict(os.environ, {gate.ENV_FLAG: "maybe"}):
            with self.assertRaisesRegex(ValueError, "not a recognised value"):
                gate.learned_gate_enabled()

    def test_method_defaults_to_the_table_and_the_env_supplies_the_default(self):
        with mock.patch.dict(os.environ, clean_env(), clear=True):
            self.assertEqual(gate.resolve_method(), "threshold_table")
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_FLAG: "1"}), clear=True):
            self.assertEqual(gate.resolve_method(), "learned")

    def test_an_explicit_method_beats_the_environment_both_ways(self):
        path = write_model(self.tmp)
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_FLAG: "1", gate.ENV_MODEL: str(path)}), clear=True):
            self.assertEqual(gate.evaluate_gate(fit(), PHOTO, FP, method="threshold_table").method, "threshold_table")
            self.assertEqual(gate.evaluate_gate(fit(), PHOTO, FP).method, "learned")  # env picks it
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_MODEL: str(path)}), clear=True):
            self.assertEqual(gate.evaluate_gate(fit(), PHOTO, FP, method="learned").method, "learned")

    def test_an_unknown_method_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown confidence method"):
            gate.evaluate_gate(fit(), PHOTO, FP, method="logistic")

    def test_the_method_strings_are_the_ones_the_contract_records(self):
        from contracts import PlacementRecord

        self.assertEqual(PlacementRecord(asset_id="x", created_at="", address_raw="").confidence_method, gate.METHOD_TABLE)


class FallbackTests(unittest.TestCase):
    """Absent model file -> the table, said out loud. Anything else wrong with a model that IS there raises."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_an_absent_model_falls_back_and_records_the_reason_and_the_path(self):
        missing = self.tmp / "nope.json"
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_MODEL: str(missing)}), clear=True):
            r = gate.evaluate_gate(fit(iou=0.6), PHOTO, FP, method="learned")
        self.assertEqual((r.method, r.requested_method), ("threshold_table", "learned"))
        self.assertIs(r.decision, R)
        self.assertIsNone(r.probability)
        self.assertIn(str(missing), r.fallback_reason)
        self.assertEqual(r.reasons[0], r.fallback_reason)  # the review queue sees it first

    def test_the_smoke_test_contract_learned_without_a_model_still_returns_a_decision(self):
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_MODEL: str(self.tmp / "nope.json")}), clear=True):
            score, decision = gate.score_confidence(fit(iou=0.91, hausdorff_m=1.1), PHOTO, FP, method="learned")
        self.assertIsInstance(score, float)
        self.assertIs(decision, Decision.AUTO_ACCEPT)

    def test_a_synthetic_model_raises_instead_of_falling_back(self):
        path = write_model(self.tmp, synthetic=True)
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_MODEL: str(path)}), clear=True):
            with self.assertRaisesRegex(ValueError, "SYNTHETIC"):
                gate.evaluate_gate(fit(), PHOTO, FP, method="learned")
        self.assertTrue(gate.load_model(path, allow_synthetic=True).trained_on_synthetic)

    def test_malformed_models_raise(self):
        bad = self.tmp / "bad.json"
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

    def test_a_model_using_an_unknown_feature_raises_instead_of_falling_back(self):
        path = write_model(self.tmp, names=["footprint_iou", "not_a_feature"], coef={})
        with mock.patch.dict(os.environ, clean_env(**{gate.ENV_MODEL: str(path)}), clear=True):
            with self.assertRaises(KeyError):
                gate.evaluate_gate(fit(), PHOTO, FP, method="learned")

    def test_only_a_missing_file_is_treated_as_absence(self):
        self.assertTrue(issubclass(gate.ModelAbsentError, FileNotFoundError))
        with self.assertRaises(gate.ModelAbsentError):
            gate.load_model(self.tmp / "nope.json")
        empty_dir = self.tmp / "dir.json"
        empty_dir.mkdir()
        with self.assertRaises(gate.ModelAbsentError):  # a directory is not a model file either
            gate.load_model(empty_dir)


class TrainArtifactTests(unittest.TestCase):
    def test_gate_reproduces_the_model_that_train_py_wrote(self):
        from confidence import train

        rows = generate_labelled(n_buildings=10, n_configs=4, n_corruptions=10, seed=3)  # 50 rows -> the 4-feature model
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
            g = gate.evaluate_gate(r.fit, r.photo, r.footprint, method="learned", model=model)
            self.assertAlmostEqual(g.probability, expected[i], places=9)


if __name__ == "__main__":
    unittest.main()
