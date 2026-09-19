"""Tests for perception/orientation.py: every branch of the A.3 precedence ladder, on contracts.py types.

Run: pytest tests/test_orientation.py
"""
import unittest

import numpy as np

from contracts import CandidateId, Disambiguator, PhotoEvidence
from perception import orientation as ori
from perception import render_compare as rc
from test_render_compare import stepped_building, stepped_mask_seen_from_front

UP = 2  # +Y, the glTF up-axis index
CANDS = [CandidateId(UP, k) for k in range(4)]
K0, K1, K2, K3 = CANDS


def facade_heading(candidate):
    """Stub for geo.disambiguate.facade_heading with fp/mo bound: the bearing FALLS 90 degrees per k, as geo's does."""
    return (30.0 - 90.0 * candidate.azimuth_k) % 360.0  # k=0:30  k=1:300  k=2:210  k=3:120


def photo(heading=None, scores=None, margin=None):
    scores = scores or {}
    if margin is None:
        top = sorted(scores.values(), reverse=True)
        margin = top[0] - top[1] if len(top) > 1 else 0.0
    return PhotoEvidence(mask=np.zeros((4, 4), bool), exif_heading_deg=heading, silhouette_scores=scores, silhouette_margin=margin)


def sil(k0=0.9, k1=0.5, k2=0.4, k3=0.3):
    return {K0: k0, K1: k1, K2: k2, K3: k3}


def choose(p, **kw):
    return ori.choose_orientation(p, kw.pop("candidates", CANDS), kw.pop("facade_heading", facade_heading), **kw)


def k(choice):
    return choice.candidate.azimuth_k


class ExifBranchTests(unittest.TestCase):
    def test_exif_decides_and_needs_no_review(self):
        # camera points at 30 -> photographed facade faces 210 -> k=2.
        r = choose(photo(30.0))
        self.assertEqual((r.candidate, r.disambiguated_by, r.needs_review), (K2, Disambiguator.EXIF_HEADING, False))
        self.assertIn("exif heading 30.0", r.reason)

    def test_exif_beats_a_confident_silhouette_that_disagrees(self):
        p = photo(30.0, sil())  # silhouette: k=0, margin 0.4
        self.assertTrue(p.has_silhouette_evidence)
        r = choose(p)
        self.assertEqual((k(r), r.disambiguated_by, r.needs_review), (2, Disambiguator.EXIF_HEADING, False))
        self.assertIs(r.exif_silhouette_disagree, True)
        self.assertIn("DISAGREES", r.reason)

    def test_exif_beats_a_low_margin_and_a_road_normal(self):
        r = choose(photo(30.0), road_normal_deg=300.0)
        self.assertEqual((k(r), r.disambiguated_by), (2, Disambiguator.EXIF_HEADING))

    def test_heading_zero_is_present_and_wraps_correctly(self):
        r = choose(photo(0.0))  # 0.0 is falsy but valid: faces 180 -> nearest is 210 (30 off) -> k=2
        self.assertEqual((k(r), r.disambiguated_by), (2, Disambiguator.EXIF_HEADING))
        self.assertEqual(k(choose(photo(359.0))), 2)  # faces 179: 210 is 31 off, 120 is 59 off
        r = choose(photo(300.0))  # faces 120 -> k=3 exactly
        self.assertEqual(k(r), 3)
        self.assertIn("0.0 deg off", r.reason)

    def test_exif_only_chooses_among_the_candidates_it_is_given(self):
        # facing 210: k=2 (210) is the best of all four, but is not offered. k=0 is 180 off, k=1 (300) is 90 off.
        self.assertEqual(k(choose(photo(30.0), candidates=[K0, K1])), 1)
        with self.assertRaises(ValueError):
            choose(photo(30.0), candidates=[])

    def test_an_exact_tie_with_no_scores_goes_to_the_first_candidate_offered(self):
        # k=1 (300) and k=3 (120) are both 90 off 210.
        self.assertEqual(k(choose(photo(30.0), candidates=[K1, K3])), 1)
        self.assertEqual(k(choose(photo(30.0), candidates=[K3, K1])), 3)

    def test_exact_tie_between_two_facades_goes_to_the_better_silhouette(self):
        # facing 75 is 45 from both 30 (k=0) and 120 (k=3).
        scores = {K0: 0.6, K1: 0.1, K2: 0.1, K3: 0.7}
        self.assertEqual(k(choose(photo(255.0, scores, margin=0.0))), 3)
        scores = {K0: 0.7, K1: 0.1, K2: 0.1, K3: 0.6}
        self.assertEqual(k(choose(photo(255.0, scores, margin=0.0))), 0)


class DisagreementTests(unittest.TestCase):
    """exif_silhouette_disagree follows the contract: None unless BOTH cues chose, then whether they differ."""

    def test_disagreement_is_true_or_false_only_when_both_cues_chose(self):
        self.assertIs(choose(photo(30.0, sil(k0=0.4, k1=0.3, k2=0.95, k3=0.2))).exif_silhouette_disagree, False)  # both say k=2
        self.assertIs(choose(photo(30.0, sil())).exif_silhouette_disagree, True)  # EXIF k=2, silhouette k=0

    def test_absent_cues_give_none_not_false(self):
        self.assertIsNone(choose(photo(None, sil())).exif_silhouette_disagree)  # no EXIF
        self.assertIsNone(choose(photo(30.0)).exif_silhouette_disagree)  # no silhouette scores at all
        weak = photo(30.0, sil(k0=0.6, k1=0.55, k2=0.5, k3=0.2))  # margin 0.05 < 0.08: has_silhouette_evidence is False
        self.assertFalse(weak.has_silhouette_evidence)
        self.assertIsNone(choose(weak).exif_silhouette_disagree)

    def test_the_threshold_is_the_contracts_0_08_inclusive(self):
        self.assertIs(choose(photo(30.0, sil(), margin=0.08)).exif_silhouette_disagree, True)
        self.assertIsNone(choose(photo(30.0, sil(), margin=0.0799)).exif_silhouette_disagree)

    def test_only_the_exif_cue_can_disagree_and_the_margin_is_always_recorded(self):
        cases = [
            choose(photo(None, sil())),  # silhouette decides
            choose(photo(None), road_normal_deg=100.0),  # road normal
            choose(photo(None)),  # nothing
            choose(photo(30.0, sil())),  # EXIF, against a confident silhouette
        ]
        self.assertEqual([r.exif_silhouette_disagree for r in cases], [None, None, None, True])
        self.assertEqual([round(r.silhouette_margin, 3) for r in cases], [0.4, 0.0, 0.0, 0.4])


class ImplausibleExifTests(unittest.TestCase):
    def test_implausible_or_absent_headings_are_ignored_and_the_ladder_falls_through(self):
        confident = sil()
        for bad in (-5.0, 360.0, 400.0, float("nan"), float("inf"), "north", True):
            r = choose(photo(bad, confident))
            self.assertEqual((k(r), r.disambiguated_by), (0, Disambiguator.SILHOUETTE), bad)
            self.assertIn("implausible", r.reason, bad)
        r = choose(photo(None, confident))
        self.assertEqual(r.disambiguated_by, Disambiguator.SILHOUETTE)
        self.assertNotIn("implausible", r.reason)

    def test_plausible_heading_boundaries(self):
        self.assertEqual(ori.plausible_heading(0), 0.0)
        self.assertEqual(ori.plausible_heading("142.5"), 142.5)
        self.assertEqual(ori.plausible_heading(359.99), 359.99)
        self.assertIsNone(ori.plausible_heading(360))
        self.assertIsNone(ori.plausible_heading(None))


class SilhouetteBranchTests(unittest.TestCase):
    def test_confident_silhouette_decides_without_calling_facade_heading(self):
        def boom(c):
            raise AssertionError("facade_heading must not be needed when the silhouette decides")

        r = choose(photo(None, sil()), facade_heading=boom)
        self.assertEqual((k(r), r.disambiguated_by, r.needs_review), (0, Disambiguator.SILHOUETTE, False))

    def test_a_road_normal_is_ignored_when_the_silhouette_is_confident(self):
        r = choose(photo(None, sil()), road_normal_deg=300.0)
        self.assertEqual((k(r), r.disambiguated_by), (0, Disambiguator.SILHOUETTE))

    def test_the_silhouette_picks_only_among_the_candidates_given(self):
        r = choose(photo(None, sil(k0=0.95, k1=0.6, k2=0.5, k3=0.4)), candidates=[K1, K2, K3])
        self.assertEqual(k(r), 1)


class RoadNormalBranchTests(unittest.TestCase):
    def test_road_normal_picks_the_nearest_facade_and_always_flags_review(self):
        r = choose(photo(None, sil(0.99, 0.98, 0.5, 0.5), margin=0.03), road_normal_deg=100.0)  # near-perfect IoU, still a guess
        self.assertEqual((k(r), r.disambiguated_by, r.needs_review), (3, Disambiguator.ROAD_NORMAL, True))  # 120 is nearest
        self.assertIn("guessing", r.reason)

    def test_bearing_wraps_and_normalises(self):
        self.assertEqual(k(choose(photo(None), road_normal_deg=359.0)), 0)  # 359 ~ 30: 31 off
        self.assertEqual(k(choose(photo(None), road_normal_deg=-60.0)), 1)  # 300
        self.assertEqual(k(choose(photo(None), road_normal_deg=750.0)), 0)  # 30

    def test_road_normal_is_used_when_exif_is_implausible(self):
        r = choose(photo(400.0), road_normal_deg=210.0)
        self.assertEqual((k(r), r.disambiguated_by, r.needs_review), (2, Disambiguator.ROAD_NORMAL, True))
        self.assertIn("implausible", r.reason)

    def test_a_nonfinite_road_normal_from_geo_raises(self):
        with self.assertRaisesRegex(ValueError, "road_normal_deg"):
            choose(photo(None), road_normal_deg=float("nan"))


class NoEvidenceTests(unittest.TestCase):
    def test_nothing_to_go_on_is_flagged_and_uses_the_contracts_fallback_value(self):
        r = choose(photo(None))
        self.assertEqual((r.disambiguated_by, r.needs_review), (Disambiguator.ASPECT_RATIO, True))
        self.assertEqual(r.candidate, K0)  # no scores: the first candidate
        self.assertIn("nothing decided", r.reason)

    def test_the_best_silhouette_candidate_is_returned_when_scores_exist_but_are_weak(self):
        r = choose(photo(None, {K0: 0.5, K1: 0.52, K2: 0.5, K3: 0.5}))
        self.assertEqual((k(r), r.needs_review), (1, True))

    def test_a_symmetric_footprint_is_recorded_as_arbitrary_symmetric(self):
        r = choose(photo(None), symmetric=True)
        self.assertEqual((r.disambiguated_by, r.needs_review), (Disambiguator.ARBITRARY_SYMMETRIC, True))

    def test_a_road_normal_beats_the_symmetric_label(self):
        r = choose(photo(None), symmetric=True, road_normal_deg=120.0)
        self.assertEqual((k(r), r.disambiguated_by, r.needs_review), (3, Disambiguator.ROAD_NORMAL, True))


class UnknownFrontTests(unittest.TestCase):
    """contracts.FacadeHeading raises ValueError when the mesh front is unknown: that cue abstains."""

    @staticmethod
    def unknown(c):
        raise ValueError("mesh front facade is unknown (MeshOutline.front_angle is None)")

    def test_exif_abstains_and_the_ladder_continues_with_the_reason_recorded(self):
        r = choose(photo(30.0, sil()), facade_heading=self.unknown)
        self.assertEqual((k(r), r.disambiguated_by), (0, Disambiguator.SILHOUETTE))
        self.assertIn("cannot be mapped to a facade", r.reason)
        self.assertIsNone(r.exif_silhouette_disagree)  # EXIF never chose, so nothing to compare

    def test_the_road_normal_abstains_too_and_nothing_is_guessed(self):
        r = choose(photo(None), road_normal_deg=100.0, facade_heading=self.unknown)
        self.assertEqual((r.disambiguated_by, r.needs_review), (Disambiguator.ASPECT_RATIO, True))
        self.assertIn("cannot be mapped to a facade", r.reason)

    def test_any_other_failure_propagates_instead_of_being_swallowed(self):
        with self.assertRaises(RuntimeError):
            choose(photo(30.0), facade_heading=lambda c: float("nan"))
        with self.assertRaises(ZeroDivisionError):
            choose(photo(30.0), facade_heading=lambda c: 1 / 0)


class HelperTests(unittest.TestCase):
    def test_bearing_from_enu(self):
        for (e, n), expected in {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270, (1, 1): 45}.items():
            self.assertAlmostEqual(ori.bearing_from_enu(e, n), expected)

    def test_angular_difference_wraps(self):
        self.assertEqual(ori.angular_diff_deg(359, 1), 2)
        self.assertEqual(ori.angular_diff_deg(0, 180), 180)
        self.assertEqual(ori.angular_diff_deg(90, 90), 0)


class WithTheHonestSilhouetteScorerTests(unittest.TestCase):
    def test_the_real_scorer_is_flat_across_k_so_exif_decides_and_nothing_disagrees(self):
        mesh, mask = stepped_building(), stepped_mask_seen_from_front()
        scores = rc.score_silhouettes(mesh, mask, CANDS, backend="real")
        p = photo(30.0, scores, margin=rc.margin_of(scores))
        self.assertFalse(p.has_silhouette_evidence)
        r = choose(p)
        self.assertEqual((k(r), r.disambiguated_by), (2, Disambiguator.EXIF_HEADING))
        self.assertIsNone(r.exif_silhouette_disagree)

    def test_without_exif_the_flat_scorer_leaves_the_decision_to_the_road_normal(self):
        mesh, mask = stepped_building(), stepped_mask_seen_from_front()
        scores = rc.score_silhouettes(mesh, mask, CANDS, backend="real")
        r = choose(photo(None, scores, margin=rc.margin_of(scores)), road_normal_deg=100.0)
        self.assertEqual((k(r), r.disambiguated_by, r.needs_review), (3, Disambiguator.ROAD_NORMAL, True))


if __name__ == "__main__":
    unittest.main()
