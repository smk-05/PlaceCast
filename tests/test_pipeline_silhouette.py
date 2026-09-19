"""pipeline.py's perception wiring: the --perception flag reaches segmentation and silhouette scoring, the
evidence records what ran, and a silhouette margin >= 0.08 decides orientation while a smaller one falls through.

Perception is stubbed at the score_silhouettes / segment_image seams; geo's real choose_orientation runs.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

import pipeline
from contracts import CandidateId, Disambiguator, Footprint, MeshOutline, PhotoEvidence
from geo import fit as fitmod
from geo.disambiguate import choose_orientation
from geo.footprint import compute_ombb, principal_orientation
from perception import segment as seg

UP = 4  # the +Z-up box the dry run scores; every candidate shares it
MASK = np.zeros((100, 200), dtype=bool)
MASK[20:80, 30:170] = True


def _rect(a, b):
    return np.array([[-a / 2, -b / 2], [a / 2, -b / 2], [a / 2, b / 2], [-a / 2, b / 2]])


def _scene():
    """An elongated rectangle: aspect 2 excludes k=1,3, and k=0/2 tie on footprint IoU (margin 0)."""
    pts = _rect(40.0, 20.0)
    theta, r = principal_orientation(pts)
    fp = Footprint(pts_enu=pts, rectilinearity=r, principal_angle=theta, ombb=compute_ombb(pts),
                   area_m2=Polygon(pts).area, match_quality="contained_and_named", geocode_location_type="ROOFTOP")
    mo = MeshOutline(pts_enu=pts, ombb=compute_ombb(pts), up_axis_idx=UP, front_angle=None)
    return fp, mo


def _evidence(**kw):
    return PhotoEvidence(mask=MASK, mask_area_frac=float(MASK.mean()), segmentation_model="test", **kw)


def _scores_with_margin(margin, best_k=2):
    scores = {CandidateId(UP, k): 0.5 for k in range(4)}
    scores[CandidateId(UP, best_k)] = 0.5 + margin
    return scores


def _attach(photo, perception="real", has_photo=True, **kw):
    fp, mo = _scene()
    return pipeline._attach_silhouette_evidence(photo, mo, None, {}, fp, perception, lambda *_: None,
                                                has_photo=has_photo, **kw)


def _choose(photo):
    fp, mo = _scene()
    scored = fitmod.score_candidates(fp, mo, fitmod.ombb_candidates(fp, mo))
    return choose_orientation(fp, mo, scored, photo=photo, building_lat=37.0, building_lon=-80.0)


# ------------------------------------------------------------- silhouette fusion


@pytest.mark.parametrize("margin", [0.10, 0.30])
def test_a_silhouette_margin_at_or_above_the_gate_decides_orientation(margin):
    with mock.patch("perception.render_compare.score_silhouettes", return_value=_scores_with_margin(margin)):
        photo = _attach(_evidence())
    assert photo.silhouette_margin == pytest.approx(margin)
    assert photo.has_silhouette_evidence
    chosen, by, _, _ = _choose(photo)
    assert by is Disambiguator.SILHOUETTE
    assert chosen == CandidateId(UP, 2)  # the silhouette's pick, not the IoU tie-break's k=0


@pytest.mark.parametrize("margin", [0.0, 0.05, 0.079])
def test_a_silhouette_margin_below_the_gate_falls_through(margin):
    with mock.patch("perception.render_compare.score_silhouettes", return_value=_scores_with_margin(margin)):
        photo = _attach(_evidence())
    assert photo.silhouette_scores and not photo.has_silhouette_evidence
    chosen, by, _, _ = _choose(photo)
    assert by is not Disambiguator.SILHOUETTE
    assert chosen != CandidateId(UP, 2)


def test_the_explicit_perception_flag_reaches_score_silhouettes_over_the_environment(monkeypatch):
    monkeypatch.setenv("PROCEDURA_PERCEPTION", "stub")
    with mock.patch("perception.render_compare.score_silhouettes", return_value=_scores_with_margin(0.1)) as score:
        _attach(_evidence(), perception="real")
    assert score.call_args.kwargs["backend"] == "real"


def test_no_photo_means_no_scoring_even_though_the_no_evidence_fixture_has_a_mask():
    photo = _evidence()
    with mock.patch("perception.render_compare.score_silhouettes", side_effect=AssertionError("scored")) as score:
        assert _attach(photo, has_photo=False) is photo
    score.assert_not_called()


def test_the_stub_backend_abstains_and_builds_no_mesh():
    with mock.patch("trimesh.creation.box", side_effect=AssertionError("no mesh on the stub path")):
        photo = _attach(_evidence(), perception="stub")
    assert set(photo.silhouette_scores.values()) == {0.5}
    assert photo.silhouette_margin == 0.0 and not photo.has_silhouette_evidence


def test_the_real_scorer_on_the_dry_run_box_is_flat_across_azimuth_so_it_cannot_yet_decide():
    """By design (perception/render_compare.py): one up-axis -> margin 0. Pinned so nobody expects SILHOUETTE here."""
    photo = _attach(_evidence(), perception="real")
    assert len(photo.silhouette_scores) == 4 and len(set(photo.silhouette_scores.values())) == 1
    assert photo.silhouette_margin == 0.0 and not photo.has_silhouette_evidence
    assert _choose(photo)[1] is not Disambiguator.SILHOUETTE


# ----------------------------------------------------- segmentation evidence


def _photo(tmp):
    path = Path(tmp) / "p.jpg"
    Image.new("RGB", (200, 100), (90, 120, 150)).save(path)
    return path


def test_segmentation_model_is_the_backend_that_ran_not_the_flag_text(monkeypatch):
    monkeypatch.setenv("PROCEDURA_PERCEPTION", "stub")  # the environment must not win over the flag
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        meta = {"discarded_area_frac": 0.0, "mask_frame_fraction": float(MASK.mean())}
        with mock.patch.object(seg, "CACHE_DIR", Path(tmp) / "cache"), mock.patch.object(seg, "_device", lambda: "cpu"), \
                mock.patch.object(seg, "segment_image", return_value=(MASK, meta)) as models:
            real = pipeline._build_photo_evidence([_photo(tmp)], "real", run_dir, lambda *_: None)
            models.assert_called_once()
        assert real.segmentation_model.startswith("real:")
        stub = pipeline._build_photo_evidence([_photo(tmp)], "stub", run_dir, lambda *_: None)
        assert stub.segmentation_model == "stub:central-box"
