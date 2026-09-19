"""
Import and contract smoke tests.

Several modules are not on the --dry-run path (terrain, generate, depth,
render_compare), so a syntax error or a bad import in them would otherwise lie
in wait until hour 20. These tests are cheap and they fail loudly.

Nothing here touches the network or a model.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

MODULES = [
    "contracts",
    "pipeline",
    "geo.coords", "geo.footprint", "geo.outline", "geo.fit",
    "geo.disambiguate", "geo.validate", "geo.height", "geo.terrain",
    "geo.overlay", "geo.exif",
    "generate.edit", "generate.lift",
    "perception.segment", "perception.render_compare",
    "confidence.gate",
    "fixtures.fake_fits", "fixtures.fake_photo_evidence",
    "server.app",
    "scripts.demo_addresses",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


# --------------------------------------------------------------------------
# The three perception signatures (contracts.py's Protocols)
# --------------------------------------------------------------------------


def test_perception_standins_satisfy_their_signatures(tmp_path):
    """The stand-ins must be drop-in compatible with the real implementations."""
    from PIL import Image

    from contracts import CandidateId
    from perception.render_compare import margin_of, score_silhouettes
    from perception.segment import assess_mask, clean, segment_building

    photo = tmp_path / "p.png"
    Image.new("RGB", (640, 480), (128, 128, 128)).save(photo)

    mask, frac, occluded = segment_building(photo)
    assert mask.shape == (480, 640) and mask.dtype == bool
    assert 0.0 < frac < 1.0 and isinstance(occluded, bool)

    cleaned = clean(mask)
    assert cleaned.shape == mask.shape

    _, _, note = assess_mask(cleaned)
    assert isinstance(note, str)

    cands = [CandidateId(4, k) for k in range(4)]
    scores = score_silhouettes(None, cleaned, cands)
    assert set(scores) == set(cands)

    # The stand-in must ABSTAIN, not vote: a flat distribution keeps
    # has_silhouette_evidence False so geo/disambiguate carries the decision.
    assert margin_of(scores) == 0.0


def test_no_evidence_fixture_abstains():
    from fixtures.fake_photo_evidence import confident, no_evidence

    assert not no_evidence().has_silhouette_evidence
    assert confident(correct_k=2).has_silhouette_evidence


def test_fake_fits_are_usable_and_imbalanced():
    """Addendum B.4: a working solver produces mostly accepts, and the model
    will learn 'always accept' unless that is handled."""
    from fixtures.fake_fits import generate

    rows = generate(n=140)
    assert len(rows) >= 130

    labels = [lab for _, _, lab in rows]
    rate = sum(labels) / len(labels)
    assert 0.5 < rate < 0.8, f"base rate {rate} — check the generator"

    fit, fp, _ = rows[0]
    assert hasattr(fit, "iou") and hasattr(fp, "rectilinearity")


# --------------------------------------------------------------------------
# The gate falls back cleanly when the learned model is absent
# --------------------------------------------------------------------------


def test_learned_gate_falls_back_to_the_table():
    """Addendum B.4's non-negotiable: the table is never unavailable."""
    from confidence.gate import score_confidence
    from contracts import Decision
    from fixtures.fake_fits import make_fit, make_footprint
    from fixtures.fake_photo_evidence import no_evidence

    fit = make_fit(iou=0.91, hausdorff=1.1, margin=0.25)
    fp = make_footprint(0.95)

    p, decision = score_confidence(fit, no_evidence(), fp, method="learned")
    assert isinstance(p, float) and isinstance(decision, Decision)
    assert decision is Decision.AUTO_ACCEPT


def test_threshold_table_rejects_a_bad_fit():
    from contracts import Decision
    from fixtures.fake_fits import make_fit, make_footprint
    from geo.validate import threshold_decision

    fit = make_fit(iou=0.22, hausdorff=11.0, area_ratio=1.9, margin=0.0)
    _, decision, reasons = threshold_decision(fit, make_footprint(0.6))

    assert decision is Decision.REJECT
    assert reasons


# --------------------------------------------------------------------------
# Spec 5.1 — the prompt guardrail
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [
    "add a collapsed tower",
    "extend the west wing",
    "partially demolished facade",
    "make the building taller",
])
def test_silhouette_mutating_prompts_are_rejected(prompt):
    from generate.edit import validate_prompt

    ok, reason = validate_prompt(prompt)
    assert not ok and "silhouette" in reason


@pytest.mark.parametrize("prompt", [
    "weathered concrete, broken windows, ivy overgrowth",
    "scorched upper floors, rusted fixtures",
])
def test_surface_level_prompts_pass(prompt):
    from generate.edit import validate_prompt

    assert validate_prompt(prompt)[0]


# --------------------------------------------------------------------------
# Spec 7 — the height priority chain
# --------------------------------------------------------------------------


def test_osm_height_beats_everything():
    from contracts import HeightSource
    from geo.height import resolve_height

    h, src, _ = resolve_height(osm_tags={"height": "20.7", "building:levels": "6"},
                               depth_estimate=31.0)
    assert h == pytest.approx(20.7)
    assert src is HeightSource.OSM_HEIGHT and src.is_authoritative


def test_levels_fallback_uses_three_metres():
    from contracts import HeightSource
    from geo.height import resolve_height

    h, src, _ = resolve_height(osm_tags={"building:levels": "6"})
    assert h == pytest.approx(18.0)
    assert src is HeightSource.OSM_LEVELS


def test_depth_disagreement_is_logged_but_osm_still_wins():
    """Addendum C.3: both can be wrong; OSM is more often right."""
    from geo.height import resolve_height

    h, _, notes = resolve_height(osm_tags={"height": "20"}, depth_estimate=40.0)
    assert h == pytest.approx(20.0)
    assert any("disagree" in n for n in notes)


def test_no_tags_falls_through_to_proportional():
    from contracts import HeightSource
    from geo.height import resolve_height

    h, src, notes = resolve_height(osm_tags={}, mesh_height_units=0.5,
                                   footprint_scale=20.0, footprint_area_m2=800.0)
    assert src is HeightSource.PROPORTIONAL_FALLBACK
    assert 2.5 <= h <= 300.0
    assert notes


def test_absurd_depth_is_rejected_by_the_sanity_band():
    from contracts import HeightSource
    from geo.height import resolve_height

    _, src, notes = resolve_height(osm_tags={}, depth_estimate=4000.0,
                                   footprint_area_m2=500.0)
    assert src is HeightSource.PROPORTIONAL_FALLBACK
    assert any("rejected" in n for n in notes)


# --------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------


def test_exif_on_a_plain_image_returns_nones_not_an_error(tmp_path):
    """Downloaded images never carry GPSImgDirection. That is normal."""
    from PIL import Image

    from geo import exif

    p = tmp_path / "plain.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(p)

    meta = exif.extract(p)
    assert meta["heading_deg"] is None
    assert meta["gps"] is None
    assert len(meta["sha256"]) == 64
    assert meta["width"] == 64


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_placement_record_serialises_to_json():
    import json
    from datetime import datetime, timezone

    from contracts import Decision, PlacementRecord
    from fixtures.fake_fits import make_fit

    rec = PlacementRecord(
        asset_id="test", created_at=datetime.now(timezone.utc).isoformat(),
        address_raw="somewhere", fit=make_fit(iou=0.9, hausdorff=1.0),
        decision=Decision.AUTO_ACCEPT,
    )
    blob = json.dumps(rec.to_json_dict(), default=str)
    back = json.loads(blob)

    assert back["decision"] == "auto_accept"
    assert back["fit"]["transform"]["frame"] == "ENU"
    assert back["height"]["authoritative"] is True
    # candidate keys must survive as the stable CandidateId string form
    assert all(k.startswith("u") for k in back["fit"]["candidate_ious"])


def test_heading_conversion_in_fitresult_matches_coords():
    from fixtures.fake_fits import make_fit
    from geo.coords import theta_to_heading

    fit = make_fit(iou=0.9, hausdorff=1.0)
    assert fit.heading_rad == pytest.approx(theta_to_heading(fit.theta))


def test_ombb_invariant_is_enforced():
    from contracts import OMBB

    with pytest.raises(ValueError, match="invariant"):
        OMBB(centre=np.zeros(2), u=np.array([1.0, 0.0]),
             v=np.array([0.0, 1.0]), a=3.0, b=9.0)
