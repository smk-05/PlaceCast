"""
facade_heading, the EXIF/road-normal facing fix, and the v1.0 contract fields.

The facing bug these tests guard against: the EXIF and road-normal filters used
to compute a candidate's facing from the mesh OMBB angle alone, never applying
the candidate's world rotation, and assumed the front was canonical +X when a
glTF mesh's front actually lands on canonical -Y. Both errors were silent — a
filter still "chose" a candidate, just the wrong one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from contracts import (
    CandidateId,
    Disambiguator,
    HeightSource,
    MeshOutline,
    PhotoEvidence,
)
from fixtures.fake_fits import make_fit, make_footprint
from geo import outline
from geo.disambiguate import choose_orientation, facade_heading
from geo.fit import ombb_candidates, score_candidates
from geo.footprint import compute_ombb, principal_orientation

GLTF_UP = 2  # +Y, the glTF convention


def _rect(a: float, b: float) -> np.ndarray:
    return np.array([[-a / 2, -b / 2], [a / 2, -b / 2],
                     [a / 2, b / 2], [-a / 2, b / 2]], dtype=float)


def _fp(pts):
    from contracts import Footprint
    from shapely.geometry import Polygon
    theta, r = principal_orientation(pts)
    return Footprint(pts_enu=pts, rectilinearity=r, principal_angle=theta,
                     ombb=compute_ombb(pts), area_m2=Polygon(pts).area,
                     match_quality="contained_and_named",
                     geocode_location_type="ROOFTOP")


def _mo(pts, front=-math.pi / 2, up=GLTF_UP):
    return MeshOutline(pts_enu=pts, ombb=compute_ombb(pts), up_axis_idx=up,
                       front_angle=front)


# --------------------------------------------------------------------------
# front_angle_canonical
# --------------------------------------------------------------------------


def test_gltf_front_lands_on_canonical_minus_y_not_plus_x():
    """+Y up, front +Z  ->  after rotating +Y to +Z, the front is -Y."""
    assert outline.front_angle_canonical(GLTF_UP) == pytest.approx(-math.pi / 2)


@pytest.mark.parametrize("idx", [4, 5])
def test_front_is_unknown_for_z_up_meshes(idx):
    """If +Z is the roof, the glTF front convention was not followed."""
    assert outline.front_angle_canonical(idx) is None


# --------------------------------------------------------------------------
# choose_up_axis — the glTF prior vs the prismatic score
# --------------------------------------------------------------------------


def _box_cloud(ext, n=6000, seed=0, base_axis=1, base_at_min=True):
    """Surface-ish point cloud of a box with a dense ground slab on base_axis."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.5, 0.5, size=(n, 3)) * np.asarray(ext)
    slab = rng.uniform(-0.5, 0.5, size=(n // 3, 3)) * np.asarray(ext)
    lo = -0.5 * ext[base_axis]
    slab[:, base_axis] = lo + rng.uniform(0, 0.05, n // 3) * ext[base_axis]
    if not base_at_min:
        slab[:, base_axis] *= -1
    return np.vstack([pts, slab])


def test_boxy_mesh_keeps_the_gltf_y_up_prior():
    """The NCB failure: a box is prismatic along every axis, so the prismatic
    score is a near-tie and must not override glTF's +Y."""
    v = _box_cloud([0.74, 0.36, 1.0])   # the real NCB mesh's extents
    idx, reason = outline.choose_up_axis(v)
    assert idx == GLTF_UP, reason


def test_upside_down_mesh_gets_minus_y():
    """The prismatic score is sign-blind; vertex mass sets the sign."""
    v = _box_cloud([0.74, 0.36, 1.0], base_at_min=False)
    idx, _ = outline.choose_up_axis(v)
    assert idx == 3   # -Y


def test_clear_prismatic_evidence_overrides_the_prior():
    """A triangular prism extruded along Z: cross-sections vary strongly along Y
    and not at all along Z, so Z should win by a wide margin."""
    rng = np.random.default_rng(5)
    n = 8000
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(0, 1, n)
    keep = np.abs(x) <= (1 - y)          # triangle in the XY plane
    z = rng.uniform(0, 3, n)
    z[: n // 4] = rng.uniform(0, 0.1, n // 4)   # dense base at min Z
    v = np.column_stack([x, y, z])[keep]

    idx, reason = outline.choose_up_axis(v)
    assert idx in (4, 5), reason
    assert "override" in reason


# --------------------------------------------------------------------------
# facade_heading
# --------------------------------------------------------------------------


def test_facade_heading_steps_by_ninety_degrees():
    pts = _rect(40, 20)
    fp, mo = _fp(pts), _mo(pts)
    h = [facade_heading(fp, mo, CandidateId(GLTF_UP, k)) for k in range(4)]

    for k in range(4):
        assert 0.0 <= h[k] < 360.0
        # theta grows CCW by 90 per k, so compass heading falls by 90 per k
        assert (h[k] - h[(k + 1) % 4]) % 360 == pytest.approx(90.0)


def test_facade_heading_known_value():
    """Identical footprint and outline -> theta_0 = 0, front -Y = South."""
    pts = _rect(40, 20)
    assert facade_heading(_fp(pts), _mo(pts), CandidateId(GLTF_UP, 0)) \
        == pytest.approx(180.0)


def test_facade_heading_refuses_to_guess_without_a_front():
    pts = _rect(40, 20)
    with pytest.raises(ValueError, match="unknown"):
        facade_heading(_fp(pts), _mo(pts, front=None), CandidateId(GLTF_UP, 0))


def test_facade_heading_rejects_mismatched_up_axis():
    pts = _rect(40, 20)
    with pytest.raises(ValueError, match="up_axis"):
        facade_heading(_fp(pts), _mo(pts), CandidateId(4, 0))


def test_facade_heading_honours_a_refined_theta():
    pts = _rect(40, 20)
    fp, mo = _fp(pts), _mo(pts)
    base = facade_heading(fp, mo, CandidateId(GLTF_UP, 0))
    nudged = facade_heading(fp, mo, CandidateId(GLTF_UP, 0), theta=math.radians(3))
    assert (base - nudged) % 360 == pytest.approx(3.0)


# --------------------------------------------------------------------------
# The EXIF filter picks the facade that faces the camera
# --------------------------------------------------------------------------

BLDG_LAT, BLDG_LON = 37.0, -80.0


def _photo(cam_lat, cam_lon, sil=None):
    return PhotoEvidence(
        mask=np.zeros((4, 4), dtype=bool),
        exif_gps=(cam_lat, cam_lon),
        silhouette_scores=sil or {},
        silhouette_margin=0.25 if sil else 0.0,
    )


@pytest.mark.parametrize("cam_dlat,expected_heading", [
    (-0.0005, 180.0),   # camera due South -> photographed facade faces South
    (+0.0005, 0.0),     # camera due North -> facade faces North
])
def test_exif_selects_the_facade_facing_the_camera(cam_dlat, expected_heading):
    """This is the test the old facing code failed: it ignored the candidate's
    world rotation and assumed a +X front, so it could only ever land on an
    East/West-facing candidate here."""
    pts = _rect(40, 20)
    fp, mo = _fp(pts), _mo(pts)
    scored = score_candidates(fp, mo, ombb_candidates(fp, mo))

    choice = choose_orientation(fp, mo, scored,
                                photo=_photo(BLDG_LAT + cam_dlat, BLDG_LON),
                                building_lat=BLDG_LAT, building_lon=BLDG_LON)

    assert choice.disambiguated_by is Disambiguator.EXIF_HEADING
    heading = facade_heading(fp, mo, choice.chosen)
    assert min(abs(heading - expected_heading),
               360 - abs(heading - expected_heading)) < 1.0


def test_exif_can_resolve_a_symmetric_footprint():
    """A square is ambiguous to IoU, not to a camera bearing."""
    pts = _rect(20, 20)
    fp, mo = _fp(pts), _mo(pts)
    scored = score_candidates(fp, mo, ombb_candidates(fp, mo))

    choice = choose_orientation(fp, mo, scored,
                                photo=_photo(BLDG_LAT - 0.0005, BLDG_LON),
                                building_lat=BLDG_LAT, building_lon=BLDG_LON)

    assert choice.disambiguated_by is Disambiguator.EXIF_HEADING
    assert facade_heading(fp, mo, choice.chosen) == pytest.approx(180.0, abs=1.0)


def test_exif_abstains_when_the_front_is_unknown():
    pts = _rect(40, 20)
    fp, mo = _fp(pts), _mo(pts, front=None)
    scored = score_candidates(fp, mo, ombb_candidates(fp, mo))

    choice = choose_orientation(fp, mo, scored,
                                photo=_photo(BLDG_LAT - 0.0005, BLDG_LON),
                                building_lat=BLDG_LAT, building_lon=BLDG_LON)

    assert choice.disambiguated_by is not Disambiguator.EXIF_HEADING
    assert any("front is unknown" in r for r in choice.reasons)


# --------------------------------------------------------------------------
# exif_silhouette_disagree
# --------------------------------------------------------------------------


def _choose_with(sil_k):
    pts = _rect(40, 20)
    fp, mo = _fp(pts), _mo(pts)
    scored = score_candidates(fp, mo, ombb_candidates(fp, mo))
    sil = None
    if sil_k is not None:
        sil = {CandidateId(GLTF_UP, k): 0.5 for k in range(4)}
        sil[CandidateId(GLTF_UP, sil_k)] = 0.75
    return choose_orientation(fp, mo, scored,
                              photo=_photo(BLDG_LAT - 0.0005, BLDG_LON, sil),
                              building_lat=BLDG_LAT, building_lon=BLDG_LON)


def test_disagreement_flagged_when_cues_conflict():
    exif_k = _choose_with(None).chosen.azimuth_k
    other_k = (exif_k + 2) % 4
    choice = _choose_with(other_k)

    assert choice.exif_silhouette_disagree is True
    assert choice.disambiguated_by is Disambiguator.EXIF_HEADING  # EXIF still wins
    assert any("conflicts" in r for r in choice.reasons)


def test_agreement_is_false_not_none():
    exif_k = _choose_with(None).chosen.azimuth_k
    assert _choose_with(exif_k).exif_silhouette_disagree is False


def test_disagreement_is_none_when_a_cue_is_missing():
    """'Nothing to compare' must stay distinguishable from 'they agree'."""
    assert _choose_with(None).exif_silhouette_disagree is None


# --------------------------------------------------------------------------
# v1.0 contract fields Owen asked for
# --------------------------------------------------------------------------


def test_geocode_rooftop_lives_on_footprint():
    assert make_footprint(0.9).geocode_rooftop is True
    assert make_footprint(0.9, geocode_location_type="APPROXIMATE").geocode_rooftop is False


def test_height_source_authoritative_on_fitresult():
    assert make_fit(iou=0.9, hausdorff=1.0).height_source_authoritative is True
    assert make_fit(iou=0.9, hausdorff=1.0,
                    authoritative_height=False).height_source_authoritative is False


def test_microsoft_is_not_authoritative():
    """Measured ~50% low against explicit OSM heights on this campus."""
    assert not HeightSource.MICROSOFT.is_authoritative
    assert HeightSource.OSM_HEIGHT.is_authoritative
    assert HeightSource.OSM_LEVELS.is_authoritative


def test_contract_is_versioned():
    import contracts
    assert contracts.CONTRACT_VERSION == "1.0"


# --------------------------------------------------------------------------
# The re-ordered height chain
# --------------------------------------------------------------------------


def test_depth_outranks_microsoft():
    from geo.height import resolve_height
    h, src, _ = resolve_height(osm_tags={}, depth_estimate=21.0,
                               microsoft_height=11.0, footprint_area_m2=3000.0)
    assert src is HeightSource.MONOCULAR_DEPTH and h == pytest.approx(21.0)


def test_microsoft_used_when_nothing_better_and_flagged():
    from geo.height import resolve_height
    h, src, notes = resolve_height(osm_tags={}, microsoft_height=11.0,
                                   footprint_area_m2=3000.0)
    assert src is HeightSource.MICROSOFT and h == pytest.approx(11.0)
    assert any("NOT authoritative" in n for n in notes)


def test_osm_still_beats_both():
    from geo.height import resolve_height
    h, src, notes = resolve_height(osm_tags={"height": "20.7"},
                                   depth_estimate=19.0, microsoft_height=11.3)
    assert src is HeightSource.OSM_HEIGHT and h == pytest.approx(20.7)
    assert any("Microsoft" in n and "disagrees" in n for n in notes)


# --------------------------------------------------------------------------
# Facade detail must corroborate the street prior, not just relabel it
# --------------------------------------------------------------------------


def _walls_with_detail(detail_angle, n=4000, seed=1):
    """Canonical-frame points on the walls of a 40 x 20 x 10 box, plus a dense
    cluster of 'windows and trim' on the side at `detail_angle`."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)
    side = rng.integers(0, 4, n)
    x = np.where(side == 0, 20, np.where(side == 2, -20, -20 + 40 * t))
    y = np.where(side == 1, 10, np.where(side == 3, -10, -10 + 20 * t))
    walls = np.c_[x, y, rng.uniform(0, 10, n)]
    d = np.array([math.cos(detail_angle), math.sin(detail_angle)])
    pos = d * np.array([20, 10])
    along = np.array([-d[1], d[0]]) * np.array([20, 10])
    k = n
    s = rng.uniform(-0.9, 0.9, k)
    cluster = np.c_[pos[0] + along[0] * s, pos[1] + along[1] * s, rng.uniform(0, 10, k)]
    return np.vstack([walls, cluster])


def _street_choice(detail_angle):
    pts = _rect(40, 20)
    fp, mo = _fp(pts), _mo(pts, front=-math.pi / 2)          # front faces canonical -Y
    scored = score_candidates(fp, mo, ombb_candidates(fp, mo))
    road = [np.array([[-100.0, -45.0], [100.0, -45.0]])]      # a street to the south
    return choose_orientation(fp, mo, scored, roads_enu=road,
                              vertices_canonical=_walls_with_detail(detail_angle))


def test_detail_on_the_front_corroborates():
    choice = _street_choice(-math.pi / 2)                     # detail on the front
    assert choice.disambiguated_by is Disambiguator.FACADE_DETAIL


def test_detail_on_the_back_does_not_upgrade_the_label():
    choice = _street_choice(+math.pi / 2)                     # detail on the BACK
    assert choice.disambiguated_by is Disambiguator.ROAD_NORMAL
    assert any("does not corroborate" in r for r in choice.reasons)


def test_no_front_means_no_corroboration():
    pts = _rect(40, 20)
    fp, mo = _fp(pts), _mo(pts, front=None)
    scored = score_candidates(fp, mo, ombb_candidates(fp, mo))
    choice = choose_orientation(fp, mo, scored,
                                roads_enu=[np.array([[-100.0, -45.0], [100.0, -45.0]])],
                                vertices_canonical=_walls_with_detail(-math.pi / 2))
    assert choice.disambiguated_by is not Disambiguator.FACADE_DETAIL
