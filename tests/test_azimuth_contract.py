"""Joint contract test: perception's azimuth / silhouette indices against geo's placement candidates.

Perception's camera azimuth (render_compare) and geometry's azimuth_k (geo.fit.ombb_candidates,
geo.disambiguate.facade_heading) are two independent conventions for "which side". This pins how
they relate, on an ASYMMETRIC mesh (an L-shaped silhouette, so no side is a mirror of another),
using geo's real functions.

Findings encoded here:
  1. Frame:      camera azimuth phi (CCW from the mesh front, seen from above) is the canonical-frame
                 angle front_angle + phi. Same sense as geo's theta, so no sign flip.
  2. Composition: side j of the mesh, under placement k, faces bearing facade_heading(k) - 90*j.
  3. Placement invariance: k rotates the PLACED mesh about the vertical, so a photo of the mesh front
                 has the same silhouette for every k. The honest score_silhouettes is flat across k.
  4. Hazard:     scoring by mesh side and passing those scores to geo as if they were placement
                 candidates makes silhouette "decide" k=0 for any photo (whatever the world
                 orientation) and raises a false exif_silhouette_disagree whenever EXIF's k != 0.
  5. Yawed mesh: if the generator's front is not the photographed side, photographed_side() finds the
                 side, and front_angle + phi corrects facade_heading so EXIF picks the right k.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from contracts import CandidateId, Footprint, PhotoEvidence
from geo import outline
from geo.coords import theta_to_heading
from geo.disambiguate import _candidate_theta, choose_orientation, facade_heading
from geo.fit import ombb_candidates, score_candidates
from geo.footprint import compute_ombb, principal_orientation
from perception import render_compare as rc

GLTF_UP = 2  # +Y
LAT0, LON0 = 37.0, -80.0
PSI = math.radians(25.0)  # footprint rotation in the world, nothing axis-aligned


# --------------------------------------------------------------------------- scene


def asymmetric_mesh():
    """36 x 34 base (x by z) with a 15 x 30 x 15 tower over its -X end: an L from the front."""
    base = trimesh.creation.box(extents=(36, 15, 34)).apply_translation((0, 7.5, 0))
    tower = trimesh.creation.box(extents=(15, 30, 15)).apply_translation((-10.5, 30, 0))
    return trimesh.util.concatenate([base, tower])


def yawed(mesh, quarter_turns):
    """The generator emitted the mesh rotated about +Y by 90 deg * quarter_turns (CCW seen from above)."""
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(quarter_turns * math.pi / 2, [0, 1, 0]))
    return out


def footprint():
    """36 x 34 rectangle rotated by PSI: aspect 1.06 < 1.1, so all four candidates are viable and the
    footprint-IoU margin is structurally ~0 (this is the case that needs another cue)."""
    a, b = 36.0, 34.0
    rect = np.array([[-a / 2, -b / 2], [a / 2, -b / 2], [a / 2, b / 2], [-a / 2, b / 2]])
    rot = np.array([[math.cos(PSI), -math.sin(PSI)], [math.sin(PSI), math.cos(PSI)]])
    pts = rect @ rot.T
    theta, r = principal_orientation(pts)
    return Footprint(pts_enu=pts, rectilinearity=r, principal_angle=theta, ombb=compute_ombb(pts),
                     area_m2=Polygon(pts).area, match_quality="contained_and_named", geocode_location_type="ROOFTOP")


def mesh_outline(mesh, **kw):
    return outline.build_mesh_outline(np.asarray(mesh.vertices), up_axis_idx=GLTF_UP, footprint_aspect=1.06, **kw)


def camera_gps(bearing_from_building_deg, metres=40.0):
    b = math.radians(bearing_from_building_deg)
    return (LAT0 + metres * math.cos(b) / 111_320.0, LON0 + metres * math.sin(b) / (111_320.0 * math.cos(math.radians(LAT0))))


def photo(mask, bearing=None, scores=None):
    return PhotoEvidence(
        mask=mask, exif_gps=None if bearing is None else camera_gps(bearing),
        silhouette_scores=scores or {}, silhouette_margin=rc.margin_of(scores) if scores else 0.0,
    )


def side_bearing(mesh, up_axis, side, theta):
    """First principles: bearing (deg) the mesh side at camera azimuth 90*side faces when the mesh is rotated
    by `theta` (math radians CCW) in the world. Uses geo's own gltf->canonical rotation."""
    toward = rc.camera_basis(up_axis, side * math.pi / 2)[2]
    canon = outline.canonical_rotation(GLTF_UP) @ toward
    return math.degrees(theta_to_heading(math.atan2(canon[1], canon[0]) + theta)) % 360.0


def angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


@pytest.fixture(scope="module")
def scene():
    mesh, fp = asymmetric_mesh(), footprint()
    mo = mesh_outline(mesh)
    cands = ombb_candidates(fp, mo)
    return dict(mesh=mesh, fp=fp, mo=mo, cands=cands, scored=score_candidates(fp, mo, cands),
                mask=rc.render_silhouette(mesh, (0, 1, 0), 0.0, 128))


# --------------------------------------------------------------------------- 1 & 2: conventions


def test_scene_is_the_case_it_claims_to_be(scene):
    mo, scored = scene["mo"], scene["scored"]
    assert mo.up_axis_idx == GLTF_UP and mo.front_angle == pytest.approx(-math.pi / 2)
    assert scene["fp"].ombb.aspect < 1.1
    assert scored[0][1] - scored[1][1] < 0.05  # footprint IoU cannot tell the candidates apart
    sides = rc.side_scores(scene["mesh"], scene["mask"])
    assert sides[0] > 0.97 and max(sides[1:]) < 0.9  # ...but the silhouette is genuinely asymmetric


def test_up_axis_table_matches_geo():
    assert np.array_equal(np.array(rc.UP_AXIS_CANDIDATES, dtype=float), outline.UP_AXIS_CANDIDATES)


@pytest.mark.parametrize("j", range(4))
def test_camera_azimuth_is_front_angle_plus_phi_in_geos_canonical_frame(scene, j):
    toward = rc.camera_basis((0, 1, 0), j * math.pi / 2)[2]
    canon = outline.canonical_rotation(GLTF_UP) @ toward
    got = math.atan2(canon[1], canon[0])
    want = scene["mo"].front_angle + j * math.pi / 2
    assert math.atan2(math.sin(got - want), math.cos(got - want)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("j", range(4))
def test_side_j_under_placement_k_faces_facade_heading_minus_90j(scene, k, j):
    cid = CandidateId(GLTF_UP, k)
    theta = _candidate_theta(scene["fp"], scene["mo"], cid)
    expected = (facade_heading(scene["fp"], scene["mo"], cid) - 90.0 * j) % 360.0
    assert angdiff(side_bearing(scene["mesh"], (0, 1, 0), j, theta), expected) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- 3: placement invariance


def test_a_photo_of_the_front_has_the_same_silhouette_for_every_placement_k(scene):
    cands = [CandidateId(GLTF_UP, k) for k in range(4)]
    scores = rc.score_silhouettes(scene["mesh"], scene["mask"], cands, backend="real")
    assert set(scores) == set(cands)
    assert len(set(round(v, 12) for v in scores.values())) == 1
    assert rc.margin_of(scores) == 0.0
    assert scores[cands[0]] > 0.97  # the mesh does match the photo; it just says nothing about k


def test_score_silhouettes_scores_the_up_axis_and_abstains_without_a_mesh(scene):
    right, wrong = CandidateId(GLTF_UP, 0), CandidateId(4, 0)  # +Y is up; +Z is not
    s = rc.score_silhouettes(scene["mesh"], scene["mask"], [right, wrong], backend="real")
    assert s[right] > s[wrong] + 0.1
    flat = rc.score_silhouettes(None, scene["mask"], [CandidateId(4, k) for k in range(4)], backend="real")
    assert rc.margin_of(flat) == 0.0


# --------------------------------------------------------------------------- EXIF picks k_true; silhouette stays quiet


@pytest.mark.parametrize("k_true", range(4))
def test_exif_picks_the_placement_whose_front_faces_the_camera_and_flat_silhouette_stays_quiet(scene, k_true):
    fp, mo = scene["fp"], scene["mo"]
    bearing = facade_heading(fp, mo, CandidateId(GLTF_UP, k_true))  # the photographer stands where the front faces
    scores = rc.score_silhouettes(scene["mesh"], scene["mask"], [c for c, _ in scene["scored"]], backend="real")
    chosen, by, _, disagree = choose_orientation(
        fp, mo, scene["scored"], photo=photo(scene["mask"], bearing, scores), building_lat=LAT0, building_lon=LON0)
    assert (chosen.azimuth_k, by.value) == (k_true, "exif_heading")
    assert disagree is None  # no silhouette evidence -> nothing to disagree with


# --------------------------------------------------------------------------- 4: the hazard


def naive_side_scores(scene):
    """What you get by reading 'azimuth k' as the mesh side (camera azimuth 90k): peaked at k=0 for a front photo."""
    return {CandidateId(GLTF_UP, j): s for j, s in enumerate(rc.side_scores(scene["mesh"], scene["mask"]))}


def test_hazard_naive_side_scores_make_silhouette_decide_k0_whatever_the_world(scene):
    naive = naive_side_scores(scene)
    assert rc.margin_of(naive) >= 0.08  # confident enough to pass geo's has_silhouette_evidence
    for k_true in range(4):
        fp, mo = scene["fp"], scene["mo"]
        chosen, by, _, _ = choose_orientation(  # no EXIF: the silhouette branch decides
            fp, mo, scene["scored"], photo=photo(scene["mask"], None, naive), building_lat=LAT0, building_lon=LON0)
        assert (chosen.azimuth_k, by.value) == (0, "silhouette")  # k_true never entered into it


@pytest.mark.parametrize("k_true", [1, 2, 3])
def test_hazard_naive_side_scores_raise_a_false_disagreement_when_exif_is_right(scene, k_true):
    fp, mo = scene["fp"], scene["mo"]
    bearing = facade_heading(fp, mo, CandidateId(GLTF_UP, k_true))
    chosen, by, _, disagree = choose_orientation(
        fp, mo, scene["scored"], photo=photo(scene["mask"], bearing, naive_side_scores(scene)), building_lat=LAT0, building_lon=LON0)
    assert chosen.azimuth_k == k_true and by.value == "exif_heading"  # EXIF is right...
    assert disagree is True  # ...and geo reports a conflict that does not exist


# --------------------------------------------------------------------------- 5: a yawed generator mesh


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
def test_photographed_side_finds_the_yaw_with_geos_rotation_sense(quarter_turns):
    truth = asymmetric_mesh()
    mask = rc.render_silhouette(truth, (0, 1, 0), 0.0, 128)  # the photo shows the true front
    got = rc.photographed_side(yawed(truth, quarter_turns), mask)
    assert got.index == quarter_turns  # CCW yaw moves the front to camera azimuth +90 * turns
    assert got.score > 0.97 and got.margin > 0.08


@pytest.mark.parametrize("quarter_turns", [1, 2, 3])
@pytest.mark.parametrize("k_true", [0, 1, 2, 3])
def test_front_angle_plus_phi_lets_exif_choose_the_right_k_for_a_yawed_mesh(quarter_turns, k_true):
    truth = asymmetric_mesh()
    mask = rc.render_silhouette(truth, (0, 1, 0), 0.0, 128)
    mesh = yawed(truth, quarter_turns)
    fp = footprint()
    mo = mesh_outline(mesh)
    scored = score_candidates(fp, mo, ombb_candidates(fp, mo))

    side = rc.photographed_side(mesh, mask).index
    mo_fixed = dataclasses.replace(mo, front_angle=mo.front_angle + side * math.pi / 2)

    # The photographer stands where the PHOTOGRAPHED side faces under placement k_true.
    theta = _candidate_theta(fp, mo, CandidateId(GLTF_UP, k_true))
    bearing = side_bearing(mesh, (0, 1, 0), side, theta)
    ph = photo(mask, bearing)

    fixed = choose_orientation(fp, mo_fixed, scored, photo=ph, building_lat=LAT0, building_lon=LON0)[0]
    naive = choose_orientation(fp, mo, scored, photo=ph, building_lat=LAT0, building_lon=LON0)[0]
    assert fixed.azimuth_k == k_true
    assert naive.azimuth_k == (k_true + side) % 4  # trusting the nominal front picks the wrong placement
    assert naive.azimuth_k != k_true


# --------------------------------------------------------------------------- backends


def test_the_stub_backend_abstains_exactly_like_the_original_stand_in(scene):
    cands = [CandidateId(GLTF_UP, k) for k in range(4)] + [CandidateId(4, 0)]
    stub = rc.score_silhouettes(scene["mesh"], scene["mask"], cands, backend="stub")
    assert stub == {c: 0.5 for c in cands} and rc.margin_of(stub) == 0.0


def test_the_environment_selects_the_backend_and_an_unknown_one_raises(scene, monkeypatch):
    cands = [CandidateId(GLTF_UP, 0), CandidateId(4, 0)]
    monkeypatch.setenv("PROCEDURA_PERCEPTION", "stub")
    assert set(rc.score_silhouettes(scene["mesh"], scene["mask"], cands).values()) == {0.5}
    monkeypatch.setenv("PROCEDURA_PERCEPTION", "real")
    assert len(set(rc.score_silhouettes(scene["mesh"], scene["mask"], cands).values())) == 2
    monkeypatch.setenv("PROCEDURA_PERCEPTION", "gpu")
    with pytest.raises(ValueError, match="unknown perception backend"):
        rc.score_silhouettes(scene["mesh"], scene["mask"], cands)
