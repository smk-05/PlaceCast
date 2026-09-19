"""
The mesh -> ENU matrix must reproduce the solver's placement exactly.

This is the guarantee that the viewer shows what overlay.png shows. The solver's
FitResult is defined on canonical coordinates; the matrix must fold in the
canonicalisation offsets the viewer cannot otherwise know.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import MultiPoint, Point, Polygon

from contracts import FitResult, Footprint
from geo import outline, placement
from geo.fit import apply_similarity, solve
from geo.footprint import compute_ombb, principal_orientation


def _apply(m, v):
    return (m @ np.c_[v, np.ones(len(v))].T).T[:, :3]


def _l_building_gltf(seed=0, n=6000):
    """An L-shaped building as a glTF mesh would store it: +Y up, off-centre,
    base not at zero — everything canonicalisation has to undo."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        x, py = rng.uniform(0, 30), rng.uniform(0, 26)
        if x < 14 or py < 12:                      # the L in plan view
            # glTF z = -plan_y: canonicalisation maps (x, y, z) -> (x, -z, y),
            # so storing +plan_y in z would produce the MIRRORED L, which no
            # rotation can fit (and the solver rightly refuses reflections).
            pts.append([x, rng.uniform(0, 18), -py])
    v = np.array(pts)
    return v * 0.03 + np.array([2.0, -0.7, 5.0])   # model units, arbitrary offset


def _fit(**kw):
    base = dict(theta=0.7, scale_x=41.0, scale_y=38.0, scale_z=12.5,
                tx=5.0, ty=-3.0, z_offset=0.4)
    base.update(kw)
    return FitResult(**base)


def test_matrix_equals_canonicalise_then_similarity():
    """Exact: the matrix is the same arithmetic as the solver, to 1e-9."""
    v = _l_building_gltf()
    rot, o, _ = outline.canonical_offsets(v, 2)
    canon, _ = outline.canonicalise(v, 2)
    fit = _fit()

    m = placement.mesh_to_enu(fit, placement.mesh_to_canonical(rot, o))
    got = _apply(m, v)

    want_xy = apply_similarity(canon[:, :2], fit.theta, fit.scale_x, fit.scale_y,
                               np.array([fit.tx, fit.ty]))
    want_z = fit.scale_z * canon[:, 2] + fit.z_offset
    assert np.allclose(got[:, :2], want_xy, atol=1e-9)
    assert np.allclose(got[:, 2], want_z, atol=1e-9)


def test_solved_mesh_lands_on_its_footprint():
    """End to end: canonicalise, solve against a known footprint, build the
    matrix, and the raw glTF vertices must land on that footprint."""
    v = _l_building_gltf()
    mo = outline.build_mesh_outline(v, up_axis_idx=2)

    # The real footprint: the mesh plan in metres, rotated and moved.
    plan = np.array([[0, 0], [30, 0], [30, 12], [14, 12], [14, 26], [0, 26]], float)
    plan = apply_similarity(plan - plan.mean(axis=0), 1.1, 1.0, 1.0, np.array([7.0, -4.0]))
    theta, r = principal_orientation(plan)
    fp = Footprint(pts_enu=plan, rectilinearity=r, principal_angle=theta,
                   ombb=compute_ombb(plan), area_m2=Polygon(plan).area,
                   match_quality="contained_and_named")

    fit = solve(fp, mo)
    assert fit.iou > 0.8

    rot, o, _ = outline.canonical_offsets(v, mo.up_axis_idx)
    world = _apply(placement.mesh_to_enu(fit, placement.mesh_to_canonical(rot, o)), v)

    footprint = Polygon(plan).buffer(1.5)
    inside = np.mean([footprint.contains(Point(p)) for p in world[:, :2]])
    assert inside > 0.95, f"only {inside:.0%} of the mesh lands on the footprint"
    assert abs(np.percentile(world[:, 2], 1)) < 0.5, "base is not on the ground"


def test_unit_box_matches_the_dry_run_outline():
    """The dry-run box path goes through the same matrix."""
    pts = np.array([[0, 0], [40, 0], [40, 18], [0, 18]], float)
    pts = apply_similarity(pts - pts.mean(axis=0), 0.4, 1, 1, np.array([3.0, 2.0]))
    ombb = compute_ombb(pts)
    fp = Footprint(pts_enu=pts, ombb=ombb, area_m2=Polygon(pts).area)

    mo = outline.outline_from_footprint(fp)
    fit = solve(fp, mo)
    fit = FitResult(**{**fit.__dict__, "scale_z": 14.0})   # a 14 m building

    m = placement.mesh_to_enu(fit, placement.unit_box_to_canonical(ombb))
    box = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    world = _apply(m, box)

    placed = MultiPoint(world[:, :2]).convex_hull
    iou = placed.intersection(Polygon(pts)).area / placed.union(Polygon(pts)).area
    assert iou > 0.99
    assert world[:, 2].min() == pytest.approx(0.0, abs=1e-9)
    assert world[:, 2].max() == pytest.approx(14.0, abs=1e-9)


def test_record_entry_column_major_is_the_transpose():
    m = np.arange(16, dtype=float).reshape(4, 4)
    e = placement.record_entry(m, "gltf_scene")
    assert e["rows"][0] == [0, 1, 2, 3]
    assert e["column_major"][:4] == [0, 4, 8, 12]   # first COLUMN


def test_record_facade_heading_uses_the_refined_rotation():
    """Owen's openings inherit this bearing: fitted theta + front normal."""
    import math

    import numpy as np

    from geo.coords import theta_to_heading
    from geo.fit import ombb_candidates, solve
    from pipeline import _facade_entry
    from tests.test_fit_synthetic import RECT, _fp, _mo

    fp = _fp(RECT)
    mo = _mo(RECT / 30.0, front_angle=-math.pi / 2)
    chosen = ombb_candidates(fp, mo)[0][0]
    fit = solve(fp, mo, chosen=chosen)
    entry = _facade_entry(fp, mo, chosen, fit, "photographed_side", lambda *_: None)
    expected = math.degrees(theta_to_heading(fit.theta - math.pi / 2))
    assert entry["front_heading_deg"] == np.float64(expected) or \
        abs(entry["front_heading_deg"] - expected) < 1e-9
    assert entry["front_source"] == "photographed_side"

    no_front = _mo(RECT / 30.0)
    assert _facade_entry(fp, no_front, chosen, fit, "gltf_prior",
                         lambda *_: None)["front_heading_deg"] is None
