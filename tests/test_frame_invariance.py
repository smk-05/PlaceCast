"""The placement must not depend on the frame a model arrives in.

"Hand us a model in any units, at any rotation, anywhere near the origin, and
we put it on the right building at the right size" is the claim the demo rests
on. scripts/scramble_demo.py checks it end to end on a real generated mesh
(measured: placed polygons agree to IoU 0.998). These are the fast unit-level
versions, so a regression shows up in the suite rather than on stage.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from contracts import Disambiguator, MeshOutline
from geo import outline
from geo.fit import apply_similarity, ombb_candidates, solve
from geo.footprint import compute_ombb
from tests.test_fit_synthetic import LSHAPE, _fp


def _prism(plan: np.ndarray, height: float = 0.4, n: int = 900) -> np.ndarray:
    """A plan extruded into a vertex cloud, Y-up as glTF writes it."""
    from shapely.geometry import Point
    rng = np.random.default_rng(0)
    poly = Polygon(plan)
    x0, y0, x1, y1 = poly.bounds
    pts = []
    while len(pts) < n:                      # rejection-sample the solid plan
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if poly.contains(Point(x, y)):
            pts.append((x, y))
    xy = np.array(pts)
    z = rng.uniform(0, height, len(xy))
    return np.stack([xy[:, 0], z, xy[:, 1]], axis=1)      # Y is up


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_outline_extraction_is_rotation_invariant():
    """The occupancy grid is axis-aligned, so it is aligned to the PLAN's own
    axes before rasterising — otherwise the same mesh rotated by 37 degrees
    discretises differently and yields a different outline."""
    verts = _prism(LSHAPE / 30.0)
    base = outline.build_mesh_outline(verts, up_axis_idx=2)
    turned = outline.build_mesh_outline(verts @ _rot_y(37.0).T, up_axis_idx=2)

    assert abs(base.ombb.aspect - turned.ombb.aspect) < 0.05
    assert abs(Polygon(base.pts_enu).area - Polygon(turned.pts_enu).area) \
        < 0.05 * Polygon(base.pts_enu).area


def _placed(fp, verts):
    mo = outline.build_mesh_outline(verts, up_axis_idx=2)
    # Let the footprint IoU pick, as the pipeline's disambiguation does. The
    # candidate ORDER is frame-dependent (k counts from the mesh's own OMBB
    # angle), so pinning index 0 would compare two different placements.
    from geo.fit import score_candidates
    chosen = score_candidates(fp, mo, ombb_candidates(fp, mo))[0][0]
    fit = solve(fp, mo, chosen=chosen, disambiguated_by=Disambiguator.EXIF_HEADING)
    moved = apply_similarity(mo.pts_enu, fit.theta, fit.scale_x, fit.scale_y,
                             np.array([fit.tx, fit.ty]))
    p = Polygon(moved)
    return (p if p.is_valid else p.buffer(0)), fit


def test_a_scrambled_model_lands_in_the_same_place():
    """Arbitrary yaw (37 deg, not a multiple of 90), x1000 scale, far offset."""
    fp = _fp(LSHAPE)
    verts = _prism(LSHAPE / 30.0)
    scrambled = (verts @ _rot_y(37.0).T) * 1000.0 + np.array([1234.5, 42.0, -987.6])

    a, fit_a = _placed(fp, verts)
    b, fit_b = _placed(fp, scrambled)

    assert a.union(b).area > 0
    assert a.intersection(b).area / a.union(b).area >= 0.97
    # The recovered model->metre scale absorbs the x1000 exactly.
    assert math.isclose(fit_a.scale_x / fit_b.scale_x, 1000.0, rel_tol=0.02)
    assert abs(fit_a.iou - fit_b.iou) < 0.02
