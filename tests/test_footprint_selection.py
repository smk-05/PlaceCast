"""
Regression tests for three footprint bugs found on live OSM data.

All three were silent: each produced a confident, plausible-looking, WRONG
answer rather than an error. They run offline against the committed cache in
fixtures/footprints/.

  1. Multi-way outer rings were truncated to the first member way, shrinking
     Burruss Hall — the primary demo building — from 6136 to 4384 m2.
  2. Multi-part relations collapsed to a single part, shrinking Lane Stadium
     from 28042 to 8930 m2.
  3. The spec 3.2 selection rule guessed instead of failing when nothing
     matched by name, returning a 360 m2 wind tunnel for a demolished
     Randolph Hall.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from geo.coords import ENUFrame
from geo.footprint import (
    _element_to_polygon,
    _stitch_rings,
    build_footprint,
    fetch_osm,
    geocode,
    select_footprint,
    to_shapely,
)


def _load(address: str):
    """Cached geocode + Overpass. No network if the cache is committed."""
    g = geocode(address)
    return g, fetch_osm(g["lat"], g["lon"])


# --------------------------------------------------------------------------
# 1. Ring stitching — the unit level, with synthetic members
# --------------------------------------------------------------------------


def _member(role: str, coords: list[tuple]) -> dict:
    return {"role": role, "geometry": [{"lon": x, "lat": y} for x, y in coords]}


def test_stitch_joins_open_segments_into_one_ring():
    """Four open ways chaining A->B->C->D->A are ONE ring, not four."""
    members = [
        _member("outer", [(0, 0), (4, 0)]),
        _member("outer", [(4, 0), (4, 3)]),
        _member("outer", [(4, 3), (0, 3)]),
        _member("outer", [(0, 3), (0, 0)]),
    ]
    rings = _stitch_rings(members, "outer")

    assert len(rings) == 1
    assert rings[0][0] == rings[0][-1]
    assert Polygon(rings[0]).area == pytest.approx(12.0)


def test_stitch_handles_reversed_segments():
    """OSM does not guarantee member ways run in a consistent direction."""
    members = [
        _member("outer", [(0, 0), (4, 0)]),
        _member("outer", [(4, 3), (4, 0)]),      # reversed
        _member("outer", [(4, 3), (0, 3)]),
        _member("outer", [(0, 0), (0, 3)]),      # reversed
    ]
    rings = _stitch_rings(members, "outer")

    assert len(rings) == 1
    assert Polygon(rings[0]).area == pytest.approx(12.0)


def test_stitch_passes_through_already_closed_rings():
    members = [_member("outer", [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])]
    rings = _stitch_rings(members, "outer")

    assert len(rings) == 1
    assert Polygon(rings[0]).area == pytest.approx(4.0)


def test_stitch_separates_genuinely_disjoint_parts():
    """Lane Stadium's shape: some open segments that join, plus closed rings."""
    members = [
        _member("outer", [(0, 0), (4, 0)]),
        _member("outer", [(4, 0), (4, 3)]),
        _member("outer", [(4, 3), (0, 3)]),
        _member("outer", [(0, 3), (0, 0)]),
        _member("outer", [(10, 0), (12, 0), (12, 2), (10, 2), (10, 0)]),
    ]
    rings = _stitch_rings(members, "outer")

    assert len(rings) == 2
    areas = sorted(Polygon(r).area for r in rings)
    assert areas == pytest.approx([4.0, 12.0])


# --------------------------------------------------------------------------
# 2. Burruss Hall — a single ring split across four member ways
# --------------------------------------------------------------------------


def test_burruss_outer_ring_is_stitched_not_truncated():
    """relation/1074686 has four OPEN outer members forming one ring.

    Taking members[0] closed it implicitly and lost 29% of the building. The
    old value was 4384 m2; the correct one is about 6136.
    """
    _, payload = _load("Burruss Hall, Blacksburg, VA")
    rel = next(el for el in payload["elements"]
               if el.get("type") == "relation" and el.get("id") == 1074686)

    outers = [m for m in rel["members"] if m.get("role") == "outer"]
    assert len(outers) == 4, "test premise changed — re-check the relation"
    assert all(m["geometry"][0] != m["geometry"][-1] for m in outers), \
        "all four members should be OPEN ways"

    geom = _element_to_polygon(rel)
    assert isinstance(geom, Polygon), "one stitched ring, not a MultiPolygon"

    frame = ENUFrame(lat0=geom.centroid.y, lon0=geom.centroid.x)
    fp = build_footprint(geom, frame, match_quality="contained_and_named")

    assert 5900 < fp.area_m2 < 6400, f"area {fp.area_m2} — truncation regressed"
    assert not fp.is_multipart


def test_burruss_is_selected_by_containment_and_name():
    g, payload = _load("Burruss Hall, Blacksburg, VA")
    el, geom, neighbours = select_footprint(payload, g["lat"], g["lon"],
                                            "Burruss Hall, Blacksburg, VA")

    assert el["id"] == 1074686
    assert el["_match_quality"] == "contained_and_named"
    assert geom.contains(Point(g["lon"], g["lat"]))


# --------------------------------------------------------------------------
# 3. Lane Stadium — genuinely disjoint outer rings
# --------------------------------------------------------------------------


def test_lane_stadium_is_a_multipolygon_with_every_stand():
    """relation/2417911 is four disjoint stands. The field is a real gap.

    Members 0 and 1 are open segments that stitch into one ring; 2, 3 and 4 are
    already closed. Four parts, not five members and not one polygon.
    """
    _, payload = _load("Lane Stadium, Blacksburg, VA")
    rel = next(el for el in payload["elements"]
               if el.get("type") == "relation" and el.get("id") == 2417911)

    geom = _element_to_polygon(rel)
    assert isinstance(geom, MultiPolygon)
    assert len(geom.geoms) == 4

    frame = ENUFrame(lat0=geom.centroid.y, lon0=geom.centroid.x)
    fp = build_footprint(geom, frame, match_quality="unnamed_sole_candidate")

    assert fp.is_multipart and len(fp.parts_enu) == 3
    assert 27_000 < fp.area_m2 < 29_000, f"area {fp.area_m2} — parts were dropped"


def test_to_shapely_keeps_every_part():
    """Anything measuring overlap must go through to_shapely, not pts_enu."""
    _, payload = _load("Lane Stadium, Blacksburg, VA")
    rel = next(el for el in payload["elements"]
               if el.get("type") == "relation" and el.get("id") == 2417911)
    geom = _element_to_polygon(rel)
    frame = ENUFrame(lat0=geom.centroid.y, lon0=geom.centroid.x)
    fp = build_footprint(geom, frame)

    full = to_shapely(fp)
    largest_only = Polygon(fp.pts_enu, [h for h in fp.holes_enu])

    assert isinstance(full, MultiPolygon)
    assert full.area == pytest.approx(fp.area_m2, rel=1e-6)
    assert full.area > largest_only.area * 1.5, \
        "reading pts_enu alone must be visibly lossy, or the test proves nothing"


def test_multipart_footprint_routes_to_review():
    from contracts import Decision
    from fixtures.fake_fits import make_fit, make_footprint
    from geo.validate import threshold_decision
    import numpy as np

    fp = make_footprint(0.95)
    fp = type(fp)(
        pts_enu=fp.pts_enu, holes_enu=fp.holes_enu,
        parts_enu=(np.array([[100.0, 100.0], [110.0, 100.0], [110.0, 108.0]]),),
        rectilinearity=fp.rectilinearity, principal_angle=fp.principal_angle,
        ombb=fp.ombb, area_m2=fp.area_m2, match_quality="contained_and_named",
    )
    _, decision, reasons = threshold_decision(
        make_fit(iou=0.92, hausdorff=0.9, margin=0.3), fp
    )

    assert decision is not Decision.AUTO_ACCEPT
    assert any("multi-part" in r for r in reasons)


# --------------------------------------------------------------------------
# 4. The selection rule must fail rather than guess
# --------------------------------------------------------------------------


def test_randolph_hall_fails_loudly_instead_of_guessing():
    """Randolph Hall is demolished; the OSM polygon is a construction site.

    The old ranking returned way/461103807, the Stability Wind Tunnel — 360 m2,
    46 m away, no name match. A wind tunnel standing in for a classroom building
    produces a plausible IoU against the wrong footprint, which is exactly the
    fictitious demo spec 14 warns about.
    """
    g, payload = _load("Randolph Hall, Blacksburg, VA")

    with pytest.raises(LookupError) as exc:
        select_footprint(payload, g["lat"], g["lon"], "Randolph Hall, Blacksburg, VA")

    msg = str(exc.value)
    assert "Refusing to guess" in msg
    # The message must name the candidates, or a human cannot act on it.
    assert "Stability Wind Tunnel" in msg


def test_sole_unnamed_candidate_is_accepted_but_flagged():
    """The one narrow exception, and it must be visibly weaker evidence."""
    g, payload = _load("Lane Stadium, Blacksburg, VA")
    el, _, _ = select_footprint(payload, g["lat"], g["lon"], "Lane Stadium, Blacksburg, VA")

    assert el["_match_quality"] == "unnamed_sole_candidate"


def test_weak_match_routes_to_review_even_with_a_perfect_fit():
    """A high IoU against a weakly-matched footprint is the most dangerous
    output this pipeline can produce, because it looks like success."""
    from contracts import Decision
    from fixtures.fake_fits import make_fit, make_footprint
    from geo.validate import threshold_decision

    fit = make_fit(iou=0.96, hausdorff=0.4, area_ratio=1.01, margin=0.4)

    _, strong, _ = threshold_decision(fit, make_footprint(0.95))
    assert strong is Decision.AUTO_ACCEPT

    _, weak, reasons = threshold_decision(
        fit, make_footprint(0.95, match_quality="unnamed_sole_candidate")
    )
    assert weak is Decision.REVIEW
    assert any("no name match" in r for r in reasons)
