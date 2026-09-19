"""
Regression tests for the issues Owen reported after the perception merge.

Each was verified against the code before fixing; see the commit message.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fixtures.fake_photo_evidence import with_exif
from geo.disambiguate import azimuth_from_exif
from geo.height import from_osm_tags, parse_osm_length
from contracts import HeightSource

BLDG = (37.23, -80.42)


# --------------------------------------------------------------------------
# OSM lengths: "82 ft" is not 82 m
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,metres", [
    ("20.7", 20.7),
    ("20.7 m", 20.7),
    ("20.7m", 20.7),
    ("20,7", 20.7),
    ("12 metres", 12.0),
    ("82 ft", 82 * 0.3048),
    ("82 feet", 82 * 0.3048),
    ("82'", 82 * 0.3048),
    ("25'6\"", 25.5 * 0.3048),
    (" 30 ", 30.0),
])
def test_parse_osm_length(raw, metres):
    assert parse_osm_length(raw) == pytest.approx(metres)


@pytest.mark.parametrize("raw", ["tall", "", "12 storeys", "~20", None, "20-25"])
def test_unparseable_lengths_return_none(raw):
    """Fall through the height chain rather than guess."""
    assert parse_osm_length(raw) is None


def test_feet_tag_is_not_three_times_too_tall():
    h, src = from_osm_tags({"height": "82 ft"})
    assert h == pytest.approx(25.0, abs=0.01)
    assert src is HeightSource.OSM_HEIGHT


def test_unparseable_height_falls_through_to_levels():
    h, src = from_osm_tags({"height": "tall", "building:levels": "4"})
    assert src is HeightSource.OSM_LEVELS and h == pytest.approx(12.0)


# --------------------------------------------------------------------------
# Nominatim 1 request/second
# --------------------------------------------------------------------------


def test_nominatim_throttle_waits_between_live_calls(monkeypatch):
    import geo.footprint as F

    clock = {"t": 1000.0}
    slept = []
    monkeypatch.setattr(F.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(F.time, "sleep", lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)))
    monkeypatch.setattr(F, "_last_nominatim_call", 0.0)

    F._throttle_nominatim()          # first call: long since the last one
    clock["t"] += 0.3
    F._throttle_nominatim()          # 0.3 s later: must wait the remaining 0.7 s

    assert slept == [pytest.approx(0.7)]


# --------------------------------------------------------------------------
# azimuth_from_exif abstains on invalid input
# --------------------------------------------------------------------------


@pytest.mark.parametrize("heading", [float("nan"), float("inf"), 400.0, 360.0, -30.0])
def test_invalid_heading_abstains(heading):
    assert azimuth_from_exif(with_exif(heading_deg=heading), *BLDG) is None


def test_valid_heading_still_works():
    az = azimuth_from_exif(with_exif(heading_deg=0.0), *BLDG)
    assert az == pytest.approx(math.pi / 2)   # camera pointing North


@pytest.mark.parametrize("gps", [
    (float("nan"), -80.42),
    (37.23, float("inf")),
    (95.0, -80.42),            # latitude out of range
    (37.23, -200.0),           # longitude out of range
    (40.0, -75.0),             # a valid fix hundreds of km away: stale, ignore it
])
def test_invalid_or_distant_gps_is_ignored(gps):
    """With the GPS rejected and no heading, the cue abstains."""
    ev = with_exif(heading_deg=None, gps=gps)
    assert azimuth_from_exif(ev, *BLDG) is None


def test_bad_gps_falls_back_to_a_good_heading():
    ev = with_exif(heading_deg=90.0, gps=(float("nan"), -80.42))
    assert azimuth_from_exif(ev, *BLDG) == pytest.approx(0.0)   # East


# --------------------------------------------------------------------------
# Road prior counts streets, not footpaths
# --------------------------------------------------------------------------


def _roads(address):
    from geo.coords import ENUFrame
    from geo.footprint import fetch_osm, geocode, select_footprint
    from pipeline import _roads_to_enu

    g = geocode(address)
    payload = fetch_osm(g["lat"], g["lon"])
    _, poly, _ = select_footprint(payload, g["lat"], g["lon"], address)
    frame = ENUFrame(lat0=poly.centroid.y, lon0=poly.centroid.x)
    return payload, _roads_to_enu(payload, frame)


def test_footpaths_are_not_roads():
    """Near Burruss, 138 of 139 highway ways are paths, footways, steps or a
    parking aisle. Only the tertiary street counts."""
    payload, roads = _roads("Burruss Hall, Blacksburg, VA")
    n_highway = sum(1 for e in payload["elements"] if "highway" in (e.get("tags") or {}))
    assert n_highway > 100
    assert len(roads) == 1


def test_ncb_keeps_its_streets():
    payload, roads = _roads("Classroom Building, Blacksburg, VA")
    streets = [e for e in payload["elements"]
               if (e.get("tags") or {}).get("highway") in ("tertiary", "unclassified", "service")
               and (e.get("tags") or {}).get("service") != "parking_aisle"]
    assert len(roads) == len(streets) > 0


# --------------------------------------------------------------------------
# front_angle from the photographed side
# --------------------------------------------------------------------------


def _asymmetric_mesh():
    """A slab with a tower at one end, glTF +Y up: every side's silhouette differs."""
    import trimesh
    slab = trimesh.creation.box(extents=[40, 12, 16])
    tower = trimesh.creation.box(extents=[10, 30, 10])
    tower.apply_translation([14, 9, 2])
    return trimesh.util.concatenate([slab, tower])


def _photo(mask, model="real:test"):
    from contracts import PhotoEvidence
    return PhotoEvidence(mask=mask, segmentation_model=model)


@pytest.mark.parametrize("side", [0, 1, 2, 3])
def test_front_angle_comes_from_the_photographed_side(tmp_path, side):
    from contracts import MeshOutline
    from geo import outline
    from perception.render_compare import render_silhouette
    from pipeline import _front_from_photo

    mesh = _asymmetric_mesh()
    glb = tmp_path / "mesh.glb"
    mesh.export(glb)

    # The "photo" is the mesh seen from `side`; the front must end up there.
    mask = render_silhouette(mesh, (0, 1, 0), side * np.pi / 2)
    mo = MeshOutline(pts_enu=np.zeros((4, 2)), up_axis_idx=2,
                     front_angle=outline.front_angle_canonical(2))

    out = _front_from_photo(mo, glb, _photo(mask), lambda *_: None)

    # glTF side 0 is canonical -90 deg; each side steps 90 deg CCW.
    expected = -np.pi / 2 + side * np.pi / 2
    d = math.atan2(math.sin(out.front_angle - expected), math.cos(out.front_angle - expected))
    assert abs(d) < 1e-6


def test_stub_mask_keeps_the_gltf_prior(tmp_path):
    from contracts import MeshOutline
    from perception.render_compare import render_silhouette
    from pipeline import _front_from_photo

    mesh = _asymmetric_mesh()
    glb = tmp_path / "mesh.glb"
    mesh.export(glb)
    mo = MeshOutline(pts_enu=np.zeros((4, 2)), up_axis_idx=2, front_angle=-np.pi / 2)

    out = _front_from_photo(mo, glb, _photo(render_silhouette(mesh, (0, 1, 0), np.pi),
                                            model="stub:central-box"), lambda *_: None)
    assert out.front_angle == pytest.approx(-np.pi / 2)


# --------------------------------------------------------------------------
# The record names the confidence method that RAN
# --------------------------------------------------------------------------


def test_record_names_the_method_that_ran(tmp_path, monkeypatch):
    """--confidence learned with no trained model falls back to the table, and
    the record must say table, not learned."""
    import pipeline

    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path)
    rec = pipeline.run("Classroom Building, Blacksburg, VA", dry_run=True,
                       confidence_method="learned")
    assert rec.confidence_method == "threshold_table"


# --------------------------------------------------------------------------
# Split buildings: Hahn Hall is two OSM buildings
# --------------------------------------------------------------------------


def _select(address):
    from geo.footprint import fetch_osm, geocode, select_footprint
    g = geocode(address)
    return select_footprint(fetch_osm(g["lat"], g["lon"]), g["lat"], g["lon"], address)


def test_split_building_names_every_part():
    """The geocoder lands on a bus stop between the halves; North sits just
    outside the normal search radius. Both must be named, and nothing chosen."""
    with pytest.raises(LookupError) as exc:
        _select("Hahn Hall, Blacksburg, VA")
    msg = str(exc.value)
    assert "Hahn Hall North" in msg and "Hahn Hall South" in msg
    assert "split" in msg


@pytest.mark.parametrize("half,way_id", [("North", 43313684), ("South", 60193988)])
def test_each_half_resolves_when_named(half, way_id):
    el, _, _ = _select(f"Hahn Hall {half}, Blacksburg, VA")
    assert el["id"] == way_id
    assert el["_match_quality"] == "contained_and_named"


def test_split_detection_needs_the_whole_name():
    """'Hall' alone must not make every hall on campus a 'part'."""
    from geo.footprint import _split_building_parts
    b = [({"tags": {"name": "Burruss Hall"}}, None), ({"tags": {"name": "Hahn Hall North"}}, None)]
    assert _split_building_parts(b, "Hall, Blacksburg, VA") == []
    assert _split_building_parts(b, "Hahn Hall, Blacksburg, VA") == ["Hahn Hall North"]
