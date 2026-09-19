"""perception/openings_prism.py on a synthetic box footprint seen by a known camera.

The prism is 20 m east-west by 10 m north-south, 8 m tall, centred on the ENU origin, so the walls are
south (y = -5, bearing 180), east (x = 10, 90), north (y = 5, 0) and west (x = -10, 270).
"""
import json
import math

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

from geo.coords import ENUFrame
from perception import openings_prism as op

W, H = 320, 240  # RASTER_MAX_SIDE is 320, so the refinement scores the full-resolution mask
F_PX = 300.0


def _prism():
    return op.prism_from_polygons([box(-10, -5, 10, 5)], 8.0, ENUFrame(37.2, -80.4, 0.0))


def _camera(x, y, yaw, z=1.5, pitch=0.0):
    return op.PinholeCamera((x, y, z), yaw, pitch, F_PX, W, H)


def _mask(prism, cam):
    r = op._Rasteriser(prism, W, H)
    assert r.scale == 1.0
    return r.render(cam)


def _wall_index(prism, bearing):
    return next(w["index"] for w in prism.walls if abs(w["bearing_deg"] - bearing) < 1e-6)


def _opening(name, cam, mask, corners_enu, type_="window", decision="ACCEPT"):
    """An openings.json-style record whose box is the projection of four ENU corners (TL, TR, BR, BL)."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    px = cam.project(corners_enu)
    uv = np.stack([(px[:, 0] - x0) / (x1 - x0), (px[:, 1] - y0) / (y1 - y0)], axis=1)
    return {
        "id": name, "type": type_, "decision": decision, "reasons": [], "score": 0.5, "group_margin": 0.6,
        "box_uv": [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
        "center_uv": list(uv.mean(axis=0)),
    }


def _on_south(name, cam, mask, cx, z0, w=2.0, h=1.5, type_="window"):
    y = -5.0
    corners = [(cx - w / 2, y, z0 + h), (cx + w / 2, y, z0 + h), (cx + w / 2, y, z0), (cx - w / 2, y, z0)]
    return _opening(name, cam, mask, corners, type_)


def _on_east(name, cam, mask, cy, z0, w=2.0, h=1.5, type_="window"):
    x = 10.0
    corners = [(x, cy - w / 2, z0 + h), (x, cy + w / 2, z0 + h), (x, cy + w / 2, z0), (x, cy - w / 2, z0)]
    return _opening(name, cam, mask, corners, type_)


def _place(prism, cam, mask, ops, **kw):
    return op.place_openings_on_prism(prism, None, mask, ops, camera=cam, refine=False, **kw)


# --------------------------------------------------------------------- walls


def test_wall_normals_and_bearings_are_exact():
    prism = _prism()
    by_bearing = {round(w["bearing_deg"]): w["normal"] for w in prism.walls}
    assert set(by_bearing) == {0, 90, 180, 270}
    assert by_bearing[180] == pytest.approx([0, -1]) and by_bearing[90] == pytest.approx([1, 0])
    assert by_bearing[0] == pytest.approx([0, 1]) and by_bearing[270] == pytest.approx([-1, 0])


def test_clockwise_input_polygon_gets_the_same_outward_normals():
    from shapely.geometry import Polygon

    cw = Polygon([(-10, -5), (-10, 5), (10, 5), (10, -5)])  # clockwise
    assert not cw.exterior.is_ccw
    prism = op.prism_from_polygons([cw], 8.0)
    assert sorted(round(w["bearing_deg"]) for w in prism.walls) == [0, 90, 180, 270]


def test_a_courtyard_wall_faces_into_the_courtyard():
    from shapely.geometry import Polygon

    prism = op.prism_from_polygons([Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)],
                                            [[(-3, -3), (3, -3), (3, 3), (-3, 3)]])], 8.0)
    # The hole's south edge (y = -3) is a wall of the solid facing north, into the courtyard.
    hole_south = next(w for w in prism.walls if np.allclose([w["p0"][1], w["p1"][1]], -3.0))
    assert hole_south["normal"] == pytest.approx([0, 1]) and hole_south["bearing_deg"] == pytest.approx(0.0)


def test_prism_from_record_recovers_the_footprint():
    frame = ENUFrame(37.2, -80.4, 0.0)
    ring = np.array([(-10, -5), (10, -5), (10, 5), (-10, 5), (-10, -5)], float)
    lat, lon, _ = frame.enu_to_geodetic(ring[:, 0], ring[:, 1], 0.0).T
    record = {"enu_origin_geodetic": [37.2, -80.4, 0.0], "height": {"value_m": 8.0},
              "footprint_geojson": {"type": "Polygon", "coordinates": [[[lo, la] for la, lo in zip(lat, lon)]]}}
    prism = op.prism_from_record(record)
    assert prism.height_m == 8.0
    assert prism.polygons[0].bounds == pytest.approx((-10, -5, 10, 5), abs=1e-3)


# -------------------------------------------------------------------- camera


def test_focal_length_from_35mm_equivalent_uses_the_diagonal():
    assert op.focal_px(26.0, 5712, 4284) == pytest.approx(26 * 7140 / 43.2666, rel=1e-4)
    assert op.focal_px(26.0, 5712, 4284) == pytest.approx(4290.6, abs=1.0)


def test_exif_focal_35mm_is_read_from_the_exif_sub_ifd(tmp_path):
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26
    Image.new("RGB", (8, 8)).save(tmp_path / "p.jpg", exif=exif)
    assert op.exif_focal_35mm(tmp_path / "p.jpg") == 26.0
    Image.new("RGB", (8, 8)).save(tmp_path / "bare.jpg")
    assert op.exif_focal_35mm(tmp_path / "bare.jpg") is None


def test_project_and_rays_are_inverse():
    cam = _camera(3, -40, 20.0, pitch=4.0)
    pts = np.array([[0, 0, 5.0], [8, 3, 2.0]])
    px = cam.project(pts)
    for p, ray in zip(pts, cam.rays(px[:, 0], px[:, 1])):
        t = np.linalg.norm(p - np.array(cam.position))
        assert np.array(cam.position) + ray * t == pytest.approx(p, abs=1e-6)


# ----------------------------------------------------------------- placement


def test_frontal_view_puts_the_opening_on_the_south_wall_with_its_bearing():
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    r, = _place(prism, cam, mask, [_on_south("w1", cam, mask, cx=3.0, z0=3.0)]).openings
    assert r["wall_index"] == _wall_index(prism, 180.0)
    assert r["bearing_deg"] == pytest.approx(180.0, abs=1e-9)
    assert r["normal_enu"] == pytest.approx([0, -1, 0])
    assert r["hit_enu"] == pytest.approx([3.0, -5.0, 3.75], abs=0.05)
    assert r["position_enu"] == pytest.approx([3.0, -5.0 - op.OFFSET_M, 3.75], abs=0.05)
    assert r["width_m"] == pytest.approx(2.0, abs=0.05) and r["height_m"] == pytest.approx(1.5, abs=0.05)
    assert r["bottom_above_ground_m"] == pytest.approx(3.0, abs=0.05)
    assert r["decision"] == "ACCEPT" and r["reasons"] == []
    assert r["lat"] is not None and r["lon"] is not None
    assert r["height_above_ground_m"] == pytest.approx(3.75, abs=0.05)


def test_corner_view_splits_openings_between_two_walls():
    # Camera south-east of the prism looking north-west (yaw 315): the SE corner (10, -5) is in the middle of the
    # view, the south wall runs off to its left and the east wall to its right.
    prism, cam = _prism(), _camera(30, -30, 315.0)
    mask = _mask(prism, cam)
    ops = [_on_south("left", cam, mask, cx=5.0, z0=3.0), _on_east("right", cam, mask, cy=-1.0, z0=3.0)]
    a, b = _place(prism, cam, mask, ops).openings
    assert a["wall_index"] == _wall_index(prism, 180.0) and b["wall_index"] == _wall_index(prism, 90.0)
    assert a["bearing_deg"] == pytest.approx(180.0) and b["bearing_deg"] == pytest.approx(90.0)
    corner_u = cam.project([[10, -5, 4]])[0, 0]
    assert cam.project([[5, -5, 4]])[0, 0] < corner_u < cam.project([[10, -1, 4]])[0, 0]
    assert a["decision"] == b["decision"] == "ACCEPT"


def test_a_raised_door_goes_to_review_but_a_raised_window_does_not():
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    ops = [_on_south("door_high", cam, mask, -5.0, z0=1.5, h=2.0, type_="door"),
           _on_south("win_high", cam, mask, 0.0, z0=1.5, type_="window"),
           _on_south("door_ok", cam, mask, 5.0, z0=0.05, h=2.0, type_="door"),
           _on_south("garage_high", cam, mask, 8.0, z0=1.4, h=2.0, type_="garage_door")]
    door, win, ok, garage = _place(prism, cam, mask, ops).openings
    assert door["decision"] == "REVIEW" and any("above ground" in why for why in door["reasons"])
    assert garage["decision"] == "REVIEW"
    assert win["decision"] == "ACCEPT" and ok["decision"] == "ACCEPT"
    assert door["bottom_above_ground_m"] == pytest.approx(1.5, abs=0.05)


def test_roof_hit_and_miss_go_to_review_and_other_is_skipped():
    prism = _prism()
    cam = _camera(0, -40, 0.0, z=30.0, pitch=-15.0)  # a camera above the roof line sees the roof
    mask = _mask(prism, cam)
    roof = _opening("roof", cam, mask, [(-1, -1, 8), (1, -1, 8), (1, 1, 8), (-1, 1, 8)])
    off = {**roof, "id": "off", "box_uv": [1.5, 0.4, 1.6, 0.5], "center_uv": [1.55, 0.45]}
    other = {**roof, "id": "x", "type": "other"}
    rows = _place(prism, cam, mask, [roof, off, other]).openings
    assert [r["id"] for r in rows] == ["roof", "off"]
    assert rows[0]["decision"] == "REVIEW" and any("roof" in why for why in rows[0]["reasons"])
    assert rows[1]["decision"] == "REVIEW" and any("missed" in why for why in rows[1]["reasons"])
    assert rows[0]["wall_index"] is None and rows[0]["bearing_deg"] is None


# ---------------------------------------------------------------- refinement


# A small search grid for the tests that are not about the grid itself: yaw +-15 (default), position +-2 m in one
# 2 m step, pitch +-2, no focal scaling, no fine stage, one process.
NARROW = {"pos_range": 2.0, "pos_step": 2.0, "refine_step": None, "pitch_range": 2.0, "scales": (1.0,), "workers": 1}


def test_ten_degrees_of_yaw_error_is_recovered():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    fit = op.fit_camera(prism, mask, true.moved(yaw_deg=10.0), **NARROW)
    assert fit.iou_initial < 0.9
    assert fit.yaw_offset_deg == pytest.approx(-10.0, abs=1.0)
    assert fit.camera.yaw_deg == pytest.approx(0.0, abs=1.0)
    assert fit.iou > 0.97 and fit.reliable and fit.at_edge == ()


def test_refinement_places_the_opening_where_the_true_camera_would():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    opening = _on_south("w1", true, mask, cx=3.0, z0=3.0)
    result = op.place_openings_on_prism(prism, None, mask, [opening], camera=true.moved(yaw_deg=10.0),
                                        fit_kwargs=NARROW)
    r, = result.openings
    assert r["decision"] == "ACCEPT" and r["hit_enu"] == pytest.approx([3.0, -5.0, 3.75], abs=0.15)


def test_a_wrong_focal_length_is_recovered_as_a_scale():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    # An EXIF focal 10% too long: the fit should scale it back by ~0.91, i.e. the 0.90 step. Position, yaw and
    # pitch are pinned so nothing else can absorb the error.
    fit = op.fit_camera(prism, mask, true.moved(scale=1.10), yaw_range=0.0, pitch_range=0.0, pos_range=0.0,
                        workers=1)
    assert fit.focal_scale == pytest.approx(0.90) and fit.iou > 0.95 and fit.iou_initial < 0.9
    assert fit.camera.f_px == pytest.approx(F_PX * 1.10 * 0.90)
    assert fit.at_edge == () and fit.reliable
    assert fit.to_dict()["focal_scale"] == pytest.approx(0.90)


def test_position_is_refined_to_one_metre_inside_the_coarse_step():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    # 6 m east of the truth: the 4 m grid has -4 and -8, the 1 m stage must land on -6.
    fit = op.fit_camera(prism, mask, true.moved(dx=6.0), yaw_range=0.0, pitch_range=0.0, scales=(1.0,), workers=1)
    assert (fit.dx_m, fit.dy_m) == pytest.approx((-6.0, 0.0)) and fit.iou > 0.99 and fit.at_edge == ()


def test_the_parallel_grid_gives_the_same_fit_as_the_sequential_one():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    kw = {"yaw_range": 5.0, "pitch_range": 0.0}  # 11 x 81 x 7 = 6237 coarse candidates: over the parallel threshold
    start = true.moved(yaw_deg=3.0, dx=2.0, scale=1.05)
    a = op.fit_camera(prism, mask, start, workers=1, **kw)
    b = op.fit_camera(prism, mask, start, workers=2, **kw)
    assert a.to_dict() == b.to_dict()


def test_a_best_value_on_the_grid_edge_sends_every_opening_to_review():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    opening = _on_south("w1", true, mask, cx=3.0, z0=3.0)
    # 6 m east of the truth but searched only +-4 m: the optimum is outside the grid.
    result = op.place_openings_on_prism(prism, None, mask, [opening], camera=true.moved(dx=6.0),
                                        fit_kwargs={**NARROW, "pos_range": 4.0, "pitch_range": 1.0, "yaw_range": 2.0})
    assert not result.fit.reliable and "east" in result.fit.at_edge
    r, = result.openings
    assert r["decision"] == "REVIEW" and any("edge of the search grid" in why for why in r["reasons"])


def test_a_poor_silhouette_fit_sends_every_opening_to_review():
    prism, true = _prism(), _camera(0, -40, 0.0)
    wrong = np.zeros((H, W), bool)
    wrong[20:60, 20:80] = True  # a mask that is nowhere near the prism
    opening = _on_south("w1", true, _mask(prism, true), cx=3.0, z0=3.0)
    result = op.place_openings_on_prism(prism, None, wrong, [opening], camera=true, refine=False)
    assert result.fit.iou < op.MIN_CAMERA_IOU and not result.fit.reliable
    assert all(r["decision"] == "REVIEW" and any("camera fit IoU" in why for why in r["reasons"])
               for r in result.openings)


# --------------------------------------------------------------- consistency


def test_facade_check_compares_the_photographed_wall_with_the_front_heading():
    ops = [{"bearing_deg": 180.0}, {"bearing_deg": 180.0}, {"bearing_deg": 90.0}]
    ok = op.facade_check({"facade": {"front_heading_deg": 175.0}}, ops)
    assert ok["applicable"] and ok["consistent"] and ok["delta_deg"] == pytest.approx(5.0)
    bad = op.facade_check({"facade": {"front_heading_deg": 200.0}}, ops)
    assert bad["applicable"] and not bad["consistent"] and bad["delta_deg"] == pytest.approx(20.0)
    wrap = op.facade_check({"facade": {"front_heading_deg": 355.0}}, [{"bearing_deg": 4.0}])
    assert wrap["consistent"] and wrap["delta_deg"] == pytest.approx(9.0)
    assert op.facade_check({"facade": {"front_heading_deg": None}}, ops)["applicable"] is False
    # A front that is only the glTF convention is not evidence: skipped even when it would "match" or "mismatch".
    for heading in (180.0, 20.0):
        skipped = op.facade_check({"facade": {"front_heading_deg": heading, "front_source": "gltf_prior"}}, ops)
        assert skipped["applicable"] is False and "gltf_prior" in skipped["reason"]
    real = op.facade_check({"facade": {"front_heading_deg": 175.0, "front_source": "photographed_side"}}, ops)
    assert real["applicable"] and real["consistent"]


def test_record_gets_the_openings(tmp_path):
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    result = _place(prism, cam, mask, [_on_south("w1", cam, mask, cx=3.0, z0=3.0)], source_photo="p.jpg")
    path = tmp_path / "record.json"
    path.write_text('{"asset_id": "x"}')
    data = op.attach_to_record(path, result)
    assert json.loads(path.read_text()) == data
    assert data["openings"][0]["id"] == "w1" and data["openings"][0]["bearing_deg"] == pytest.approx(180.0)
    assert data["openings_camera"]["camera_iou"] == pytest.approx(1.0)
    assert data["openings_facade_check"]["applicable"] is False
