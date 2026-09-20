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


def test_implausible_sizes_go_to_review_as_a_grazing_view():
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    ops = [_on_south("win_wide", cam, mask, -4.0, z0=2.0, w=5.0, h=1.5, type_="window"),
           _on_south("win_ok", cam, mask, 6.0, z0=2.0, w=3.0, h=1.5, type_="window"),
           _on_south("door_wide", cam, mask, 0.0, z0=0.05, w=5.0, h=2.0, type_="door"),  # only windows have a width cap
           _on_south("tall", cam, mask, 8.0, z0=0.05, w=1.0, h=7.0, type_="door")]
    wide, ok, door, tall = _place(prism, cam, mask, ops).openings
    assert wide["width_m"] == pytest.approx(5.0, abs=0.1) and tall["height_m"] == pytest.approx(7.0, abs=0.1)
    assert wide["decision"] == "REVIEW" and "implausible size (grazing view)" in wide["reasons"]
    assert tall["decision"] == "REVIEW" and "implausible size (grazing view)" in tall["reasons"]
    assert ok["decision"] == "ACCEPT" and door["decision"] == "ACCEPT"


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
# 2 m step, pitch 0-2, no focal scaling, no fine stage, one process.
SCALE_GRID = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)  # the pipeline does NOT search scale (frozen 1.0)
NARROW = {"pos_range": 2.0, "pos_step": 2.0, "refine_step": None, "pitches": (0.0, 1.0, 2.0), "scales": (1.0,), "workers": 1}


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


def test_focal_scale_is_frozen_at_one_by_default():
    assert op.FOCAL_SCALES == (1.0,)
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    fit = op.fit_camera(prism, mask, true.moved(scale=1.10), yaw_range=0.0, pitches=(0.0,), pos_range=0.0,
                        workers=1)
    assert fit.focal_scale == 1.0 and fit.camera.f_px == pytest.approx(F_PX * 1.10)  # not searched, not corrected
    assert fit.at_edge == ()  # a one-value axis has no edge


def test_a_wrong_focal_length_is_recovered_as_a_scale_when_scale_is_searched():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    # An EXIF focal 10% too long: the fit should scale it back by ~0.91, i.e. the 0.90 step. Position, yaw and
    # pitch are pinned so nothing else can absorb the error.
    fit = op.fit_camera(prism, mask, true.moved(scale=1.10), yaw_range=0.0, pitches=(0.0,), pos_range=0.0,
                        scales=SCALE_GRID, workers=1)
    assert fit.focal_scale == pytest.approx(0.90) and fit.iou > 0.95 and fit.iou_initial < 0.9
    assert fit.camera.f_px == pytest.approx(F_PX * 1.10 * 0.90)
    assert fit.at_edge == () and fit.reliable
    assert fit.to_dict()["focal_scale"] == pytest.approx(0.90)


def test_pitch_is_searched_upward_from_the_horizon():
    prism = _prism()
    true = _camera(0, -40, 0.0, pitch=8.0)
    mask = _mask(prism, true)
    # No EXIF pitch (0): the fit finds the nearest 2.5 deg step to the true 8 deg, and 0 is not an edge.
    fit = op.fit_camera(prism, mask, true.moved(pitch_deg=-8.0), yaw_range=0.0, pos_range=0.0, scales=(1.0,),
                        workers=1)
    assert fit.camera.pitch_deg == pytest.approx(7.5) and fit.iou > 0.9 and fit.at_edge == ()
    level = _camera(0, -40, 0.0)
    flat = op.fit_camera(prism, _mask(prism, level), level, yaw_range=0.0, pos_range=0.0, scales=(1.0,), workers=1)
    assert flat.camera.pitch_deg == 0.0 and "pitch" not in flat.at_edge  # the floor is not an edge


def test_a_pitch_beyond_the_top_of_the_grid_is_an_edge_hit():
    prism = _prism()
    true = _camera(0, -40, 0.0, pitch=22.0)
    fit = op.fit_camera(prism, _mask(prism, true), true.moved(pitch_deg=-22.0), yaw_range=0.0, pos_range=0.0,
                        scales=(1.0,), workers=1)
    assert fit.camera.pitch_deg == 15.0 and "pitch" in fit.at_edge and not fit.reliable


def test_position_is_refined_to_one_metre_inside_the_coarse_step():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    # 6 m east of the truth: the 4 m grid has -4 and -8, the 1 m stage must land on -6.
    fit = op.fit_camera(prism, mask, true.moved(dx=6.0), yaw_range=0.0, pitches=(0.0,), scales=(1.0,), workers=1)
    assert (fit.dx_m, fit.dy_m) == pytest.approx((-6.0, 0.0)) and fit.iou > 0.99 and fit.at_edge == ()


def test_the_parallel_grid_gives_the_same_fit_as_the_sequential_one():
    prism, true = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, true)
    kw = {"yaw_range": 10.0, "pitches": (0.0,), "pos_range": 8.0, "scales": SCALE_GRID}  # 4725 candidates: parallel
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
                                        fit_kwargs={**NARROW, "pos_range": 4.0, "yaw_range": 2.0})
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


# ------------------------------------------------- corners on the wall's plane


def test_corners_are_intersected_with_the_walls_plane_not_the_prism_mesh():
    # An opening at the east end of the south wall whose right corners overhang the building's corner by 0.4 m: those
    # rays miss the mesh (they pass beside the prism) but land on the wall's plane, within the +-0.5 m tolerance.
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    op_ = _on_south("edge_window", cam, mask, cx=9.4, z0=3.0, w=2.0, h=1.5)
    r, = _place(prism, cam, mask, [op_]).openings
    assert r["decision"] == "ACCEPT" and r["reasons"] == [] and r["corner_hits"] == 4
    assert r["width_m"] == pytest.approx(2.0, abs=0.05) and r["height_m"] == pytest.approx(1.5, abs=0.05)
    assert r["bottom_above_ground_m"] == pytest.approx(3.0, abs=0.05)


def test_a_corner_beyond_the_walls_ends_by_more_than_half_a_metre_is_flagged():
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    op_ = _on_south("overhang", cam, mask, cx=9.9, z0=3.0, w=2.0, h=1.5)  # right corners 0.9 m past the end
    r, = _place(prism, cam, mask, [op_]).openings
    assert r["decision"] == "REVIEW" and r["corner_hits"] == 2 and r["width_m"] is None
    assert any("2 of 4 corner rays missed the wall" in why for why in r["reasons"])


def test_corners_may_reach_one_metre_below_ground_but_not_further():
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    ok = _on_south("half_metre_down", cam, mask, cx=0.0, z0=-0.5, w=2.0, h=3.0, type_="window")
    deep = _on_south("one_and_a_half_down", cam, mask, cx=0.0, z0=-1.5, w=2.0, h=3.5, type_="window")
    a, b = _place(prism, cam, mask, [ok, deep]).openings
    assert a["decision"] == "ACCEPT" and a["bottom_above_ground_m"] == pytest.approx(-0.5, abs=0.05)
    assert b["decision"] == "REVIEW" and b["corner_hits"] == 2  # only the two top corners are on the wall
    assert any("below ground" in why for why in b["reasons"])


def test_a_corner_above_the_roof_is_flagged():
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    op_ = _on_south("up_and_over", cam, mask, cx=0.0, z0=6.0, w=2.0, h=3.0)  # top corners at z = 9 m > the 8 m roof
    r, = _place(prism, cam, mask, [op_]).openings
    assert r["decision"] == "REVIEW" and r["corner_hits"] == 2


def test_an_oblique_view_still_sizes_an_opening_on_the_wall_it_is_on():
    # From the SE corner camera the east wall is seen at a grazing angle. Corners are placed on the east wall's plane.
    prism, cam = _prism(), _camera(30, -30, 315.0)
    mask = _mask(prism, cam)
    r, = _place(prism, cam, mask, [_on_east("e", cam, mask, cy=-1.0, z0=2.0, w=2.0, h=1.5)]).openings
    assert r["wall_index"] == _wall_index(prism, 90.0) and r["corner_hits"] == 4
    assert r["width_m"] == pytest.approx(2.0, abs=0.1) and r["height_m"] == pytest.approx(1.5, abs=0.1)
    assert r["decision"] == "ACCEPT"


# ------------------------------------------- distance-scaled corner tolerance


def test_corner_tolerance_is_the_angular_uncertainty_with_a_floor():
    assert op.corner_tolerance(5.0) == 0.5 and op.corner_tolerance(20.0) == 0.5  # the 0.5 m floor
    assert op.corner_tolerance(114.0) == pytest.approx(114.0 * math.tan(math.radians(1.25)), rel=1e-12)
    assert op.corner_tolerance(114.0) == pytest.approx(2.49, abs=0.01)
    assert op.corner_tolerance(50.0) < op.corner_tolerance(114.0) < op.corner_tolerance(200.0)


def test_the_tolerance_is_recorded_per_opening_from_the_centre_ray_distance():
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    a, b = _place(prism, cam, mask, [_on_south("near_ok", cam, mask, cx=0.0, z0=3.0),
                                     _opening_off_prism("miss", cam, mask)]).openings
    dist = float(np.linalg.norm(np.asarray(a["hit_enu"]) - np.asarray(cam.position)))
    assert a["corner_tolerance_m"] == pytest.approx(op.corner_tolerance(dist))
    assert a["corner_tolerance_m"] == pytest.approx(35.0 * math.tan(math.radians(1.25)), rel=0.02)
    assert b["corner_tolerance_m"] is None  # no wall, so no corner check and nothing to record


def _opening_off_prism(name, cam, mask):
    return {**_on_south(name, cam, mask, cx=0.0, z0=3.0), "box_uv": [1.5, 0.4, 1.6, 0.5], "center_uv": [1.55, 0.45]}


def test_a_far_camera_forgives_what_a_near_camera_rejects():
    # The same wall, the same opening: its right corners overhang the building's east end by 1.0 m and its bottom
    # corners sit 1.0 m below ground. At 114 m the tolerance is 2.5 m; at 8 m it is the 0.5 m floor.
    prism = _prism()
    far, near = _camera(8, -119, 0.0), _camera(8, -13, 0.0)
    results = {}
    for name, cam in (("far", far), ("near", near)):
        mask = _mask(prism, cam)
        overhang = _on_south("overhang", cam, mask, cx=9.0, z0=3.0, w=4.0, h=1.5)
        low = _on_south("low", cam, mask, cx=3.0, z0=-1.0, w=2.0, h=4.0)
        results[name] = _place(prism, cam, mask, [overhang, low]).openings
    for r in results["far"]:
        assert r["corner_tolerance_m"] > 2.0 and r["corner_hits"] == 4 and r["decision"] == "ACCEPT", r["id"]
    for r in results["near"]:
        assert r["corner_tolerance_m"] == pytest.approx(0.5) and r["corner_hits"] == 2 and r["decision"] == "REVIEW"
        assert any("0.5 m below ground" in why or "past its ends" in why for why in r["reasons"])


def test_the_roof_limit_is_not_widened_by_the_tolerance():
    prism, cam = _prism(), _camera(8, -119, 0.0)  # a wide tolerance (2.5 m) that must not lift the roof limit
    mask = _mask(prism, cam)
    r, = _place(prism, cam, mask, [_on_south("over_roof", cam, mask, cx=3.0, z0=6.0, w=2.0, h=3.0)]).openings
    assert r["corner_hits"] == 2 and r["decision"] == "REVIEW"  # the top corners (z = 9 m) are above the 8 m roof


def test_the_stored_camera_carries_the_photos_field_of_view():
    prism, cam = _prism(), _camera(0, -40, 0.0)
    mask = _mask(prism, cam)
    d = _place(prism, cam, mask, [_on_south("w1", cam, mask, cx=0.0, z0=3.0)]).fit.to_dict()
    assert d["image_size"] == [320, 240]
    assert d["hfov_deg"] == pytest.approx(math.degrees(2 * math.atan(160 / 300)))  # 55.9 deg
    assert d["vfov_deg"] == pytest.approx(math.degrees(2 * math.atan(120 / 300)))  # 43.6 deg
    assert d["camera_yaw_deg"] == 0.0 and d["camera_position_enu"] == [0, -40, 1.5]
