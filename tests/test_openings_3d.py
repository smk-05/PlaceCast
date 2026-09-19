"""perception/openings_3d.py on synthetic meshes: ray-cast placement, wall snapping, bearing, glTF export.

The mesh is a glTF-convention box (+Y up, front +Z) so every expectation can be worked out by hand.
"""
import math

import numpy as np
import pytest
import trimesh

from contracts import CandidateId, FitResult, MeshOutline
from geo import outline, placement
from geo.coords import theta_to_heading
from geo.disambiguate import facade_heading
from perception import openings_3d as o3
from perception.render_compare import render_silhouette

UP = (0, 1, 0)
CUBE = 6.0


def _cube(size=CUBE):
    return trimesh.creation.box(extents=(size, size, size))


def _opening(name, uv, half=0.05, type_="window", decision="ACCEPT"):
    u, v = uv
    return {
        "id": name, "type": type_, "decision": decision, "reasons": [], "score": 0.5, "group_margin": 0.6,
        "box_uv": [u - half, v - half, u + half, v + half], "center_uv": [u, v],
    }


def _cast(mesh, openings, azimuth, elevation=0.0, mask_wh=(1.0, 1.0), m2e=None):
    return o3.cast_openings(mesh, openings, o3.Camera(UP, azimuth, elevation, 1.0), mask_wh, m2e, "synthetic.jpg")


def _m2e(theta, scale=(1.0, 1.0, 1.0)):
    """mesh -> ENU exactly as pipeline.py builds it for a +Y-up mesh: geo.placement over geo.outline's canonical frame."""
    fit = FitResult(theta=theta, scale_x=scale[0], scale_y=scale[1], scale_z=scale[2], tx=0.0, ty=0.0, z_offset=0.0)
    pre = placement.mesh_to_canonical(outline.canonical_rotation(2), np.zeros(3))
    return placement.mesh_to_enu(fit, pre)


# --------------------------------------------------------------- placement


def test_frontal_camera_places_opening_on_the_front_wall():
    op = _opening("w1", (0.5, 0.5), half=0.1)
    (r,) = _cast(_cube(), [op], azimuth=0.0)
    half = CUBE / 2
    assert r["hit_mesh"] == pytest.approx([0.0, 0.0, half], abs=1e-6)  # centre of the +Z face
    assert r["normal_mesh_raw"] == pytest.approx([0, 0, 1], abs=1e-9)
    assert r["normal_mesh"] == pytest.approx([0, 0, 1], abs=1e-9)
    assert r["snap_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert r["position_mesh"] == pytest.approx([0.0, 0.0, half + o3.OFFSET_M], abs=1e-6)  # 2 cm outward
    assert r["width_m"] == pytest.approx(0.2 * CUBE, abs=1e-6)
    assert r["height_m"] == pytest.approx(0.2 * CUBE, abs=1e-6)
    assert r["decision"] == "ACCEPT" and r["reasons"] == []
    assert r["source_photo"] == "synthetic.jpg"


def test_off_centre_opening_lands_at_its_uv():
    # uv (0.75, 0.25): three quarters across to the right (+X on the front wall), a quarter down from the top.
    (r,) = _cast(_cube(), [_opening("w1", (0.75, 0.25))], azimuth=0.0)
    assert r["hit_mesh"] == pytest.approx([0.25 * CUBE, 0.25 * CUBE, CUBE / 2], abs=1e-6)


def test_corner_camera_puts_left_and_right_openings_on_different_walls():
    # Camera at azimuth 45 sits over the +X/+Z edge, which projects to the middle of the silhouette. With
    # camera_basis's right = (+X - Z)/sqrt(2), left of centre is the +Z wall and right of centre is the +X wall.
    left, right = _opening("left", (0.25, 0.5)), _opening("right", (0.75, 0.5))
    a, b = _cast(_cube(), [left, right], azimuth=45.0)
    assert a["normal_mesh"] == pytest.approx([0, 0, 1], abs=1e-9)
    assert b["normal_mesh"] == pytest.approx([1, 0, 0], abs=1e-9)
    assert a["hit_mesh"][2] == pytest.approx(CUBE / 2) and b["hit_mesh"][0] == pytest.approx(CUBE / 2)
    assert a["snap_angle_deg"] == pytest.approx(0.0, abs=1e-6) and b["snap_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert a["decision"] == b["decision"] == "ACCEPT"


def test_pitched_camera_still_hits_the_wall_and_flips_nothing():
    (r,) = _cast(_cube(), [_opening("w1", (0.5, 0.5))], azimuth=0.0, elevation=15.0)
    assert r["normal_mesh"] == pytest.approx([0, 0, 1], abs=1e-9)
    assert r["snap_angle_deg"] == pytest.approx(0.0, abs=1e-6)


# -------------------------------------------------------------------- snap


def test_snap_picks_the_nearest_wall_and_reports_the_angle():
    walls = o3.wall_directions(_cube(), UP)
    assert {tuple(np.round(w, 6)) for w in walls} == {(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)}
    n = np.array([math.sin(math.radians(20)), 0.0, math.cos(math.radians(20))])  # 20 deg off +Z toward +X
    snapped, angle, _ = o3.snap_normal(n, UP, walls)
    assert snapped == pytest.approx([0, 0, 1], abs=1e-9)
    assert angle == pytest.approx(20.0, abs=1e-6)
    _, angle, _ = o3.snap_normal([math.sin(math.radians(40)), 0.0, math.cos(math.radians(40))], UP, walls)
    assert angle == pytest.approx(40.0, abs=1e-6)  # over SNAP_REVIEW_DEG: cast_openings turns this into REVIEW


def test_a_roof_hit_is_not_a_wall():
    walls = o3.wall_directions(_cube(), UP)
    _, angle, frac = o3.snap_normal([0.0, 1.0, 0.0], UP, walls, toward_camera=[0, 0.3, 1])
    assert angle == 90.0 and frac == pytest.approx(0.0)


def test_wall_directions_follow_a_rotated_mesh():
    mesh = trimesh.creation.box(extents=(10, 4, 6))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(30), UP))
    walls = o3.wall_directions(mesh, UP)
    expect = trimesh.transformations.rotation_matrix(math.radians(30), UP)[:3, :3] @ [0, 0, 1]
    assert max(walls @ expect) == pytest.approx(1.0, abs=1e-6)


def test_snap_over_thirty_degrees_sends_the_opening_to_review():
    # A camera pitched to look almost straight down at a box sees the roof: not a wall.
    (r,) = _cast(_cube(), [_opening("roof", (0.5, 0.5))], azimuth=0.0, elevation=80.0)
    assert r["snap_angle_deg"] > o3.SNAP_REVIEW_DEG
    assert r["decision"] == "REVIEW"
    assert any("wall-snap angle" in why for why in r["reasons"])


# ------------------------------------------------------------------ misses


def test_a_missed_ray_is_review_with_a_reason():
    off = _opening("off", (1.4, 0.5), half=0.02)  # well right of the silhouette
    partial = _opening("edge", (0.98, 0.5), half=0.05)  # centre on the wall, right corners off the edge
    a, b = _cast(_cube(), [off, partial], azimuth=0.0)
    assert a["decision"] == "REVIEW" and a["position_mesh"] is None and a["bearing_deg"] is None
    assert any("centre ray missed" in why for why in a["reasons"])
    assert b["decision"] == "REVIEW" and b["position_mesh"] is not None and b["width_m"] is None
    assert any("corner rays missed" in why for why in b["reasons"])


def test_upstream_review_reasons_survive_and_other_is_skipped():
    weak = _opening("weak", (0.5, 0.5), decision="REVIEW")
    weak["reasons"] = ["weak detection: combined score 0.13 < 0.2"]
    rows = _cast(_cube(), [weak, _opening("x", (0.3, 0.3), type_="other")], azimuth=0.0)
    assert [r["id"] for r in rows] == ["weak"]
    assert rows[0]["decision"] == "REVIEW" and rows[0]["reasons"] == ["weak detection: combined score 0.13 < 0.2"]


# ------------------------------------------------------------------- bearing


@pytest.mark.parametrize("theta_deg", [0.0, 90.0, 33.0, -140.0])
def test_bearing_of_the_front_wall_matches_geo_facade_heading(theta_deg):
    theta = math.radians(theta_deg)
    mo = MeshOutline(pts_enu=np.zeros((3, 2)), up_axis_idx=2, front_angle=outline.front_angle_canonical(2))
    expected = facade_heading(None, mo, CandidateId(2, 0), theta=theta)  # the +Z wall is the mesh front
    (r,) = _cast(_cube(), [_opening("w1", (0.5, 0.5))], azimuth=0.0, m2e=_m2e(theta))
    assert r["bearing_deg"] == pytest.approx(expected, abs=1e-6)
    assert r["bearing_deg"] == pytest.approx(math.degrees(theta_to_heading(theta - math.pi / 2)), abs=1e-6)


def test_known_rotation_gives_expected_bearings():
    # theta = 0: canonical +X is East, and the glTF front (+Z) lands on canonical -Y = South.
    # theta = 90 deg (CCW): the front now faces East and the mesh's +X wall faces North.
    front, side = _opening("front", (0.5, 0.5)), _opening("side", (0.75, 0.5))
    for theta_deg, front_bearing, side_bearing in [(0, 180, 90), (90, 90, 0)]:
        m2e = _m2e(math.radians(theta_deg))
        a = _cast(_cube(), [front], azimuth=0.0, m2e=m2e)[0]
        assert a["bearing_deg"] == pytest.approx(front_bearing, abs=1e-6)
        b = _cast(_cube(), [side], azimuth=45.0, m2e=m2e)[0]  # from the corner camera the +X wall is right of centre
        assert b["normal_mesh"] == pytest.approx([1, 0, 0], abs=1e-9)
        assert b["bearing_deg"] == pytest.approx(side_bearing, abs=1e-6)


def test_metres_come_from_mesh_to_enu_scale():
    m2e = _m2e(0.0, scale=(2.0, 3.0, 0.5))  # sx, sy are plan scales (canonical X, Y), sz the height scale
    (r,) = _cast(_cube(), [_opening("w1", (0.5, 0.5), half=0.1)], azimuth=0.0, m2e=m2e)
    assert r["width_m"] == pytest.approx(0.2 * CUBE * 2.0, abs=1e-6)  # glTF X -> canonical X (x2)
    assert r["height_m"] == pytest.approx(0.2 * CUBE * 0.5, abs=1e-6)  # glTF Y (up) -> canonical Z (x0.5)
    offset = np.linalg.norm(np.asarray(r["position_mesh"]) - r["hit_mesh"])
    assert offset * 3.0 == pytest.approx(o3.OFFSET_M, abs=1e-9)  # glTF Z -> canonical -Y (x3): still 2 cm in metres
    assert r["bearing_deg"] == pytest.approx(180.0, abs=1e-6)


# -------------------------------------------------------------------- camera


def _asymmetric_building():
    body = trimesh.creation.box(extents=(20, 6, 6))
    wing = trimesh.creation.box(extents=(4, 3, 10))
    wing.apply_translation((8.0, -1.5, 3.0))  # a low wing on the +X end, sticking out toward +Z
    return trimesh.util.concatenate([body, wing])


@pytest.mark.parametrize("azimuth, elevation", [(0.0, 0.0), (42.0, 0.0), (-24.0, 10.0)])
def test_fit_camera_recovers_the_view_that_made_the_mask(azimuth, elevation):
    # Azimuths are chosen off the silhouette's stationary points (a 20 x 6 body's width is flat near 17 deg), where
    # the normalised silhouette cannot tell neighbouring azimuths apart at any resolution.
    mesh = _asymmetric_building()
    b_level = np.stack(o3.camera_basis(UP, math.radians(azimuth)), axis=1)
    b_pitched = np.stack(o3.camera_frame(UP, azimuth, elevation), axis=1)  # the photo, rendered pitched
    tilted = trimesh.Trimesh(mesh.vertices @ (b_level @ b_pitched.T).T, mesh.faces, process=False)
    photo = render_silhouette(tilted, UP, math.radians(azimuth), size=256)
    cam = o3.fit_camera(mesh, photo, UP)
    assert abs((cam.azimuth_deg - azimuth + 180) % 360 - 180) <= 3.0
    assert cam.elevation_deg == pytest.approx(elevation, abs=5.0)
    assert cam.iou > 0.9


# ---------------------------------------------------------------------- glTF


def _write_building_glb(path, rotation_deg=25.0):
    """A cube under a node that carries a rotation and translation, so child transforms must be parent-relative."""
    scene = trimesh.Scene()
    node_matrix = trimesh.transformations.rotation_matrix(math.radians(rotation_deg), UP)
    node_matrix[:3, 3] = (5.0, 1.0, -2.0)
    scene.add_geometry(_cube(), node_name="building", geom_name="building_mesh", transform=node_matrix)
    path.write_bytes(scene.export(file_type="glb"))
    return node_matrix


def test_glb_gets_one_coloured_child_node_per_opening_with_extras(tmp_path):
    from pygltflib import GLTF2

    src, out = tmp_path / "mesh.glb", tmp_path / "out.glb"
    _write_building_glb(src)
    scene_mesh = trimesh.load(str(src), force="scene")
    baked = trimesh.util.concatenate(scene_mesh.dump())  # the frame place_openings works in
    ops = [
        _opening("window_1", (0.3, 0.4)), _opening("door_1", (0.6, 0.6), type_="door"),
        _opening("garage_1", (0.5, 0.5), type_="garage_door"), _opening("ent_1", (0.4, 0.5), type_="entrance"),
        _opening("weak_1", (0.7, 0.3), decision="REVIEW"), _opening("other_1", (0.5, 0.5), type_="other"),
        _opening("miss_1", (1.5, 0.5)),
    ]
    ops[4]["reasons"] = ["weak detection: combined score 0.13 < 0.2"]
    camera = o3.Camera(UP, 0.0, 0.0, 1.0)  # the baked cube is turned 25 deg: walls are then off-axis, still snapped
    placement_ = o3.place_openings(baked, _rect_mask(baked, camera), ops, mesh_to_enu=_m2e(0.4), camera=camera,
                                   source_photo="p.jpg")
    o3.export_openings_glb(src, placement_.openings, out)

    gltf = GLTF2().load(str(out))
    names = {n.name: n for n in gltf.nodes}
    parent = names["building"]
    kids = {gltf.nodes[i].name for i in parent.children}
    assert kids == {"window_1", "door_1", "garage_1", "ent_1", "weak_1"}  # not other_1, not the missed one
    for name in kids:
        extras = names[name].extras
        assert {"type", "decision", "score", "group_margin", "normal_mesh", "bearing_deg", "width_m", "height_m",
                "source_photo"} <= set(extras)
        assert extras["source_photo"] == "p.jpg"
    assert names["weak_1"].extras["decision"] == "REVIEW"
    assert names["weak_1"].extras["reasons"] == ["weak detection: combined score 0.13 < 0.2"]
    assert "reasons" not in names["door_1"].extras  # pygltflib omits null and empty extras; absent means none

    def rgb(name):  # marker colour: the material's baseColorFactor or the vertex COLOR_0
        prim = gltf.meshes[names[name].mesh].primitives[0]
        acc = gltf.accessors[prim.attributes.COLOR_0]
        return tuple(round(c * 255) for c in _read_first_colour(gltf, acc)[:3])

    assert rgb("door_1") == (46, 160, 67) and rgb("ent_1") == (0, 150, 150)
    assert rgb("garage_1") == (245, 140, 20) and rgb("window_1") == (40, 110, 230) and rgb("weak_1") == (220, 40, 40)

    # World placement survives the parent's rotation: the marker sits where the ray hit the (baked) building.
    world = trimesh.load(str(out), force="scene")
    got = {n: world.graph.get(n)[0] for n in kids}
    by_id = {r["id"]: r for r in placement_.openings}
    for name in kids:
        assert got[name][:3, 3] == pytest.approx(by_id[name]["position_mesh"], abs=1e-4)
        assert got[name][:3, 2] == pytest.approx(by_id[name]["normal_mesh"], abs=1e-6)  # local +Z is the snapped normal


def _rect_mask(mesh, camera):
    return o3.render_silhouette(mesh, camera.up_axis, math.radians(camera.azimuth_deg), size=128)


def _read_first_colour(gltf, acc):
    view = gltf.bufferViews[acc.bufferView]
    data = gltf.binary_blob()[view.byteOffset + (acc.byteOffset or 0):]
    comps = {"VEC3": 3, "VEC4": 4}[acc.type]
    if acc.componentType == 5126:
        return np.frombuffer(data, np.float32, comps)
    scale = {5121: 255.0, 5123: 65535.0}[acc.componentType]
    return np.frombuffer(data, {5121: np.uint8, 5123: np.uint16}[acc.componentType], comps) / scale


def test_record_gets_the_same_openings(tmp_path):
    import json

    rec = tmp_path / "record.json"
    rec.write_text(json.dumps({"asset_id": "x"}))
    plc = o3.place_openings(_cube(), _rect_mask(_cube(), o3.Camera(UP, 0.0, 0.0, 1.0)),
                            [_opening("w1", (0.5, 0.5))], camera=o3.Camera(UP, 0.0, 0.0, 0.87), source_photo="p.jpg")
    data = o3.attach_to_record(rec, plc)
    assert json.loads(rec.read_text()) == data
    assert data["openings"][0]["id"] == "w1" and "_depth_mesh" not in data["openings"][0]
    assert data["openings_camera"]["camera_iou"] == pytest.approx(0.87)


def test_marker_thickness_and_offset_are_metres_on_a_unit_scale_mesh(tmp_path):
    # TRELLIS normalises the building to ~1 unit; mesh_to_enu scales it to tens of metres, and not uniformly.
    # 0.1 m thickness and the 2 cm offset are metres, so in mesh units they are ~0.003 / ~0.0007: check them
    # in the exported glb's world size, not only in the result dict.
    scale = (30.0, 20.0, 12.0)  # plan X, plan Y, height; the front (+Z) wall's normal maps to canonical -Y (x20)
    m2e = _m2e(math.radians(40.0), scale)
    cube = _cube(1.0)
    src, out = tmp_path / "mesh.glb", tmp_path / "out.glb"
    scene = trimesh.Scene()
    scene.add_geometry(cube, node_name="building", geom_name="building_mesh")
    src.write_bytes(scene.export(file_type="glb"))

    ops = [_opening("front", (0.5, 0.5), half=0.1), _opening("side", (0.75, 0.5), half=0.05)]
    front = o3.place_openings(cube, _rect_mask(cube, o3.Camera(UP, 0.0, 0.0, 1.0)), [ops[0]], m2e,
                              camera=o3.Camera(UP, 0.0, 0.0, 1.0))
    side = o3.place_openings(cube, _rect_mask(cube, o3.Camera(UP, 45.0, 0.0, 1.0)), [ops[1]], m2e,
                             camera=o3.Camera(UP, 45.0, 0.0, 1.0))
    placed = front.openings + side.openings
    o3.export_openings_glb(src, placed, out)

    world = trimesh.load(str(out), force="scene")
    m3 = m2e[:3, :3]
    for r in placed:
        assert r["_depth_mesh"] < 0.01  # a mesh-unit depth of 0.1 would be 2 m thick on this building
        # offset from the hit, in metres
        assert np.linalg.norm(m3 @ (np.asarray(r["position_mesh"]) - r["hit_mesh"])) == pytest.approx(o3.OFFSET_M, abs=1e-9)
        # marker world size: each local axis, times its box extent, through the marker's node and mesh_to_enu
        transform, geom_name = world.graph[r["id"]]
        extents = world.geometry[geom_name].extents
        size_m = [np.linalg.norm(m3 @ transform[:3, i]) * extents[i] for i in range(3)]
        assert size_m == pytest.approx([r["width_m"], r["height_m"], o3.MARKER_DEPTH_M], rel=1e-6)
    assert placed[0]["width_m"] == pytest.approx(0.2 * 30.0) and placed[0]["height_m"] == pytest.approx(0.2 * 12.0)
    # 45 deg camera: the silhouette is sqrt(2) wide and the wall foreshortens by cos 45, so a 0.1-uv box is 0.2 wide;
    # the +X wall's marker runs along glTF Z -> canonical -Y (x20)
    assert placed[1]["width_m"] == pytest.approx(0.2 * 20.0)
