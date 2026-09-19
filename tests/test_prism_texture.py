"""perception/prism_texture.py on a synthetic box prism, a known camera and a rendered checkerboard.

The prism is 20 m east-west by 10 m north-south and 8 m tall, centred on the ENU origin. The camera stands 40 m south
of it looking north, so it sees only the south wall (y = -5, outward normal south). The photo is rendered by casting
every pixel's ray onto that wall and colouring the point from a 2 m checkerboard in wall coordinates, so the baked
texture has an exactly known answer.
"""
import json

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box

from geo.coords import ENUFrame
from perception import openings_prism as op
from perception import prism_texture as pt

W, H = 320, 240
F_PX = 300.0
WHITE, BLACK, SKY = 235, 25, (60, 120, 200)


def _prism(polys=None, height=8.0):
    return op.prism_from_polygons(polys or [box(-10, -5, 10, 5)], height, ENUFrame(37.2, -80.4, 0.0))


def _camera(x=0.0, y=-40.0, yaw=0.0, z=1.5):
    return op.PinholeCamera((x, y, z), yaw, 0.0, F_PX, W, H)


def _checker(x, z):
    return (np.floor((x + 10.0) / 2.0) + np.floor(z / 2.0)) % 2 == 0


def _photo(cam):
    """(image (H, W, 3) uint8, mask (H, W) bool): the checkerboard south wall as the camera sees it."""
    ys, xs = np.mgrid[0:H, 0:W]
    d = cam.rays(xs.ravel() + 0.5, ys.ravel() + 0.5)
    o = np.array(cam.position)
    t = (-5.0 - o[1]) / d[:, 1]
    p = o + d * t[:, None]
    on = (t > 0) & (np.abs(p[:, 0]) <= 10) & (p[:, 2] >= 0) & (p[:, 2] <= 8)
    grey = np.where(_checker(p[:, 0], p[:, 2]), WHITE, BLACK).astype(np.uint8)
    img = np.empty((H * W, 3), np.uint8)
    img[:] = SKY
    img[on] = grey[on][:, None]
    return img.reshape(H, W, 3), on.reshape(H, W)


def _south(prism):
    return next(w for w in prism.walls if abs(w["bearing_deg"] - 180.0) < 1e-6)["index"]


def _texel_xz(prism, wall_index, tex):
    """Wall coordinates (x along the wall, z up) of every texel centre of a baked wall texture (south wall: x is east)."""
    wall = prism.walls[wall_index]
    h, w = tex.visible.shape
    s = (np.arange(w) + 0.5) / w
    x = wall["p0"][0] + s * (wall["p1"][0] - wall["p0"][0])
    z = prism.height_m * (1.0 - (np.arange(h) + 0.5) / h)
    return np.meshgrid(x, z)


# ------------------------------------------------------------------ baking


def test_the_baked_south_wall_matches_the_rectified_checkerboard():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, mask)
    tex = bake.walls[_south(prism)]
    assert tex.source == "photo" and tex.coverage > 0.8  # only the eroded mask border is lost
    xx, zz = _texel_xz(prism, tex.index, tex)
    # Ignore texels within 3 texels of a checker line: 2.7x magnification blurs an edge across a few of them.
    tol = 3.0 / bake.ppm
    fx, fz = (xx + 10.0) / 2.0, zz / 2.0
    near_edge = (np.minimum(fx % 1, 1 - fx % 1) * 2 < tol) | (np.minimum(fz % 1, 1 - fz % 1) * 2 < tol)
    sel = tex.visible & ~near_edge
    baked_white = tex.rgb[..., 0] > 130
    assert sel.sum() > 0.6 * tex.visible.size
    assert (baked_white[sel] == _checker(xx, zz)[sel]).mean() > 0.98
    assert tex.rgb[..., 0][sel & _checker(xx, zz)].mean() == pytest.approx(WHITE, abs=8)
    assert tex.rgb[..., 0][sel & ~_checker(xx, zz)].mean() == pytest.approx(BLACK, abs=8)


def test_walls_facing_away_are_not_visible_and_take_a_darker_donor():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, mask)
    south = bake.walls[_south(prism)]
    others = [t for t in bake.walls if t.index != south.index]
    assert len(others) == 3
    for t in others:
        assert not t.visible.any() and t.coverage == 0.0  # the camera is behind or edge-on to them
        assert t.source == "donor" and t.donor == south.index
        assert t.rgb.mean() / south.rgb.mean() == pytest.approx(pt.DARKEN, abs=0.03)  # darker, not blank


def test_a_wall_needs_a_view_to_be_baked_at_all():
    prism, cam = _prism(), _camera()
    img, _ = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, np.zeros((H, W), bool))  # nothing is "building"
    assert all(t.source == "neutral" and not t.visible.any() for t in bake.walls)
    assert len({tuple(t.rgb.reshape(-1, 3)[0]) for t in bake.walls}) == 1  # one neutral colour, no garbage


def test_texels_behind_a_hole_in_the_mask_are_not_visible():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    tree = mask.copy()
    tree[100:140, 150:175] = False  # a tree trunk / lamp post standing in front of the wall
    tex = pt.bake_textures(prism, cam, img, tree).walls[_south(prism)]
    xx, zz = _texel_xz(prism, tex.index, tex)
    px = cam.project(np.column_stack([xx.ravel(), np.full(xx.size, -5.0), zz.ravel()]))
    inside = ((px[:, 0] >= 150) & (px[:, 0] < 175) & (px[:, 1] >= 100) & (px[:, 1] < 140)).reshape(xx.shape)
    clear = ((px[:, 0] < 150 - 12) | (px[:, 0] > 175 + 12) | (px[:, 1] < 100 - 12) | (px[:, 1] > 140 + 12)).reshape(
        xx.shape)
    interior = tex.visible | inside  # ignore the eroded outer border of the whole wall
    assert not tex.visible[inside].any()
    assert tex.visible[clear & (interior)].size > 0
    ok = pt.bake_textures(prism, cam, img, mask).walls[_south(prism)].visible
    assert tex.visible[clear].mean() == pytest.approx(ok[clear].mean(), abs=1e-9)  # nothing else changed


def test_a_wall_behind_another_building_is_occluded_texel_by_texel():
    back = box(-10, 0, 10, 10)
    front = box(-12, -20, 0, -10)  # stands between the camera (0, -40) and the west half of the back wall
    prism = _prism([back, front])
    cam = _camera()
    mask = op._Rasteriser(prism, W, H).render(cam)
    img = np.full((H, W, 3), 120, np.uint8)
    bake = pt.bake_textures(prism, cam, img, mask)
    wall = next(w for w in prism.walls if abs(w["p0"][1] - 0.0) < 1e-9 and abs(w["p1"][1] - 0.0) < 1e-9
                and abs(w["bearing_deg"] - 180.0) < 1e-6)  # the back building's south wall (y = 0)
    vis = bake.walls[wall["index"]].visible
    x = -10 + 20 * (np.arange(vis.shape[1]) + 0.5) / vis.shape[1]
    west, east = vis[:, x < -0.5], vis[:, x > 0.5]
    assert west.mean() == 0.0  # every west texel's sight line crosses the front building
    assert east.mean() > 0.5  # the east half is in plain view (minus the eroded mask border)


def test_wall_visibility_is_geometry_only_and_needs_the_front_side():
    prism, cam = _prism(), _camera()
    south = prism.walls[_south(prism)]
    size = pt.wall_size(south["length_m"], prism.height_m, 10.0)
    assert pt.wall_visibility(prism, cam, south, size).all()
    behind = _camera(y=+40.0, yaw=180.0)  # the camera is on the north side
    assert not pt.wall_visibility(prism, behind, south, size).any()


def test_atlas_layout_fits_and_shrinks_pixels_per_metre_for_a_big_building():
    big = _prism([box(-250, -100, 250, 100)], 30.0)
    ppm, sizes = pt.choose_ppm(big)
    assert ppm < pt.PPM and max(max(s) for s in sizes) <= pt.MAX_SIDE
    pos, (aw, ah) = pt.plan_layout(sizes)
    assert aw <= pt.ATLAS_MAX and ah <= pt.ATLAS_MAX
    small_ppm, _ = pt.choose_ppm(_prism())
    assert small_ppm == pt.PPM


# ----------------------------------------------------------------------- glb


def _accessor(gltf, index):
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc.type]
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[acc.componentType]
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    data = np.frombuffer(gltf.binary_blob(), dtype, acc.count * comps, offset)
    return data.reshape(acc.count, comps)


def _glb(tmp_path, prism=None, cam=None):
    from pygltflib import GLTF2

    prism, cam = prism or _prism(), cam or _camera()
    img, mask = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, mask)
    out = pt.build_glb(prism, bake, tmp_path / "prism_photo.glb", source="synthetic.png")
    return prism, bake, GLTF2().load(str(out))


def test_glb_is_in_enu_with_the_walls_and_roof_materials(tmp_path):
    prism, bake, gltf = _glb(tmp_path)
    by_name = {n.name: n for n in gltf.nodes}
    assert {"walls", "roof"} <= set(by_name)
    walls_prim = gltf.meshes[by_name["walls"].mesh].primitives[0]
    roof_prim = gltf.meshes[by_name["roof"].mesh].primitives[0]
    pos = _accessor(gltf, walls_prim.attributes.POSITION)
    assert pos.min(axis=0) == pytest.approx([-10, -5, 0]) and pos.max(axis=0) == pytest.approx([10, 5, 8])  # ENU metres
    roof = _accessor(gltf, roof_prim.attributes.POSITION)
    assert roof[:, 2] == pytest.approx(8.0)
    wall_mat, roof_mat = gltf.materials[walls_prim.material], gltf.materials[roof_prim.material]
    wp, rp = wall_mat.pbrMetallicRoughness, roof_mat.pbrMetallicRoughness
    assert (wp.metallicFactor, wp.roughnessFactor) == pytest.approx((0.4, 0.75))
    assert wp.baseColorTexture is not None and len(gltf.images) == 1
    assert rp.baseColorTexture is None and rp.baseColorFactor[:3] == pytest.approx(pt.ROOF_RGBA[:3], abs=4e-3)  # 8-bit
    assert max(rp.baseColorFactor[:3]) < 0.25  # dark
    assert "ENU" in by_name["walls"].extras["frame"] and by_name["roof"].extras["source_image"] == "synthetic.png"


def test_glb_uvs_address_each_walls_rectangle_in_the_atlas(tmp_path):
    prism, bake, gltf = _glb(tmp_path)
    by_name = {n.name: n for n in gltf.nodes}
    prim = gltf.meshes[by_name["walls"].mesh].primitives[0]
    pos, uv = _accessor(gltf, prim.attributes.POSITION), _accessor(gltf, prim.attributes.TEXCOORD_0)
    a_h, a_w = bake.atlas.shape[:2]
    for wall in prism.walls:
        i = wall["index"]
        x, y, w, h = bake.rects[i]
        corners = pos[4 * i : 4 * i + 4]  # p0-bottom, p1-bottom, p1-top, p0-top
        assert corners[2, 2] == pytest.approx(prism.height_m) and corners[0, 2] == 0.0
        # glTF UVs: origin top-left, v down. p0-top is the texture's top-left corner.
        assert uv[4 * i + 3] * [a_w, a_h] == pytest.approx([x, y], abs=0.01)
        assert uv[4 * i + 1] * [a_w, a_h] == pytest.approx([x + w, y + h], abs=0.01)
        # ... and the atlas texel there really is that wall's top-left texel
        assert (bake.atlas[y, x] == bake.walls[i].rgb[0, 0]).all()
        assert (bake.atlas[y + h - 1, x + w - 1] == bake.walls[i].rgb[-1, -1]).all()


def test_wall_triangles_face_outward(tmp_path):
    prism, bake, gltf = _glb(tmp_path)
    prim = gltf.meshes[{n.name: n for n in gltf.nodes}["walls"].mesh].primitives[0]
    pos, idx = _accessor(gltf, prim.attributes.POSITION), _accessor(gltf, prim.indices).ravel()
    for tri in idx.reshape(-1, 3):
        a, b, c = pos[tri]
        normal = np.cross(b - a, c - a)
        mid = (a + b + c) / 3
        assert np.dot(normal[:2], mid[:2]) > 0  # away from the prism's centre at the origin


# -------------------------------------------------------------------- record


def _run_dir(tmp_path, reliable=True):
    frame = ENUFrame(37.2, -80.4, 0.0)
    ring = np.array([(-10, -5), (10, -5), (10, 5), (-10, 5), (-10, -5)], float)
    lat, lon, _ = frame.enu_to_geodetic(ring[:, 0], ring[:, 1], 0.0).T
    cam = _camera()
    record = {
        "asset_id": "synthetic", "enu_origin_geodetic": [37.2, -80.4, 0.0], "height": {"value_m": 8.0},
        "footprint_geojson": {"type": "Polygon", "coordinates": [[[lo, la] for la, lo in zip(lat, lon)]]},
        "openings_camera": {"camera_position_enu": list(cam.position), "camera_yaw_deg": 0.0,
                            "camera_pitch_deg": 0.0, "f_px": F_PX, "reliable": reliable},
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / "record.json").write_text(json.dumps(record))
    img, mask = _photo(cam)
    Image.fromarray(img).save(tmp_path / "photo.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(tmp_path / "mask.png")
    return run


def test_bake_record_writes_the_glbs_previews_and_lists_them(tmp_path):
    run = _run_dir(tmp_path)
    Image.fromarray(np.full((H, W, 3), 90, np.uint8)).save(tmp_path / "edit.png")
    rec = pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png", scorched=tmp_path / "edit.png",
                         scorched_mask=np.ones((H, W), bool))
    assert rec["textured_glbs"] == {"photo": "prism_photo.glb", "scorched": "prism_scorched.glb"}
    for name in ("prism_photo.glb", "prism_scorched.glb", "prism_photo_atlas.png", "prism_photo_atlas_visibility.png",
                 "prism_scorched_atlas.png"):
        assert (run / name).is_file(), name
    on_disk = json.loads((run / "record.json").read_text())
    assert on_disk["textured_glbs"] == rec["textured_glbs"] and on_disk["texture_bake"]["photo"]["walls"]


def test_a_missing_scorched_edit_is_skipped_not_an_error(tmp_path):
    run = _run_dir(tmp_path)
    rec = pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png", scorched=tmp_path / "not_yet.png")
    assert rec["textured_glbs"] == {"photo": "prism_photo.glb"} and not (run / "prism_scorched.glb").exists()


def test_the_scorched_image_may_be_a_different_resolution(tmp_path):
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    half = np.array(Image.fromarray(img).resize((W // 2, H // 2), Image.BILINEAR))  # an edit at half size
    a = pt.bake_textures(prism, cam, img, mask).walls[_south(prism)].rgb.astype(float)
    b = pt.bake_textures(prism, cam, half, mask).walls[_south(prism)].rgb.astype(float)
    assert np.median(np.abs(a - b)) < 3  # same picture, just softer (the mean is dominated by checker edges)


def test_an_unreliable_camera_is_not_textured(tmp_path):
    run = _run_dir(tmp_path, reliable=False)
    with pytest.raises(ValueError, match="not reliable"):
        pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png")
    assert pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png", force=True)["textured_glbs"]


# --------------------------------------------------------------- edit mask


def _fringed_edit(cam):
    """An 'edit' whose building is a few pixels smaller than in the photo (a generated edit never lands exactly on
    the original silhouette) and whose background is white. -> (edit image, the photo's mask, the edit's own mask).
    Wall texels project into the photo's silhouette, so the ring between the two silhouettes is exactly where the
    original mask alone would let the edit's white background in."""
    import cv2

    img, tight = _photo(cam)
    smaller = cv2.erode(tight.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    edit = img.copy()
    edit[~smaller] = 255  # brighter than any checker square (235)
    return edit, tight, smaller


def test_an_edit_bakes_no_fringe_only_inside_both_masks():
    prism, cam = _prism(), _camera()
    edit, original_mask, edit_mask = _fringed_edit(cam)
    south = _south(prism)
    unmasked = pt.bake_textures(prism, cam, edit, original_mask).walls[south]  # the original mask alone
    both = pt.bake_textures(prism, cam, edit, original_mask, edit_mask=edit_mask).walls[south]
    assert (unmasked.rgb[unmasked.visible] == 255).any()  # the fringe leaks in without the edit's mask
    assert not (both.rgb[both.visible] == 255).any()  # ... and cannot with it
    assert both.visible.sum() < unmasked.visible.sum() and both.coverage > 0.6  # the border ring is now filled


def test_a_texel_outside_the_edits_mask_is_never_visible():
    prism, cam = _prism(), _camera()
    edit, original_mask, _ = _fringed_edit(cam)
    hole = np.ones((H, W), bool)
    hole[:, 140:190] = False  # the edit's own mask has a gap the original mask does not
    tex = pt.bake_textures(prism, cam, edit, original_mask, edit_mask=hole).walls[_south(prism)]
    xx, zz = _texel_xz(prism, tex.index, tex)
    px = cam.project(np.column_stack([xx.ravel(), np.full(xx.size, -5.0), zz.ravel()]))
    in_gap = ((px[:, 0] >= 140) & (px[:, 0] < 190)).reshape(xx.shape)
    assert not tex.visible[in_gap].any() and tex.visible[~in_gap].any()


def test_the_edit_mask_may_be_at_the_edits_resolution():
    prism, cam = _prism(), _camera()
    edit, original_mask, edit_mask = _fringed_edit(cam)
    half = np.array(Image.fromarray(edit).resize((W // 2, H // 2), Image.NEAREST))
    half_mask = np.array(Image.fromarray(edit_mask.astype(np.uint8) * 255).resize((W // 2, H // 2))) > 127
    tex = pt.bake_textures(prism, cam, half, original_mask, edit_mask=half_mask).walls[_south(prism)]
    assert tex.visible.mean() > 0.4  # the 9 px erosion is 9 px at HALF resolution here: a wide border is lost
    with pytest.raises(ValueError, match="edit_mask"):
        pt.bake_textures(prism, cam, half, original_mask, edit_mask=edit_mask)  # full-size mask for a half-size edit


def test_bake_record_uses_the_edit_mask_and_skips_the_scorched_bake_without_one(tmp_path, monkeypatch):
    run = _run_dir(tmp_path)
    edit, _, edit_mask = _fringed_edit(_camera())
    Image.fromarray(edit).save(tmp_path / "edit.png")
    Image.fromarray((edit_mask * 255).astype(np.uint8)).save(tmp_path / "edit_mask.png")
    rec = pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png", scorched=tmp_path / "edit.png",
                         scorched_mask=tmp_path / "edit_mask.png")
    assert rec["textured_glbs"]["scorched"] == "prism_scorched.glb" and rec["texture_bake"]["scorched"]["edit_mask"]
    assert (run / "prism_scorched_editmask.png").is_file()

    def no_segmenter(_path):
        raise RuntimeError("no weights on this machine")

    monkeypatch.setattr(pt, "segment_edit", no_segmenter)
    (tmp_path / "again").mkdir()
    run2 = _run_dir(tmp_path / "again")
    rec = pt.bake_record(run2, tmp_path / "again" / "photo.png", tmp_path / "again" / "mask.png",
                         scorched=tmp_path / "edit.png")
    assert "scorched" not in rec["textured_glbs"] and "no edit mask" in rec["texture_bake"]["scorched"]["skipped"]
    assert not (run2 / "prism_scorched.glb").exists()  # never baked with the original mask alone


# ------------------------------------------------------------- edit halo


def test_an_edits_mask_is_eroded_by_about_one_percent_of_the_building_width():
    assert pt.edit_erode_size(150) == pt.MASK_ERODE_PX  # small building: the fixed minimum wins
    assert pt.edit_erode_size(3873) == 39  # the Burruss mask: 1% of 3873 px, made odd
    assert pt.edit_erode_size(4000) % 2 == 1 and pt.edit_erode_size(4000) == 41
    # ... and it really is the eroded intersection that decides visibility: a wide building loses a wide border
    mask = np.zeros((60, 4000), bool)
    mask[10:50, 100:3900] = True
    both = pt._erode(mask, pt.edit_erode_size(pt._mask_width(mask)))
    assert np.nonzero(both.any(axis=0))[0][0] - 100 == pt.edit_erode_size(3800) // 2


def _painted_edit(cam):
    """A uniform mid-grey wall with three painted vertical stripes, all beyond the 9 px erosion:
    beige near the left boundary (halo), beige in the middle (a real light wall), red near the boundary (saturated)."""
    img, mask = _photo(cam)
    edit = img.copy()
    edit[mask] = 120
    left = np.nonzero(mask.any(axis=0))[0][0]
    stripes = {"halo": (left + 12, left + 22, (236, 226, 206)), "red": (left + 26, left + 36, (200, 40, 40)),
               "centre": (left + 70, left + 80, (236, 226, 206))}
    for x0, x1, colour in stripes.values():
        edit[:, x0:x1][mask[:, x0:x1]] = colour
    return edit, mask, stripes


def _stripe_columns(prism, cam, tex, stripes, mask):
    """Texels inside each stripe, 2 px clear of its blended edge and 30 px inside the mask's top and bottom, so they are
    farther from every boundary than the halo band (and clear of the ordinary mask erosion)."""
    xx, zz = _texel_xz(prism, tex.index, tex)
    px = cam.project(np.column_stack([xx.ravel(), np.full(xx.size, -5.0), zz.ravel()]))
    rows = np.nonzero(mask.any(axis=1))[0]
    inside = ((px[:, 1] > rows[0] + 30) & (px[:, 1] < rows[-1] - 30)).reshape(xx.shape)
    x = px[:, 0].reshape(xx.shape)
    return {name: (x >= x0 + 2) & (x < x1 - 2) & inside for name, (x0, x1, _) in stripes.items()}


def test_halo_coloured_texels_near_the_boundary_are_not_baked_but_the_same_colour_inside_is():
    prism, cam = _prism(), _camera()
    edit, mask, stripes = _painted_edit(cam)
    tex = pt.bake_textures(prism, cam, edit, mask, edit_mask=mask, halo_band_frac=0.15).walls[_south(prism)]
    cols = _stripe_columns(prism, cam, tex, stripes, mask)
    assert cols["halo"].any() and not tex.visible[cols["halo"]].any()  # beige and near the boundary: filled instead
    assert tex.visible[cols["centre"]].all()  # the same beige in the middle of the wall is real wall
    assert tex.visible[cols["red"]].all()  # near the boundary but saturated: kept
    assert not (tex.rgb[cols["halo"]] == (236, 226, 206)).all(axis=1).any()  # what was baked there is not the halo


def test_the_halo_rule_is_limited_to_the_band_and_to_edits():
    prism, cam = _prism(), _camera()
    edit, mask, stripes = _painted_edit(cam)
    narrow = pt.bake_textures(prism, cam, edit, mask, edit_mask=mask, halo_band_frac=0.01).walls[_south(prism)]
    cols = _stripe_columns(prism, cam, narrow, stripes, mask)
    assert narrow.visible[cols["halo"]].all()  # a 1% band does not reach the stripe
    photo = pt.bake_textures(prism, cam, edit, mask).walls[_south(prism)]  # no edit mask: this is a photo bake
    assert photo.visible[cols["halo"]].all()  # the rule never touches an original photo
