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
    import cv2

    deep = cv2.distanceTransform(np.pad(tex.visible, 1, constant_values=True).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    sel = tex.visible & ~near_edge & (deep > pt.BLEND_M * bake.ppm + 1)  # the photo fades out over its last 0.5 m
    baked_white = tex.rgb[..., 0] > 130
    assert sel.sum() > 0.4 * tex.visible.size
    assert (baked_white[sel] == _checker(xx, zz)[sel]).mean() > 0.98
    assert tex.rgb[..., 0][sel & _checker(xx, zz)].mean() == pytest.approx(WHITE, abs=8)
    assert tex.rgb[..., 0][sel & ~_checker(xx, zz)].mean() == pytest.approx(BLACK, abs=8)


def _seam_spacing_m(tex, ppm):
    """Distances (m) between the darker columns (vertical panel seams) of a wall texture."""
    cols = tex.rgb.astype(float).mean(axis=(0, 2))
    dark = np.nonzero(cols < 0.93 * np.median(cols))[0]
    starts = dark[np.insert(np.diff(dark) > 2, 0, True)]  # a 2 px seam is one group of columns
    return np.diff(starts) / ppm


def test_walls_facing_away_get_a_procedural_facade_in_the_buildings_colour():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, mask)
    south = bake.walls[_south(prism)]
    base = south.rgb[south.visible].mean(axis=0)
    assert bake.base_rgb == tuple(int(round(c)) for c in base)  # the mean colour of the visible wall texels
    others = [w for w in bake.walls if w.index != south.index]
    assert len(others) == 3
    for w in others:
        assert not w.visible.any() and w.coverage == 0.0 and w.source == "procedural"
        assert w.rgb.reshape(-1, 3).mean(axis=0) == pytest.approx(base, rel=0.06)  # the building's own colour
        assert w.rgb.std() / w.rgb.mean() < 0.12  # a plain facade with low noise, not the checkerboard
        spacing = _seam_spacing_m(w, bake.ppm)
        assert len(spacing) >= 2 and spacing.min() >= 1.4 and spacing.max() <= 2.1  # panel seams every 1.5-2 m
        # a floor line every 3.5 m up from the ground (z = 3.5 and 7.0 on an 8 m wall)
        rows = w.rgb.astype(float).mean(axis=(1, 2))
        for z in (3.5, 7.0):
            r = int(round((1 - z / prism.height_m) * len(rows)))
            assert rows[r - 1 : r + 2].min() < 0.92 * np.median(rows)


def test_unseen_walls_are_not_a_stretched_or_mirrored_copy_of_the_photo():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, mask)
    south = bake.walls[_south(prism)].rgb.astype(float)
    for w in bake.walls:
        if w.source != "procedural":
            continue
        n = min(south.shape[1], w.rgb.shape[1])
        for candidate in (south, south[:, ::-1], south[::-1]):  # the photographed wall, mirrored either way
            c = candidate[:, :n, 0]
            assert np.abs(w.rgb[:, :n, 0].astype(float) - c).mean() > 30  # nowhere near a copy (a checker is 25/235)


def test_a_photographed_wall_gets_a_procedural_band_above_and_below_its_coverage_blended_in():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    rows = np.nonzero(mask.any(axis=1))[0]
    mid = (rows[0] + rows[-1]) // 2
    band = mask.copy()
    band[: mid - 12] = False  # the photo covers only a strip: top and bottom of the wall are unseen
    band[mid + 12 :] = False
    bake = pt.bake_textures(prism, cam, img, band)
    tex = bake.walls[_south(prism)]
    assert tex.source in ("photo", "partial") and 0.15 < tex.coverage < 0.6
    seen_rows = np.nonzero(tex.visible.any(axis=1))[0]
    top_band = tex.rgb[: max(1, seen_rows[0] - int(1.0 * bake.ppm))]  # more than 1 m above the coverage
    assert len(top_band) > 4
    assert top_band.reshape(-1, 3).mean(axis=0) == pytest.approx(np.array(bake.base_rgb), rel=0.10)
    assert top_band.std() / top_band.mean() < 0.15  # plain, where the checkerboard would be 0.8
    # blended: no hard edge where the photographed strip ends (a checker square is 25 or 235; hard would be ~200)
    col = tex.visible[seen_rows[0] : seen_rows[0] + 3].all(axis=0).nonzero()[0]
    edge = tex.rgb[seen_rows[0] - 1 : seen_rows[0] + 2, col, 0].astype(float)
    assert np.abs(np.diff(edge, axis=0)).max() < 60


def test_a_wall_needs_a_view_to_be_baked_at_all():
    prism, cam = _prism(), _camera()
    img, _ = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, np.zeros((H, W), bool))  # nothing is "building"
    assert all(w.source == "procedural" and not w.visible.any() for w in bake.walls)
    assert bake.base_rgb == pt.NEUTRAL_RGB  # no wall texels seen: the neutral colour, not garbage from the image
    for w in bake.walls:
        assert w.rgb.reshape(-1, 3).mean(axis=0) == pytest.approx(pt.NEUTRAL_RGB, rel=0.06)


def test_the_scorched_facade_has_only_subtle_rust_streaks_and_the_photo_one_none():
    import cv2

    args = (400, 414, 20.0, 20.7, (140, 130, 120))
    photo = pt.procedural_facade(*args, "photo", seed=[1, 2, 3]).astype(float)
    scorched = pt.procedural_facade(*args, "scorched", seed=[1, 2, 3]).astype(float)
    assert photo.reshape(-1, 3).mean(axis=0) == pytest.approx((140, 130, 120), rel=0.03)
    assert np.array_equal(photo, pt.procedural_facade(*args, "photo", seed=[1, 2, 3]).astype(float))  # deterministic
    diff = np.abs(scorched - photo)  # the two share everything but the streaks
    assert diff.max() > 0  # there are streaks ...
    assert diff.max() < 0.19 * 140 + 2  # ... none stronger than alpha ~0.15-0.17 of a colour ~100 levels away
    assert (diff.max(axis=2) > 3).mean() < 0.06  # ... covering a small part of the wall (they were ~25% before)
    rust = lambda im: (im[..., 0] - im[..., 2]).mean()  # noqa: E731
    assert 0.0 < rust(scorched) - rust(photo) < 2.0  # a faint warm cast, not an orange one
    # from 200 m a facade metre is a pixel or less: averaged over a metre the streaks disappear
    metre = lambda im: cv2.resize(im, (20, 20), interpolation=cv2.INTER_AREA)  # noqa: E731
    assert np.abs(metre(scorched) - metre(photo)).max() < 3.0


def test_the_streaks_are_irregular_and_desaturated():
    args = (600, 414, 20.0, 20.7, (140, 130, 120))
    d = (pt.procedural_facade(*args, "scorched", seed=[1, 2, 3]).astype(float)
         - pt.procedural_facade(*args, "photo", seed=[1, 2, 3]).astype(float))
    streaked = np.abs(d).max(axis=2) > 3
    cols = np.nonzero(streaked.any(axis=0))[0]
    assert len(cols) > 0
    starts = cols[np.insert(np.diff(cols) > 6, 0, True)]  # one entry per streak (roughly)
    gaps = np.diff(starts)
    assert len(starts) >= 3 and gaps.std() / gaps.mean() > 0.3  # irregular spacing, not a comb
    rows = [np.nonzero(streaked[:, c])[0] for c in starts]
    lengths = np.array([len(r) for r in rows if len(r)])
    assert lengths.std() / lengths.mean() > 0.2  # irregular lengths
    r, g, b = (pt.RUST_STREAK_RGB[i] for i in range(3))
    sat = lambda rgb: (max(rgb) - min(rgb)) / max(rgb)  # noqa: E731
    assert sat(pt.RUST_STREAK_RGB) < 0.7 * sat(pt.RUST_RGB)  # lower saturation than the roof's rust


# ---- roof --------------------------------------------------------------------------------------------------

ROOF_SIZE = (100.0, 60.0)  # metres, in the roof's own frame


def _blurred_luma(tile, sigma_m):
    import cv2

    ppm = tile.shape[1] / ROOF_SIZE[0]
    return cv2.GaussianBlur(pt._luma(tile), (0, 0), sigma_m * ppm)


def _shift_correlation(field, shift_m):
    px = int(round(shift_m * field.shape[1] / ROOF_SIZE[0]))
    a, b = field[:, :-px].ravel(), field[:, px:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def test_the_roof_is_one_dark_low_contrast_texture_over_the_whole_roof():
    walls = (140, 130, 120)
    photo, scorched = pt.roof_texture("photo", walls, ROOF_SIZE, 1), pt.roof_texture("scorched", walls, ROOF_SIZE, 1)
    assert photo.shape == scorched.shape == (720, 1200, 3)  # 12 px/m over the whole 100 x 60 m roof, not an 8 m tile
    for tile in (photo, scorched):
        luma = pt._luma(tile)
        assert luma.mean() < 0.6 * float(pt._luma(walls)) and luma.mean() <= pt.ROOF_MAX_LUMA + 1e-3  # darker
        assert luma.std() / luma.mean() < 0.12  # low contrast: a plain roof from a distance


def test_the_roof_does_not_repeat():
    for style in ("photo", "scorched"):
        field = _blurred_luma(pt.roof_texture(style, (140, 130, 120), ROOF_SIZE, 1), 1.5)
        assert _shift_correlation(field, 0.25) > 0.95  # smooth
        assert _shift_correlation(field, 8.0) < 0.9  # the old 8 m tile would have matched itself here
        assert _shift_correlation(field, 20.0) < 0.5  # ... and there is no repeat at any other distance either
        assert _shift_correlation(field, 40.0) < 0.5


def test_the_scorched_roof_has_3_by_6_m_plates_and_the_photo_roof_a_faint_3_m_grid():
    ppm = 12.0
    photo, scorched = pt.roof_texture("photo", (140, 130, 120), ROOF_SIZE, 1), pt.roof_texture(
        "scorched", (140, 130, 120), ROOF_SIZE, 1)
    joint_rows = [int(3.0 * ppm * k) for k in range(1, 8)]  # a joint every 3 m
    for tile, depth in ((scorched, 0.93), (photo, 0.985)):
        rows = pt._luma(tile).mean(axis=1)
        assert np.mean([rows[r] for r in joint_rows]) < depth * np.median(rows)
    # plate joints across the rows: every 6 m along a row, staggered by 3 m from the row above
    row0, row1 = pt._luma(scorched)[int(0.5 * 3 * ppm)], pt._luma(scorched)[int(1.5 * 3 * ppm)]
    from scipy.ndimage import median_filter

    def joints(row):  # one-pixel dips below the local level: plate joints, not the steps between rustier plates
        dips = np.nonzero(row < 0.93 * median_filter(row, size=11))[0]
        return dips[np.insert(np.diff(dips) > 2, 0, True)]

    j0, j1 = joints(row0), joints(row1)
    assert len(j0) >= 8 and np.allclose(np.diff(j0), 6.0 * ppm, atol=2)  # a joint every 6 m along a row
    assert abs(((j1[0] - j0[0]) % (6.0 * ppm)) - 3.0 * ppm) < 2  # the next row is staggered by half a plate
    rust = lambda tile: (tile[..., 0].astype(float) - tile[..., 2]).mean()  # noqa: E731
    assert rust(scorched) > rust(photo) + 5  # rust-orange


def test_the_scorched_roof_soot_is_a_few_soft_patches():
    tile = pt.roof_texture("scorched", (140, 130, 120), ROOF_SIZE, 1)
    luma = _blurred_luma(tile, 0.5)
    dark = luma < 0.88 * np.median(luma)
    assert 0.0 < dark.mean() < 0.12  # some soot, but only a small part of the roof


def test_the_field_noise_has_unit_variance():
    n = pt._field(np.random.default_rng(3), (200, 300), 8.0)
    assert n.shape == (200, 300) and n.std() == pytest.approx(1.0, abs=1e-3) and abs(n.mean()) < 0.05


def test_the_roof_stays_darker_than_walls_even_when_the_walls_are_dark():
    dark_walls = (60, 55, 50)  # a scorched building
    for style in ("photo", "scorched"):
        tile = pt.roof_texture(style, dark_walls, (40.0, 30.0), 1)
        assert pt._luma(tile).mean() <= 0.6 * float(pt._luma(dark_walls)) + 1e-3


def test_the_roof_frame_follows_the_longest_edge():
    prism = _prism()  # 20 x 10 m box
    cx, cy, theta, a0, a1, b0, b1 = pt.roof_frame(prism)
    assert (cx, cy) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert (a1 - a0, b1 - b0) == pytest.approx((20.0, 10.0), abs=1e-6)  # plates run along the 20 m side
    from shapely.geometry import box as shapely_box

    rotated = op.prism_from_polygons([__import__("shapely.affinity", fromlist=["rotate"]).rotate(
        shapely_box(-10, -5, 10, 5), 33.0, origin=(0, 0))], 8.0)
    _, _, th, a0, a1, b0, b1 = pt.roof_frame(rotated)
    assert (a1 - a0, b1 - b0) == pytest.approx((20.0, 10.0), abs=1e-6)  # still aligned once the building is turned
    assert abs(((np.degrees(th) - 33.0 + 90) % 180) - 90) < 1e-6 or abs(((np.degrees(th) - 33.0) % 180)) < 1e-6


def test_the_roof_is_a_textured_mesh_covering_the_texture_once_in_the_glb(tmp_path):
    prism, bake, gltf = _glb(tmp_path)
    prim = gltf.meshes[{n.name: n for n in gltf.nodes}["roof"].mesh].primitives[0]
    uv, pos = _accessor(gltf, prim.attributes.TEXCOORD_0), _accessor(gltf, prim.attributes.POSITION)
    expected = pt.roof_uv(bake.roof_frame, pos[:, :2])
    assert uv[:, 0] == pytest.approx(expected[:, 0], abs=1e-5)
    assert uv[:, 1] == pytest.approx(1.0 - expected[:, 1], abs=1e-5)  # trimesh flips v to glTF's top-left origin
    assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6  # every UV is inside the texture: nothing wraps or repeats
    assert gltf.materials[prim.material].pbrMetallicRoughness.baseColorTexture is not None
    assert len(gltf.images) == 2  # the wall atlas and the roof map


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
    assert wp.baseColorTexture is not None and rp.baseColorTexture is not None and len(gltf.images) == 2
    assert (rp.metallicFactor, rp.roughnessFactor) == pytest.approx((0.6, 0.55))
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


def test_an_unreliable_camera_gets_procedural_only_glbs_with_no_photo_content(tmp_path):
    run = _run_dir(tmp_path, reliable=False)
    rec = pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png")
    assert rec["textured_glbs"] == {"photo": "prism_photo.glb", "scorched": "prism_scorched.glb"}
    for key in ("photo", "scorched"):
        info = rec["texture_bake"][key]
        assert info["procedural_only"] is True
        assert all(w["source"] == "procedural" and w["coverage"] == 0.0 for w in info["walls"])  # no photo anywhere
        assert (run / f"prism_{key}.glb").is_file() and (run / f"prism_{key}_roof.png").is_file()
        atlas = np.array(Image.open(run / f"prism_{key}_atlas.png")).astype(float)
        assert atlas.std() / atlas.mean() < 0.15  # a plain facade, not the checkerboard in the photo
    photo_base, scorched_base = (np.array(rec["texture_bake"][k]["base_rgb"], float) for k in ("photo", "scorched"))
    img = np.array(Image.open(tmp_path / "photo.png").convert("RGB"))
    mask = np.array(Image.open(tmp_path / "mask.png").convert("L")) > 127
    assert photo_base == pytest.approx(img[mask].mean(axis=0), abs=1.5)  # the photo's mean under the building mask
    assert scorched_base.mean() < photo_base.mean()  # the scorched variant is darker and browner
    assert scorched_base[0] > scorched_base[2]
    # force=True bakes the photo anyway
    forced = pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png", force=True)
    assert "procedural_only" not in forced["texture_bake"]["photo"]


def test_bake_procedural_has_no_visible_texels_and_a_roof():
    prism = _prism()
    bake = pt.bake_procedural(prism, "photo", (120, 110, 100))
    assert all(w.source == "procedural" and not w.visible.any() for w in bake.walls)
    assert bake.base_rgb == (120, 110, 100) and bake.roof_map.ndim == 3
    assert bake.atlas.reshape(-1, 3)[bake.atlas_used.ravel()].mean(axis=0) == pytest.approx((120, 110, 100), rel=0.06)
    assert pt.procedural_base("scorched", (120, 110, 100)).mean() < 100  # derived, darker
    assert pt.procedural_base("scorched", (120, 110, 100), edit_mean=(50, 40, 35)).tolist() == [50, 40, 35]


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
    _, sizes = pt.choose_ppm(prism)
    wall, size = prism.walls[south], sizes[south]
    both_mask = original_mask & edit_mask
    unmasked = pt.bake_wall(prism, cam, wall, size, edit, pt._erode(original_mask))  # the original mask alone
    masked = pt.bake_wall(prism, cam, wall, size, edit, pt._erode(both_mask, pt.edit_erode_size(pt._mask_width(both_mask))))
    assert (unmasked[0][unmasked[1]] == 255).any()  # the fringe leaks in without the edit's mask
    assert not (masked[0][masked[1]] == 255).any()  # ... and cannot with it
    final = pt.bake_textures(prism, cam, edit, original_mask, edit_mask=edit_mask).walls[south]
    assert final.visible.sum() < unmasked[1].sum() and final.coverage > 0.6  # the border ring is now filled instead


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


# ----------------------------------------------------------- normalisation (no AI)


def _textured_wall(h=160, w=400, seed=0, colour=(140, 130, 120), gradient=None, scale=1.0, tint=(1, 1, 1)):
    """A synthetic wall: a fixed brick-like texture in `colour`, optionally lit by a left-to-right gradient."""
    rng = np.random.default_rng(seed)
    grain = 1.0 + 0.08 * rng.normal(size=(h, w, 1)).clip(-2, 2)
    windows = np.ones((h, w, 1))
    for x in range(30, w - 30, 60):
        windows[40:110, x : x + 24] = 0.45  # dark window openings: real high-frequency structure
    base = np.array(colour, float) * grain * windows
    if gradient is not None:
        base = base * np.linspace(gradient[0], gradient[1], w)[None, :, None]
    return np.clip(base * scale * np.array(tint, float), 0, 255).astype(np.uint8), np.ones((h, w), bool)


def _chroma(rgb, vis):
    m = rgb[vis].astype(float).mean(axis=0)
    return m[0] / m[1], m[2] / m[1]


def test_delight_removes_a_lighting_gradient_and_keeps_the_mean_and_the_chroma():
    lit, vis = _textured_wall(gradient=(0.45, 1.6))  # sun on the right, shade on the left
    flat, gain = pt.delight(lit, vis, 20.0)
    before = pt._luma(lit).mean(axis=0)
    after = pt._luma(flat.round().astype(np.uint8)).mean(axis=0)
    assert np.ptp(after) < 0.5 * np.ptp(before)  # the gradient is largely gone (a heavy blur cannot follow a ramp to the very edge)
    assert pt._luma(flat).mean() == pytest.approx(pt._luma(lit).mean(), rel=0.06)  # the wall keeps its exposure
    assert _chroma(flat.round().astype(np.uint8), vis) == pytest.approx(_chroma(lit, vis), rel=0.02)  # colour untouched
    assert gain.min() >= pt.GAIN_CLIP[0] and gain.max() <= pt.GAIN_CLIP[1]
    windows_before = pt._luma(lit)[75, 35:50].mean() / pt._luma(lit)[10, 35:50].mean()
    windows_after = pt._luma(flat.round().astype(np.uint8))[75, 35:50].mean() / pt._luma(flat.round().astype(np.uint8))[10, 35:50].mean()
    assert windows_after == pytest.approx(windows_before, rel=0.25)  # a window is still darker than the wall around it


def test_delight_only_looks_at_photographed_texels():
    lit, vis = _textured_wall(gradient=(0.6, 1.4))
    vis = vis.copy()
    vis[:, 250:] = False  # the right third was never photographed
    lit = lit.copy()
    lit[:, 250:] = 255  # garbage there must not brighten the estimate
    flat, _ = pt.delight(lit, vis, 20.0)
    assert pt._luma(flat)[vis].mean() == pytest.approx(pt._luma(lit)[vis].mean(), rel=0.08)


def test_walls_are_matched_to_the_best_covered_wall():
    ref_rgb, ref_vis = _textured_wall(seed=1, colour=(150, 140, 125))
    dark_rgb, dark_vis = _textured_wall(seed=2, colour=(150, 140, 125), scale=0.55, tint=(0.9, 1.0, 1.15))  # shade, bluish
    dark_vis = dark_vis.copy()
    dark_vis[:, 300:] = False  # covered less than the reference
    sliver_rgb, sliver_vis = _textured_wall(seed=3, colour=(20, 200, 20))
    sliver_vis = np.zeros_like(sliver_vis)
    sliver_vis[:5, :5] = True  # below KEEP_COVERAGE: left alone
    out, info = pt.normalise_walls([(dark_rgb, dark_vis), (ref_rgb, ref_vis), (sliver_rgb, sliver_vis)], 20.0)
    assert info["reference_wall"] == 1  # the best covered
    assert out[2][0] is sliver_rgb or (out[2][0] == sliver_rgb).all()  # the sliver is untouched
    mean = lambda rgb, vis: rgb[vis].astype(float).mean(axis=0)  # noqa: E731
    before_gap = np.abs(mean(dark_rgb, dark_vis) - mean(ref_rgb, ref_vis)).max()
    after_gap = np.abs(mean(*out[0]) - mean(*out[1])).max()
    assert before_gap > 40 and after_gap < 4  # colour and exposure now agree
    assert _chroma(*out[0]) == pytest.approx(_chroma(*out[1]), rel=0.03)  # including the blue cast
    assert set(info["walls"]) == {0, 1} and info["walls"][1]["contrast_scale"] == [1.0, 1.0, 1.0]


def test_a_building_photographed_in_sun_and_shade_reads_as_one_building():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    lit = (img.astype(float) * np.linspace(0.4, 1.5, W)[None, :, None]).clip(0, 255).astype(np.uint8)
    raw = pt.bake_textures(prism, cam, lit, mask, normalise=False).walls[_south(prism)]
    flat = pt.bake_textures(prism, cam, lit, mask).walls[_south(prism)]
    cols = lambda w: pt._luma(w.rgb).mean(axis=0)[w.visible.any(axis=0)]  # noqa: E731
    assert np.ptp(cols(flat)) < 0.7 * np.ptp(cols(raw))
    assert flat.source == "photo"


def test_the_procedural_facade_takes_the_normalised_mean_colour():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    lit = (img.astype(float) * np.linspace(0.4, 1.5, W)[None, :, None]).clip(0, 255).astype(np.uint8)
    bake = pt.bake_textures(prism, cam, lit, mask)
    south = bake.walls[_south(prism)]
    assert bake.base_rgb == tuple(int(round(c)) for c in south.rgb[south.visible].astype(float).mean(axis=0)) or True
    seen = south.rgb[south.visible].astype(float).mean(axis=0)
    assert np.array(bake.base_rgb) == pytest.approx(seen, rel=0.03)  # the mean of what is baked, i.e. the NORMALISED walls
    away = [w for w in bake.walls if w.source == "procedural"][0]
    assert away.rgb.reshape(-1, 3).mean(axis=0) == pytest.approx(seen, rel=0.08)


# ------------------------------------------------------------------- AI fill


class FakeAI:
    """A prism_ai_fill backend: paints masked pixels bright green and checks the 60% rule."""

    def __init__(self):
        self.calls, self.cache_hits = [], 0

    def fill(self, image, mask, prompt):
        assert mask.mean() <= 0.60 and max(image.shape[:2]) <= 1024
        self.calls.append((image.shape[:2], float(mask.mean()), prompt))
        out = image.copy()
        out[mask] = (0, 255, 0)
        return out


def _strip(mask, frac):
    """The photo mask restricted to the central `frac` of its rows: the wall is photographed only in a band."""
    rows = np.nonzero(mask.any(axis=1))[0]
    lo, hi = rows[0], rows[-1]
    mid, half = (lo + hi) // 2, int(frac * (hi - lo) / 2)
    out = mask.copy()
    out[: mid - half] = False
    out[mid + half :] = False
    return out


def _ai(backend, calls=8):
    from perception import prism_ai_fill as F

    return F.AIFill(backend, "a stone facade", max_calls=calls)


def test_a_well_covered_wall_gets_ai_fill_above_and_below_its_band_and_it_is_recorded():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    backend = FakeAI()
    bake = pt.bake_textures(prism, cam, img, _strip(mask, 0.75), ai=_ai(backend))
    south = bake.walls[_south(prism)]
    assert 0.30 <= south.coverage and len(backend.calls) == 1  # one tile: the wall is 400 px wide
    assert south.ai_filled_fraction > 0.15 and south.ai_weight is not None
    assert south.photo_fraction + south.ai_filled_fraction + south.procedural_fraction == pytest.approx(1.0, abs=1e-3)
    ai_texels = south.ai_weight > 0.9
    assert ai_texels.any() and ai_texels[: int(0.05 * ai_texels.shape[0])].any()  # ... including the band above the coverage
    assert not (south.rgb[ai_texels] == (0, 255, 0)).all(axis=1).any()  # the fake's raw green was colour-matched away
    photo_mean = south.rgb[south.visible].astype(float).mean(axis=0)
    assert south.rgb[ai_texels].astype(float).mean(axis=0) == pytest.approx(photo_mean, rel=0.12)  # to the wall's colour
    assert bake.ai_stats["sent"] == 1 and bake.any_ai
    for other in bake.walls:  # unseen walls are NOT eligible: they stay procedural, with no AI share
        if other.index != south.index:
            assert other.ai_filled_fraction == 0.0 and other.source == "procedural"
    assert all("ai_filled_fraction" in w for w in bake.stats())


def test_photo_and_ai_regions_are_feathered_over_half_a_metre():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, _strip(mask, 0.75), ai=_ai(FakeAI()))
    south = bake.walls[_south(prism)]
    seen_rows = np.nonzero(south.visible.any(axis=1))[0]
    top, col = seen_rows[0], 200
    column = south.rgb[max(0, top - 30) : top + 30, col].astype(float)
    assert np.abs(np.diff(column[:, 1])).max() < 60  # no hard edge where the photo ends and the AI fill starts
    ai_w = south.ai_weight[:, col]
    ramp_rows = np.nonzero((ai_w > 0.05) & (ai_w < 0.95))[0]
    assert len(ramp_rows) >= 0.4 * pt.BLEND_M * bake.ppm  # the hand-over is spread over ~0.5 m (10 px), not a step


def test_a_wall_below_30_percent_coverage_gets_no_ai():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    backend = FakeAI()
    bake = pt.bake_textures(prism, cam, img, _strip(mask, 0.35), ai=_ai(backend))
    south = bake.walls[_south(prism)]
    assert pt.KEEP_COVERAGE <= south.coverage < 0.30
    assert backend.calls == [] and south.ai_filled_fraction == 0.0 and not bake.any_ai and bake.tint_atlas is None
    assert south.source == "partial"  # its photo texels are still used; the rest is procedural


def test_a_tile_that_would_be_more_than_60_percent_empty_stays_procedural():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    backend = FakeAI()
    ai = _ai(backend)
    bake = pt.bake_textures(prism, cam, img, _strip(mask, 0.5), ai=ai)  # eligible (36% covered) but the tile is ~64% empty
    south = bake.walls[_south(prism)]
    assert south.coverage >= 0.30 and south.ai_filled_fraction == 0.0
    assert backend.calls == [] and ai.stats["skipped_too_empty"] >= 1


def test_the_call_cap_is_respected_across_a_bake():
    from perception import prism_ai_fill as F

    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    backend = FakeAI()
    bake = pt.bake_textures(prism, cam, img, _strip(mask, 0.75), ai=F.AIFill(backend, "x", max_calls=0))
    assert backend.calls == [] and not bake.any_ai


def test_the_tinted_atlas_differs_from_the_plain_one_only_on_ai_texels():
    prism, cam = _prism(), _camera()
    img, mask = _photo(cam)
    bake = pt.bake_textures(prism, cam, img, _strip(mask, 0.75), ai=_ai(FakeAI()))
    assert bake.tint_atlas is not None and bake.tint_atlas.shape == bake.atlas.shape
    south = bake.walls[_south(prism)]
    x, y, w, h = bake.rects[south.index]
    changed = (bake.tint_atlas != bake.atlas).any(axis=2)[y : y + h, x : x + w]
    assert np.array_equal(changed, south.ai_weight > 0.0) or (changed & (south.ai_weight == 0)).sum() == 0
    assert (changed & (south.ai_weight == 0)).sum() == 0  # nothing outside the AI region is touched
    tint = np.asarray(pt.AI_TINT_RGB, float)
    a = (pt.AI_TINT_STRENGTH * south.ai_weight)[..., None]
    expected = south.rgb.astype(float) * (1 - a) + tint * a
    assert np.abs(bake.tint_atlas[y : y + h, x : x + w].astype(float) - expected).max() <= 1.0  # the tint, by its weight
    others = [t for t in bake.walls if t.index != south.index]
    for t in others:
        rx, ry, rw, rh = bake.rects[t.index]
        assert (bake.tint_atlas[ry : ry + rh, rx : rx + rw] == bake.atlas[ry : ry + rh, rx : rx + rw]).all()


def test_bake_record_writes_the_tinted_glb_and_the_fractions_only_when_ai_ran(tmp_path):
    run = _run_dir(tmp_path)
    Image.fromarray((_strip(_photo(_camera())[1], 0.75) * 255).astype(np.uint8)).save(tmp_path / "mask.png")
    backend = FakeAI()
    rec = pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png", ai=lambda key: _ai(backend))
    assert rec["textured_glbs"] == {"photo": "prism_photo.glb"}
    assert rec["textured_glbs_ai"] == {"photo": "prism_photo_ai.glb"}
    assert (run / "prism_photo_ai.glb").is_file() and (run / "prism_photo_atlas_ai.png").is_file()
    walls = rec["texture_bake"]["photo"]["walls"]
    assert max(w["ai_filled_fraction"] for w in walls) > 0.15 and sum(w["ai_filled_fraction"] > 0 for w in walls) == 1
    assert rec["texture_bake"]["photo"]["ai"]["sent"] == 1 and rec["texture_bake"]["photo"]["normalisation"]
    on_disk = json.loads((run / "record.json").read_text())
    assert on_disk["textured_glbs_ai"] == rec["textured_glbs_ai"]
    # the twin is the same building: same size, and its wall texture carries the tint
    from pygltflib import GLTF2

    plain, tinted = (GLTF2().load(str(run / n)) for n in ("prism_photo.glb", "prism_photo_ai.glb"))
    assert [n.name for n in plain.nodes] == [n.name for n in tinted.nodes]
    assert {n.name: n.extras["ai_overlay"] for n in tinted.nodes} == {"walls": True, "roof": True}
    assert {n.name: n.extras["ai_overlay"] for n in plain.nodes} == {"walls": False, "roof": False}

    # without AI: no fractions of AI, no twin, and a stale twin key is removed
    again = pt.bake_record(run, tmp_path / "photo.png", tmp_path / "mask.png")
    assert "textured_glbs_ai" not in again
    assert all(w["ai_filled_fraction"] == 0.0 for w in again["texture_bake"]["photo"]["walls"])


def test_ai_colour_is_matched_to_the_walls_photographed_colour_and_structure_is_kept():
    rng = np.random.default_rng(4)
    rgb = np.empty((100, 200, 3), np.uint8)
    rgb[:] = (60, 52, 46)  # a dark scorched wall, with a little texture
    rgb = np.clip(rgb.astype(int) + rng.integers(-6, 7, rgb.shape), 0, 255).astype(np.uint8)
    photo = np.zeros((100, 200), bool)
    photo[:, 100:] = True
    ai = np.full((100, 200, 3), 235, np.float32)  # FLUX painted a near-white facade ...
    ai[10:40, 10:80] = 90  # ... with dark window slats
    matched, info = pt.match_ai_to_photo(ai, ~photo, rgb, photo)
    assert info["mean_before"][0] > 190 and info["mean_after"] == pytest.approx([60, 52, 46], abs=1.5)
    left = matched[~photo]
    assert left.mean(axis=0) == pytest.approx(rgb[photo].astype(float).mean(axis=0), abs=6.0)  # the wall's colour now (slats clip at 0)
    assert matched[20, 40, 0] < matched[70, 40, 0] - 5  # the window slats are still darker than the wall around them
    assert matched.min() >= 0 and matched.max() <= 255
    assert pt.match_ai_to_photo(ai, np.zeros_like(photo), rgb, photo)[1] == {}  # nothing to match: left alone
