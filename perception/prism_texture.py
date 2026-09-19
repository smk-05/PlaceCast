#!/usr/bin/env python
"""Textured footprint prism: bake a photo onto the prism's walls through the refined camera, write a glb.

    bake = bake_textures(prism, camera, image, mask)
    build_glb(prism, bake, "prism_photo.glb")
    bake_record(run_dir, photo, mask, scorched=None)      # the three-step driver used from the command line

perception/openings_prism.py fits a pinhole camera to each photo and stores it in record["openings_camera"]. This
module reuses that camera: it does not fit anything.

Baking is by INVERSE MAPPING, per wall. A wall gets a rectified texture (about 20 px/m, no side over 4096 px, all of
them packed into one atlas no bigger than 4096 x 4096). For each texel the 3-D point on the wall in ENU is projected
through the camera and the photo is sampled bilinearly there. Vertex UVs cannot do this: a perspective image is not an
affine function of a wall's plan coordinates, and interpolating UVs across a large quad bends the texture.

A texel is VISIBLE only if
  * the camera is on the wall's outer side,
  * the camera ray to the texel reaches it before it meets any other wall (the prism's first hit is this wall; done
    analytically: the plan segment camera -> texel against every other wall, then the height of the crossing),
  * the projected pixel is inside the image, and
  * it is inside the building mask (eroded a few pixels), which drops trees, lamps and signs standing in front, and
  * for an EDITED image (the scorched bake), it is also inside the EDIT'S OWN building mask: the edit's background
    (sky, ground, a white surround) differs from the photo's, so a texel the original mask accepts can land on a
    fringe of that background. A texel must be inside both masks (intersected at photo resolution), the
    intersection is eroded by ~1% of the building width (EDIT_ERODE_FRAC; a photo's mask only by MASK_ERODE_PX), and a
    texel near the boundary (within HALO_BAND_FRAC of the width) whose baked colour is light and low-saturation (the
    edits carry a beige halo) is not visible either. Everything else goes to the fill step.

Texels the camera cannot see are filled, never left blank:
  * a wall with coverage >= STRONG_COVERAGE keeps its visible texels; small holes are cv2.inpaint'ed and large ones
    filled from a low-resolution inpaint;
  * a wall the photo barely shows (coverage < STRONG_COVERAGE) takes a DONOR: the well-covered wall whose outward
    normal is most similar, mirror-tiled along its length and darkened by DARKEN. Its own visible texels are kept if
    coverage >= KEEP_COVERAGE, otherwise it is entirely donor;
  * with no well-covered wall at all, a neutral colour from the visible texels.
`WallTexture.source` records which of these happened to each wall ("photo", "partial", "donor", "neutral").
The roof is a dark weathered-steel material, not a photo texture (it is not visible from the ground).

glb. Vertices are ENU metres (x east, y north, z up): the frame web/src/main.js uses for the opening markers, with
the record's origin and ground. That is not glTF's Y-up convention on purpose: Cesium is told upAxis Z / forwardAxis X,
exactly as for the generated-mesh path, so it applies no axis correction and modelMatrix = enuToFixed places the model.
Walls: PBR, metallic 0.4, roughness 0.75, one atlas texture. Roof: metallic 0.6, roughness 0.55, no texture.

Usage: python perception/prism_texture.py RUN_DIR PHOTO MASK [--scorched EDIT.png [--edit-mask MASK.png]] [--force]
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `python perception/prism_texture.py` puts perception/ first, not the repo root
    sys.path.insert(0, str(ROOT))

from perception.openings_prism import PinholeCamera, prism_from_record  # noqa: E402

PPM = 20.0
MAX_SIDE = 4096
ATLAS_MAX = 4096
PAD = 2  # replicated border around each wall in the atlas, so linear filtering cannot bleed a neighbour in
ATLAS_FILL = 0.7  # use at most this fraction of the atlas area before shrinking px/m
MASK_ERODE_PX = 9  # at photo resolution
EDIT_ERODE_FRAC = 0.01  # an edit's (original & edit) mask is eroded by this fraction of the building width
HALO_BAND_FRAC = 0.03  # ... and texels this close to its boundary are checked for halo colour
HALO_MIN_V, HALO_MAX_S = 0.75, 0.30  # HSV value >= / saturation <= : "light and low-saturation"
EPS_M = 0.05  # an obstruction closer than this to the texel is the wall itself, not something in front
STRONG_COVERAGE = 0.4
KEEP_COVERAGE = 0.15
SMALL_HOLE_PX = 1500
DARKEN = 0.85
NEUTRAL_RGB = (128, 126, 120)
METALLIC_WALL, ROUGHNESS_WALL = 0.4, 0.75
ROOF_RGBA = (0.16, 0.17, 0.19, 1.0)
METALLIC_ROOF, ROUGHNESS_ROOF = 0.6, 0.55
NEAR_M = 0.5

FRAME_NOTE = "ENU metres (x east, y north, z up); load with Cesium upAxis Z / forwardAxis X and modelMatrix = enuToFixed"


@dataclass
class WallTexture:
    index: int
    rgb: np.ndarray  # (h, w, 3) uint8, row 0 is the TOP of the wall, column 0 is at p0
    visible: np.ndarray  # (h, w) bool: baked from the photo (False = filled)
    coverage: float
    source: str = "photo"  # photo | partial | donor | neutral
    donor: int = -1


@dataclass
class Bake:
    walls: list
    ppm: float
    atlas: np.ndarray = None  # (A_h, A_w, 3) uint8
    atlas_visible: np.ndarray = None  # (A_h, A_w) bool
    atlas_used: np.ndarray = None  # (A_h, A_w) bool: inside some wall's rectangle
    rects: list = field(default_factory=list)  # per wall (x, y, w, h) in the atlas, excluding the PAD ring

    def stats(self):
        return [{"wall": t.index, "coverage": round(t.coverage, 3), "source": t.source,
                 **({"donor": t.donor} if t.donor >= 0 else {})} for t in self.walls]


# ------------------------------------------------------------------- layout


def wall_size(length_m, height_m, ppm):
    return max(4, math.ceil(length_m * ppm)), max(4, math.ceil(height_m * ppm))


def plan_layout(sizes, atlas_max=ATLAS_MAX):
    """Shelf-pack (w, h) rectangles (each with a PAD ring) into an atlas at most `atlas_max` wide.
    -> (rects [(x, y)], (atlas_w, atlas_h)) or None when the atlas would be taller than `atlas_max`."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    x = y = shelf = used_w = 0
    pos = [None] * len(sizes)
    for i in order:
        w, h = sizes[i][0] + 2 * PAD, sizes[i][1] + 2 * PAD
        if w > atlas_max:
            return None
        if x + w > atlas_max:
            x, y, shelf = 0, y + shelf, 0
        pos[i] = (x + PAD, y + PAD)
        x, shelf, used_w = x + w, max(shelf, h), max(used_w, x + w)
    total_h = y + shelf
    if total_h > atlas_max:
        return None
    return pos, (math.ceil(used_w / 4) * 4, math.ceil(total_h / 4) * 4)


def choose_ppm(prism, ppm=PPM):
    """Largest px/m <= `ppm` for which every wall is under MAX_SIDE and the atlas fits in ATLAS_MAX^2."""
    longest = max(w["length_m"] for w in prism.walls)
    ppm = min(ppm, (MAX_SIDE - 2 * PAD) / max(longest, prism.height_m))
    while True:
        sizes = [wall_size(w["length_m"], prism.height_m, ppm) for w in prism.walls]
        area = sum((w + 2 * PAD) * (h + 2 * PAD) for w, h in sizes)
        if area <= ATLAS_FILL * ATLAS_MAX ** 2 and plan_layout(sizes) is not None:
            return ppm, sizes
        ppm *= 0.92


# ------------------------------------------------------------------ camera


def camera_from_record(record, width, height):
    """The refined pinhole camera openings_prism.py stored in record['openings_camera']."""
    c = record.get("openings_camera")
    if not c:
        raise ValueError("record has no openings_camera: run perception/openings_prism.py --write-record first")
    return PinholeCamera(tuple(c["camera_position_enu"]), float(c["camera_yaw_deg"]), float(c["camera_pitch_deg"]),
                         float(c["f_px"]), int(width), int(height))


# --------------------------------------------------------------- visibility


def _plan_crossings(cam_xy, targets, walls, skip):
    """T[i, k]: fraction along camera -> targets[i] where that plan segment crosses wall k (inf: it does not, or the
    crossing is within EPS_M of the target, which is the target's own wall or a shared corner)."""
    a = np.array([w["p0"] for w in walls])
    e = np.array([w["p1"] for w in walls]) - a
    d = targets - cam_xy
    length = np.linalg.norm(d, axis=1)
    ac = a - cam_xy
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    tnum = ac[None, :, 0] * e[None, :, 1] - ac[None, :, 1] * e[None, :, 0]
    snum = ac[None, :, 0] * d[:, None, 1] - ac[None, :, 1] * d[:, None, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t, s = tnum / denom, snum / denom
    hit = ((np.abs(denom) > 1e-12) & (s >= -1e-9) & (s <= 1 + 1e-9) & (t > 1e-9)
           & (t < 1 - EPS_M / np.maximum(length, 1e-9)[:, None]))
    hit[:, skip] = False
    return np.where(hit, t, np.inf)


def wall_visibility(prism, camera, wall, size):
    """(h, w) bool: texels of `wall` whose camera ray reaches them first (geometry only; no image or mask)."""
    w_px, h_px = size
    cam = np.asarray(camera.position, float)
    p0, p1, n = np.asarray(wall["p0"]), np.asarray(wall["p1"]), np.asarray(wall["normal"])
    if np.dot(cam[:2] - p0, n) <= 1e-9:  # the camera is behind this wall
        return np.zeros((h_px, w_px), bool)
    s = (np.arange(w_px) + 0.5) / w_px
    targets = p0[None] + s[:, None] * (p1 - p0)[None]
    crossings = _plan_crossings(cam[:2], targets, prism.walls, wall["index"])
    keep = np.isfinite(crossings).any(axis=0)  # only walls that cross some column's sight line matter
    z = prism.height_m * (1.0 - (np.arange(h_px) + 0.5) / h_px)
    if not keep.any():
        return np.ones((h_px, w_px), bool)
    t = crossings[:, keep]
    finite = np.isfinite(t)
    t = np.where(finite, t, 0.0)
    blocked = np.zeros((h_px, w_px), bool)
    rows = max(1, int(8e6 // max(1, w_px * t.shape[1])))
    for r0 in range(0, h_px, rows):
        zr = z[r0 : r0 + rows]
        z_hit = cam[2] + t[None] * (zr[:, None, None] - cam[2])  # height at which the sight line crosses each wall
        blocked[r0 : r0 + rows] = (finite[None] & (z_hit >= 0.0) & (z_hit <= prism.height_m)).any(axis=2)
    return ~blocked


def _texel_points(prism, wall, size):
    w_px, h_px = size
    s = (np.arange(w_px) + 0.5) / w_px
    z = prism.height_m * (1.0 - (np.arange(h_px) + 0.5) / h_px)
    xy = np.asarray(wall["p0"])[None] + s[:, None] * (np.asarray(wall["p1"]) - np.asarray(wall["p0"]))[None]
    pts = np.empty((h_px, w_px, 3))
    pts[..., :2] = xy[None]
    pts[..., 2] = z[:, None]
    return pts.reshape(-1, 3)


def bake_wall(prism, camera, wall, size, image, mask_eroded, halo_zone=None):
    """-> (rgb (h, w, 3) uint8, visible (h, w) bool) for one wall. See the module docstring. `mask_eroded` is at PHOTO
    resolution. `halo_zone` (an edit bake only) is the photo-resolution band along the mask boundary in which
    light, low-saturation texels are treated as the edit's halo."""
    w_px, h_px = size
    pts = _texel_points(prism, wall, size)
    px = camera.project(pts)
    right, up, forward = camera.basis()
    in_front = ((pts - np.asarray(camera.position)) @ forward > NEAR_M).reshape(h_px, w_px)
    ix, iy = np.floor(px[:, 0]).astype(int), np.floor(px[:, 1]).astype(int)
    inside = ((ix >= 0) & (ix < camera.width) & (iy >= 0) & (iy < camera.height)).reshape(h_px, w_px)
    in_mask = np.zeros(h_px * w_px, bool)
    ok = inside.reshape(-1)
    in_mask[ok] = mask_eroded[iy[ok], ix[ok]]
    visible = wall_visibility(prism, camera, wall, size) & in_front & inside & in_mask.reshape(h_px, w_px)

    sx, sy = image.shape[1] / camera.width, image.shape[0] / camera.height
    map_x = (px[:, 0] * sx - 0.5).astype(np.float32).reshape(h_px, w_px)  # cv2 puts pixel centres on integers
    map_y = (px[:, 1] * sy - 0.5).astype(np.float32).reshape(h_px, w_px)
    rgb = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if halo_zone is not None:  # near the boundary, a light and desaturated texel is the edit's halo, not the wall
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # S and V are 0-255 for uint8
        light = (hsv[..., 2] >= HALO_MIN_V * 255) & (hsv[..., 1] <= HALO_MAX_S * 255)
        near = np.zeros(h_px * w_px, bool)
        near[ok] = halo_zone[iy[ok], ix[ok]]
        visible &= ~(near.reshape(h_px, w_px) & light)
    return rgb, visible


# -------------------------------------------------------------------- filling


def _inpaint_small(rgb, hole):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    small = np.zeros(hole.shape, bool)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] <= SMALL_HOLE_PX:
            small |= labels == i
    if small.any():
        rgb = cv2.inpaint(rgb, small.astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)
    return rgb, hole & ~small


def _inpaint_lowres(rgb, hole, scale=4):
    """Fill large holes from a downscaled inpaint: blurred, but it borrows the wall's own colour."""
    if not hole.any():
        return rgb
    h, w = hole.shape
    sh, sw = max(1, h // scale), max(1, w // scale)
    valid = (~hole).astype(np.float32)
    total = cv2.resize(rgb.astype(np.float32) * valid[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
    weight = cv2.resize(valid, (sw, sh), interpolation=cv2.INTER_AREA)
    small = (total / np.maximum(weight[..., None], 1e-6)).clip(0, 255).astype(np.uint8)
    small = cv2.inpaint(small, (weight < 0.5).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    out = rgb.copy()
    out[hole] = up[hole]
    return out


def _mirror_tile(rgb, width):
    """`rgb` repeated (alternately mirrored, so there is no seam) to `width` columns, centred on the original."""
    h, w, _ = rgb.shape
    reps = math.ceil(width / w) + 2
    tiles = [rgb if i % 2 == 0 else rgb[:, ::-1] for i in range(reps)]
    wide = np.concatenate(tiles, axis=1)
    start = (wide.shape[1] - width) // 2
    return wide[:, start : start + width]


def _darken(rgb):
    return (rgb.astype(np.float32) * DARKEN).clip(0, 255).astype(np.uint8)


def fill_walls(prism, raw):
    """raw: list of (rgb, visible). -> list of WallTexture with every texel filled. See the module docstring."""
    out = []
    for i, (rgb, vis) in enumerate(raw):
        out.append(WallTexture(i, rgb.copy(), vis.copy(), float(vis.mean())))
    strong = [t for t in out if t.coverage >= STRONG_COVERAGE]
    for t in strong:
        rgb, big = _inpaint_small(t.rgb, ~t.visible)
        t.rgb, t.source = _inpaint_lowres(rgb, big & ~t.visible), "photo"
    seen = np.concatenate([t.rgb[t.visible] for t in out if t.visible.any()] or [np.zeros((0, 3), np.uint8)])
    neutral = np.median(seen, axis=0) if len(seen) else np.array(NEUTRAL_RGB, float)
    for t in out:
        if t.coverage >= STRONG_COVERAGE:
            continue
        keep = t.visible if t.coverage >= KEEP_COVERAGE else np.zeros_like(t.visible)
        donors = [d for d in strong if d.index != t.index]
        if donors:
            n = np.asarray(prism.walls[t.index]["normal"])
            best = max(donors, key=lambda d: (float(np.dot(n, prism.walls[d.index]["normal"])), d.coverage))
            fill, t.donor, t.source = _darken(_mirror_tile(best.rgb, t.rgb.shape[1])), best.index, "donor"
        else:
            fill = np.empty_like(t.rgb)
            fill[:] = _darken(np.asarray(neutral, np.uint8)[None, None])[0, 0]
            t.source = "neutral"
        t.rgb = np.where(keep[..., None], t.rgb, fill)
        if keep.any():
            t.source = "partial"
        t.visible = keep
    return out


# ---------------------------------------------------------------------- atlas


def build_atlas(prism, walls, ppm, sizes):
    pos, (a_w, a_h) = plan_layout(sizes)
    atlas = np.full((a_h, a_w, 3), NEUTRAL_RGB, np.uint8)
    visible = np.zeros((a_h, a_w), bool)
    used = np.zeros((a_h, a_w), bool)
    rects = []
    for t, (x, y), (w, h) in zip(walls, pos, sizes):
        padded = cv2.copyMakeBorder(t.rgb, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        atlas[y - PAD : y + h + PAD, x - PAD : x + w + PAD] = padded
        visible[y : y + h, x : x + w] = t.visible
        used[y : y + h, x : x + w] = True
        rects.append((x, y, w, h))
    return atlas, visible, used, rects


def _erode(mask, size=MASK_ERODE_PX):
    return cv2.erode(np.asarray(mask).astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)


def _mask_width(mask):
    xs = np.nonzero(np.asarray(mask).any(axis=0))[0]
    return int(xs[-1] - xs[0] + 1) if len(xs) else 0


def edit_erode_size(width_px):
    """Odd erosion kernel for an edit: ~EDIT_ERODE_FRAC of the building width, never below MASK_ERODE_PX."""
    size = max(MASK_ERODE_PX, round(EDIT_ERODE_FRAC * width_px))
    return size + 1 - size % 2


def bake_textures(prism, camera, image, mask, ppm=PPM, edit_mask=None, halo_band_frac=HALO_BAND_FRAC):
    """Bake `image` (H, W, 3 uint8 RGB, the same framing as the photo the camera was fitted to) onto every wall.

    `mask` is the ORIGINAL photo's building mask at the PHOTO's resolution (bool, camera.height x camera.width);
    `image` may be a different resolution (a generated edit): pixel coordinates are scaled. For an edit, pass
    `edit_mask`, the edit's own building mask at the edit's resolution: the two are intersected at photo resolution,
    eroded by ~1% of the building width, and halo-coloured texels near the boundary are dropped (module docstring)."""
    mask = np.asarray(mask)
    if mask.shape != (camera.height, camera.width):
        raise ValueError(f"mask is {mask.shape}, camera expects {(camera.height, camera.width)}")
    halo_zone = None
    if edit_mask is None:
        eroded = _erode(mask)
    else:
        edit_mask = np.asarray(edit_mask)
        if edit_mask.shape != image.shape[:2]:
            raise ValueError(f"edit_mask is {edit_mask.shape}, the edit image is {image.shape[:2]}")
        at_photo = cv2.resize(edit_mask.astype(np.uint8), (camera.width, camera.height),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        both = mask.astype(bool) & at_photo
        width = _mask_width(both)
        eroded = _erode(both, edit_erode_size(width))
        band = max(1, round(halo_band_frac * width))
        halo_zone = both & (cv2.distanceTransform(both.astype(np.uint8), cv2.DIST_L2, 3) < band)
    ppm, sizes = choose_ppm(prism, ppm)
    raw = [bake_wall(prism, camera, wall, size, image, eroded, halo_zone) for wall, size in zip(prism.walls, sizes)]
    walls = fill_walls(prism, raw)
    atlas, visible, used, rects = build_atlas(prism, walls, ppm, sizes)
    return Bake(walls, ppm, atlas, visible, used, rects)


def write_previews(bake, prefix):
    """<prefix>_atlas.png (what is baked) and <prefix>_atlas_visibility.png (green: from the photo, red: filled)."""
    prefix = str(prefix)
    Image.fromarray(bake.atlas).save(prefix + "_atlas.png")
    tint = bake.atlas.astype(np.float32)
    real = bake.atlas_visible[..., None]
    tint = np.where(real, tint * 0.65 + np.array([0, 90, 0]), tint * 0.5 + np.array([120, 0, 0]))
    tint = np.where(bake.atlas_used[..., None], tint, 30.0)  # space no wall uses
    Image.fromarray(tint.clip(0, 255).astype(np.uint8)).save(prefix + "_atlas_visibility.png")
    return prefix + "_atlas.png", prefix + "_atlas_visibility.png"


# ------------------------------------------------------------------------ glb


def _roof_mesh(prism):
    verts, faces = [], []
    for poly in prism.polygons:
        v2, f = trimesh.creation.triangulate_polygon(poly)
        f = np.asarray(f)
        for tri in f:  # keep every triangle counter-clockwise seen from above
            a, b, c = v2[tri]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                tri = tri[::-1]
            verts.append(np.column_stack([v2[tri], np.full(3, prism.height_m)]))
            faces.append(np.arange(3) + 3 * len(faces))
    roof = trimesh.Trimesh(np.concatenate(verts), np.array(faces), process=False)
    roof.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=ROOF_RGBA, metallicFactor=METALLIC_ROOF, roughnessFactor=ROUGHNESS_ROOF))
    return roof


def wall_uvs(bake, index):
    """Atlas UVs (glTF convention: v down from the top-left) of wall `index`'s corners p0-bottom, p1-bottom,
    p1-top, p0-top."""
    x, y, w, h = bake.rects[index]
    a_h, a_w = bake.atlas.shape[:2]
    u0, u1, v0, v1 = x / a_w, (x + w) / a_w, y / a_h, (y + h) / a_h
    return [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]


def build_glb(prism, bake, out_glb, source=None):
    """Write the textured prism. trimesh's uv convention is OpenGL's (v up) and its exporter flips it back to glTF's,
    so v is passed as 1 - v_image here."""
    verts, faces, uvs = [], [], []
    for wall in prism.walls:
        (x0, y0), (x1, y1), h = wall["p0"], wall["p1"], prism.height_m
        base = len(verts)
        verts += [(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, h), (x0, y0, h)]
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]  # outward for a CCW exterior ring
        uvs += [(u, 1.0 - v) for u, v in wall_uvs(bake, wall["index"])]
    walls = trimesh.Trimesh(np.array(verts), np.array(faces), process=False)
    walls.visual = trimesh.visual.TextureVisuals(uv=np.array(uvs), material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(bake.atlas), metallicFactor=METALLIC_WALL, roughnessFactor=ROUGHNESS_WALL))
    scene = trimesh.Scene()
    scene.add_geometry(walls, node_name="walls", geom_name="walls")
    scene.add_geometry(_roof_mesh(prism), node_name="roof", geom_name="roof")
    out_glb = Path(out_glb)
    out_glb.write_bytes(scene.export(file_type="glb"))

    from pygltflib import GLTF2  # the frame is not glTF's default, so say so inside the file

    gltf = GLTF2().load(str(out_glb))
    for node in gltf.nodes:  # (pygltflib drops Asset.extras on save, so the note rides on the nodes)
        node.extras = {"frame": FRAME_NOTE, "source_image": source, "px_per_m": round(bake.ppm, 2)}
    gltf.save(str(out_glb))
    return out_glb


# --------------------------------------------------------------------- driver


def _load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def _load_mask(path):
    return np.array(Image.open(path).convert("L")) > 127


def segment_edit(edit_path):
    """The edit's own building mask, from the real segmenter (perception.segment.segment_file, cached by image hash,
    so an edit already segmented for the alignment check costs nothing). Raises when it cannot be computed."""
    from perception.segment import segment_file

    return segment_file(edit_path)[0]


def bake_record(run_dir, photo, mask, scorched=None, force=False, scorched_mask=None):
    """Bake prism_photo.glb (and prism_scorched.glb when `scorched` exists) into a run directory and list them in
    record.json as record['textured_glbs'] (+ 'texture_bake' stats). -> the record dict.

    The scorched bake needs the edit's own building mask (`scorched_mask`: a path or array; else it is recomputed with
    segment_edit). Without one it is SKIPPED and says why in record['texture_bake']['scorched'], rather than baked with
    the original mask alone (which lets the edit's background leak in as a fringe)."""
    run_dir = Path(run_dir)
    record_path = run_dir / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    cam_info = record.get("openings_camera") or {}
    if not cam_info.get("reliable") and not force:
        raise ValueError(f"{run_dir.name}: the openings camera fit is not reliable; texturing it would smear the "
                         "photo across the wrong walls (pass force=True to override)")
    mask_arr = _load_mask(mask)
    camera = camera_from_record(record, mask_arr.shape[1], mask_arr.shape[0])
    prism = prism_from_record(record)

    listed, stats = {}, {}
    for key, image_path in (("photo", photo), ("scorched", scorched)):
        if image_path is None or not Path(image_path).is_file():
            continue
        edit_mask = None
        if key == "scorched":
            try:
                edit_mask = (_load_mask(scorched_mask) if isinstance(scorched_mask, (str, Path))
                             else scorched_mask if scorched_mask is not None else segment_edit(image_path))
                Image.fromarray((np.asarray(edit_mask) * 255).astype(np.uint8)).save(
                    run_dir / "prism_scorched_editmask.png")
            except Exception as exc:  # noqa: BLE001 - no segmenter / no weights: skip, do not bake unmasked
                stats[key] = {"skipped": f"no edit mask ({type(exc).__name__}: {exc})"}
                continue
        bake = bake_textures(prism, camera, _load_image(image_path), mask_arr, edit_mask=edit_mask)
        glb = build_glb(prism, bake, run_dir / f"prism_{key}.glb", source=Path(image_path).name)
        write_previews(bake, run_dir / f"prism_{key}")
        listed[key] = glb.name
        stats[key] = {"px_per_m": round(bake.ppm, 2), "atlas": list(bake.atlas.shape[1::-1]), "walls": bake.stats(),
                      "edit_mask": edit_mask is not None}
    record["textured_glbs"] = listed
    record["texture_bake"] = stats
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("photo")
    parser.add_argument("mask")
    parser.add_argument("--scorched", help="the edited photo (same framing); baked into prism_scorched.glb if it exists")
    parser.add_argument("--edit-mask", help="the edit's own building mask (default: recompute with the segmenter)")
    parser.add_argument("--force", action="store_true", help="bake even when the camera fit is not reliable")
    args = parser.parse_args()
    record = bake_record(args.run_dir, args.photo, args.mask, args.scorched, args.force, args.edit_mask)
    for key, info in record["texture_bake"].items():
        if "skipped" in info:
            print(f"{key}: SKIPPED, {info['skipped']}")
            continue
        counts = {}
        for w in info["walls"]:
            counts[w["source"]] = counts.get(w["source"], 0) + 1
        print(f"{key}: {len(info['walls'])} walls {counts}, {info['px_per_m']} px/m, atlas {info['atlas']}")
    print("textured_glbs:", record["textured_glbs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
