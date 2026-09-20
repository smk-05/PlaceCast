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

Before filling, the photographed walls are NORMALISED (no AI): each wall's low-frequency lighting is removed (divided
by a heavily blurred luminance, the mean restored), then its colour and exposure are matched to the building's
best-covered wall, so walls photographed in sun and shade read as one building.

Texels the camera cannot see are filled, never left blank and NEVER stretched or mirrored from the photo:
  * a small gap inside the photographed area is cv2.inpaint'ed;
  * on a wall with at least 30% photo coverage the rest may be inpainted by FLUX Fill (perception/prism_ai_fill.py:
    tiles of at most 1024 px, 128 px overlap, never a tile more than 60% empty, a hard call budget). What was
    AI-filled is recorded per wall (`ai_filled_fraction`) and tinted in the `prism_<variant>_ai.glb` twin;
  * everything else unseen (a whole wall the photo does not show, and any band above, below or beside a photographed
    wall's coverage) is a PROCEDURAL facade in the mean colour of the building's visible wall texels (per variant):
    vertical panel seams every 1.5-2 m, a floor line every 3.5 m, low noise, and for the scorched variant rust
    streaks. The photographed area fades into it over BLEND_M (0.5 m), so there is no hard edge; so does AI fill.
    The procedural colour is the NORMALISED mean of the building's photographed walls.
`WallTexture.source` records "photo" (coverage >= STRONG_COVERAGE), "partial" (>= KEEP_COVERAGE) or "procedural"
(a wall whose own visible texels are discarded as too few).
The roof is procedural too (not visible from the ground): ONE texture over the whole roof, not a repeating tile,
in the footprint's own axes. Photo variant: a dark membrane with a faint panel grid. Scorched: steel plates 3 x 6 m,
each slightly more or less rusty, a few soft soot patches. Low contrast, so from far away it reads as a plain weathered
roof; always darker than the walls.

glb. Vertices are ENU metres (x east, y north, z up): the frame web/src/main.js uses for the opening markers, with
the record's origin and ground. That is not glTF's Y-up convention on purpose: Cesium is told upAxis Z / forwardAxis X,
exactly as for the generated-mesh path, so it applies no axis correction and modelMatrix = enuToFixed places the model.
Walls: PBR, metallic 0.4, roughness 0.75, one atlas texture. Roof: metallic 0.6, roughness 0.55, one texture over
the whole roof (UV = position in `roof_frame`, so it never wraps).

Usage: python perception/prism_texture.py RUN_DIR PHOTO MASK [--scorched EDIT.png [--edit-mask MASK.png]] [--force]
       [--ai-plan | --ai [--ai-calls N] [--style 'stone facade']]     (--ai spends Replicate credit; <= 40 calls ever)
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
NEUTRAL_RGB = (128, 126, 120)
SEED = 20260919
BLEND_M = 0.5  # the photographed area fades into the procedural facade over this distance
FACADE_SEAM_M = (1.5, 2.0)  # vertical panel seams
FLOOR_PITCH_M = 3.5  # a floor line every this many metres up from the ground
NOISE_SIGMA = 0.035
PANEL_TONE_SIGMA = 0.025
SEAM_DARKEN, FLOOR_LINE_DARKEN = 0.84, 0.80
RUST_RGB = (150, 72, 32)  # the saturated rust of the scorched ROOF plates
RUST_STREAK_RGB = (128, 88, 64)  # wall streaks: the same hue, desaturated
STREAK_PER_M = 0.25  # about one streak per 4 m of wall
STREAK_LENGTH_M = (0.6, 3.5)
STREAK_ALPHA = (0.08, 0.17)
SCORCHED_NEUTRAL_RGB = (66, 54, 47)  # a scorched facade with no photograph to take its colour from
DELIGHT_SIGMA_M = 4.0  # the lighting scale removed from each wall: the luminance blurred this heavily
GAIN_CLIP = (0.5, 2.0)  # the largest brightening / darkening the de-lighting may apply to a texel
MATCH_STD_CLIP = (0.8, 1.25)  # how far a wall's contrast may be stretched to the reference wall's
AI_TINT_RGB, AI_TINT_STRENGTH = (255, 0, 200), 0.5  # what the "AI-filled" overlay paints over AI texels
AI_STD_CLIP = (0.6, 1.6)  # how far AI-filled contrast may be stretched to the wall's photographed contrast
ROOF_PPM, ROOF_MAX_SIDE = 12.0, 2048  # one texture for the whole roof: px/m, and its longest side
ROOF_PLATE_M = (3.0, 6.0)  # scorched steel plates: 3 m tall rows of 6 m plates
ROOF_MEMBRANE_RGB = (56.0, 58.0, 62.0)
ROOF_MAX_LUMA = 0.20  # the roof's mean luminance is at most this ...
ROOF_WALL_RATIO = 0.6  # ... and at most this fraction of the walls'
METALLIC_WALL, ROUGHNESS_WALL = 0.4, 0.75
METALLIC_ROOF, ROUGHNESS_ROOF = 0.6, 0.55
NEAR_M = 0.5

FRAME_NOTE = "ENU metres (x east, y north, z up); load with Cesium upAxis Z / forwardAxis X and modelMatrix = enuToFixed"


@dataclass
class WallTexture:
    index: int
    rgb: np.ndarray  # (h, w, 3) uint8, row 0 is the TOP of the wall, column 0 is at p0
    visible: np.ndarray  # (h, w) bool: baked from the photo (False = filled)
    coverage: float
    source: str = "photo"  # photo | partial | procedural
    ai_weight: np.ndarray = None  # (h, w) float in [0, 1]: how much of each texel's colour is AI-filled (None: none)
    photo_fraction: float = 0.0  # the shares of this wall's colour that come from the photo / from FLUX Fill /
    ai_filled_fraction: float = 0.0  # from the procedural facade (they sum to 1)
    procedural_fraction: float = 1.0


@dataclass
class Bake:
    walls: list
    ppm: float
    atlas: np.ndarray = None  # (A_h, A_w, 3) uint8
    atlas_visible: np.ndarray = None  # (A_h, A_w) bool
    atlas_used: np.ndarray = None  # (A_h, A_w) bool: inside some wall's rectangle
    roof_map: np.ndarray = None  # (H, W, 3) uint8: one texture spanning the whole roof, no tiling
    roof_frame: tuple = ()  # (cx, cy, theta, a_min, a_max, b_min, b_max): where roof_map sits on the roof (roof_frame())
    base_rgb: tuple = ()  # the mean colour of the visible wall texels: what the procedural facade is built from
    rects: list = field(default_factory=list)  # per wall (x, y, w, h) in the atlas, excluding the PAD ring
    tint_atlas: np.ndarray = None  # the atlas with AI-filled texels tinted (None when nothing was AI-filled)
    normalisation: dict = field(default_factory=dict)  # {"reference_wall": i, "gains": {wall: ...}}
    ai_stats: dict = field(default_factory=dict)  # what the AI fill did: tiles sent, failed, skipped, cache hits

    def stats(self):
        return [{"wall": t.index, "coverage": round(t.coverage, 3), "source": t.source,
                 "photo_fraction": round(t.photo_fraction, 4), "ai_filled_fraction": round(t.ai_filled_fraction, 4),
                 "procedural_fraction": round(t.procedural_fraction, 4)} for t in self.walls]

    @property
    def any_ai(self):
        return any(t.ai_filled_fraction > 0 for t in self.walls)


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
    """cv2.inpaint the SMALL connected components of `hole`. -> (rgb, the components left unfilled)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    small = np.zeros(hole.shape, bool)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] <= SMALL_HOLE_PX:
            small |= labels == i
    if small.any():
        rgb = cv2.inpaint(rgb, small.astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)
    return rgb, hole & ~small


def _interior_holes(visible):
    """Unseen texels that have a seen texel above AND below them in their column: gaps INSIDE the photographed area
    (a lamp post, a branch). What lies above the top or below the bottom of a column's coverage is a band, not a
    hole, and so are columns with no coverage at all."""
    if not visible.any():
        return np.zeros_like(visible)
    has = visible.any(axis=0)
    first = np.where(has, visible.argmax(axis=0), visible.shape[0])
    last = np.where(has, visible.shape[0] - 1 - visible[::-1].argmax(axis=0), -1)
    rows = np.arange(visible.shape[0])[:, None]
    return (rows >= first[None]) & (rows <= last[None]) & ~visible


def _photo_weight(photo_mask, ppm):
    """1 deep inside the photographed area, falling smoothly to 0 over BLEND_M at its boundary with unseen texels.
    The outer border of the texture is not a boundary (the neighbouring wall is not a hole)."""
    if not photo_mask.any():
        return np.zeros(photo_mask.shape, np.float32)
    padded = cv2.copyMakeBorder(photo_mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    w = np.clip(dist / (BLEND_M * ppm), 0.0, 1.0)
    return (w * w * (3.0 - 2.0 * w) * photo_mask).astype(np.float32)


def procedural_facade(w_px, h_px, ppm, height_m, base_rgb, style, seed):
    """A plain building facade in the building's own colour: (h, w, 3) uint8 for a wall texture, row 0 at the top.

    `base_rgb` is the mean colour of the building's visible wall texels. On it: vertical panel seams every 1.5-2 m
    (jittered), each panel a touch lighter or darker; a floor line every 3.5 m up from the ground; low noise. The
    'scorched' style adds rust streaks running down from the seams and floor lines. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    img = np.empty((h_px, w_px, 3), np.float32)
    img[:] = np.asarray(base_rgb, np.float32)

    edges, x = [0], 0.0
    while True:
        x += rng.uniform(*FACADE_SEAM_M) * ppm
        if x >= w_px - 0.5 * ppm:
            break
        edges.append(int(round(x)))
    edges.append(w_px)
    for a, b in zip(edges[:-1], edges[1:]):
        img[:, a:b] *= 1.0 + rng.normal(0.0, PANEL_TONE_SIGMA)
    for e in edges[1:-1]:
        img[:, max(0, e - 1) : e + 1] *= SEAM_DARKEN
    floor_rows = []
    z = FLOOR_PITCH_M
    while z < height_m - 0.3:
        r = int(round((1.0 - z / height_m) * h_px))
        floor_rows.append(r)
        img[max(0, r - 1) : r + 2] *= FLOOR_LINE_DARKEN
        z += FLOOR_PITCH_M

    noise = cv2.GaussianBlur(rng.normal(0.0, 1.0, (h_px, w_px)).astype(np.float32), (0, 0), 0.8)
    img *= 1.0 + NOISE_SIGMA * (noise / max(noise.std(), 1e-6))[..., None]

    if style == "scorched":
        # Rust streaks: few, thin, irregular in position, length, width and path, desaturated, alpha ~0.15 at most.
        # From 200 m they average away to nothing; up close they are a stain, not a stripe.
        alpha = np.zeros((h_px, w_px), np.float32)
        for _ in range(max(1, int(w_px / ppm * STREAK_PER_M))):
            x = float(rng.integers(0, w_px))
            r0 = int(rng.integers(0, max(1, int(h_px * 0.8))))
            r1 = min(h_px, r0 + int(rng.uniform(*STREAK_LENGTH_M) * ppm))
            n = r1 - r0
            if n < 4:
                continue
            path = x + np.cumsum(rng.normal(0.0, 0.35, n))  # the streak wanders
            width = np.clip(rng.uniform(1.0, 2.6) * (1.0 + 0.4 * rng.normal(0.0, 1.0, n).clip(-1, 1)), 0.8, 3.5)
            fade = np.linspace(1.0, 0.0, n, dtype=np.float32) ** 1.5 * (0.6 + 0.4 * rng.random(n))
            peak = rng.uniform(*STREAK_ALPHA)
            for i in range(n):
                lo, hi = int(round(path[i] - width[i] / 2)), int(round(path[i] + width[i] / 2)) + 1
                lo, hi = max(0, lo), min(w_px, hi)
                if hi > lo:
                    alpha[r0 + i, lo:hi] = np.maximum(alpha[r0 + i, lo:hi], fade[i] * peak)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)[..., None]
        img = img * (1.0 - alpha) + np.asarray(RUST_STREAK_RGB, np.float32) * alpha
    return img.clip(0, 255).astype(np.uint8)


def _lowfreq(values, weight, sigma_px):
    """Normalised-convolution blur of `values` over the texels where `weight` is 1, computed at 1/4 resolution."""
    h, w = values.shape
    k = 4
    sh, sw = max(1, h // k), max(1, w // k)
    v = cv2.resize(values * weight, (sw, sh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(weight, (sw, sh), interpolation=cv2.INTER_AREA)
    sigma = max(1.0, sigma_px / k)
    num, den = cv2.GaussianBlur(v, (0, 0), sigma), cv2.GaussianBlur(m, (0, 0), sigma)
    low = cv2.resize(num / np.maximum(den, 1e-4), (w, h), interpolation=cv2.INTER_LINEAR)
    known = cv2.resize(den, (w, h), interpolation=cv2.INTER_LINEAR) > 0.02
    return np.where(known, low, float(values[weight > 0].mean()))


def delight(rgb, visible, ppm):
    """Remove one wall's low-frequency lighting: divide by the luminance blurred over ~DELIGHT_SIGMA_M (over the
    photographed texels only), then multiply by its mean, so the wall keeps its exposure but loses the gradient
    (sun on one side, a cast shadow, a dark corner). The gain is one number per texel, so chroma is untouched.
    -> (float32 (h, w, 3) in 0..255, the gain map)."""
    lum = _luma(rgb)
    low = _lowfreq(lum, visible.astype(np.float32), DELIGHT_SIGMA_M * ppm)
    mean_low = float(low[visible].mean())
    gain = np.clip(mean_low / np.maximum(low, 1e-3), *GAIN_CLIP)
    return np.clip(rgb.astype(np.float32) * gain[..., None], 0.0, 255.0), gain


def normalise_walls(raw, ppm):
    """(rgb, visible) per wall -> (the same, normalised, the normalisation record). See the module docstring.

    Walls with too little coverage (< KEEP_COVERAGE) are left alone: their photo texels are discarded anyway. The
    reference is the best-covered wall: every other wall's per-channel mean is moved onto it and its contrast
    stretched toward it (within MATCH_STD_CLIP), so colour and exposure agree across the building."""
    cover = [float(vis.mean()) for _, vis in raw]
    kept = [i for i, c in enumerate(cover) if c >= KEEP_COVERAGE]
    if not kept:
        return list(raw), {}
    flat = {i: delight(raw[i][0], raw[i][1], ppm)[0] for i in kept}
    ref = max(kept, key=lambda i: cover[i])
    ref_px = flat[ref][raw[ref][1]]
    mu_ref, sd_ref = ref_px.mean(axis=0), ref_px.std(axis=0)
    out, info = list(raw), {"reference_wall": ref, "walls": {}}
    for i in kept:
        px = flat[i][raw[i][1]]
        mu, sd = px.mean(axis=0), px.std(axis=0)
        scale = np.ones(3, np.float32) if i == ref else np.clip(sd_ref / np.maximum(sd, 1e-3), *MATCH_STD_CLIP)
        matched = flat[i] if i == ref else (flat[i] - mu) * scale + mu_ref
        out[i] = (np.clip(matched, 0, 255).round().astype(np.uint8), raw[i][1])
        info["walls"][i] = {"mean_before": [round(float(v), 1) for v in mu], "mean_after": [round(float(v), 1) for v in
                            (mu_ref if i != ref else mu)], "contrast_scale": [round(float(v), 3) for v in scale]}
    return out, info


def match_ai_to_photo(ai_rgb, region, rgb, photo):
    """FLUX Fill invents colour as well as structure (on a dark scorched facade it painted a near-white block). Move
    the AI-filled texels (`region`) onto the wall's own photographed statistics: per-channel mean to the photo's, and
    its contrast toward the photo's within AI_STD_CLIP. Structure (window rows, courses) is kept; colour and exposure
    are the wall's. -> (matched float32 array, {"mean_before": ..., "mean_after": ...})."""
    if not region.any() or not photo.any():
        return ai_rgb, {}
    a, p = ai_rgb[region].astype(np.float32), rgb[photo].astype(np.float32)
    mu_a, sd_a, mu_p, sd_p = a.mean(axis=0), a.std(axis=0), p.mean(axis=0), p.std(axis=0)
    scale = np.clip(sd_p / np.maximum(sd_a, 1e-3), *AI_STD_CLIP)
    matched = np.clip((ai_rgb - mu_a) * scale + mu_p, 0.0, 255.0)
    return matched, {"mean_before": [round(float(v), 1) for v in mu_a], "mean_after": [round(float(v), 1) for v in mu_p]}


def _ai_context(rgb, photo, proc):
    """The image FLUX Fill sees for a wall: the photo where there is photo, the procedural facade elsewhere (so the
    masked area is surrounded by plausible colour rather than garbage)."""
    return np.where(photo[..., None], rgb, proc)


def fill_walls(prism, raw, ppm, style="photo", base_rgb=None, ai=None, normalise=True):
    """raw: list of (rgb, visible). -> (list of WallTexture with every texel filled, the building's mean colour,
    the normalisation record).

    1. The photographed walls are normalised (`normalise_walls`).
    2. Nothing is ever stretched or mirrored from the photo. What the camera did not see is a PROCEDURAL facade
       (`procedural_facade`) in the NORMALISED mean colour of the building's visible wall texels: a whole wall the
       photo does not show, and any band above or below (or beside) a photographed wall's coverage.
    3. Small gaps inside the photographed area are cv2.inpaint'ed. If `ai` (a prism_ai_fill.AIFill) is given, the
       remaining unseen texels of every wall with >= ai.min_coverage of photo are offered to FLUX Fill instead of
       being left procedural (tiles under its rules); its result replaces the procedural facade there, after its
       colour and exposure are matched to the wall's own photographed texels (`match_ai_to_photo`).
    4. Photo -> fill and AI -> procedural boundaries fade over BLEND_M, so there is no hard edge.
    `base_rgb` replaces the mean colour of the visible texels (bake_procedural has none)."""
    norm = {}
    if normalise:
        raw, norm = normalise_walls(raw, ppm)
    out = [WallTexture(i, rgb.copy(), vis.copy(), float(vis.mean())) for i, (rgb, vis) in enumerate(raw)]
    kept = [t for t in out if t.coverage >= KEEP_COVERAGE]
    seen = [t.rgb[t.visible] for t in kept if t.visible.any()]
    base = np.concatenate(seen).mean(axis=0) if seen else np.array(NEUTRAL_RGB, np.float32)
    if base_rgb is not None:  # no photograph to take the colour from (or one to override it)
        base = np.asarray(base_rgb, np.float32)
    style_id = {"photo": 0, "scorched": 1}.get(style, 2)

    prep = {}
    for t in out:
        vis = t.visible if t.coverage >= KEEP_COVERAGE else np.zeros_like(t.visible)
        holes = _interior_holes(vis)
        rgb, unfilled = _inpaint_small(t.rgb, holes)
        photo = vis | (holes & ~unfilled)
        h_px, w_px = vis.shape
        proc = procedural_facade(w_px, h_px, ppm, prism.height_m, base, style, seed=[SEED, t.index, style_id])
        prep[t.index] = (vis, rgb, photo, proc)

    ai_out = {}
    if ai is not None:
        eligible = {i: ~photo for i, (vis, rgb, photo, proc) in prep.items()
                    if out[i].coverage >= ai.min_coverage and (~photo).any()}
        context = {i: _ai_context(prep[i][1], prep[i][2], prep[i][3]) for i in eligible}
        ai_out = ai.run(context, eligible)
        ai.stats["colour_match"] = {}
        for i, (ai_rgb, cover) in list(ai_out.items()):  # the AI's colour is matched to the wall's photographed colour
            matched, info = match_ai_to_photo(ai_rgb, cover & ~prep[i][2], prep[i][1], prep[i][2])
            ai_out[i] = (matched, cover)
            ai.stats["colour_match"][i] = info

    for t in out:
        vis, rgb, photo, proc = prep[t.index]
        w_photo = _photo_weight(photo, ppm)
        layer = proc.astype(np.float32)
        ai_weight = np.zeros(vis.shape, np.float32)
        if t.index in ai_out:
            ai_rgb, cover = ai_out[t.index]
            w_ai = _photo_weight(cover | photo, ppm)  # AI fades into the procedural facade where its tiles end
            layer = ai_rgb * w_ai[..., None] + layer * (1.0 - w_ai[..., None])
            ai_weight = ((1.0 - w_photo) * w_ai).astype(np.float32)
        w = w_photo[..., None]
        t.rgb = (rgb.astype(np.float32) * w + layer * (1.0 - w)).round().astype(np.uint8)
        t.visible = vis
        t.ai_weight = ai_weight if ai_weight.any() else None
        t.photo_fraction = float(w_photo.mean())
        t.ai_filled_fraction = float(ai_weight.mean())
        t.procedural_fraction = float(max(0.0, 1.0 - t.photo_fraction - t.ai_filled_fraction))
        t.source = "photo" if t.coverage >= STRONG_COVERAGE else "partial" if t.coverage >= KEEP_COVERAGE else "procedural"
    return out, base, norm


# ------------------------------------------------------------ procedural roof


def roof_frame(prism):
    """A frame aligned with the footprint's longest edge, and the roof's bounds in it: (cx, cy, theta, a_min, a_max,
    b_min, b_max). The roof texture is laid out in this frame so its plates run parallel to the building."""
    edge = max(prism.walls, key=lambda w: w["length_m"])
    d = np.asarray(edge["p1"]) - np.asarray(edge["p0"])
    theta = math.atan2(d[1], d[0])
    pts = np.concatenate([np.asarray(p.exterior.coords)[:-1] for p in prism.polygons])  # without the closing repeat
    cx, cy = pts.mean(axis=0)
    a, b = _to_frame(pts, cx, cy, theta)
    return (float(cx), float(cy), theta, float(a.min()), float(a.max()), float(b.min()), float(b.max()))


def _to_frame(xy, cx, cy, theta):
    x, y = np.asarray(xy)[:, 0] - cx, np.asarray(xy)[:, 1] - cy
    c, s = math.cos(theta), math.sin(theta)
    return x * c + y * s, -x * s + y * c


def roof_uv(frame, xy):
    """UV (v up, OpenGL convention: trimesh flips it on export) of ENU (x, y) points in the roof texture's frame."""
    cx, cy, theta, a0, a1, b0, b1 = frame
    a, b = _to_frame(xy, cx, cy, theta)
    return np.column_stack([(a - a0) / (a1 - a0), (b - b0) / (b1 - b0)])


def _luma(rgb):
    rgb = np.asarray(rgb, np.float32)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) / 255.0


def _field(rng, shape, sigma_px):
    """Unit-variance smooth noise over `shape` (Gaussian-filtered white noise): one field across the WHOLE roof."""
    fy, fx = np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1])
    gauss = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (fy[:, None] ** 2 + fx[None, :] ** 2))
    n = np.fft.ifft2(np.fft.fft2(rng.normal(0.0, 1.0, shape)) * gauss).real.astype(np.float32)
    n -= n.mean()
    return n / max(float(n.std()), 1e-6)


def roof_texture(style, wall_rgb, size_m, seed=0):
    """One texture for the whole roof, (H, W, 3) uint8, `size_m` = (width, height) of the roof in its frame. Not a
    repeating tile: every noise field spans the entire roof, so nothing repeats. Low contrast: from 200 m it should
    read as a plain weathered roof, with detail (plate joints, rivets, grain) only up close.

    photo:    dark grey membrane, a faint panel grid every 3 m, broad tonal drift, fine grain.
    scorched: steel plates 3 x 6 m in staggered rows, each a little more or less rusty, a few soft soot patches,
              faint joints and rivets.
    Scaled so its mean luminance stays below both ROOF_MAX_LUMA and ROOF_WALL_RATIO x the walls' (`wall_rgb`)."""
    rng = np.random.default_rng([seed, {"photo": 0, "scorched": 1}.get(style, 2)])
    ppm = min(ROOF_PPM, ROOF_MAX_SIDE / max(size_m))
    w_px, h_px = max(8, math.ceil(size_m[0] * ppm)), max(8, math.ceil(size_m[1] * ppm))
    shape = (h_px, w_px)
    yy, xx = np.mgrid[0:h_px, 0:w_px].astype(np.float32)
    broad = _field(rng, shape, 6.0 * ppm)
    mid = _field(rng, shape, 1.2 * ppm)
    fine = _field(rng, shape, 0.8)

    if style == "scorched":
        steel, rust = np.array([72.0, 66.0, 62.0]), np.array(RUST_RGB, np.float32)
        plate_h, plate_w = ROOF_PLATE_M[0] * ppm, ROOF_PLATE_M[1] * ppm
        row = np.floor(yy / plate_h).astype(int)
        shifted = xx + (row % 2) * plate_w / 2  # alternate rows stagger by half a plate
        col = np.floor(shifted / plate_w).astype(int)
        amount = rng.uniform(0.0, 0.3, (row.max() + 1, col.max() + 1))[row, col]
        mix = np.clip(amount + 0.06 * broad + 0.03 * mid, 0.0, 0.5)[..., None]
        img = steel * (1.0 - mix) + rust * mix
        seam = ((yy % plate_h) < 1.0) | ((shifted % plate_w) < 1.0)
        img *= np.where(seam, 0.86, 1.0)[..., None]  # faint joints
        step, inset = max(2, int(round(0.5 * ppm))), max(1, int(round(0.15 * ppm)))
        rows = (np.arange(0, h_px, plate_h)[:, None] + np.array([inset, plate_h - inset])[None]).ravel().astype(int)
        cols = np.arange(0, w_px, step)
        rr, cc = np.meshgrid(rows[rows < h_px], cols)
        rivet = np.zeros(shape, np.float32)
        rivet[rr.ravel(), cc.ravel()] = 1.0
        img *= (1.0 + 0.10 * cv2.GaussianBlur(rivet, (0, 0), 0.7) * 4.0)[..., None]
        soot = 1.0 - 0.25 * np.clip((_field(rng, shape, 2.5 * ppm) - 1.5) / 1.0, 0.0, 1.0)  # only the tail: few patches
        img *= soot[..., None]
        img *= (1.0 + 0.02 * fine + 0.02 * mid)[..., None]
    else:
        img = np.empty((h_px, w_px, 3), np.float32)
        img[:] = ROOF_MEMBRANE_RGB
        grid = 3.0 * ppm
        lines = ((xx % grid) < 1.0) | ((yy % grid) < 1.0)
        img *= np.where(lines, 0.95, 1.0)[..., None]
        img *= (1.0 + 0.03 * broad + 0.015 * mid + 0.015 * fine)[..., None]

    target = min(ROOF_MAX_LUMA, ROOF_WALL_RATIO * float(_luma(wall_rgb)))
    current = float(_luma(img).mean())
    if current > target > 0:
        img *= target / current
    return img.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------- atlas


def _tinted(t):
    """A wall's texture with its AI-filled texels painted over with AI_TINT_RGB (for the overlay glb)."""
    if t.ai_weight is None:
        return t.rgb
    a = (AI_TINT_STRENGTH * t.ai_weight)[..., None]
    return (t.rgb.astype(np.float32) * (1.0 - a) + np.asarray(AI_TINT_RGB, np.float32) * a).round().astype(np.uint8)


def build_atlas(prism, walls, ppm, sizes, tinted=False):
    pos, (a_w, a_h) = plan_layout(sizes)
    atlas = np.full((a_h, a_w, 3), NEUTRAL_RGB, np.uint8)
    visible = np.zeros((a_h, a_w), bool)
    used = np.zeros((a_h, a_w), bool)
    rects = []
    for t, (x, y), (w, h) in zip(walls, pos, sizes):
        padded = cv2.copyMakeBorder(_tinted(t) if tinted else t.rgb, PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
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


def bake_textures(prism, camera, image, mask, ppm=PPM, edit_mask=None, halo_band_frac=HALO_BAND_FRAC,
                  style="photo", ai=None, normalise=True):
    """Bake `image` (H, W, 3 uint8 RGB, the same framing as the photo the camera was fitted to) onto every wall.

    `mask` is the ORIGINAL photo's building mask at the PHOTO's resolution (bool, camera.height x camera.width);
    `image` may be a different resolution (a generated edit): pixel coordinates are scaled. For an edit, pass
    `edit_mask`, the edit's own building mask at the edit's resolution: the two are intersected at photo resolution,
    eroded by ~1% of the building width, and halo-coloured texels near the boundary are dropped (module docstring).
    `style` ("photo" or "scorched") picks the procedural facade and roof that fill what the camera did not see.
    `ai` (a prism_ai_fill.AIFill) turns on FLUX Fill for well-covered walls; `normalise=False` skips the de-lighting and
    colour matching (the tests that want the raw bake)."""
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
    walls, base, norm = fill_walls(prism, raw, ppm, style, ai=ai, normalise=normalise)
    atlas, visible, used, rects = build_atlas(prism, walls, ppm, sizes)
    frame = roof_frame(prism)
    bake = Bake(walls, ppm, atlas=atlas, atlas_visible=visible, atlas_used=used, rects=rects,
                roof_map=roof_texture(style, base, (frame[4] - frame[3], frame[6] - frame[5]), SEED), roof_frame=frame,
                base_rgb=tuple(int(round(c)) for c in base), normalisation=norm,
                ai_stats=dict(ai.stats) if ai is not None else {})
    if bake.any_ai:
        bake.tint_atlas = build_atlas(prism, walls, ppm, sizes, tinted=True)[0]
    return bake


def bake_procedural(prism, style, base_rgb, ppm=PPM):
    """A bake with NO photo content at all: every wall a procedural facade in `base_rgb`, and the procedural roof.
    For a building whose camera fit is unreliable, so a photo cannot be projected onto its walls but the viewer can
    still show it like the others."""
    ppm, sizes = choose_ppm(prism, ppm)
    raw = [(np.zeros((h, w, 3), np.uint8), np.zeros((h, w), bool)) for w, h in sizes]
    walls, base, _ = fill_walls(prism, raw, ppm, style, base_rgb=base_rgb, normalise=False)
    atlas, visible, used, rects = build_atlas(prism, walls, ppm, sizes)
    frame = roof_frame(prism)
    return Bake(walls, ppm, atlas=atlas, atlas_visible=visible, atlas_used=used, rects=rects,
                roof_map=roof_texture(style, base, (frame[4] - frame[3], frame[6] - frame[5]), SEED), roof_frame=frame,
                base_rgb=tuple(int(round(c)) for c in base))


def mean_colour(image, mask):
    """Mean RGB of `image` under `mask` (bool; may be another resolution: nearest-neighbour), None if empty."""
    mask = np.asarray(mask).astype(np.uint8)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    sel = mask.astype(bool)
    return image[sel].reshape(-1, 3).mean(axis=0) if sel.any() else None


def procedural_base(style, photo_mean, edit_mean=None):
    """The wall colour of a procedural-only building. photo: the photo's mean colour under the building mask;
    scorched: the edit's if there is one, else the photo's pulled toward SCORCHED_NEUTRAL_RGB."""
    photo_mean = np.array(NEUTRAL_RGB, np.float32) if photo_mean is None else np.asarray(photo_mean, np.float32)
    if style != "scorched":
        return photo_mean
    if edit_mean is not None:
        return np.asarray(edit_mean, np.float32)
    return 0.4 * photo_mean + 0.6 * np.array(SCORCHED_NEUTRAL_RGB, np.float32)


def write_previews(bake, prefix):
    """<prefix>_atlas.png (what is baked), <prefix>_atlas_visibility.png (green: from the photo, red: filled) and
    <prefix>_roof.png (the roof texture)."""
    prefix = str(prefix)
    Image.fromarray(bake.atlas).save(prefix + "_atlas.png")
    Image.fromarray(bake.roof_map).save(prefix + "_roof.png")
    if bake.tint_atlas is not None:
        Image.fromarray(bake.tint_atlas).save(prefix + "_atlas_ai.png")  # AI-filled texels tinted
    tint = bake.atlas.astype(np.float32)
    real = bake.atlas_visible[..., None]
    tint = np.where(real, tint * 0.65 + np.array([0, 90, 0]), tint * 0.5 + np.array([120, 0, 0]))
    tint = np.where(bake.atlas_used[..., None], tint, 30.0)  # space no wall uses
    Image.fromarray(tint.clip(0, 255).astype(np.uint8)).save(prefix + "_atlas_visibility.png")
    return prefix + "_atlas.png", prefix + "_atlas_visibility.png"


# ------------------------------------------------------------------------ glb


def _roof_mesh(prism, bake):
    """The roof, triangulated, textured with the single roof map: UV = the vertex's position in the roof frame."""
    verts, faces = [], []
    for poly in prism.polygons:
        v2, f = trimesh.creation.triangulate_polygon(poly)
        for tri in np.asarray(f):  # keep every triangle counter-clockwise seen from above
            a, b, c = v2[tri]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                tri = tri[::-1]
            verts.append(np.column_stack([v2[tri], np.full(3, prism.height_m)]))
            faces.append(np.arange(3) + 3 * len(faces))
    verts = np.concatenate(verts)
    roof = trimesh.Trimesh(verts, np.array(faces), process=False)
    roof.visual = trimesh.visual.TextureVisuals(uv=roof_uv(bake.roof_frame, verts[:, :2]),
                                                material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(bake.roof_map), metallicFactor=METALLIC_ROOF, roughnessFactor=ROUGHNESS_ROOF))
    return roof


def wall_uvs(bake, index):
    """Atlas UVs (glTF convention: v down from the top-left) of wall `index`'s corners p0-bottom, p1-bottom,
    p1-top, p0-top."""
    x, y, w, h = bake.rects[index]
    a_h, a_w = bake.atlas.shape[:2]
    u0, u1, v0, v1 = x / a_w, (x + w) / a_w, y / a_h, (y + h) / a_h
    return [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]


def build_glb(prism, bake, out_glb, source=None, tinted=False):
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
        baseColorTexture=Image.fromarray(bake.tint_atlas if tinted and bake.tint_atlas is not None else bake.atlas),
        metallicFactor=METALLIC_WALL, roughnessFactor=ROUGHNESS_WALL))
    scene = trimesh.Scene()
    scene.add_geometry(walls, node_name="walls", geom_name="walls")
    scene.add_geometry(_roof_mesh(prism, bake), node_name="roof", geom_name="roof")
    out_glb = Path(out_glb)
    out_glb.write_bytes(scene.export(file_type="glb"))

    from pygltflib import GLTF2  # the frame is not glTF's default, so say so inside the file

    gltf = GLTF2().load(str(out_glb))
    for node in gltf.nodes:  # (pygltflib drops Asset.extras on save, so the note rides on the nodes)
        node.extras = {"frame": FRAME_NOTE, "source_image": source, "px_per_m": round(bake.ppm, 2),
                       "ai_overlay": bool(tinted)}
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


def bake_record(run_dir, photo, mask, scorched=None, force=False, scorched_mask=None, ai=None):
    """Bake prism_photo.glb (and prism_scorched.glb when `scorched` exists) into a run directory and list them in
    record.json as record['textured_glbs'] (+ 'texture_bake' stats). -> the record dict.

    The scorched bake needs the edit's own building mask (`scorched_mask`: a path or array; else it is recomputed with
    segment_edit). Without one it is SKIPPED and says why in record['texture_bake']['scorched'], rather than baked with
    the original mask alone (which lets the edit's background leak in as a fringe).

    `ai`: None, or a callable `variant -> prism_ai_fill.AIFill | None` that says whether (and with what backend, prompt
    and call cap) each variant may use FLUX Fill on its well-covered walls. It is never used for a procedural-only
    bake. When anything was AI-filled the record gets `textured_glbs_ai` (a tinted twin glb per variant, for the
    viewer's "AI-filled" overlay) and every wall in `texture_bake` carries `ai_filled_fraction`.

    A record whose camera fit is NOT reliable (and no `force`) gets PROCEDURAL-ONLY glbs, photo and scorched: walls and
    roof with no photo content, coloured from the photo's mean under the building mask, so it renders like the others
    without smearing a photograph across the wrong walls."""
    run_dir = Path(run_dir)
    record_path = run_dir / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    cam_info = record.get("openings_camera") or {}
    mask_arr = _load_mask(mask)
    prism = prism_from_record(record)
    if not cam_info.get("reliable") and not force:
        return _bake_record_procedural(record, record_path, run_dir, prism, photo, mask_arr, scorched)
    camera = camera_from_record(record, mask_arr.shape[1], mask_arr.shape[0])

    listed, stats, listed_ai = {}, {}, {}
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
        variant_ai = ai(key) if ai is not None else None
        bake = bake_textures(prism, camera, _load_image(image_path), mask_arr, edit_mask=edit_mask, style=key,
                             ai=variant_ai)
        glb = build_glb(prism, bake, run_dir / f"prism_{key}.glb", source=Path(image_path).name)
        write_previews(bake, run_dir / f"prism_{key}")
        listed[key] = glb.name
        if bake.any_ai:
            listed_ai[key] = build_glb(prism, bake, run_dir / f"prism_{key}_ai.glb", source=Path(image_path).name,
                                       tinted=True).name
        stats[key] = {"px_per_m": round(bake.ppm, 2), "atlas": list(bake.atlas.shape[1::-1]), "walls": bake.stats(),
                      "edit_mask": edit_mask is not None, "base_rgb": list(bake.base_rgb),
                      "normalisation": bake.normalisation, "ai": bake.ai_stats}
    record["textured_glbs"] = listed
    if listed_ai:
        record["textured_glbs_ai"] = listed_ai
    else:
        record.pop("textured_glbs_ai", None)
    record["texture_bake"] = stats
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def _bake_record_procedural(record, record_path, run_dir, prism, photo, mask_arr, scorched):
    photo_mean = mean_colour(_load_image(photo), mask_arr) if photo is not None and Path(photo).is_file() else None
    edit_mean = None
    if scorched is not None and Path(scorched).is_file():  # an edit exists: take its colour, under its own mask
        try:
            edit_mean = mean_colour(_load_image(scorched), segment_edit(scorched))  # its own mask is enough for a colour
        except Exception:  # noqa: BLE001 - no segmenter: fall back to the derived scorched colour
            edit_mean = None
    listed, stats = {}, {}
    for key in ("photo", "scorched"):
        base = procedural_base(key, photo_mean, edit_mean if key == "scorched" else None)
        bake = bake_procedural(prism, key, base)
        glb = build_glb(prism, bake, run_dir / f"prism_{key}.glb", source="procedural (camera fit unreliable)")
        write_previews(bake, run_dir / f"prism_{key}")
        listed[key] = glb.name
        stats[key] = {"procedural_only": True, "px_per_m": round(bake.ppm, 2), "atlas": list(bake.atlas.shape[1::-1]),
                      "walls": bake.stats(), "edit_mask": False, "base_rgb": list(bake.base_rgb)}
    record["textured_glbs"] = listed
    record["texture_bake"] = stats
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def _ai_factory(args):
    """The `ai` callable for bake_record built from the command line: one shared persistent budget, a per-run cap per
    variant, a per-run disk cache, and a prompt that names the building and (for scorched) its weathering."""
    from dotenv import load_dotenv

    from perception import prism_ai_fill as F

    load_dotenv(ROOT / ".env")
    plan_only = args.ai_plan and not args.ai
    budget = F.PersistentBudget(cap=args.ai_budget)
    backend = None if plan_only else F.FluxFillBackend(budget, Path(args.run_dir) / "ai_fill_cache")

    def make(key):
        style = args.style + (", weathered, soot-stained and scorched, dark aged masonry" if key == "scorched" else "")
        return F.AIFill(backend, F.BASE_PROMPT.format(style=style), max_calls=args.ai_calls)

    return make


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("photo")
    parser.add_argument("mask")
    parser.add_argument("--scorched", help="the edited photo (same framing); baked into prism_scorched.glb if it exists")
    parser.add_argument("--edit-mask", help="the edit's own building mask (default: recompute with the segmenter)")
    parser.add_argument("--force", action="store_true", help="bake even when the camera fit is not reliable")
    parser.add_argument("--ai", action="store_true", help="FLUX Fill (Replicate, billed) on the well-covered walls")
    parser.add_argument("--ai-plan", action="store_true", help="plan the AI tiles for each variant; spend nothing")
    parser.add_argument("--ai-calls", type=int, default=6, help="Replicate calls allowed PER VARIANT in this run")
    parser.add_argument("--ai-budget", type=int, default=40, help="hard cap on ALL calls ever, across runs (max 40)")
    parser.add_argument("--style", default="a stone masonry university building facade", help="the building, for the prompt")
    args = parser.parse_args()
    ai = None
    if args.ai or args.ai_plan:
        ai = _ai_factory(args)
    record = bake_record(args.run_dir, args.photo, args.mask, args.scorched, args.force, args.edit_mask, ai=ai)
    for key, info in record["texture_bake"].items():
        if "skipped" in info:
            print(f"{key}: SKIPPED, {info['skipped']}")
            continue
        if info.get("procedural_only"):
            print(f"{key}: PROCEDURAL ONLY (camera fit unreliable), base colour {info['base_rgb']}")
            continue
        counts = {}
        for w in info["walls"]:
            counts[w["source"]] = counts.get(w["source"], 0) + 1
        a = info.get("ai") or {}
        if a:
            share = np.mean([w["ai_filled_fraction"] for w in info["walls"]])
            print(f"{key}: AI tiles candidate {a.get('candidate_tiles')} (fill {a.get('candidate_fill_px')} px), sent "
                  f"{a.get('sent')}, failed {a.get('failed')}, skipped-too-empty {a.get('skipped_too_empty')}, cache hits "
                  f"{a.get('cache_hits')}; mean AI share {share:.3f}")
        print(f"{key}: {len(info['walls'])} walls {counts}, {info['px_per_m']} px/m, atlas {info['atlas']}")
    print("textured_glbs:", record["textured_glbs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
