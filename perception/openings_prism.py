#!/usr/bin/env python
"""Facade openings on a footprint PRISM, placed with the photo's own (EXIF) camera.

    prism = prism_from_record(record)
    result = place_openings_on_prism(record, photo, mask, openings, source_photo="patton_1.jpg")
    attach_to_record(record_path, result)

The demo draws the OSM footprint extruded to the resolved height, not a generated mesh (perception/openings_3d.py
handles meshes). Here the 3-D model is exact, so the unknown is the camera.

Where the inputs live
  outputs/openings/<stem>.json   boxes normalised to the mask bbox (perception/openings.py)
  outputs/masks/<stem>.png       the building mask, full photo resolution
  the photo                      geo/exif.py: gps, heading_deg (pitch_deg is absent from iPhone photos). It does NOT
                                 return the focal length: FocalLengthIn35mmFilm is in the Exif sub-IFD, which
                                 geo/exif.py does not read, so `exif_focal_35mm` reads it here.
  <run>/record.json              footprint_geojson (lon/lat rings), enu_origin_geodetic (lat0, lon0, h0), height.value_m,
                                 facade.front_heading_deg (None for a prism: no mesh, no front). No terrain: h0 is 0 and
                                 the viewer samples the ground itself (spec 8), so ground is the ENU z = 0 plane here
                                 unless `ground_delta_m` (camera ground minus building ground) is given.
  demo_assets.json               asset ids and photo names only; the records are in data/runs (untracked) on the
                                 machine that ran the pipeline.

Camera. Pinhole at the EXIF GPS -> ENU, `camera_height_m` (1.5) above the ground; yaw = EXIF heading (compass,
clockwise from north), pitch = EXIF pitch or 0, roll 0. Focal length in pixels from the 35 mm equivalent, which is
defined on the frame DIAGONAL (43.27 mm): f_px = f35 * diag_px / 43.27.

Refinement (fit_camera). The prism is projected into the image and its silhouette scored against the building mask
by plain IoU (position and scale matter here: the camera is metric, unlike perception/render_compare's normalised
score), over yaw +-15 deg (1 deg), east/north +-24 m (4 m) and pitch 0 to +15 deg (2.5 deg; absolute, because
phones tilt up at tall buildings and carry no EXIF pitch). The focal length is NOT searched: it is fixed at the
EXIF-derived value (focal scale 1.0), because focal length and camera distance trade off almost exactly and a
searched scale drifted to the grid edge on four of six photos. The best coarse candidate is then refined in
position only, at 1 m within one coarse step. A best value on the edge of any axis (yaw, east, north, or the TOP
of the pitch grid; pitch 0 is a floor, not an edge) means the optimum is outside the grid, so it is not trusted.
camera_iou < MIN_CAMERA_IOU or an edge value sends every opening from the photo to REVIEW; a photo that still
fails is REVIEW and the grid is not widened further (camera work is frozen). The coarse grid is ~37k silhouettes,
scored in parallel worker processes (up to 8).

Size sanity. A window wider than 4 m, or any opening taller than 6 m, is REVIEW ("implausible size (grazing view)"):
at a grazing angle the corner rays stretch a box along the wall.

Placement. Perspective rays from the camera through each opening's centre and four box corners hit the prism
(trimesh extrusion of the ENU footprint); the first hit is the surface. A wall hit is mapped to the footprint edge
it lies on (`wall_index`, over the exterior ring then any holes) and takes that edge's EXACT outward normal.
`bearing_deg` is that normal's compass bearing through geo.coords.theta_to_heading. Roof hits and misses are REVIEW.
The marker position is offset OFFSET_M (2 cm) outward along the normal. The four box corners are intersected with
the INFINITE plane of the wall the centre ray hit (not the prism mesh); a corner is valid within the wall's extent
+- 0.5 m and from 1 m below ground to the roof, and "corner rays missed" is flagged only if one fails that.
width_m / height_m come from those corner points, bottom_above_ground_m from the two bottom
corners. Doors, entrances and garage doors whose bottom is more than
DOOR_MAX_BOTTOM_M above ground go to REVIEW: a real door reaches the ground.

Viewer. There is no glb to add child nodes to (web/src/main.js draws the prism as a Cesium polygon extrusion), so the
markers are Cesium primitives read from record["openings"]: box (width x height x 0.1 m) at position_enu, local Z on
normal_enu, local Y up, coloured as in perception/openings_3d.py.

Usage: python perception/openings_prism.py RECORD.json PHOTO.jpg MASK.png OPENINGS.json [--write-record] [--terrain]
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
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `python perception/openings_prism.py` puts perception/ first, not the repo root
    sys.path.insert(0, str(ROOT))

from geo.coords import ENUFrame, theta_to_heading  # noqa: E402
from geo.exif import extract as extract_exif  # noqa: E402
from perception.openings_3d import (  # noqa: E402  (the rules and the extras serialisation are shared)
    MARKER_DEPTH_M,
    OFFSET_M,
    SKIP_TYPES,
    _jsonable,
    marker_colour,
)

CAMERA_HEIGHT_M = 1.5
YAW_RANGE_DEG, YAW_STEP_DEG = 15.0, 1.0
POS_RANGE_M, POS_STEP_M, REFINE_STEP_M = 24.0, 4.0, 1.0
# Absolute pitch above the horizon. Phones are tilted up at tall buildings and carry no EXIF pitch; 0 is the floor
# (a level camera), so a best value at 0 is not an edge hit, only one at the top of the grid is.
PITCH_GRID_DEG = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
# x the EXIF-derived focal length in pixels. FROZEN at 1.0: focal length and camera distance trade off almost exactly,
# so the IoU is flat along that ridge and a searched scale runs to the grid edge on noise. fit_camera still accepts a
# `scales` sequence (the tests use it); the pipeline does not search it.
FOCAL_SCALES = (1.0,)
MAX_WINDOW_WIDTH_M = 4.0
MAX_OPENING_HEIGHT_M = 6.0
MIN_CAMERA_IOU = 0.6
DOOR_MAX_BOTTOM_M = 1.0
DOOR_TYPES = ("door", "entrance", "garage_door")
FRONT_TOLERANCE_DEG = 10.0
DIAGONAL_35MM_MM = 43.2666
DEFAULT_F35_MM = 26.0  # the iPhone main camera, when a photo carries no focal length at all
RASTER_MAX_SIDE = 320  # the refinement scores silhouettes on a downscaled mask
NEAR_M = 0.5
TIE_TOL = 1e-9
WALL_HIT_TOL_M = 0.05  # a wall hit must lie this close to a footprint edge in plan
CORNER_EXTENT_TOL_M = 0.5  # a corner may lie this far beyond the wall's ends ...
CORNER_BELOW_GROUND_M = 1.0  # ... and this far below ground; its upper limit is the roof


# --------------------------------------------------------------------- prism


@dataclass
class Prism:
    frame: ENUFrame
    polygons: list  # shapely polygons in ENU metres, exterior CCW, holes CW
    height_m: float
    mesh: trimesh.Trimesh
    walls: list = field(default_factory=list)  # dicts: index, p0, p1, normal (unit, outward), bearing_deg, length_m

    def centroid(self):
        c = MultiPolygon(self.polygons).centroid
        return np.array([c.x, c.y])


def _walls(polygons):
    walls = []
    for poly in polygons:
        for ring in [poly.exterior, *poly.interiors]:
            pts = np.asarray(ring.coords)[:-1]
            for a, b in zip(pts, np.roll(pts, -1, axis=0)):
                d = b - a
                length = float(np.hypot(*d))
                if length < 1e-6:
                    continue
                normal = np.array([d[1], -d[0]]) / length  # right of a CCW exterior / CW hole = away from the solid
                walls.append({
                    "index": len(walls), "p0": a, "p1": b, "normal": normal, "length_m": length,
                    "bearing_deg": math.degrees(theta_to_heading(math.atan2(normal[1], normal[0]))),
                })
    return walls


def prism_from_polygons(polygons, height_m, frame=None):
    """Build the prism from ENU polygons (the tests use this directly)."""
    polygons = [orient(p, 1.0) for p in polygons]
    mesh = trimesh.util.concatenate([trimesh.creation.extrude_polygon(p, height_m) for p in polygons])
    return Prism(frame or ENUFrame(0.0, 0.0, 0.0), polygons, float(height_m), mesh, _walls(polygons))


def prism_from_record(record):
    """The footprint in ENU, the height and the ENU frame, all from a placement record.json."""
    lat0, lon0, h0 = record["enu_origin_geodetic"]
    frame = ENUFrame(lat0, lon0, h0)
    geom = shape(record["footprint_geojson"])
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

    def ring_enu(ring):
        lon, lat = np.asarray(ring.coords).T
        return frame.geodetic_to_enu(lat, lon, h0)[:, :2]

    enu = [Polygon(ring_enu(p.exterior), [ring_enu(r) for r in p.interiors]) for p in polys]
    return prism_from_polygons(enu, record["height"]["value_m"], frame)


# -------------------------------------------------------------------- camera


@dataclass(frozen=True)
class PinholeCamera:
    position: tuple  # ENU metres (east, north, up); up is height above the building's ground plane
    yaw_deg: float  # compass heading of the optical axis, clockwise from north
    pitch_deg: float  # above the horizon
    f_px: float
    width: int
    height: int

    def basis(self):
        """(right, up, forward) unit vectors in ENU. Roll is 0: right stays horizontal."""
        yaw, pitch = math.radians(self.yaw_deg), math.radians(self.pitch_deg)
        forward = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        return right, np.cross(right, forward), forward

    def rays(self, px, py):
        """Unit ray directions (N, 3) through image pixels (in the full-resolution photo)."""
        right, up, forward = self.basis()
        px, py = np.asarray(px, float), np.asarray(py, float)
        d = (forward[None] + right[None] * ((px - self.width / 2) / self.f_px)[:, None]
             - up[None] * ((py - self.height / 2) / self.f_px)[:, None])
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, points):
        """Full-resolution pixel coordinates (N, 2) of ENU points; the inverse of `rays` along a ray."""
        right, up, forward = self.basis()
        rel = np.atleast_2d(np.asarray(points, float)) - np.asarray(self.position)
        z = rel @ forward
        return np.stack([self.width / 2 + self.f_px * (rel @ right) / z, self.height / 2 - self.f_px * (rel @ up) / z], axis=1)

    def moved(self, yaw_deg=0.0, dx=0.0, dy=0.0, pitch_deg=0.0, scale=1.0):
        """A camera offset from this one; `scale` multiplies the focal length in pixels."""
        x, y, z = self.position
        return PinholeCamera((x + dx, y + dy, z), self.yaw_deg + yaw_deg, self.pitch_deg + pitch_deg,
                             self.f_px * scale, self.width, self.height)


def exif_focal_35mm(photo):
    """FocalLengthIn35mmFilm (mm) from the Exif sub-IFD, or None. geo/exif.py reads IFD0 only and misses it."""
    try:
        with Image.open(photo) as im:
            return float(im.getexif().get_ifd(0x8769).get(0xA405)) or None
    except Exception:  # noqa: BLE001 - no EXIF is normal for a downloaded image
        return None


def focal_px(f35_mm, width, height):
    """35 mm equivalence is defined on the frame diagonal."""
    return f35_mm * math.hypot(width, height) / DIAGONAL_35MM_MM


def camera_from_exif(prism, photo, width, height, camera_height_m=CAMERA_HEIGHT_M, ground_delta_m=0.0):
    """-> (PinholeCamera, notes) from the photo's EXIF. Raises ValueError without GPS or heading."""
    exif = extract_exif(photo)
    if exif["gps"] is None or exif["heading_deg"] is None:
        raise ValueError(f"{photo}: no EXIF GPS + heading, so there is no camera to start from")
    lat, lon = exif["gps"]
    e, n, _ = prism.frame.geodetic_to_enu(lat, lon, prism.frame.h0)
    f35 = exif_focal_35mm(photo)
    notes = {"exif_heading_deg": exif["heading_deg"], "exif_pitch_deg": exif["pitch_deg"], "exif_gps": exif["gps"],
             "exif_focal_35mm": f35, "focal_assumed": f35 is None}
    f35 = f35 or DEFAULT_F35_MM
    cam = PinholeCamera((float(e), float(n), camera_height_m + ground_delta_m), float(exif["heading_deg"]),
                        float(exif["pitch_deg"] or 0.0), focal_px(f35, width, height), int(width), int(height))
    return cam, notes


# ---------------------------------------------------------------- silhouette


class _Rasteriser:
    """The prism's silhouette from a camera, on a downscaled canvas. Faces are the wall quads plus the roof; the
    silhouette of a solid is the union of its projected faces, so hidden faces do no harm."""

    def __init__(self, prism, width, height):
        self.scale = RASTER_MAX_SIDE / max(width, height)
        self.size = (max(1, round(height * self.scale)), max(1, round(width * self.scale)))
        h = prism.height_m
        corners, self.faces = [], []
        for w in prism.walls:
            start = len(corners)
            (x0, y0), (x1, y1) = w["p0"], w["p1"]
            corners += [(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, h), (x0, y0, h)]
            self.faces.append((start, 4))
        for poly in prism.polygons:
            ring = np.asarray(poly.exterior.coords)[:-1]
            self.faces.append((len(corners), len(ring)))
            corners += [(x, y, h) for x, y in ring]
        self.corners = np.asarray(corners, float)
        self.width, self.height = width, height

    def render(self, cam):
        right, up, forward = cam.basis()
        rel = self.corners - np.asarray(cam.position)
        z = rel @ forward
        canvas = np.zeros(self.size, np.uint8)
        if z.min() > NEAR_M:
            s = cam.f_px * self.scale / z
            px = (rel @ right) * s + self.width * self.scale / 2
            py = -(rel @ up) * s + self.height * self.scale / 2
            pts = np.round(np.stack([px, py], axis=1) * 16).astype(np.int32)
            for start, count in self.faces:
                cv2.fillPoly(canvas, [pts[start : start + count]], 1, shift=4)
            return canvas.astype(bool)
        for start, count in self.faces:  # slow path: something is behind the near plane, clip each face to it
            poly = _clip_near(np.stack([rel[start : start + count] @ right, rel[start : start + count] @ up,
                                        z[start : start + count]], axis=1))
            if len(poly) >= 3:
                s = cam.f_px * self.scale / poly[:, 2]
                pts = np.stack([poly[:, 0] * s + self.width * self.scale / 2,
                                -poly[:, 1] * s + self.height * self.scale / 2], axis=1)
                cv2.fillPoly(canvas, [np.round(pts * 16).astype(np.int32)], 1, shift=4)
        return canvas.astype(bool)


def _clip_near(poly):
    """Sutherland-Hodgman against z = NEAR_M on a polygon in camera coordinates (x, y, z)."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ina, inb = a[2] >= NEAR_M, b[2] >= NEAR_M
        if ina:
            out.append(a)
        if ina != inb:
            out.append(a + (b - a) * ((NEAR_M - a[2]) / (b[2] - a[2])))
    return np.asarray(out)


def _mask_array(mask):
    if isinstance(mask, (str, Path)):
        mask = np.array(Image.open(mask).convert("L")) > 127
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("mask must be a non-empty 2-D array")
    return mask


def silhouette_iou(prism_raster, cam, target):
    sil = prism_raster.render(cam)
    union = np.logical_or(sil, target).sum()
    return float(np.logical_and(sil, target).sum() / union) if union else 0.0


# ---------------------------------------------------------------- refinement


@dataclass(frozen=True)
class CameraFit:
    camera: PinholeCamera  # the refined camera
    initial: PinholeCamera  # the EXIF camera
    iou: float
    iou_initial: float
    yaw_offset_deg: float
    dx_m: float
    dy_m: float
    pitch_offset_deg: float
    focal_scale: float  # refined f_px / EXIF f_px
    at_edge: tuple  # names of the axes whose best value sits on the grid boundary
    reliable: bool
    why: tuple  # reasons the fit is not reliable

    def to_dict(self):
        return {
            "camera_iou": self.iou, "camera_iou_initial": self.iou_initial,
            "camera_yaw_deg": self.camera.yaw_deg % 360, "exif_heading_deg": self.initial.yaw_deg,
            "yaw_offset_deg": self.yaw_offset_deg, "camera_pitch_deg": self.camera.pitch_deg,
            "camera_position_enu": list(self.camera.position), "position_offset_m": [self.dx_m, self.dy_m],
            "focal_scale": self.focal_scale, "f_px": self.camera.f_px, "f_px_exif": self.initial.f_px,
            "at_grid_edge": list(self.at_edge), "reliable": self.reliable, "reasons": list(self.why),
        }


def _offsets(half_range, step):
    n = int(round(half_range / step)) if step else 0
    return [0.0] + [s * k * step for k in range(1, n + 1) for s in (-1, 1)]


_WORKER = {}  # per-process state for the parallel grid: (rasteriser, target, camera)


def _init_worker(prism, width, height, target, camera):
    _WORKER["state"] = (_Rasteriser(prism, width, height), target, camera)


def _score_chunk(candidates):
    raster, target, camera = _WORKER["state"]
    return [silhouette_iou(raster, camera.moved(*c), target) for c in candidates]


def _score_all(candidates, raster, target, camera, prism, workers):
    """IoU of every (dyaw, dx, dy, dpitch, scale) candidate, in order. Chunks go to worker processes when there
    are enough of them to pay for the start-up; the scores do not depend on how the work was split."""
    if workers > 1 and len(candidates) > 4000:
        from concurrent.futures import ProcessPoolExecutor

        size = math.ceil(len(candidates) / (workers * 4))
        chunks = [candidates[i : i + size] for i in range(0, len(candidates), size)]
        with ProcessPoolExecutor(workers, initializer=_init_worker,
                                 initargs=(prism, camera.width, camera.height, target, camera)) as pool:
            return [v for part in pool.map(_score_chunk, chunks) for v in part]
    return [silhouette_iou(raster, camera.moved(*c), target) for c in candidates]


def _first_best(candidates, scores, floor_iou):
    """The first candidate (in enumeration order) that beats every earlier one by more than TIE_TOL, so an exact
    tie keeps the candidate nearest the EXIF camera."""
    best, best_iou = None, floor_iou
    for c, s in zip(candidates, scores):
        if s > best_iou + TIE_TOL:
            best, best_iou = c, s
    return best, best_iou


def fit_camera(prism, mask, camera, *, refine=True, yaw_range=YAW_RANGE_DEG, pitches=PITCH_GRID_DEG,
               pos_range=POS_RANGE_M, pos_step=POS_STEP_M, refine_step=REFINE_STEP_M, scales=FOCAL_SCALES,
               workers=None):
    """Grid-refine the EXIF camera against the building mask. See the module docstring.

    Coarse grid over yaw x pitch (`pitches`, absolute degrees) x (east, north) x focal scale; then position only, at `refine_step`, within one
    coarse step of the best (never past `pos_range`, so an optimum outside the grid still reads as an edge hit).
    Ties keep the candidate nearest the EXIF camera: candidates are enumerated nearest-first and only a real
    improvement replaces the best. `workers`: processes for the coarse grid (default: up to 8 cores)."""
    import os

    full = _mask_array(mask)
    raster = _Rasteriser(prism, camera.width, camera.height)
    target = cv2.resize(full.astype(np.float32), (raster.size[1], raster.size[0]), interpolation=cv2.INTER_AREA) >= 0.5
    iou0 = silhouette_iou(raster, camera, target)
    if not refine:
        return CameraFit(camera, camera, iou0, iou0, 0.0, 0.0, 0.0, 0.0, 1.0, (), *_reliability(iou0, ()))
    workers = max(1, min(os.cpu_count() or 1, 8)) if workers is None else workers

    scales = tuple(sorted(scales, key=lambda s: abs(s - 1.0)))  # nearest 1.0 first
    dpitches = sorted((p - camera.pitch_deg for p in pitches), key=abs)  # offsets from the camera, nearest first
    pitch_span = max(abs(d) for d in dpitches) or 1.0
    poss = _offsets(pos_range, pos_step)
    # Candidates are (dyaw, dx, dy, dpitch, scale): the argument order of PinholeCamera.moved.
    coarse = [(dyaw, dx, dy, dpitch, s)
              for dyaw in _offsets(yaw_range, YAW_STEP_DEG) for dpitch in dpitches
              for dx in poss for dy in poss for s in scales]
    coarse.sort(key=lambda c: abs(c[0]) / (yaw_range or 1) + math.hypot(c[1], c[2]) / (pos_range or 1)
                + abs(c[3]) / pitch_span + abs(c[4] - 1.0))  # stable: ties keep grid order
    best, best_iou = _first_best(coarse, _score_all(coarse, raster, target, camera, prism, workers), iou0)
    best = best or (0.0, 0.0, 0.0, 0.0, 1.0)

    if refine_step and refine_step < pos_step:
        radius = pos_step - refine_step
        fine = [(best[0], best[1] + a, best[2] + b, best[3], best[4])
                for a in _offsets(radius, refine_step) for b in _offsets(radius, refine_step)
                if abs(best[1] + a) <= pos_range + 1e-9 and abs(best[2] + b) <= pos_range + 1e-9]
        found, found_iou = _first_best(fine, [silhouette_iou(raster, camera.moved(*c), target) for c in fine],
                                       best_iou)
        best, best_iou = found or best, found_iou

    dyaw, dx, dy, dpitch, scale = best
    axes = [("yaw", dyaw, yaw_range), ("east", dx, pos_range), ("north", dy, pos_range)]
    edge = [name for name, v, r in axes if r > 0 and abs(v) >= r - 1e-9]
    if len(dpitches) > 1 and camera.pitch_deg + dpitch >= max(pitches) - 1e-9:  # only the top: 0 is a floor
        edge.append("pitch")
    if len(scales) > 1 and (scale <= min(scales) + 1e-9 or scale >= max(scales) - 1e-9):
        edge.append("focal scale")
    return CameraFit(camera.moved(dyaw, dx, dy, dpitch, scale), camera, best_iou, iou0, dyaw, dx, dy, dpitch, scale,
                     tuple(edge), *_reliability(best_iou, edge))


def _reliability(iou, edge):
    why = []
    if iou < MIN_CAMERA_IOU:
        why.append(f"camera fit IoU {iou:.2f} < {MIN_CAMERA_IOU}: the prism does not match the photographed building")
    if edge:
        why.append(f"refined camera {'/'.join(edge)} sits on the edge of the search grid: the optimum is outside it")
    return (not why), tuple(why)


# ------------------------------------------------------------------ ray-cast


def _corner_uvs(op):
    u0, v0, u1, v1 = op["box_uv"]
    return [tuple(op["center_uv"]), (u0, v0), (u1, v0), (u1, v1), (u0, v1)]  # centre, TL, TR, BR, BL


def _wall_of(prism, xy):
    """(wall index, distance in plan) of the footprint edge nearest to `xy`."""
    best, best_d = -1, np.inf
    for w in prism.walls:
        a, b = w["p0"], w["p1"]
        t = np.clip(np.dot(xy - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
        d = float(np.linalg.norm(xy - (a + t * (b - a))))
        if d < best_d:
            best, best_d = w["index"], d
    return best, best_d


def cast_openings(prism, camera, openings, bbox_px, source_photo=None):
    """Ray-cast every non-'other' opening from `camera` into the prism. `bbox_px` = (x0, y0, x1, y1), the mask
    bounding box in full-resolution pixels that box_uv is normalised to. -> list of per-opening dicts."""
    kept = [(i, op) for i, op in enumerate(openings) if op.get("type") not in SKIP_TYPES]
    x0, y0, x1, y1 = bbox_px
    uv = np.array([p for _, op in kept for p in _corner_uvs(op)], float).reshape(-1, 2)
    dirs = camera.rays(x0 + uv[:, 0] * (x1 - x0), y0 + uv[:, 1] * (y1 - y0))
    origin = np.asarray(camera.position, float)
    hits = np.full((len(dirs), 3), np.nan)
    tris = np.full(len(dirs), -1)
    if len(dirs):
        loc, ray_idx, tri_idx = prism.mesh.ray.intersects_location(np.tile(origin, (len(dirs), 1)), dirs,
                                                                   multiple_hits=False)
        if len(ray_idx):  # nothing hit: trimesh returns an empty (0,) location array
            hits[ray_idx], tris[ray_idx] = loc, tri_idx
    is_wall = np.zeros(len(dirs), bool)  # a hit on a side face (roof and floor hits are not walls)
    is_wall[tris >= 0] = np.abs(prism.mesh.face_normals[tris[tris >= 0]][:, 2]) < 0.5
    return [_place_one(i, op, prism, origin, dirs[5 * n : 5 * n + 5], hits[5 * n : 5 * n + 5],
                       is_wall[5 * n : 5 * n + 5], source_photo) for n, (i, op) in enumerate(kept)]


def _corner_points(prism, wall_index, origin, rays):
    """Intersect the four corner rays (TL, TR, BR, BL) with the INFINITE vertical plane of a wall.

    The corners of a box on a tall or oblique wall often fall on a neighbouring surface, past a corner of the
    footprint or over an edge; the plane is the surface the opening is ON, so that is where they belong.
    -> (points (4, 3), valid (4,)). A corner is valid when its ray reaches the plane's outer face and it lands within
    the wall's horizontal extent +- CORNER_EXTENT_TOL_M and between CORNER_BELOW_GROUND_M below ground and the roof."""
    wall = prism.walls[wall_index]
    n = np.array([wall["normal"][0], wall["normal"][1], 0.0])
    p0 = np.array([wall["p0"][0], wall["p0"][1], 0.0])
    u = (np.asarray(wall["p1"]) - np.asarray(wall["p0"])) / wall["length_m"]
    denom = rays @ n
    toward = denom < -1e-9  # the ray travels against the outward normal: it reaches the wall from outside
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(toward, ((p0 - origin) @ n) / denom, np.nan)
    pts = origin + t[:, None] * rays
    along = (pts[:, :2] - np.asarray(wall["p0"])) @ u
    valid = (toward & (t > 0) & (along >= -CORNER_EXTENT_TOL_M) & (along <= wall["length_m"] + CORNER_EXTENT_TOL_M)
             & (pts[:, 2] >= -CORNER_BELOW_GROUND_M) & (pts[:, 2] <= prism.height_m))
    return pts, valid


def _place_one(index, op, prism, origin, rays, hits, is_wall, source_photo):
    reasons = list(op.get("reasons") or [])
    decision = op.get("decision", "REVIEW")
    rec = {
        "id": op.get("id") or f"opening_{index}", "type": op["type"], "score": op.get("score"),
        "group_margin": op.get("group_margin"), "source_photo": source_photo,
        "wall_index": None, "normal_enu": None, "bearing_deg": None, "hit_enu": None, "position_enu": None,
        "lat": None, "lon": None, "height_above_ground_m": None, "width_m": None, "height_m": None,
        "bottom_above_ground_m": None, "corner_hits": 0,
    }

    def review(reason):
        nonlocal decision
        decision = "REVIEW"
        reasons.append(reason)

    if not np.isfinite(hits[0, 0]):
        review("centre ray missed the prism: no 3-D position")
    elif not is_wall[0]:
        review("centre ray hit the roof, not a wall")
    else:
        wall, dist = _wall_of(prism, hits[0, :2])
        if dist > WALL_HIT_TOL_M:
            review(f"hit point is {dist:.2f} m from the nearest wall edge: no wall assigned")
        else:
            n2 = prism.walls[wall]["normal"]
            normal = np.array([n2[0], n2[1], 0.0])
            position = hits[0] + normal * OFFSET_M
            lat, lon, h = prism.frame.enu_to_geodetic(*position)
            rec.update(wall_index=wall, normal_enu=normal, bearing_deg=prism.walls[wall]["bearing_deg"],
                       hit_enu=hits[0], position_enu=position, lat=float(lat), lon=float(lon),
                       height_above_ground_m=float(h - prism.frame.h0))

    bottoms = []
    if rec["wall_index"] is not None:  # the corners only mean something on the wall the centre ray found
        pts, valid = _corner_points(prism, rec["wall_index"], origin, rays[1:])
        rec["corner_hits"] = int(valid.sum())
        if valid.all():
            tl, tr, br, bl = pts
            rec["width_m"] = float((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
            rec["height_m"] = float((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
        else:
            review(f"{int((~valid).sum())} of 4 corner rays missed the wall (past its ends by more than "
                   f"{CORNER_EXTENT_TOL_M} m, or below ground / above the roof): size unreliable")
        bottoms = [pts[k, 2] for k in (2, 3) if valid[k]]  # BR, BL
    too_wide = op["type"] == "window" and rec["width_m"] is not None and rec["width_m"] > MAX_WINDOW_WIDTH_M
    too_tall = rec["height_m"] is not None and rec["height_m"] > MAX_OPENING_HEIGHT_M
    if too_wide or too_tall:  # a grazing view stretches boxes along the wall
        review("implausible size (grazing view)")
    if bottoms:
        rec["bottom_above_ground_m"] = float(np.mean(bottoms))
    bottom = rec["bottom_above_ground_m"]
    if op["type"] in DOOR_TYPES and bottom is not None and bottom > DOOR_MAX_BOTTOM_M:
        review(f"{op['type']} bottom is {bottom:.1f} m above ground (> {DOOR_MAX_BOTTOM_M:.1f} m): "
               "a door reaches the ground")

    rec.update(decision=decision, reasons=reasons)
    return rec


# ------------------------------------------------------------- orchestration


@dataclass
class PrismPlacement:
    fit: CameraFit
    openings: list
    walls: list
    facade_check: dict
    notes: dict

    def per_wall(self):
        """{wall_index: (bearing_deg, [opening ids])} for the openings that landed on a wall."""
        out = {}
        for op in self.openings:
            if op["wall_index"] is not None:
                out.setdefault(op["wall_index"], (op["bearing_deg"], []))[1].append(op["id"])
        return out


def _bearing_gap(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def facade_check(record, openings):
    """Do the openings' wall bearings agree with record['facade']['front_heading_deg'] (within 10 deg)?

    Applies only when the record has a front heading that is EVIDENCE: a prism record has none (no mesh, so no
    front), and a front_source of "gltf_prior" is the glTF +Z convention, so agreeing or disagreeing with it means
    nothing. A made-up default would be worse than no check."""
    facade = (record or {}).get("facade") or {}
    if facade.get("front_source") == "gltf_prior":  # the glTF +Z convention, not evidence: a mismatch means nothing
        return {"applicable": False, "reason": "front_source is gltf_prior: the front is a convention, not evidence"}
    front = facade.get("front_heading_deg")
    if front is None:
        return {"applicable": False, "reason": "record has no facade.front_heading_deg (prism: no mesh, no front)"}
    bearings = [op["bearing_deg"] for op in openings if op["bearing_deg"] is not None]
    if not bearings:
        return {"applicable": True, "front_heading_deg": front, "consistent": None, "reason": "no opening on a wall"}
    values, counts = np.unique(np.round(bearings, 3), return_counts=True)
    photo_wall = float(values[np.argmax(counts)])  # the wall most openings landed on: the photographed facade
    gap = _bearing_gap(photo_wall, front)
    return {"applicable": True, "front_heading_deg": front, "photo_wall_bearing_deg": photo_wall, "delta_deg": gap,
            "tolerance_deg": FRONT_TOLERANCE_DEG, "consistent": gap <= FRONT_TOLERANCE_DEG}


def place_openings_on_prism(record, photo, mask, openings, *, bbox_px=None, source_photo=None, camera=None,
                            camera_height_m=CAMERA_HEIGHT_M, ground_delta_m=0.0, refine=True, fit_kwargs=None):
    """EXIF camera -> refinement -> ray-cast -> per-opening dicts. `openings` is the list from
    outputs/openings/<stem>.json; `record` is a record dict or a Prism. Every opening goes to REVIEW when the camera
    fit is unreliable. Pass `camera` to skip EXIF (the tests do) and `fit_kwargs` to narrow the search grid."""
    prism = prism_from_record(record) if not isinstance(record, Prism) else record
    full = _mask_array(mask)
    height, width = full.shape
    notes = {}
    if camera is None:
        camera, notes = camera_from_exif(prism, photo, width, height, camera_height_m, ground_delta_m)
    notes["ground_delta_m"] = ground_delta_m
    fit = fit_camera(prism, full, camera, refine=refine, **(fit_kwargs or {}))
    if bbox_px is None:
        ys, xs = np.nonzero(full)
        bbox_px = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    placed = cast_openings(prism, fit.camera, openings, bbox_px, source_photo)
    if not fit.reliable:
        for op in placed:
            op["decision"] = "REVIEW"
            op["reasons"] = list(op["reasons"]) + list(fit.why)
    check = facade_check(record if isinstance(record, dict) else None, placed)
    return PrismPlacement(fit, placed, prism.walls, check, notes)


def attach_to_record(record_path, placement):
    """Add `openings`, `openings_camera` and `openings_facade_check` to an asset's record.json in place. The viewer
    (web/src/main.js drawOpenings) draws record['openings']."""
    path = Path(record_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["openings"] = _jsonable(placement.openings)
    record["openings_camera"] = _jsonable({**placement.fit.to_dict(), **placement.notes})
    record["openings_facade_check"] = _jsonable(placement.facade_check)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def _ground_delta(prism, photo):
    """Camera ground minus building ground from USGS 3DEP (network); None when either sample is unavailable."""
    from geo.terrain import sample_usgs_3dep

    lat, lon = extract_exif(photo)["gps"]
    c = prism.centroid()
    blat, blon, _ = prism.frame.enu_to_geodetic(c[0], c[1], 0.0)
    cam, bld = sample_usgs_3dep(lat, lon), sample_usgs_3dep(float(blat), float(blon))
    return None if cam is None or bld is None else cam - bld


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("record")
    parser.add_argument("photo")
    parser.add_argument("mask")
    parser.add_argument("openings")
    parser.add_argument("--write-record", action="store_true", help="write openings back into record.json")
    parser.add_argument("--terrain", action="store_true", help="use USGS 3DEP for the camera/building ground offset")
    args = parser.parse_args()

    record = json.loads(Path(args.record).read_text(encoding="utf-8"))
    data = json.loads(Path(args.openings).read_text(encoding="utf-8"))
    delta = 0.0
    if args.terrain:
        got = _ground_delta(prism_from_record(record), args.photo)
        print(f"terrain: camera ground - building ground = {got}" if got is not None else "terrain: unavailable, flat")
        delta = got or 0.0
    result = place_openings_on_prism(record, args.photo, args.mask, data["openings"], bbox_px=data["mask_bbox_px"],
                                     source_photo=data.get("photo"), ground_delta_m=delta)
    fit = result.fit
    print(f"camera_iou={fit.iou:.3f} (EXIF camera {fit.iou_initial:.3f}); yaw {fit.camera.yaw_deg % 360:.1f} vs EXIF "
          f"{fit.initial.yaw_deg:.1f} (offset {fit.yaw_offset_deg:+.0f}); position offset "
          f"({fit.dx_m:+.0f}, {fit.dy_m:+.0f}) m; pitch {fit.camera.pitch_deg:.1f} deg; "
          f"focal scale {fit.focal_scale:.2f}; edge={list(fit.at_edge)} reliable={fit.reliable}")
    for wall, (bearing, ids) in sorted(result.per_wall().items()):
        print(f"  wall {wall}: bearing {bearing:.1f}  {len(ids)} openings")
    print("facade check:", result.facade_check)
    if args.write_record:
        attach_to_record(args.record, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
