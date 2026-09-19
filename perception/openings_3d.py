#!/usr/bin/env python
"""Facade openings: 2-D photo boxes -> 3-D mesh positions -> glTF child nodes.

    place_openings(mesh, mask, openings_json, mesh_to_enu=..., up_axis=(0, 1, 0)) -> Placement
    export_openings_glb(mesh_glb, placement.openings, out_glb)
    attach_to_record(record_json, placement)

Input is outputs/openings/<stem>.json (perception/openings.py), whose boxes are normalised (box_uv, center_uv) to
the building-mask bounding box. Nothing here modifies that module or its output.

Camera (fit_camera). The render_compare orthographic camera, started at the photographed_side() azimuth and
searched +-AZ_RANGE_DEG in AZ_STEP_DEG steps and elevation 0..EL_MAX_DEG in EL_STEP_DEG steps, maximising the normalised-silhouette
IoU against the photo mask (render_compare's own scoring: both shapes cropped to their bounding boxes, longer side
scaled to a common canvas, centred). A joint grid (41 x 5 = 205 silhouettes, ~0.2 s each on an 80k-face mesh):
azimuth and elevation are coupled, and a coarse-to-fine search settles at elevation 0. Ties go to the smallest
departure from the starting camera.

Ray-cast (cast_openings). A box_uv point in the photo's mask bbox is mapped to the orthographic camera plane by the
same bbox-to-canvas normalisation the IoU used, so the point that sat at a bbox fraction in the photo sits at the same
fraction of the render's silhouette. The ray runs along the view direction from in front of the mesh; the first hit is
the surface. The hit face's normal, flipped to face the camera, is snapped to the nearest of the mesh's four
OMBB-aligned wall directions (horizontal component only); the snap angle is kept as a quality value.

Units. Hit points and normals are in the MESH frame (the scene frame of the glb, after node transforms are baked in).
Metres exist only through `mesh_to_enu` (geo/placement.py, written to the placement record): width_m / height_m /
the 2 cm offset are converted with it. Without it mesh units are treated as metres, which is only right for a
synthetic mesh.

Bearing. `bearing_deg` is the compass bearing (0 = N, 90 = E) of the snapped normal after placement. The fitted
rotation, the up-axis canonicalisation and any anisotropic scale are all inside `mesh_to_enu`, so the bearing is
mathematical theta of the transformed normal converted with heading = pi/2 - theta: the same composition as
geo.disambiguate.facade_heading (theta + front_angle) and geo.coords.theta_to_heading. perception must not import
geo/ (see render_compare), so tests/test_openings_3d.py asserts that agreement instead.

glTF (export_openings_glb). One child node per opening under the building mesh node: a thin box (width x height x
MARKER_DEPTH_M) at position_mesh, local +Z along the snapped normal, local +Y up. glTF +Z is the asset's *front*;
Cesium does not treat it as a heading here (web/src/main.js loads the mesh with upAxis Z / forwardAxis X so neither
of Cesium's axis corrections applies, and every rotation comes from mesh_to_enu), so a marker's +Z is placed in
the world by exactly the matrix that places the building. Extras (type, decision, reasons, ...) are attached with
pygltflib after trimesh writes the file, because trimesh does not export per-node extras. pygltflib omits None and
empty containers from extras (0, 0.0 and False survive), so an empty `reasons` or a null `bearing_deg` is an absent key
in the glb; the record's `openings` list keeps them.

Usage: python perception/openings_3d.py MESH.glb MASK.(png|npy) OPENINGS.json [--record record.json] [--out out.glb]
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `python perception/openings_3d.py` puts perception/ first, not the repo root
    sys.path.insert(0, str(ROOT))

from perception.render_compare import (  # noqa: E402
    CANVAS_PX,
    _as_mask,
    _as_mesh,
    camera_basis,
    normalise_mask,
    normalised_iou,
    photographed_side,
    render_silhouette,
)

AZ_RANGE_DEG = 60.0
AZ_STEP_DEG = 3.0
EL_MAX_DEG = 20.0
EL_STEP_DEG = 5.0
SNAP_REVIEW_DEG = 30.0  # a hit whose wall-snap angle exceeds this is not a wall we can place on
NOT_A_WALL_HORIZ = 0.5  # |horizontal component| of a unit normal below this = a surface >60 deg off vertical
OFFSET_M = 0.02
MARKER_DEPTH_M = 0.1
TIE_TOL = 1e-9
SKIP_TYPES = ("other",)

COLOURS = {  # RGBA
    "door": (46, 160, 67, 255),
    "entrance": (0, 150, 150, 255),
    "garage_door": (245, 140, 20, 255),
    "window": (40, 110, 230, 255),
    "review": (220, 40, 40, 255),
}
FALLBACK_COLOUR = (150, 150, 150, 255)


# ------------------------------------------------------------------ camera


@dataclass(frozen=True)
class Camera:
    up_axis: tuple
    azimuth_deg: float
    elevation_deg: float
    iou: float

    def frame(self):
        return camera_frame(self.up_axis, self.azimuth_deg, self.elevation_deg)


def camera_frame(up_axis, azimuth_deg, elevation_deg=0.0):
    """(right, up, toward_camera): render_compare.camera_basis, then pitched so the camera looks down by
    `elevation_deg`. Azimuth 0 / elevation 0 is render_compare's front view."""
    right, up0, toward0 = camera_basis(up_axis, math.radians(azimuth_deg))
    e = math.radians(elevation_deg)
    toward = math.cos(e) * toward0 + math.sin(e) * up0
    up = math.cos(e) * up0 - math.sin(e) * toward0
    return right, up, toward


def _silhouette_iou(mesh, up_axis, azimuth_deg, elevation_deg, target, size):
    """render_silhouette knows azimuth only. Pitch the mesh into the level camera's basis instead: rotating v by
    R = B_level . B_pitched^T gives x = v.right and y = v.up_pitched, so the level render IS the pitched one."""
    az = math.radians(azimuth_deg)
    if elevation_deg:
        b_level = np.stack(camera_basis(up_axis, az), axis=1)
        b_pitched = np.stack(camera_frame(up_axis, azimuth_deg, elevation_deg), axis=1)
        mesh = SimpleNamespace(vertices=mesh.vertices @ (b_level @ b_pitched.T).T, faces=mesh.faces)
    return normalised_iou(render_silhouette(mesh, up_axis, az, size), target)


def _offsets(half_range, step):
    n = int(round(half_range / step))
    return [0.0] + [s * k * step for k in range(1, n + 1) for s in (-1, 1)]  # nearest to the start first


def fit_camera(mesh, mask, up_axis=(0, 1, 0), size=CANVAS_PX):
    """Best (azimuth, elevation) of the orthographic camera for this photo mask. See the module docstring."""
    mesh, target = _as_mesh(mesh), normalise_mask(_as_mask(mask), size)
    up_axis = tuple(float(v) for v in up_axis)
    az0 = 90.0 * photographed_side(mesh, mask, up_axis, size).index
    elevations = np.arange(0.0, EL_MAX_DEG + 1e-9, EL_STEP_DEG)

    best = None
    for d in _offsets(AZ_RANGE_DEG, AZ_STEP_DEG):  # nearest the start first: an exact tie keeps the incumbent
        for el in elevations:
            iou = _silhouette_iou(mesh, up_axis, az0 + d, el, target, size)
            if best is None or iou > best.iou + TIE_TOL:
                best = Camera(up_axis, float((az0 + d) % 360), float(el), float(iou))
    return best


# ------------------------------------------------------------ wall directions


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _horizontal_basis(up):
    ref = np.eye(3)[(int(np.argmax(np.abs(up))) + 1) % 3]
    e1 = _unit(ref - ref.dot(up) * up)
    return e1, np.cross(up, e1)


def wall_directions(mesh, up_axis=(0, 1, 0)):
    """The mesh's four OMBB-aligned outward wall directions (unit vectors in the horizontal plane): +-u, +-v of the
    minimum-area rectangle around the plan-view convex hull (rotating calipers over the hull edges)."""
    from scipy.spatial import ConvexHull

    up = _unit(up_axis)
    e1, e2 = _horizontal_basis(up)
    xy = mesh.vertices @ np.stack([e1, e2], axis=1)
    hull = xy[ConvexHull(xy).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    best_area, best_angle = np.inf, 0.0
    for angle in np.arctan2(edges[:, 1], edges[:, 0]):
        c, s = math.cos(angle), math.sin(angle)
        p = hull @ np.array([[c, -s], [s, c]])  # coordinates along (c, s) and (-s, c)
        area = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        if area < best_area - TIE_TOL:
            best_area, best_angle = area, angle
    c, s = math.cos(best_angle), math.sin(best_angle)
    u = c * e1 + s * e2
    v = -s * e1 + c * e2
    return np.stack([u, v, -u, -v])


def snap_normal(normal, up_axis, walls, toward_camera=None):
    """-> (snapped unit normal, snap angle in degrees, horizontal fraction of the raw normal).

    Snaps the horizontal component of `normal` to the nearest of `walls`. The angle is measured between that
    component and the wall. A surface more than 60 deg off vertical (roof, ground) has no meaningful horizontal
    direction: it is reported as 90 deg and snapped to the wall nearest `toward_camera`."""
    up = _unit(up_axis)
    n = _unit(normal)
    h = n - n.dot(up) * up
    frac = float(np.linalg.norm(h))
    if frac < NOT_A_WALL_HORIZ:
        ref = h if frac > 1e-9 else None
        if ref is None and toward_camera is not None:
            t = np.asarray(toward_camera, dtype=float)
            ref = t - t.dot(up) * up
        if ref is None or np.linalg.norm(ref) < 1e-9:
            ref = walls[0]
        return walls[int(np.argmax(walls @ _unit(ref)))].copy(), 90.0, frac
    h = h / frac
    cosines = walls @ h
    k = int(np.argmax(cosines))
    return walls[k].copy(), float(np.degrees(np.arccos(np.clip(cosines[k], -1.0, 1.0)))), frac


# --------------------------------------------------------------------- world


def bearing_deg(normal_mesh, mesh_to_enu):
    """Compass bearing (0 = N, 90 = E, clockwise) of a mesh-frame normal after placement by `mesh_to_enu`.

    A normal transforms by the inverse transpose, which agrees with the plain rotation when the scale is uniform
    and stays perpendicular to its surface when it is not. Returns None without a matrix."""
    if mesh_to_enu is None:
        return None
    m = np.asarray(mesh_to_enu, dtype=float)[:3, :3]
    east, north = (np.linalg.inv(m).T @ np.asarray(normal_mesh, dtype=float))[:2]
    theta = math.atan2(north, east)  # mathematical, CCW from East (spec 2.4)
    return float(np.degrees(np.mod(math.pi / 2.0 - theta, 2.0 * math.pi)))  # geo.coords.theta_to_heading


def mesh_to_enu_from_record(record):
    """The 4x4 from a placement record dict (pipeline.py writes record['mesh_to_enu']['rows']); None if absent."""
    entry = (record or {}).get("mesh_to_enu")
    return np.asarray(entry["rows"], dtype=float) if entry else None


def _metres(mesh_to_enu, vec):
    vec = np.asarray(vec, dtype=float)
    return float(np.linalg.norm(vec if mesh_to_enu is None else np.asarray(mesh_to_enu)[:3, :3] @ vec))


# ------------------------------------------------------------------ ray-cast


def mask_bbox_size(mask):
    ys, xs = np.nonzero(_as_mask(mask))
    return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


def _corner_uvs(op):
    u0, v0, u1, v1 = op["box_uv"]
    return [tuple(op["center_uv"]), (u0, v0), (u1, v0), (u1, v1), (u0, v1)]  # centre, TL, TR, BR, BL


def _view_setup(mesh, camera):
    right, up, toward = camera.frame()
    used = mesh.vertices[np.unique(mesh.faces)]
    x, y, depth = used @ right, used @ up, used @ toward
    lo, hi = np.array([x.min(), y.min()]), np.array([x.max(), y.max()])
    return SimpleNamespace(right=right, up=up, toward=toward, centre=(lo + hi) / 2,
                           extent=float((hi - lo).max()), front=float(depth.max()))


def cast_openings(mesh, openings, camera, mask_wh, mesh_to_enu=None, source_photo=None):
    """Ray-cast every non-'other' opening. -> list of per-opening dicts (see the module docstring).

    `mask_wh` is the photo mask's bounding-box (width, height) in pixels: the aspect the box_uv coordinates were
    normalised against."""
    mesh = _as_mesh(mesh)
    up_axis = camera.up_axis
    walls = wall_directions(mesh, up_axis)
    view = _view_setup(mesh, camera)
    pw, ph = mask_wh
    a_p, b_p = pw / max(pw, ph), ph / max(pw, ph)  # photo bbox in units of its longer side

    kept = [(i, op) for i, op in enumerate(openings) if op.get("type") not in SKIP_TYPES]
    uvs = np.array([uv for _, op in kept for uv in _corner_uvs(op)], dtype=float).reshape(-1, 2)
    px = view.centre[0] + (uvs[:, 0] - 0.5) * a_p * view.extent  # bbox-to-canvas normalisation, PAD cancels
    py = view.centre[1] - (uvs[:, 1] - 0.5) * b_p * view.extent  # image v grows downward
    origins = (px[:, None] * view.right + py[:, None] * view.up
               + (view.front + view.extent) * view.toward)
    dirs = np.tile(-view.toward, (len(origins), 1))

    hits = np.full((len(origins), 3), np.nan)
    tris = np.full(len(origins), -1)
    if len(origins):
        loc, ray_idx, tri_idx = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        hits[ray_idx], tris[ray_idx] = loc, tri_idx

    out = []
    for n, (i, op) in enumerate(kept):
        h, t = hits[5 * n : 5 * n + 5], tris[5 * n : 5 * n + 5]
        out.append(_place_one(i, op, mesh, view, walls, up_axis, h, t, mesh_to_enu, source_photo))
    return out


def _place_one(index, op, mesh, view, walls, up_axis, hits, tris, mesh_to_enu, source_photo):
    reasons = list(op.get("reasons") or [])
    decision = op.get("decision", "REVIEW")
    rec = {
        "id": op.get("id") or f"opening_{index}",
        "type": op["type"],
        "score": op.get("score"),
        "group_margin": op.get("group_margin"),
        "source_photo": source_photo,
        "hit_mesh": None, "position_mesh": None, "normal_mesh_raw": None, "normal_mesh": None,
        "snap_angle_deg": None, "bearing_deg": None, "width_m": None, "height_m": None,
        "marker_size_mesh": None, "corner_hits": int(np.isfinite(hits[1:, 0]).sum()),
    }

    def review(reason):
        nonlocal decision
        decision = "REVIEW"
        reasons.append(reason)

    if tris[0] < 0:
        review("centre ray missed the mesh: no 3-D position")
    else:
        raw = mesh.face_normals[tris[0]].copy()
        if raw.dot(view.toward) < 0:  # a face wound away from the camera: report the side that faces it
            raw = -raw
        snapped, snap, _ = snap_normal(raw, up_axis, walls, view.toward)
        per_unit = _metres(mesh_to_enu, snapped)  # metres per mesh unit along the normal
        position = hits[0] + snapped * (OFFSET_M / per_unit)
        rec.update(
            hit_mesh=hits[0], position_mesh=position, normal_mesh_raw=raw, normal_mesh=snapped,
            snap_angle_deg=snap, bearing_deg=bearing_deg(snapped, mesh_to_enu),
        )
        rec["_depth_mesh"] = MARKER_DEPTH_M / per_unit
        if snap > SNAP_REVIEW_DEG:
            review(f"wall-snap angle {snap:.0f} deg > {SNAP_REVIEW_DEG:.0f} deg: hit face is not a wall")

    if tris[1:].min() < 0:
        review(f"{int((tris[1:] < 0).sum())} of 4 corner rays missed the mesh: size unreliable")
    else:
        tl, tr, br, bl = hits[1:]
        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        rec["width_m"] = (_metres(mesh_to_enu, tr - tl) + _metres(mesh_to_enu, br - bl)) / 2
        rec["height_m"] = (_metres(mesh_to_enu, bl - tl) + _metres(mesh_to_enu, br - tr)) / 2
        rec["marker_size_mesh"] = (float(width), float(height))

    rec.update(decision=decision, reasons=reasons)
    return rec


# ------------------------------------------------------------- orchestration


@dataclass
class Placement:
    camera: Camera
    openings: list
    metric: bool  # widths, heights and the offset are true metres (a mesh_to_enu was supplied)


def load_openings_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["openings"], data.get("photo")


def place_openings(mesh, mask, openings, mesh_to_enu=None, up_axis=(0, 1, 0), source_photo=None, camera=None):
    """Fit the camera (unless one is given) and ray-cast every opening. `openings` is the list from
    outputs/openings/<stem>.json; `source_photo` its 'photo' field."""
    mesh = _as_mesh(mesh)
    if camera is None:
        camera = fit_camera(mesh, mask, up_axis)
    placed = cast_openings(mesh, openings, camera, mask_bbox_size(mask), mesh_to_enu, source_photo)
    return Placement(camera, placed, mesh_to_enu is not None)


# ---------------------------------------------------------------------- glTF


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items() if not k.startswith("_")}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def marker_colour(op):
    return COLOURS["review"] if op["decision"] == "REVIEW" else COLOURS.get(op["type"], FALLBACK_COLOUR)


def marker_transform(position, normal, up_axis=(0, 1, 0)):
    """4x4 with local +Z on `normal`, local +Y on the vertical, local +X = up x normal (right-handed)."""
    up, z = _unit(up_axis), _unit(normal)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:  # normal along the vertical: any horizontal axis will do
        x = _horizontal_basis(up)[0]
    x = _unit(x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, position
    return m


def _building_node(scene):
    """The geometry node that is the building: the one with the largest bounding-box volume."""
    nodes = list(scene.graph.nodes_geometry)
    if not nodes:
        raise ValueError("glb has no mesh node")

    def volume(node):
        _, geom_name = scene.graph[node]
        return float(np.prod(np.ptp(scene.geometry[geom_name].bounds, axis=0)))

    return max(nodes, key=volume)


def export_openings_glb(mesh_glb, openings, out_glb, up_axis=(0, 1, 0)):
    """Write `mesh_glb` plus one child node per placed opening (named by its id, under the building node), with
    each opening's fields attached as glTF extras. Openings with no position (centre ray missed) get no node."""
    scene = trimesh.load(str(mesh_glb), force="scene")
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    parent = _building_node(scene)
    to_local = np.linalg.inv(scene.graph.get(parent)[0])  # positions are in the scene frame; children are parent-local

    drawn = []
    for op in openings:
        if op["type"] in SKIP_TYPES or op["position_mesh"] is None:
            continue
        w, h = op["marker_size_mesh"] or _fallback_size(op)
        box = trimesh.creation.box(extents=(w, h, op["_depth_mesh"]))
        box.visual.face_colors = marker_colour(op)
        scene.add_geometry(
            box, node_name=op["id"], geom_name=f"{op['id']}_marker", parent_node_name=parent,
            transform=to_local @ marker_transform(op["position_mesh"], op["normal_mesh"], up_axis),
        )
        drawn.append(op)
    out_glb = Path(out_glb)
    out_glb.write_bytes(scene.export(file_type="glb"))
    _attach_extras(out_glb, drawn)
    return out_glb


def _fallback_size(op):
    """Marker size when a corner ray missed: 1 m x 1 m in mesh units (width_m / height_m stay None)."""
    per_unit = op["_depth_mesh"] / MARKER_DEPTH_M
    return 1.0 * per_unit, 1.0 * per_unit


def _attach_extras(glb_path, drawn):
    from pygltflib import GLTF2

    gltf = GLTF2().load(str(glb_path))
    by_name = {op["id"]: _jsonable(op) for op in drawn}
    for node in gltf.nodes:
        if node.name in by_name:
            node.extras = by_name.pop(node.name)
    if by_name:
        raise RuntimeError(f"nodes missing from the exported glb: {sorted(by_name)}")
    gltf.save(str(glb_path))


# -------------------------------------------------------------------- record


def attach_to_record(record_path, placement):
    """Add `openings` and `openings_camera` to an asset's record.json in place. Same list as the glTF extras."""
    path = Path(record_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["openings"] = _jsonable(placement.openings)
    record["openings_camera"] = {
        "camera_azimuth_deg": placement.camera.azimuth_deg,
        "camera_elevation_deg": placement.camera.elevation_deg,
        "camera_iou": placement.camera.iou,
        "metric": placement.metric,
    }
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def _load_mask(path):
    path = Path(path)
    return np.load(path) if path.suffix == ".npy" else path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mesh")
    parser.add_argument("mask")
    parser.add_argument("openings")
    parser.add_argument("--record", help="record.json: its mesh_to_enu gives metres and bearings; openings are written back")
    parser.add_argument("--out", help="output .glb (default: <mesh>_openings.glb beside the mesh)")
    args = parser.parse_args()

    record = json.loads(Path(args.record).read_text(encoding="utf-8")) if args.record else None
    m2e = mesh_to_enu_from_record(record)
    if m2e is None:
        print("no mesh_to_enu: widths/heights are in mesh units and bearing_deg is None", file=sys.stderr)
    openings, photo = load_openings_json(args.openings)
    mesh = trimesh.load(args.mesh, force="scene")
    placement = place_openings(mesh, _load_mask(args.mask), openings, m2e, source_photo=photo)
    cam = placement.camera
    print(f"camera azimuth {cam.azimuth_deg:.1f} elevation {cam.elevation_deg:.0f} iou {cam.iou:.3f}")
    for op in placement.openings:
        print(f"{op['id']:<16}{op['decision']:<8} bearing={op['bearing_deg']} snap={op['snap_angle_deg']} "
              f"{'; '.join(op['reasons'])}")
    out = args.out or str(Path(args.mesh).with_name(Path(args.mesh).stem + "_openings.glb"))
    export_openings_glb(args.mesh, placement.openings, out)
    if args.record:
        attach_to_record(args.record, placement)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
