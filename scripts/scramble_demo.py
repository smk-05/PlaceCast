"""
Prove the placement does not care what frame a model arrives in.

The claim the demo rests on is: hand us a 3D model in ANY units, at ANY
rotation, anywhere near the origin, and we will put it on the right building at
the right size and facing the right way. This script tests exactly that, on a
REAL generated mesh rather than a synthetic box.

It rewrites the run's mesh into a deliberately hostile frame —

    * a rotation about the up axis by an arbitrary angle (not a multiple of 90)
    * a swap of which axis is "up" (Y-up glTF -> Z-up, the classic trap)
    * a uniform scale of 1000 (millimetres, say, instead of metres)
    * a large translation off the origin

— clones it into a new run, re-solves, and compares the PLACED polygon (the
mesh put into ENU by the solver) against the original run's. Same building on
the ground, or the invariance claim is false.

Nothing is regenerated: FLUX and TRELLIS both cache on their output paths, so
this costs nothing.

    uv run python scripts/scramble_demo.py ASSET_ID
    uv run python scripts/scramble_demo.py ASSET_ID --keep   # leave the run
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geo import fit as fitmod, outline  # noqa: E402
from geo.coords import ENUFrame  # noqa: E402
from geo.footprint import (  # noqa: E402
    build_footprint,
    fetch_osm,
    geocode,
    select_footprint,
)
from scripts.clone_run import RUNS, clone  # noqa: E402

# Arbitrary on purpose: 37 degrees is not a multiple of 90, so a solver that
# only ever tried the four OMBB candidates of the ORIGINAL frame would fail.
YAW_DEG = 37.0
SCALE = 1000.0
OFFSET = (1234.5, -987.6, 42.0)


def scramble_matrix(swap_up: bool = False) -> np.ndarray:
    """Yaw about the model's own up, then scale, then translate. det > 0.

    `swap_up` additionally rewrites Y-up into Z-up, which BREAKS the glTF
    convention the mesh was written with. That is a different claim from
    frame-invariance and it degrades the result on purpose: "the asset faces +Z"
    is only meaningful inside the convention, so a Z-up model has no front, EXIF
    can no longer be tied to a facade, and orientation falls back to the weaker
    aspect-ratio cue. Measured on NCB: the azimuth then lands 90 degrees out.
    """
    up = [0, 0, 1] if swap_up else [0, 1, 0]
    m = (trimesh.transformations.translation_matrix(OFFSET)
         @ np.diag([SCALE, SCALE, SCALE, 1.0])
         @ trimesh.transformations.rotation_matrix(math.radians(YAW_DEG), up))
    if swap_up:
        m = m @ trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    return m


def scramble_glb(src: Path, dst: Path, swap_up: bool = False) -> None:
    scene = trimesh.load(str(src), force="scene")
    mesh = trimesh.util.concatenate(list(scene.geometry.values()))
    mesh.apply_transform(scramble_matrix(swap_up))
    trimesh.Scene(mesh).export(str(dst))


def placed_polygon(asset_id: str):
    """Re-derive where this run put the mesh, in its own ENU frame."""
    rec = json.loads((RUNS / asset_id / "record.json").read_text(encoding="utf-8"))
    sc = trimesh.load(str(RUNS / asset_id / "mesh.glb"), force="scene")
    verts = np.vstack([g.vertices for g in sc.geometry.values()])
    up, _ = outline.choose_up_axis(verts)
    mo = outline.build_mesh_outline(verts, up_axis_idx=up)
    # The serialised fit nests the pose under "transform" (contracts.py's
    # to_json_dict), not as flat theta/scale_x fields.
    t = rec["fit"]["transform"]
    sx, sy = t["scale"][0], t["scale"][1]
    moved = fitmod.apply_similarity(mo.pts_enu, t["rotation_z_rad"], sx, sy,
                                    np.array(t["translation_m"][:2]))
    from shapely.geometry import Polygon
    p = Polygon(moved)
    return (p if p.is_valid else p.buffer(0)), rec


def footprint_of(address: str):
    g = geocode(address)
    payload = fetch_osm(g["lat"], g["lon"])
    el, poly, _ = select_footprint(payload, g["lat"], g["lon"], address)
    frame = ENUFrame(lat0=poly.centroid.y, lon0=poly.centroid.x)
    return build_footprint(poly, frame, match_quality=el.get("_match_quality", ""))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("asset_id")
    ap.add_argument("--keep", action="store_true", help="keep the scrambled run")
    ap.add_argument("--swap-up", action="store_true",
                    help="also rewrite Y-up as Z-up, breaking the glTF "
                         "convention: shows where the front prior is lost")
    args = ap.parse_args()

    original, rec = placed_polygon(args.asset_id)
    inputs = rec.get("inputs") or {}
    address = rec["address_raw"]

    dst_id = clone(args.asset_id)
    scramble_glb(RUNS / args.asset_id / "mesh.glb", RUNS / dst_id / "mesh.glb",
                 args.swap_up)
    print(f"scrambled: yaw {YAW_DEG}°, scale x{SCALE:.0f}, offset {OFFSET}"
          + (", Y-up -> Z-up" if args.swap_up else ""))

    cmd = [sys.executable, "pipeline.py", "--address", address,
           "--asset-id", dst_id, "--perception", inputs.get("perception", "stub")]
    for p in inputs.get("photos", []):
        cmd += ["--photo", p]
    if inputs.get("conform"):
        cmd.append("--conform")
    if subprocess.run(cmd, cwd=ROOT).returncode not in (0, 2):  # 2 = REJECT
        print("the re-solve failed")
        return 1

    scrambled, rec2 = placed_polygon(dst_id)
    inter = original.intersection(scrambled).area
    union = original.union(scrambled).area
    agreement = inter / union if union else 0.0

    fp = footprint_of(address)
    print(f"\n{'':22}{'original':>12}{'scrambled':>12}")
    t1, t2 = rec["fit"]["transform"], rec2["fit"]["transform"]
    for name, a, b in [
        ("IoU vs footprint", rec["fit"]["iou"], rec2["fit"]["iou"]),
        ("Hausdorff (m)", rec["fit"]["hausdorff_m"], rec2["fit"]["hausdorff_m"]),
        ("heading (deg)", math.degrees(t1["heading_rad"]) % 360,
         math.degrees(t2["heading_rad"]) % 360),
        ("scale (model->m)", t1["scale"][0], t2["scale"][0]),
        ("height (m)", rec["height"]["value_m"], rec2["height"]["value_m"]),
    ]:
        print(f"{name:22}{a:>12.3f}{b:>12.3f}")
    print(f"\nplaced polygons agree: IoU {agreement:.4f} "
          f"(footprint area {fp.area_m2:.0f} m2)")
    print(f"scrambled run: {dst_id}")

    if not args.keep:
        import shutil
        shutil.rmtree(RUNS / dst_id)
        print("(removed; pass --keep to inspect it in the viewer)")
    return 0 if agreement >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
