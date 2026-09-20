"""
Can we place ANY correctly-proportioned building model, with no photo?

The generated meshes are shallow shells, which confounds two different
questions: "is the placement maths right?" and "is the generated model any
good?". This separates them. Every demo building already has a correctly
proportioned model available — its own footprint extruded to its OSM height —
so each one is exported as if a modeller had handed it over:

    * written Y-up, the glTF convention, not in our ENU frame
    * rotated by an arbitrary yaw (not a multiple of 90 degrees)
    * in arbitrary units (a random scale over four orders of magnitude)
    * centred wherever, far from the origin

and then placed from the ADDRESS ALONE. No photo, no EXIF, no mask: the
orientation has only the street prior and the footprint's own shape to go on,
which is the hardest honest case and the one a games studio with an asset
library would actually be in.

Truth is exact, because the model was built from the footprint: a perfect
placement reproduces the footprint. Reported per building:

    IoU       placed plan vs the real footprint (1.0 = exact)
    rot err   recovered rotation vs the known scramble, absolute and mod 90
    scale err recovered model-units-to-metres vs the known scramble
    facing    whether the building ends up pointing the right way (<10 deg)

    uv run python scripts/blind_placement_test.py
    uv run python scripts/blind_placement_test.py --seed 7 --reps 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contracts import Decision  # noqa: E402
from geo import disambiguate, fit as fitmod, outline, placement, validate  # noqa: E402
from geo.coords import ENUFrame  # noqa: E402
from geo.footprint import (  # noqa: E402
    build_footprint,
    fetch_osm,
    geocode,
    select_footprint,
    to_shapely,
)
from perception.openings_prism import prism_from_record  # noqa: E402
from pipeline import _roads_to_enu  # noqa: E402

RUNS = ROOT / "data" / "runs"
FACING_TOL_DEG = 10.0


def model_transform(yaw_deg: float, scale: float, offset: np.ndarray) -> np.ndarray:
    """The 4x4 taking ENU metres to the handed-over model's own frame.

    Kept explicit so the test can push KNOWN ENU points into model space and
    back out through the recovered placement: comparing positions in metres is
    convention-free, where comparing angles is not (the recovered theta is
    stated in the canonical frame, whose relation to the model's yaw depends on
    which up-axis was detected, so an angle comparison silently mis-scores).
    """
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    yaw = np.eye(4)
    yaw[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    to_y_up = np.eye(4)
    to_y_up[:3, :3] = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    scl = np.diag([scale, scale, scale, 1.0])
    mov = np.eye(4)
    mov[:3, 3] = offset
    return mov @ scl @ to_y_up @ yaw


def as_third_party_model(prism_mesh, yaw_deg: float, scale: float,
                         offset: np.ndarray, seed: int = 0) -> np.ndarray:
    """ENU prism -> vertices as an exported model: Y-up, yawed, scaled, moved.

    The prism is subdivided first. trimesh's extrusion puts vertices ONLY on
    the top and bottom rings, so the up-axis test (which measures cross-section
    area in 20 slabs) sees empty slabs and returns -inf for every axis. Any
    real model has geometry down its walls; without this the test would be
    measuring an artefact of how the prism was built, not the placement.
    """
    # SAMPLE THE SURFACE rather than take the vertices. geo.outline rasterises
    # a vertex cloud, so how well it works depends on how finely the model
    # happens to be tessellated: an extruded prism has vertices only on its top
    # and bottom rings, and even subdivided to 0.6 m facets its plan came out
    # 34x48 m for an 86x75 m building, because the rasterised ring has gaps the
    # closing radius cannot bridge. A generated mesh never shows this — its
    # vertices are dense and scattered. Sampling 60k points off the faces is
    # what any dense model looks like, and it keeps this test about placement
    # rather than about tessellation. (Rasterising faces directly is the real
    # fix; see the note in the summary.)
    # Seeded: an unseeded sampler draws a different cloud every run, and the
    # same scramble then scored 8/18 once and 3/18 the next time. The spread is
    # real — near-square plans are genuinely unstable — but it has to be
    # measured over scrambles, not accidentally re-rolled per run.
    pts, _ = trimesh.sample.sample_surface(prism_mesh, 60000, seed=seed)
    v = np.asarray(pts, dtype=float)
    m = model_transform(yaw_deg, scale, offset)
    return v @ m[:3, :3].T + m[:3, 3]


def _wrap(deg: float, period: float = 360.0) -> float:
    return abs((deg + period / 2) % period - period / 2)


def place_candidate(fp, verts, cand):
    """Solve with a GIVEN azimuth, for the oracle column."""
    up, _ = outline.choose_up_axis(verts)
    mo = outline.build_mesh_outline(verts, up_axis_idx=up,
                                    footprint_aspect=fp.ombb.aspect)
    rot, o, _ = outline.canonical_offsets(verts, mo.up_axis_idx)
    fit = fitmod.solve(fp, mo, chosen=cand,
                       disambiguated_by=disambiguate.Disambiguator.EXIF_HEADING)
    return placement.mesh_to_enu(fit, placement.mesh_to_canonical(rot, o))


def candidates_of(fp, verts):
    up, _ = outline.choose_up_axis(verts)
    mo = outline.build_mesh_outline(verts, up_axis_idx=up,
                                    footprint_aspect=fp.ombb.aspect)
    return [c for c, _ in fitmod.ombb_candidates(fp, mo)]


def place_blind(fp, roads_enu, verts: np.ndarray):
    """The pipeline's mesh path with NO photo evidence."""
    from fixtures.fake_photo_evidence import no_evidence

    up, _ = outline.choose_up_axis(verts)
    mo = outline.build_mesh_outline(verts, up_axis_idx=up,
                                    footprint_aspect=fp.ombb.aspect)
    rot, o, _ = outline.canonical_offsets(verts, mo.up_axis_idx)
    scored = fitmod.score_candidates(fp, mo, fitmod.ombb_candidates(fp, mo))
    chosen, by, _, _ = disambiguate.choose_orientation(
        fp, mo, scored, photo=no_evidence(), roads_enu=roads_enu,
        vertices_canonical=outline.canonicalise(verts, mo.up_axis_idx)[0])
    fit = fitmod.solve(fp, mo, chosen=chosen, disambiguated_by=by)
    placed = Polygon(fitmod.apply_similarity(
        mo.pts_enu, fit.theta, fit.scale_x, fit.scale_y,
        np.array([fit.tx, fit.ty])))
    m2e = placement.mesh_to_enu(fit, placement.mesh_to_canonical(rot, o))
    return (placed if placed.is_valid else placed.buffer(0)), fit, by, m2e


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1, help="scrambles per building")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # The prism records: a placement with no generated mesh.
    records = []
    for f in sorted(RUNS.glob("*/record.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        frame = (rec.get("mesh_to_enu") or {}).get("mesh_frame")
        if frame == "unit_box_centred" and rec.get("height"):
            records.append(rec)
    seen, unique = set(), []
    for rec in records:                       # newest record per address wins
        if rec["address_raw"] not in seen:
            seen.add(rec["address_raw"])
            unique.append(rec)

    print(f"{len(unique)} buildings, {args.reps} scramble(s) each, no photo evidence\n")
    print(f"{'building':<20}{'aspect':>7}{'yaw':>7}{'scale':>9}{'IoU':>7}"
          f"{'corner err':>11}{'scale err':>10}  facing  decided by")
    ok = 0
    total = 0
    rows: list[tuple] = []
    for rec in unique:
        address = rec["address_raw"]
        g = geocode(address)
        payload = fetch_osm(g["lat"], g["lon"])
        el, poly, _ = select_footprint(payload, g["lat"], g["lon"], address)
        frame = ENUFrame(lat0=poly.centroid.y, lon0=poly.centroid.x)
        fp = build_footprint(poly, frame, match_quality=el.get("_match_quality", ""),
                             geocode_location_type=g["location_type"])
        roads = _roads_to_enu(payload, frame)
        truth = to_shapely(fp)
        prism = prism_from_record(rec).mesh

        for _ in range(args.reps):
            yaw = float(rng.uniform(0, 360))
            scale = float(10 ** rng.uniform(-2, 2))     # 0.01x .. 100x
            offset = rng.uniform(-500, 500, 3)
            verts = as_third_party_model(prism, yaw, scale, offset,
                                         seed=int(rng.integers(1 << 31)))

            placed, fit, by, m2e = place_blind(fp, roads, verts)
            iou = placed.intersection(truth).area / placed.union(truth).area

            # Facing, measured in METRES and free of any angle convention: take
            # the footprint's own corners, carry them into the model's frame
            # with the scramble we applied, then back out through the recovered
            # placement. A correct placement returns each corner to where it
            # started; a building turned 180 degrees returns the WRONG corner to
            # each spot, which is exactly the failure IoU cannot see.
            corners = np.column_stack([fp.pts_enu, np.full(len(fp.pts_enu), 1.0)])
            to_model = model_transform(yaw, scale, offset)
            in_model = corners @ to_model[:3, :3].T + to_model[:3, 3]
            back = in_model @ m2e[:3, :3].T + m2e[:3, 3]
            corner_err = float(np.linalg.norm(back[:, :2] - corners[:, :2],
                                              axis=1).mean())
            tol = 0.1 * math.sqrt(fp.area_m2)                   # 10% of its size
            facing = corner_err < tol
            # Oracle: would ANY of the four azimuths have been right? This
            # separates "the geometry cannot do it" from "the evidence picked
            # the wrong one of four" - a distinction the headline number hides.
            oracle = False
            for cand in candidates_of(fp, verts):
                m = place_candidate(fp, verts, cand)
                b = in_model @ m[:3, :3].T + m[:3, 3]
                if float(np.linalg.norm(b[:, :2] - corners[:, :2], axis=1).mean()) < tol:
                    oracle = True
                    break
            scale_err = abs(fit.scale_x * scale - 1.0)
            _, decision, _ = validate.threshold_decision(fit, fp)

            ok += facing
            total += 1
            rows.append((facing, decision is Decision.AUTO_ACCEPT, corner_err,
                         scale_err, iou, oracle))
            print(f"{address.split(',')[0]:<20}{fp.ombb.aspect:>7.2f}{yaw:>7.0f}"
                  f"{scale:>9.3f}{iou:>7.3f}{corner_err:>10.1f}m{scale_err:>10.4f}"
                  f"  {'yes' if facing else 'NO ':>6}  {'o' if oracle else '-'} {by.value}"
                  f" ({'auto' if decision is Decision.AUTO_ACCEPT else decision.value})")

    autos = [r for r in rows if r[1]]
    good = [r for r in rows if r[0]]
    print(f"\nplaced correctly, including which way it faces: {ok}/{total}")
    if autos:
        wrong = [r for r in autos if not r[0]]
        print(f"auto-accepted: {len(autos)}/{total}, of which WRONG: {len(wrong)}"
              " — the rest were sent to review or rejected, not placed wrongly")
    print(f"an azimuth existed that WOULD have been right: "
          f"{sum(r[5] for r in rows)}/{total} (the 'o' column) - so the geometry "
          "can place these; picking which of the four needs evidence")
    if good:
        print(f"when right: corner error median {np.median([r[2] for r in good]):.2f} m, "
              f"scale error median {np.median([r[3] for r in good]) * 100:.2f}%, "
              f"IoU median {np.median([r[4] for r in good]):.3f}")
    print("'corner err' is how far the footprint's own corners land from where "
          "they started, in metres, after a round trip through the scramble and "
          "the recovered placement. A 180-degree error leaves the plan sitting "
          "perfectly on the footprint (high IoU) while every corner is swapped — "
          "the failure IoU cannot see, and the reason the pipeline asks for "
          "photo evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
