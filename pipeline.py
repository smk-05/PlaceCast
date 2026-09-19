"""
Stage wiring. One direction of flow, one immutable record out.

    address -> geocode -> footprint -> [photo -> edit -> lift] -> outline
            -> disambiguate -> solve -> height -> terrain -> validate
            -> PlacementRecord + overlay.png

Spec 14: every stage consumes and returns an immutable dataclass and no stage
reaches backwards, which makes the whole pipeline replayable from any
intermediate state. Every intermediate artifact is logged to disk under the
asset UUID — disk is free; a failure you cannot reproduce is not.

Execution order follows addendum D.1, and two of its constraints are
load-bearing rather than stylistic:

  1. 6.4 bas-relief rejection runs BEFORE any silhouette comparison.
  2. Segmentation happens at ingest, not at fit time — you will re-run the
     solver hundreds of times while debugging and it must never re-run a model.

Usage:
    uv run python pipeline.py --address "Burruss Hall, Blacksburg, VA" --dry-run
    uv run python pipeline.py --address "..." --photo shot.jpg --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Tokens (REPLICATE_API_TOKEN, CESIUM_ION_TOKEN, ...) live in .env, which is
# gitignored. Without this they are silently absent and generation fails.
load_dotenv(Path(__file__).resolve().parent / ".env")

from contracts import Decision, PlacementRecord  # noqa: E402
from geo import disambiguate, exif, fit as fitmod, height as heightmod, outline, placement, validate
from geo.coords import ENUFrame
from geo.footprint import build_footprint, fetch_osm, geocode, select_footprint, to_shapely
from geo.overlay import render_overlay

DATA_DIR = Path(__file__).resolve().parent / "data" / "runs"


def run(address: str,
        *,
        photos: list[Path] | None = None,
        prompt: str = "weathered concrete, ivy overgrowth, scorched upper floors",
        dry_run: bool = False,
        perception: str = "stub",
        confidence_method: str = "threshold_table",
        allow_anisotropy: bool = False,
        seed: int = 42,
        asset_id: str | None = None) -> PlacementRecord:

    asset_id = asset_id or str(uuid.uuid4())
    run_dir = DATA_DIR / asset_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _Logger(run_dir / "run.log")
    log(f"asset {asset_id}  |  {address}")

    # -- 3.1 geocode --------------------------------------------------------
    g = geocode(address)
    log(f"geocode: {g['provider']} ({g['lat']:.5f}, {g['lon']:.5f}) {g['location_type']}")

    # -- 3.2 footprint + neighbours + roads, one Overpass call --------------
    payload = fetch_osm(g["lat"], g["lon"])
    element, poly_lonlat, neighbours_lonlat = select_footprint(
        payload, g["lat"], g["lon"], address
    )
    tags = element.get("tags") or {}
    log(f"footprint: {element['type']} {element['id']} "
        f"{tags.get('name', '(unnamed)')} | {len(neighbours_lonlat)} neighbours")

    # ENU anchored at the footprint centroid (spec 2.2)
    frame = ENUFrame(lat0=poly_lonlat.centroid.y, lon0=poly_lonlat.centroid.x)
    fp = build_footprint(poly_lonlat, frame,
                         match_quality=element.get("_match_quality", ""),
                         geocode_location_type=g["location_type"])
    log(f"conditioned: {len(fp.pts_enu)} verts, {fp.area_m2:.0f} m2, "
        f"R={fp.rectilinearity:.3f}, aspect={fp.ombb.aspect:.2f}, "
        f"parts={1 + len(fp.parts_enu)}, match={fp.match_quality}")
    if fp.is_ill_posed:
        log("  WARNING: R < 0.6 — rotation is intrinsically ill-posed (spec 4.2)")

    neighbours_enu = [_to_enu_polygon(n, frame) for n in neighbours_lonlat]
    roads_enu = _roads_to_enu(payload, frame)

    # -- ingest: EXIF (free) then segmentation (once, cached) ---------------
    photo_ev = _build_photo_evidence(photos, perception, run_dir, log)

    # -- 5 generation -------------------------------------------------------
    mesh_vertices = None
    models: list[dict] = []
    if not dry_run and photos:
        mesh_vertices, models = _generate(photos, prompt, run_dir, seed, log)

    # -- 6.1-6.4 canonicalise -----------------------------------------------
    if mesh_vertices is not None:
        up_idx, up_reason = outline.choose_up_axis(np.asarray(mesh_vertices, dtype=float))
        log(f"up-axis: {up_reason}")
        mo = outline.build_mesh_outline(
            mesh_vertices, up_axis_idx=up_idx, footprint_aspect=fp.ombb.aspect
        )
        log(f"outline: {len(mo.pts_enu)} verts, up_axis={mo.up_axis_idx}, "
            f"extent_ratio={mo.extent_ratio:.3f}")
        if mo.is_bas_relief:
            # Ordering constraint: reject reliefs BEFORE trusting silhouettes.
            log("  REJECTED: bas-relief mesh (spec 6.4) — regenerate with more views")
            return _fail_record(asset_id, address, g, element, fp, frame,
                                "bas-relief mesh rejected (spec 6.4)", run_dir)
    else:
        mo = outline.outline_from_footprint(fp)
        log("outline: STUB (footprint OMBB) — a plain box, the hour-6 milestone object")

    # -- A.3: which side of the mesh the photo shows -> MeshOutline.front_angle --
    # Replaces the glTF "+Z is the front" ASSUMPTION with evidence, when there is
    # any. Must precede disambiguation: EXIF and road-normal both match a bearing
    # to the front facade.
    if mesh_vertices is not None:
        mo = _front_from_photo(mo, run_dir / "mesh.glb", photo_ev, log)

    # -- A.3 silhouette render-and-compare: after M_0 and the 6.4 check, before 6.6 ----
    photo_ev = _attach_silhouette_evidence(
        photo_ev, mo, run_dir / "mesh.glb" if mesh_vertices is not None else None,
        tags, fp, perception, log, has_photo=bool(photos))

    # -- 6.6 disambiguate, then 6.5-6.9 solve -------------------------------
    cands = fitmod.ombb_candidates(fp, mo)
    scored = fitmod.score_candidates(fp, mo, cands)
    chosen, by, reasons, exif_sil_disagree = disambiguate.choose_orientation(
        fp, mo, scored,
        photo=photo_ev,
        building_lat=poly_lonlat.centroid.y,
        building_lon=poly_lonlat.centroid.x,
        roads_enu=roads_enu,
        # CANONICAL vertices: facade_detail_scores reads sides in the canonical
        # XY plane. Raw glTF vertices are Y-up, which made its "sides"
        # meaningless (NCB reported an implausible 0.927 spread).
        vertices_canonical=(outline.canonicalise(mesh_vertices, mo.up_axis_idx)[0]
                            if mesh_vertices is not None else None),
    )
    log(f"orientation: k={chosen.azimuth_k} via {by.value}")
    for r in reasons:
        log(f"  note: {r}")

    result = fitmod.solve(fp, mo, chosen=chosen, disambiguated_by=by,
                          allow_anisotropy=allow_anisotropy)
    result = _replace(result, exif_silhouette_disagree=exif_sil_disagree)
    log(f"fit: IoU={result.iou:.3f} hausdorff={result.hausdorff_m:.2f}m "
        f"area_ratio={result.area_ratio:.3f} margin={result.rotation_margin_footprint:.3f}")

    # -- 9.2 neighbour collision --------------------------------------------
    metrics = validate.compute_metrics(fp, mo, {
        "theta": result.theta, "sx": result.scale_x,
        "sy": result.scale_y, "t": np.array([result.tx, result.ty]),
    })
    overlap = validate.neighbour_overlap(metrics["placed_polygon"], neighbours_enu)
    result = _replace(result, max_neighbor_overlap=overlap)
    if overlap > 0.02:
        log(f"  neighbour overlap {overlap:.3f}")

    # -- 7 height ------------------------------------------------------------
    plan_scale = 0.5 * (result.scale_x + result.scale_y)
    h, h_src, h_notes = heightmod.resolve_height(
        osm_tags=tags,
        mesh_height_units=mo.mesh_height_units,
        footprint_scale=plan_scale,
        footprint_area_m2=fp.area_m2,
    )
    result = _replace(result, height_m=h, height_source=h_src,
                      scale_z=h / max(mo.mesh_height_units, 1e-6))
    log(f"height: {h:.2f} m via {h_src.value}")
    for n in h_notes:
        log(f"  note: {n}")

    # -- 9.1 the gate --------------------------------------------------------
    from confidence.gate import evaluate_gate
    gate = evaluate_gate(result, photo_ev, fp, method=confidence_method)
    score, decision, gate_reasons = gate.score, gate.decision, list(gate.reasons)
    # Record the method that RAN. "learned" falls back to the table when no
    # trained model exists; logging the request would claim a model decided.
    method_ran = gate.method
    all_reasons = reasons + gate_reasons
    log(f"decision: {decision.value} (score {score:.3f}, method ran: {method_ran}"
        + (f", requested {gate.requested_method}" if gate.requested_method != method_ran else "")
        + ")")
    if gate.fallback_reason:
        log(f"  gate fallback: {gate.fallback_reason}")
    for r in gate_reasons:
        log(f"  {r}")

    # -- 9.3 overlay + 12.1 record ------------------------------------------
    overlay_path = render_overlay(
        run_dir / "overlay.png", fp, mo, result,
        neighbours=neighbours_enu, decision=decision,
        reasons=all_reasons,
        address=f"{tags.get('name') or address}  [{asset_id[:8]}]",
    )
    log(f"overlay: {overlay_path}")

    record = PlacementRecord(
        asset_id=asset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        address_raw=address,
        address_normalized=g.get("normalized", ""),
        geocode_provider=g["provider"],
        lat=g["lat"], lon=g["lon"], location_type=g["location_type"],
        footprint_source=fp.source,
        osm_type=element["type"], osm_id=element["id"],
        osm_version=int(element.get("version", 0)),
        footprint_geojson=_geojson(poly_lonlat),
        rectilinearity=fp.rectilinearity,
        footprint_ombb_m=(fp.ombb.a, fp.ombb.b),
        enu_origin_geodetic=(frame.lat0, frame.lon0, frame.h0),
        height_datum="cesium_ion_ellipsoidal",
        prompt=prompt,
        models=tuple(models),
        fit=result,
        decision=decision,
        confidence_p=score,
        confidence_method=method_ran,
        review_reasons=tuple(all_reasons),
    )

    # The matrix the viewer needs: the FitResult alone is defined on canonical
    # coordinates the viewer cannot reconstruct. Added to the record dict here
    # rather than to PlacementRecord, which lives in the frozen contract.
    if mesh_vertices is not None:
        rot, o, _ = outline.canonical_offsets(mesh_vertices, mo.up_axis_idx)
        pre, frame_name = placement.mesh_to_canonical(rot, o), "gltf_scene"
    else:
        pre, frame_name = placement.unit_box_to_canonical(fp.ombb), "unit_box_centred"
    m2e = placement.mesh_to_enu(result, pre)

    record_dict = record.to_json_dict()
    record_dict["mesh_to_enu"] = placement.record_entry(m2e, frame_name)

    record_path = run_dir / "record.json"
    record_path.write_text(json.dumps(record_dict, indent=2, default=str),
                           encoding="utf-8")
    log(f"record: {record_path}")
    log.close()
    return record


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _build_photo_evidence(photos, perception, run_dir, log):
    """EXIF first (free, spec 6.6 Filter 2), then segmentation once at ingest."""
    from contracts import PhotoEvidence

    if not photos:
        from fixtures.fake_photo_evidence import no_evidence
        log("photo: none supplied — using the no-evidence fixture")
        return no_evidence()

    primary = Path(photos[0])
    meta = exif.extract(primary)
    log(f"exif: heading={meta['heading_deg']} pitch={meta['pitch_deg']} "
        f"gps={meta['gps']} focal={meta['focal_mm']}")

    from perception.segment import assess_mask, clean, segment_building_evidence

    # Pass --perception through explicitly. perception/backend.py defaults to the
    # REAL models (torch + GPU) when the caller says nothing, so omitting this
    # made every photo run try to load Grounding DINO + SAM 2, even with the
    # default --perception stub.
    mask, _, seg_occluded, seg = segment_building_evidence(primary, backend=perception)
    mask = clean(mask)
    log(f"segmentation: {seg['segmentation_model']} (backend that ran: {seg['backend']})")

    frac, mask_occluded, note = assess_mask(mask)
    # Keep perception's own occlusion verdict: it knows how much of SAM's mask
    # clean-up discarded, which the finished mask alone cannot show.
    occluded = bool(seg_occluded or mask_occluded)
    if note:
        log(f"  mask: {note}")

    np.save(run_dir / "mask.npy", mask)   # cached: never re-run at fit time

    return PhotoEvidence(
        mask=mask,
        mask_area_frac=frac,
        occlusion_flag=occluded,
        photo_path=primary,
        photo_sha256=meta["sha256"],
        exif_heading_deg=meta["heading_deg"],
        exif_pitch_deg=meta["pitch_deg"],
        exif_focal_mm=meta["focal_mm"],
        exif_gps=meta["gps"],
        segmentation_model=seg["segmentation_model"],
    )


# Addendum A.3: the silhouette decides only above this margin.
FRONT_FROM_PHOTO_MIN_MARGIN = 0.08


def _front_from_photo(mo, mesh_glb, photo_ev, log):
    """Set MeshOutline.front_angle from perception's photographed_side().

    The photographed facade faces the camera, so the side the photo shows IS the
    facade EXIF and the road prior must match. Without this, front_angle is the
    glTF convention's guess (front = +Z), which TRELLIS does not guarantee.

    The side index is turned into an angle via the camera direction itself,
    rotated into the canonical frame by the same rotation canonicalisation used
    — so it holds for any up-axis, not only glTF's +Y.

    Abstains (keeps the prior) unless the mask came from the REAL segmenter —
    the stub's central box carries no shape — and the margin clears A.3's 0.08.
    """
    from dataclasses import replace

    if not photo_ev.segmentation_model.startswith("real:"):
        log(f"front: glTF prior kept (mask from {photo_ev.segmentation_model}, "
            "which carries no shape evidence)")
        return mo
    if not Path(mesh_glb).is_file():
        return mo

    import trimesh

    from perception.render_compare import camera_basis, photographed_side

    up = outline.UP_AXIS_CANDIDATES[mo.up_axis_idx]
    mesh = trimesh.load(str(mesh_glb), force="scene")
    side = photographed_side(mesh, photo_ev.mask, up_axis=tuple(up))

    if side.margin < FRONT_FROM_PHOTO_MIN_MARGIN:
        log(f"front: glTF prior kept (photographed-side margin {side.margin:.3f} "
            f"< {FRONT_FROM_PHOTO_MIN_MARGIN})")
        return mo

    _, _, toward = camera_basis(up, side.index * np.pi / 2)
    t = outline.canonical_rotation(mo.up_axis_idx) @ toward
    angle = float(np.arctan2(t[1], t[0]))
    prior = mo.front_angle
    log(f"front: photo shows mesh side {side.index} (margin {side.margin:.3f}); "
        f"front_angle {np.degrees(angle):.0f} deg"
        + ("" if prior is None else f" (glTF prior was {np.degrees(prior):.0f} deg)"))
    return replace(mo, front_angle=angle)


def _attach_silhouette_evidence(photo_ev, mo, mesh_glb, tags, fp, perception, log, *, has_photo):
    """Score M_0 against the photo mask and attach the scores and margin (addendum A.3, stages 2-3).

    Runs after the bas-relief check (D.1) and before orientation is chosen. No photo -> nothing to score: the
    no-evidence fixture's mask is not a photograph. `mesh_glb` is the generated mesh; None on --dry-run, where
    M_0 is the footprint's own OMBB and is scored as a box extruded to the OSM/fallback height.
    score_silhouettes scores the up-axis only and is flat across azimuth_k (perception/render_compare.py), so on
    one up-axis the margin is 0 and has_silhouette_evidence stays False: this wires the cue, it does not yet decide.
    """
    from perception.render_compare import margin_of, score_silhouettes

    mask = photo_ev.mask
    if not has_photo or mask is None or not np.asarray(mask).any():
        return photo_ev

    mesh = None
    if perception != "stub":  # the stub abstains without looking at the mesh, so do not build (or import) one
        import trimesh
        if mesh_glb is not None:
            mesh = trimesh.load(str(mesh_glb), force="scene") if Path(mesh_glb).is_file() else None
        else:
            h, _, _ = heightmod.resolve_height(osm_tags=tags, mesh_height_units=mo.mesh_height_units,
                                               footprint_scale=1.0, footprint_area_m2=fp.area_m2)
            mesh = trimesh.creation.box(extents=[mo.ombb.a, mo.ombb.b, h])
        if mesh is None:
            log(f"silhouette: skipped, no mesh at {mesh_glb}")
            return photo_ev

    cands = [cid for cid, _ in fitmod.ombb_candidates(fp, mo)]
    scores = score_silhouettes(mesh, mask, cands, backend=perception)
    margin = margin_of(scores)
    photo_ev = _replace(photo_ev, silhouette_scores=scores, silhouette_margin=margin)
    log(f"silhouette[{perception}]: " + " ".join(f"{c}={s:.3f}" for c, s in scores.items())
        + f" margin={margin:.3f} decides={photo_ev.has_silhouette_evidence}")
    return photo_ev


def _generate(photos, prompt, run_dir, seed, log):
    from generate.edit import edit_image
    from generate.lift import lift_to_mesh, load_vertices

    edited = []
    for i, p in enumerate(photos):
        out = run_dir / f"edited_{i}.png"
        log(f"edit: {p.name} -> {out.name}")
        edited.append(edit_image(Path(p), prompt, out, seed=seed))

    glb = run_dir / "mesh.glb"
    log("lift: TRELLIS (10-60 s)")
    glb, params = lift_to_mesh(edited, glb, seed=seed)

    models = [
        {"stage": "edit", "name": "flux-kontext-pro", "seed": seed},
        {"stage": "lift", "name": "trellis", "seed": seed, "params": params},
    ]
    return load_vertices(glb), models


def _to_enu_polygon(poly_lonlat, frame):
    from shapely.geometry import Polygon
    arr = np.asarray(poly_lonlat.exterior.coords)[:-1]
    enu = np.asarray(frame.geodetic_to_enu(arr[:, 1], arr[:, 0], 0.0))[:, :2]
    p = Polygon(enu)
    return p if p.is_valid else p.buffer(0)


# Streets a facade can face. Spec 6.6 Filter 3 says highway=*, but on a campus
# that is mostly footpaths: near NCB 67 of 90 highway ways were path, footway,
# steps or cycleway, and near Burruss 138 of 139 were not streets at all. Every
# wall has a sidewalk beside it, so "nearest highway" pointed at an arbitrary
# side. Only roads a building actually addresses count.
STREET_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
EXCLUDED_SERVICE = {"parking_aisle", "drive-through", "emergency_access"}


def _roads_to_enu(payload, frame) -> list[np.ndarray]:
    """Street ways in ENU, for spec 6.6 Filter 3's street-facing prior."""
    out = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        if tags.get("highway") not in STREET_HIGHWAYS or not el.get("geometry"):
            continue
        if tags.get("service") in EXCLUDED_SERVICE:
            continue
        arr = np.array([[p["lon"], p["lat"]] for p in el["geometry"]])
        enu = np.asarray(frame.geodetic_to_enu(arr[:, 1], arr[:, 0], 0.0))[:, :2]
        out.append(enu)
    return out


def _geojson(poly) -> dict:
    from shapely.geometry import mapping
    return dict(mapping(poly))


def _replace(fit, **kw):
    from dataclasses import replace
    return replace(fit, **kw)


def _fail_record(asset_id, address, g, element, fp, frame, reason, run_dir):
    """Spec 14: structured errors, never silent defaults."""
    rec = PlacementRecord(
        asset_id=asset_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        address_raw=address,
        geocode_provider=g["provider"], lat=g["lat"], lon=g["lon"],
        location_type=g["location_type"],
        osm_type=element["type"], osm_id=element["id"],
        rectilinearity=fp.rectilinearity,
        enu_origin_geodetic=(frame.lat0, frame.lon0, frame.h0),
        decision=Decision.REJECT,
        review_reasons=(reason,),
    )
    (run_dir / "record.json").write_text(
        json.dumps(rec.to_json_dict(), indent=2, default=str), encoding="utf-8"
    )
    return rec


class _Logger:
    def __init__(self, path: Path):
        # Append: a re-run with --asset-id keeps the earlier run's log above it.
        self.f = open(path, "a", encoding="utf-8")
        self.f.write(f"\n===== run {datetime.now(timezone.utc).isoformat()} =====\n")

    def __call__(self, msg: str) -> None:
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Photo -> map-ready 3D building.")
    ap.add_argument("--address", required=True)
    ap.add_argument("--photo", action="append", type=Path, default=[],
                    help="repeatable; multi-view conditioning (spec 5.2)")
    ap.add_argument("--prompt",
                    default="weathered concrete, ivy overgrowth, scorched upper floors")
    ap.add_argument("--dry-run", action="store_true",
                    help="skip generation; use the footprint OMBB as M_0")
    ap.add_argument("--perception", choices=["stub", "real"], default="stub")
    ap.add_argument("--confidence", choices=["threshold_table", "learned"],
                    default="threshold_table")
    ap.add_argument("--anisotropy", action="store_true",
                    help="allow limited non-uniform scaling (spec 6.9)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asset-id", default=None,
                    help="re-use an existing run directory: its cached edit and "
                         "mesh are used, so no model is re-run and nothing is billed")
    args = ap.parse_args(argv)

    try:
        rec = run(args.address, photos=list(args.photo), prompt=args.prompt,
                  dry_run=args.dry_run, perception=args.perception,
                  confidence_method=args.confidence,
                  allow_anisotropy=args.anisotropy, seed=args.seed,
                  asset_id=args.asset_id)
    except (LookupError, RuntimeError, ValueError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\n{rec.decision.value.upper()}  asset {rec.asset_id}")
    return 0 if rec.decision is not Decision.REJECT else 2


if __name__ == "__main__":
    raise SystemExit(main())
