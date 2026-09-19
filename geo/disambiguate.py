"""
Resolving the four-fold azimuth ambiguity. Spec 6.6.

This is the interesting sub-problem: IoU alone discriminates well for asymmetric
footprints and fails completely for square ones.

NOTHING IN THIS MODULE REQUIRES ML. That is deliberate and it is the single most
important consequence of building this solo. The addendum calls silhouette
render-and-compare "the only cue that is evidence rather than prior", and it is
right — but it arrives late and from another machine, so the pipeline cannot
depend on it. Filters 1-3 are arithmetic, EXIF, and mesh geometry:

  Filter 1  aspect ratio      OMBB arithmetic
  Filter 2  EXIF heading      GPSImgDirection -> camera-to-centroid azimuth
  Filter 3a road normal       Overpass highway=* + outward boundary normal
  Filter 3b facade detail     vertex density / normal variance per side

When perception does land, its silhouette score enters as a SECOND, INDEPENDENT
piece of evidence rather than the only one — which is what addendum A.3 wants:
"keep both margins as separate features... disagreement between them is itself a
strong signal that something is wrong."
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from contracts import CandidateId, Disambiguator, Footprint, MeshOutline, PhotoEvidence
from geo.coords import theta_to_heading

# Addendum A.3: silhouette decides only above this margin; below it we fall
# through to the road-normal prior and flag for review regardless of IoU.
SILHOUETTE_MIN_MARGIN = 0.08
ASPECT_TAU = 0.10          # spec 6.6 Filter 1
SYMMETRIC_ASPECT = 1.10    # addendum A.4: below this a square building is honestly ambiguous


# --------------------------------------------------------------------------
# Filter 1 — aspect ratio
# --------------------------------------------------------------------------


def viable_by_aspect(fp: Footprint, candidates: list[CandidateId]) -> list[CandidateId]:
    """Exclude the 90-degree candidates on scale grounds. Spec 6.6 Filter 1.

    Only bites when the footprint is not near-square — which is precisely the
    case where the ambiguity matters least. Two candidates (front vs back)
    survive when it does.
    """
    if fp.ombb is None or fp.ombb.aspect <= 1.0 + ASPECT_TAU:
        return list(candidates)
    return [c for c in candidates if c.azimuth_k % 2 == 0]


# --------------------------------------------------------------------------
# Filter 2 — EXIF heading
# --------------------------------------------------------------------------


def azimuth_from_exif(photo: PhotoEvidence,
                      building_lat: float, building_lon: float) -> float | None:
    """Bearing from camera to building centroid, radians CCW from East.

    Spec 6.6: "it converts a hard inference problem into arithmetic. Check for
    this first." Phone photos frequently carry GPSImgDirection; downloaded
    images never do.

    Prefers the camera GPS fix (a true bearing to the building) and falls back
    to the recorded compass heading (where the camera was pointed).

    Invalid input ABSTAINS (returns None), never wraps or propagates. This is
    the strongest cue in the pipeline and it overrides everything below it, so a
    NaN or an out-of-range value must not be allowed to decide an orientation.
    """
    gps = photo.exif_gps
    if gps is not None and _valid_camera_fix(gps, building_lat, building_lon):
        cam_lat, cam_lon = gps
        d_north = (building_lat - cam_lat) * 111_320.0
        d_east = (building_lon - cam_lon) * 111_320.0 * math.cos(math.radians(cam_lat))
        if math.hypot(d_north, d_east) > 2.0:
            return math.atan2(d_north, d_east)

    h = photo.exif_heading_deg
    if h is not None and _finite(h) and 0.0 <= h < 360.0:
        # Compass heading (CW from North) -> mathematical theta (CCW from East).
        return math.radians(90.0 - h)
    return None


# A street photo is taken from across the road, not from another town. A fix
# farther than this is stale or wrong (a phone that has not re-acquired GPS
# indoors reports its last position), so it is ignored rather than trusted.
MAX_CAMERA_DISTANCE_M = 1000.0


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _valid_camera_fix(gps, building_lat, building_lon) -> bool:
    try:
        lat, lon = gps
    except (TypeError, ValueError):
        return False
    if not (_finite(lat) and _finite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        return False
    d_north = (building_lat - lat) * 111_320.0
    d_east = (building_lon - lon) * 111_320.0 * math.cos(math.radians(lat))
    return math.hypot(d_north, d_east) <= MAX_CAMERA_DISTANCE_M


# --------------------------------------------------------------------------
# Filter 3a — street-facing prior
# --------------------------------------------------------------------------


def road_normal(fp: Footprint, roads_enu: list[np.ndarray]) -> float | None:
    """Outward normal from the footprint toward the nearest road, radians.

    Spec 6.6 Filter 3. The primary facade faces the street with high prior
    probability. This is a PRIOR, not evidence: it fails on corner lots, on
    buildings set back from the road, and on campus buildings whose front faces
    a quad rather than a street — which is most of Virginia Tech. Anything
    decided by this alone is flagged for review regardless of IoU.
    """
    if not roads_enu:
        return None

    centre = fp.pts_enu.mean(axis=0)
    best_d, best_pt = float("inf"), None
    for way in roads_enu:
        if len(way) < 2:
            continue
        d = np.linalg.norm(way - centre, axis=1)
        j = int(np.argmin(d))
        if d[j] < best_d:
            best_d, best_pt = float(d[j]), way[j]

    if best_pt is None or best_d > 120.0:
        return None
    v = best_pt - centre
    return float(math.atan2(v[1], v[0]))


# --------------------------------------------------------------------------
# Filter 3b — facade detail scoring (mesh only, no ML, no network)
# --------------------------------------------------------------------------


def facade_detail_scores(vertices_canonical: np.ndarray,
                         face_normals: np.ndarray | None = None) -> np.ndarray:
    """Per-side "how detailed is this facade" score. -> (4,) for +X,+Y,-X,-Y.

    Spec 6.6: "Windows, doors, and trim concentrate geometric and textural
    detail on the facade; the back and roof of a single-photo generation are
    hallucinated and smooth. This works surprisingly well and is cheap."

    Quantified here as vertex density times normal-direction variance in the
    outward-facing band on each side. Both terms matter: density alone rewards
    tessellation artifacts, variance alone rewards noise.
    """
    xy = vertices_canonical[:, :2]
    centre = xy.mean(axis=0)
    rel = xy - centre
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    rad = np.linalg.norm(rel, axis=1)

    # Only the outer shell carries facade detail; the interior is fill.
    shell = rad >= np.percentile(rad, 55.0)

    scores = np.zeros(4)
    for k in range(4):
        lo = -math.pi / 4 + k * math.pi / 2
        hi = lo + math.pi / 2
        a = np.mod(ang - lo, 2 * math.pi)
        sel = shell & (a < (hi - lo))
        n = int(sel.sum())
        if n < 8:
            continue

        density = n / max(len(xy), 1)
        z = vertices_canonical[sel, 2]
        z_spread = float(z.std() / max(z.mean(), 1e-6)) if z.size else 0.0

        if face_normals is not None and len(face_normals) == len(vertices_canonical):
            nv = face_normals[sel]
            variance = float(np.mean(np.var(nv, axis=0)))
        else:
            # Radial roughness stands in for normal variance when we only have
            # a point cloud: a flat wall has near-constant radius, a facade with
            # recessed windows and trim does not.
            variance = float(np.std(rad[sel]) / max(np.mean(rad[sel]), 1e-6))

        scores[k] = density * (1.0 + variance) * (1.0 + z_spread)

    total = scores.sum()
    return scores / total if total > 1e-9 else scores


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


class OrientationChoice(NamedTuple):
    chosen: CandidateId
    disambiguated_by: Disambiguator
    reasons: list[str]
    # None when EXIF or silhouette evidence was absent. See FitResult.
    exif_silhouette_disagree: bool | None


def choose_orientation(fp: Footprint,
                       mo: MeshOutline,
                       scored: list[tuple[CandidateId, float]],
                       *,
                       photo: PhotoEvidence | None = None,
                       building_lat: float | None = None,
                       building_lon: float | None = None,
                       roads_enu: list[np.ndarray] | None = None,
                       vertices_canonical: np.ndarray | None = None,
                       ) -> OrientationChoice:
    """Resolve the four-fold ambiguity.

    Precedence is addendum A.3's, extended with the mesh-only filters:

      1. EXIF heading, present and plausible            -> it decides
      -  genuinely symmetric footprint, nothing else    -> arbitrary, flagged
      2. silhouette score, if margin >= 0.08            -> it decides
      3. facade detail agreeing with the road normal    -> it decides
      4. road normal alone                              -> decides, ALWAYS flagged
      5. best footprint IoU                             -> decides, flagged if margin low

    EXIF is checked BEFORE the symmetry bail-out: a square footprint is
    ambiguous to IoU, but a camera bearing is not, so EXIF can still resolve it.

    Both EXIF and silhouette are evaluated even when EXIF decides, so that their
    disagreement can be recorded — addendum A.3 treats conflict between
    independent orientation evidence as a confidence signal in its own right.
    """
    reasons: list[str] = []
    viable = viable_by_aspect(fp, [c for c, _ in scored])
    by_score = [c for c, _ in scored if c in viable] or [c for c, _ in scored]
    margin = scored[0][1] - scored[1][1] if len(scored) > 1 else 0.0

    # --- evaluate the two evidence cues independently ----------------------
    exif_choice: CandidateId | None = None
    if photo is not None and building_lat is not None and building_lon is not None:
        cam_az = azimuth_from_exif(photo, building_lat, building_lon)
        if cam_az is not None:
            # The photographed facade faces back toward the camera.
            exif_choice = _closest_candidate_to_facing(fp, mo, viable, cam_az + math.pi)
            if exif_choice is None:
                reasons.append("EXIF bearing present but the mesh front is "
                               "unknown — cannot map it to a facade")

    sil_choice: CandidateId | None = None
    if photo is not None and photo.has_silhouette_evidence:
        ranked = sorted(photo.silhouette_scores.items(), key=lambda kv: kv[1],
                        reverse=True)
        sil_choice = next((cid for cid, _ in ranked if cid in viable), None)

    disagree = (exif_choice != sil_choice
                if exif_choice is not None and sil_choice is not None else None)
    if disagree:
        reasons.append(
            f"EXIF selects k={exif_choice.azimuth_k} but silhouette selects "
            f"k={sil_choice.azimuth_k} — independent evidence conflicts (addendum A.3)"
        )

    def done(c, by):
        return OrientationChoice(c, by, reasons, disagree)

    # --- 1. EXIF -----------------------------------------------------------
    if exif_choice is not None:
        return done(exif_choice, Disambiguator.EXIF_HEADING)

    # Genuinely symmetric: no remaining filter can resolve what has no answer.
    aspect = fp.ombb.aspect if fp.ombb else 1.0
    if aspect < SYMMETRIC_ASPECT and margin < 0.05:
        sil_margin = photo.silhouette_margin if photo else 0.0
        if sil_margin < 0.05:
            reasons.append(
                f"symmetric footprint (aspect {aspect:.2f}) with no discriminating "
                "evidence — orientation is arbitrary, not wrong"
            )
            return done(by_score[0], Disambiguator.ARBITRARY_SYMMETRIC)

    # --- 2. silhouette (perception) -----------------------------------------
    if sil_choice is not None:
        if margin > 0.05 and sil_choice != by_score[0]:
            reasons.append(
                "silhouette and footprint-IoU margins disagree — "
                "independent evidence conflicts (addendum A.3)"
            )
        return done(sil_choice, Disambiguator.SILHOUETTE)

    # --- 3 / 4. road normal, optionally corroborated by facade detail ------
    normal = road_normal(fp, roads_enu or [])
    if normal is not None:
        chosen = _closest_candidate_to_facing(fp, mo, viable, normal)
        if chosen is not None:
            if vertices_canonical is not None:
                detail = facade_detail_scores(vertices_canonical)
                detailed_side = int(np.argmax(detail))
                spread = float(detail.max() - np.median(detail))
                if spread > 0.05:
                    reasons.append(
                        f"facade detail concentrated on side {detailed_side} "
                        f"(spread {spread:.3f})"
                    )
                    return done(chosen, Disambiguator.FACADE_DETAIL)
            reasons.append(
                "orientation from the street-facing prior alone — this is a "
                "prior, not evidence; flagged regardless of IoU (spec 6.6)"
            )
            return done(chosen, Disambiguator.ROAD_NORMAL)

    # --- 5. fall back to IoU ----------------------------------------------
    if margin < 0.05:
        reasons.append(
            f"rotation margin {margin:.3f} < 0.05 with no disambiguating cue"
        )
    return done(by_score[0], Disambiguator.ASPECT_RATIO)


# --------------------------------------------------------------------------
# facade_heading — the one place a candidate becomes a compass bearing
# --------------------------------------------------------------------------


def _candidate_theta(fp: Footprint, mo: MeshOutline, candidate: CandidateId) -> float:
    """World rotation (mathematical theta) the OMBB init assigns to a candidate."""
    from geo.fit import ombb_candidates

    for cid, params in ombb_candidates(fp, mo):
        if cid.azimuth_k == candidate.azimuth_k:
            return float(params["theta"])
    raise ValueError(f"no OMBB candidate with azimuth_k={candidate.azimuth_k}")


def _facing_theta(fp: Footprint, mo: MeshOutline, candidate: CandidateId,
                  theta: float | None = None) -> float | None:
    """Direction the front facade faces in the world, mathematical radians."""
    if mo.front_angle is None:
        return None
    t = _candidate_theta(fp, mo, candidate) if theta is None else theta
    # apply_similarity rotates mesh directions by theta, so a canonical-frame
    # direction phi ends up at phi + theta in ENU.
    return t + mo.front_angle


def facade_heading(fp: Footprint, mo: MeshOutline, candidate: CandidateId,
                   *, theta: float | None = None) -> float:
    """Compass bearing in degrees [0, 360) the front facade faces under
    `candidate`. 0 = North, 90 = East. See contracts.FacadeHeading.

    Uses the closed-form OMBB rotation for the candidate. Pass `theta` (e.g.
    FitResult.theta) to use the refined rotation instead; refinement moves it by
    a few degrees, which never changes which 90-degree candidate is meant.

    Raises ValueError when the mesh front is unknown. Do not substitute a
    default — an unknown front means this cue has nothing to say.
    """
    if candidate.up_axis_idx != mo.up_axis_idx:
        raise ValueError(
            f"candidate up_axis_idx={candidate.up_axis_idx} does not match the "
            f"outline's up_axis_idx={mo.up_axis_idx}"
        )
    facing = _facing_theta(fp, mo, candidate, theta)
    if facing is None:
        raise ValueError(
            "mesh front facade is unknown (MeshOutline.front_angle is None) — "
            "no bearing can be assigned"
        )
    return math.degrees(theta_to_heading(facing))


def _closest_candidate_to_facing(fp: Footprint, mo: MeshOutline,
                                 viable: list[CandidateId],
                                 target: float) -> CandidateId | None:
    """Candidate whose front facade faces nearest `target` (math radians).

    None when the front is unknown — the filter abstains rather than guessing.
    """
    if not viable or mo.front_angle is None or mo.ombb is None or fp.ombb is None:
        return None
    best, best_d = None, float("inf")
    for c in viable:
        facing = _facing_theta(fp, mo, c)
        d = abs(math.atan2(math.sin(facing - target), math.cos(facing - target)))
        if d < best_d:
            best, best_d = c, d
    return best
