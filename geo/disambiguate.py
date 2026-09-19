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

import numpy as np

from contracts import CandidateId, Disambiguator, Footprint, MeshOutline, PhotoEvidence

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
    """
    if photo.exif_gps is not None:
        cam_lat, cam_lon = photo.exif_gps
        d_north = (building_lat - cam_lat) * 111_320.0
        d_east = (building_lon - cam_lon) * 111_320.0 * math.cos(math.radians(cam_lat))
        if math.hypot(d_north, d_east) > 2.0:
            return math.atan2(d_north, d_east)

    if photo.exif_heading_deg is not None:
        # Compass heading (CW from North) -> mathematical theta (CCW from East).
        return math.radians(90.0 - photo.exif_heading_deg)
    return None


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


def choose_orientation(fp: Footprint,
                       mo: MeshOutline,
                       scored: list[tuple[CandidateId, float]],
                       *,
                       photo: PhotoEvidence | None = None,
                       building_lat: float | None = None,
                       building_lon: float | None = None,
                       roads_enu: list[np.ndarray] | None = None,
                       vertices_canonical: np.ndarray | None = None,
                       ) -> tuple[CandidateId, Disambiguator, list[str]]:
    """-> (chosen candidate, which filter decided, review reasons).

    Precedence is addendum A.3's, extended with the mesh-only filters:

      1. EXIF heading, present and plausible            -> it decides
      2. silhouette score, if margin >= 0.08            -> it decides
      3. facade detail agreeing with the road normal    -> it decides
      4. road normal alone                              -> decides, ALWAYS flagged
      5. best footprint IoU                             -> decides, flagged if margin low

    Do not let a later filter silently override a confident earlier one.
    """
    reasons: list[str] = []
    viable = viable_by_aspect(fp, [c for c, _ in scored])
    by_score = [c for c, _ in scored if c in viable] or [c for c, _ in scored]
    margin = scored[0][1] - scored[1][1] if len(scored) > 1 else 0.0

    # Genuinely symmetric: no filter can resolve what has no answer.
    aspect = fp.ombb.aspect if fp.ombb else 1.0
    if aspect < SYMMETRIC_ASPECT and margin < 0.05:
        sil_margin = photo.silhouette_margin if photo else 0.0
        if sil_margin < 0.05:
            reasons.append(
                f"symmetric footprint (aspect {aspect:.2f}) with no discriminating "
                "evidence — orientation is arbitrary, not wrong"
            )
            return by_score[0], Disambiguator.ARBITRARY_SYMMETRIC, reasons

    # --- 1. EXIF -----------------------------------------------------------
    if photo is not None and building_lat is not None and building_lon is not None:
        cam_az = azimuth_from_exif(photo, building_lat, building_lon)
        if cam_az is not None:
            chosen = _closest_candidate_to_facing(mo, viable, cam_az + math.pi)
            if chosen is not None:
                return chosen, Disambiguator.EXIF_HEADING, reasons

    # --- 2. silhouette (perception; absent until the friend's code lands) ---
    if photo is not None and photo.has_silhouette_evidence:
        ranked = sorted(photo.silhouette_scores.items(), key=lambda kv: kv[1],
                        reverse=True)
        for cid, _ in ranked:
            if cid in viable:
                if margin > 0.05 and cid != by_score[0]:
                    reasons.append(
                        "silhouette and footprint-IoU margins disagree — "
                        "independent evidence conflicts (addendum A.3)"
                    )
                return cid, Disambiguator.SILHOUETTE, reasons

    # --- 3 / 4. road normal, optionally corroborated by facade detail ------
    normal = road_normal(fp, roads_enu or [])
    if normal is not None:
        chosen = _closest_candidate_to_facing(mo, viable, normal)
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
                    return chosen, Disambiguator.FACADE_DETAIL, reasons
            reasons.append(
                "orientation from the street-facing prior alone — this is a "
                "prior, not evidence; flagged regardless of IoU (spec 6.6)"
            )
            return chosen, Disambiguator.ROAD_NORMAL, reasons

    # --- 5. fall back to IoU ----------------------------------------------
    if margin < 0.05:
        reasons.append(
            f"rotation margin {margin:.3f} < 0.05 with no disambiguating cue"
        )
    return by_score[0], Disambiguator.ASPECT_RATIO, reasons


def _closest_candidate_to_facing(mo: MeshOutline, viable: list[CandidateId],
                                 target_az: float) -> CandidateId | None:
    """Pick the candidate whose front face points nearest `target_az`.

    The mesh's front is +X in the canonical frame (spec 6.1: glTF's asset front
    faces +Z, rotated to +X when we canonicalise the up-axis to +Z), so
    candidate k faces base_angle + k*pi/2.
    """
    if not viable or mo.ombb is None:
        return None
    base = mo.ombb.angle
    best, best_d = None, float("inf")
    for c in viable:
        facing = base + c.azimuth_k * math.pi / 2.0
        d = abs(math.atan2(math.sin(facing - target_az), math.cos(facing - target_az)))
        if d < best_d:
            best, best_d = c, d
    return best
