"""Orientation fusion ladder (ML_ADDENDUM A.3 "Fusion with the other filters"), on contracts.py types.

    choice = choose_orientation(photo, candidates, facade_heading, road_normal_deg=None, symmetric=False)
    choice.candidate, choice.disambiguated_by, choice.needs_review, choice.reason,
    choice.silhouette_margin, choice.exif_silhouette_disagree

The pipeline's own ladder is geo.disambiguate.choose_orientation (five levels, and it owns the facade-to-bearing
mapping). This is the perception-side statement of the A.3 ladder over the same objects, kept so the precedence
rules and their edge cases are pinned by tests independently of geo. It uses the same vocabulary, and the same
meaning of every field.

Precedence (never let render-and-compare silently override a confident EXIF heading):
  1. EXIF GPSImgDirection present and plausible  -> it decides      Disambiguator.EXIF_HEADING
  2. else PhotoEvidence.has_silhouette_evidence  -> silhouette       Disambiguator.SILHOUETTE   (margin >= 0.08)
  3. else road-normal prior (SPEC 6.6 Filter 3)  -> nearest facade   Disambiguator.ROAD_NORMAL, needs_review
     regardless of IoU: you are now guessing.
  If nothing has evidence, the best-silhouette candidate (else the first) is returned, flagged for review, as
  ARBITRARY_SYMMETRIC when the caller says the footprint is symmetric (A.4), else ASPECT_RATIO: the value
  contracts.py gives "nothing else decided".

`exif_silhouette_disagree` follows the contract (FitResult.exif_silhouette_disagree): None unless BOTH cues
produced a choice, then whether they differ. It is recorded and never acted on; EXIF still wins.

Inputs
  photo         a contracts.PhotoEvidence: exif_heading_deg, silhouette_scores {CandidateId: iou},
                silhouette_margin, has_silhouette_evidence.
  candidates    the viable geo placement candidates (CandidateId), e.g. geo.disambiguate.viable_by_aspect. EXIF and
                the road normal choose among them; the up-axis is not something either can resolve.
  facade_heading  callable(CandidateId) -> compass bearing (deg clockwise from north) the mesh front faces under
                that placement: geo.disambiguate.facade_heading with fp and mo bound. It raises ValueError when the
                mesh front is unknown; that means "no bearing" (the cue abstains, and the reason says so), never
                a default. Any other failure, including a non-finite bearing, propagates.
  road_normal_deg  bearing of the footprint's outward normal toward the nearest road (fetched by geo/, never
                here). `bearing_from_enu` converts (east, north).

The EXIF heading is the camera's pointing direction, so the photographed facade faces the opposite bearing
(heading + 180). EXIF is assumed relative to true north (PhotoEvidence carries no GPSImgDirectionRef). An
implausible value (outside [0, 360), non-finite, non-numeric) is ignored and the reason says so. Ties go to the
better silhouette score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from contracts import CandidateId, Disambiguator


@dataclass(frozen=True)
class OrientationChoice:
    candidate: CandidateId
    disambiguated_by: Disambiguator
    needs_review: bool
    reason: str  # human-readable audit trail for the intermediate-artifact log
    silhouette_margin: float  # photo.silhouette_margin, whichever cue decided (a confidence-gate feature)
    exif_silhouette_disagree: Optional[bool] = None  # None when either cue was absent


def bearing_from_enu(east: float, north: float) -> float:
    """Compass bearing (degrees clockwise from north, [0, 360)) of an ENU vector."""
    return math.degrees(math.atan2(east, north)) % 360.0


def angular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def plausible_heading(value) -> Optional[float]:
    """EXIF GPSImgDirection as a float in [0, 360), else None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        heading = float(value)
    except (TypeError, ValueError):
        return None
    return heading if math.isfinite(heading) and 0.0 <= heading < 360.0 else None


def _nearest_facade(candidates, bearing, facade_heading, scores):
    """(candidate, its facade bearing, angular difference) closest to `bearing`; ties -> better silhouette score.

    Raises ValueError when facade_heading does ("mesh front unknown"); the caller treats that as abstaining.
    """
    ranked = []
    for order, c in enumerate(candidates):
        facing = float(facade_heading(c))
        if not math.isfinite(facing):
            raise RuntimeError(f"facade_heading returned {facing!r} for candidate {c}")
        ranked.append((round(angular_diff_deg(facing, bearing), 6), -scores.get(c, 0.0), order, c, facing))
    diff, _, _, chosen, facing = min(ranked, key=lambda r: r[:3])
    return chosen, facing, diff


def _best_by_silhouette(candidates, scores):
    return max(candidates, key=lambda c: scores.get(c, float("-inf")))  # first maximum: stable


def choose_orientation(photo, candidates, facade_heading: Callable, road_normal_deg=None, symmetric=False) -> OrientationChoice:
    candidates = list(candidates)
    if not candidates:
        raise ValueError("no candidates to choose among")
    scores = dict(photo.silhouette_scores)
    margin = photo.silhouette_margin
    notes = []

    # --- evaluate both evidence cues independently, so their disagreement can be recorded ---------------------
    raw = photo.exif_heading_deg
    heading = plausible_heading(raw)
    if raw is not None and heading is None:
        notes.append(f"exif heading {raw!r} implausible, ignored")
    exif_choice = exif_reason = None
    if heading is not None:
        facing = (heading + 180.0) % 360.0  # the photographed facade faces back toward the camera
        try:
            exif_choice, bearing, diff = _nearest_facade(candidates, facing, facade_heading, scores)
            exif_reason = (f"exif heading {heading:.1f} -> photographed facade faces {facing:.1f}; "
                           f"nearest candidate faces {bearing:.1f} ({diff:.1f} deg off)")
        except ValueError as exc:
            notes.append(f"EXIF bearing present but cannot be mapped to a facade ({exc})")

    sil_choice = None
    if photo.has_silhouette_evidence:
        in_play = [c for c in candidates if c in scores]
        sil_choice = _best_by_silhouette(in_play, scores) if in_play else None

    disagree = (exif_choice != sil_choice) if exif_choice is not None and sil_choice is not None else None
    lead = "; ".join(notes) + ("; " if notes else "")

    def done(candidate, by, review, reason):
        return OrientationChoice(candidate, by, review, lead + reason, margin, disagree)

    # 1. EXIF decides.
    if exif_choice is not None:
        extra = f"; DISAGREES with a confident silhouette (margin {margin:.3f})" if disagree else ""
        return done(exif_choice, Disambiguator.EXIF_HEADING, False, exif_reason + extra)

    # 2. A confident silhouette decides.
    if sil_choice is not None:
        return done(sil_choice, Disambiguator.SILHOUETTE, False, f"silhouette margin {margin:.3f} >= 0.08")

    # 3. Road-normal prior; a guess, so always review.
    if road_normal_deg is not None:
        road = float(road_normal_deg)
        if not math.isfinite(road):
            raise ValueError(f"road_normal_deg must be finite, got {road_normal_deg!r}")
        try:
            chosen, bearing, diff = _nearest_facade(candidates, road % 360.0, facade_heading, scores)
        except ValueError as exc:
            lead += f"road normal cannot be mapped to a facade ({exc}); "
        else:
            return done(chosen, Disambiguator.ROAD_NORMAL, True,
                        f"silhouette margin {margin:.3f} < 0.08 and no usable EXIF; road normal {road % 360.0:.1f} -> "
                        f"nearest candidate faces {bearing:.1f} ({diff:.1f} deg off); guessing")

    # No evidence left: the best silhouette candidate, visibly unreliable.
    by = Disambiguator.ARBITRARY_SYMMETRIC if symmetric else Disambiguator.ASPECT_RATIO
    return done(_best_by_silhouette(candidates, scores), by, True,
                f"silhouette margin {margin:.3f} < 0.08, no usable EXIF, no road normal; nothing decided")
