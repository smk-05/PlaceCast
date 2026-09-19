"""
Validation, metrics, and the threshold-table confidence gate.
Spec sections 9.1 and 9.2.

This module owns the PRIMARY accept/review/reject decision. Addendum B.4's
non-negotiable: the threshold table ships first and stays in the codebase as the
fallback path; the learned logistic regression is a second code path selected by
a flag. A learned gate should be an upgrade to a working system, never a
dependency.

IoU and Hausdorff are complementary and both are required. IoU is an area
measure dominated by the bulk of the shape: a building whose main mass is
correctly placed but whose 8-metre wing points the wrong way still scores 0.82.
Hausdorff is a worst-case boundary measure and catches exactly that. A fit
passes only if both pass.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from contracts import Decision, FitResult, Footprint, MeshOutline
from geo.fit import _poly, apply_similarity, iou

# Spec 9.1. Every number here is a documented guess, which is precisely the
# criticism addendum B.1 makes of it — these cutoffs are reasonable values
# pulled from the air, and whether they are THE values for this pipeline on
# these buildings is an empirical question the spec 11 benchmark answers.
THRESHOLDS = {
    "iou":            {"accept": 0.75, "reject": 0.50},
    "hausdorff_m":    {"accept": 2.0,  "reject": 5.0},
    "area_ratio":     {"accept": (0.85, 1.15), "reject": (0.70, 1.30)},
    "rotation_margin": {"accept": 0.05},
    "rectilinearity": {"accept": 0.75},
    "anisotropy":     {"accept": 0.05, "reject": 0.15},
    "neighbor_overlap": {"accept": 0.02, "reject": 0.10},
}


def symmetric_hausdorff(a, b) -> float:
    """max( sup_{p in dA} d(p, dB), sup_{q in dB} d(q, dA) ), in metres.

    Uses `.boundary` rather than `.exterior` so MultiPolygon footprints work —
    a multi-part building has no single exterior ring.
    """
    if a.is_empty or b.is_empty:
        return float("inf")
    ba, bb = a.boundary, b.boundary
    return float(max(ba.hausdorff_distance(bb), bb.hausdorff_distance(ba)))


def compute_metrics(fp: Footprint, mo: MeshOutline, params: dict) -> dict:
    """The spec 9.1 metric set for one placement."""
    from geo.fit import _target_geom
    target = _target_geom(fp)
    moved = apply_similarity(mo.pts_enu, params["theta"], params["sx"],
                             params["sy"], params["t"])
    holes = tuple(apply_similarity(h, params["theta"], params["sx"],
                                   params["sy"], params["t"])
                  for h in mo.holes_enu)
    placed = _poly(moved, holes)

    return {
        "iou": iou(placed, target),
        "hausdorff_m": symmetric_hausdorff(placed, target),
        "area_ratio": float(placed.area / target.area) if target.area > 1e-9 else 0.0,
        "placed_polygon": placed,
    }


def neighbour_overlap(placed: Polygon, neighbours: list[Polygon]) -> float:
    """max_j |T(M_0) ^ F_j| / |T(M_0)|. Spec 9.2.

    A building encroaching into a neighbour — or into a road buffer — is an
    obvious visual failure in a game world, and it is invisible to IoU against
    the target footprint alone.
    """
    if not neighbours or placed.is_empty or placed.area < 1e-9:
        return 0.0
    return float(max(placed.intersection(n).area for n in neighbours) / placed.area)


def threshold_decision(fit: FitResult, fp: Footprint) -> tuple[float, Decision, list[str]]:
    """Spec 9.1's table. -> (pseudo-probability, decision, reasons).

    The returned scalar is a monotone score, NOT a calibrated probability — it
    exists so the pipeline has one numeric field whether the gate is this table
    or addendum B's fitted model. Do not read it as a likelihood.
    """
    reasons: list[str] = []
    reject = False
    review = False

    t = THRESHOLDS
    if fit.iou < t["iou"]["reject"]:
        reject = True
        reasons.append(f"iou {fit.iou:.3f} < {t['iou']['reject']}")
    elif fit.iou < t["iou"]["accept"]:
        review = True
        reasons.append(f"iou {fit.iou:.3f} below auto-accept {t['iou']['accept']}")

    if fit.hausdorff_m > t["hausdorff_m"]["reject"]:
        reject = True
        reasons.append(f"hausdorff {fit.hausdorff_m:.2f} m > {t['hausdorff_m']['reject']} m")
    elif fit.hausdorff_m > t["hausdorff_m"]["accept"]:
        review = True
        reasons.append(f"hausdorff {fit.hausdorff_m:.2f} m above auto-accept")

    lo_r, hi_r = t["area_ratio"]["reject"]
    lo_a, hi_a = t["area_ratio"]["accept"]
    if not (lo_r <= fit.area_ratio <= hi_r):
        reject = True
        reasons.append(f"area ratio {fit.area_ratio:.3f} outside [{lo_r}, {hi_r}]")
    elif not (lo_a <= fit.area_ratio <= hi_a):
        review = True
        reasons.append(f"area ratio {fit.area_ratio:.3f} outside auto-accept band")

    if fit.rotation_margin_footprint < t["rotation_margin"]["accept"]:
        review = True
        reasons.append(
            f"rotation margin {fit.rotation_margin_footprint:.3f} < "
            f"{t['rotation_margin']['accept']} — near-square footprint, "
            "orientation is not confidently determined"
        )

    # Spec 4.2 / 10.2 failure 8: a round or organic building has a meaningless
    # rotation. Flag regardless of IoU — this is correct behaviour, not a bug.
    if fp.rectilinearity < t["rectilinearity"]["accept"]:
        review = True
        reasons.append(
            f"rectilinearity {fp.rectilinearity:.3f} < {t['rectilinearity']['accept']}"
            + (" — rotation is intrinsically ill-posed" if fp.is_ill_posed else "")
        )

    if fit.anisotropy_log_ratio >= t["anisotropy"]["reject"]:
        reject = True
        reasons.append(f"anisotropy {fit.anisotropy_log_ratio:.3f} >= {t['anisotropy']['reject']}")
    elif fit.anisotropy_log_ratio > t["anisotropy"]["accept"]:
        review = True
        reasons.append(f"anisotropy {fit.anisotropy_log_ratio:.3f} above auto-accept")

    if fit.max_neighbor_overlap > t["neighbor_overlap"]["reject"]:
        reject = True
        reasons.append(f"neighbour overlap {fit.max_neighbor_overlap:.3f} > "
                       f"{t['neighbor_overlap']['reject']}")
    elif fit.max_neighbor_overlap > t["neighbor_overlap"]["accept"]:
        review = True
        reasons.append(f"neighbour overlap {fit.max_neighbor_overlap:.3f} above auto-accept")

    # Spec 3.2: a footprint accepted on thin evidence taints every metric above
    # it — a high IoU against the WRONG building is the most dangerous output
    # this pipeline can produce, because it looks like success.
    if fp.match_quality == "unnamed_sole_candidate":
        review = True
        reasons.append(
            "footprint matched only as the sole nearby candidate, with no name "
            "match — verify it is the right building before accepting"
        )
    elif not fp.match_quality:
        review = True
        reasons.append("footprint match quality unrecorded — treat as unverified")

    if fp.is_multipart:
        review = True
        reasons.append(
            f"multi-part footprint ({1 + len(fp.parts_enu)} disjoint outer rings) "
            "— the OMBB and rotation come from the largest part only"
        )

    # Spec 1.2: reflections are rejected outright, never accepted as a mirror.
    if fit.is_mirrored:
        reject = True
        reasons.append("mirrored mesh — det(R) < 0 (spec 6.8, Umeyama S correction)")

    if reject:
        decision = Decision.REJECT
    elif review:
        decision = Decision.REVIEW
    else:
        decision = Decision.AUTO_ACCEPT

    score = _pseudo_score(fit, fp)
    return score, decision, reasons


def _pseudo_score(fit: FitResult, fp: Footprint) -> float:
    """A monotone 0-1 summary for the record. Not calibrated. Not a probability.

    Deliberately crude: its only job is to give the placement record one
    comparable number under either gate. Addendum B replaces it with a fitted
    logistic regression whose coefficients are readable, which is MORE
    inspectable than seven independently chosen cutoffs, not less.
    """
    terms = [
        np.clip(fit.iou / 0.85, 0, 1),
        np.clip(1.0 - fit.hausdorff_m / 6.0, 0, 1),
        np.clip(1.0 - abs(math.log(max(fit.area_ratio, 1e-6))) / 0.35, 0, 1),
        np.clip(fit.rotation_margin_footprint / 0.20, 0, 1),
        np.clip(fp.rectilinearity, 0, 1),
    ]
    return float(np.mean(terms))
