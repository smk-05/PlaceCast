"""
Height estimation. Spec section 7.

Image-to-3D produces no absolute scale. The footprint supplies two constraints —
two in-plane extents — leaving the vertical unconstrained: a two-storey house and
a twenty-storey tower with the same floor plan are indistinguishable to the
footprint fit.

Priority order, and ALWAYS record which tier fired:
  7.1  authoritative tags   OSM height -> OSM building:levels -> Microsoft
  7.2  single-view metrology                     (not built; see the cut list)
  7.3  monocular metric depth                    (perception, conditional)
  7.4  proportional fallback + sanity band
"""

from __future__ import annotations

import math

from contracts import HeightSource

# Spec 7.1: the OSM convention is 3 m per level for 3D rendering when explicit
# height tags are absent. Some renderers assume 4 m. We pick 3 m and record it.
LEVEL_HEIGHT_M = 3.0

# Spec 7.4 sanity band.
MIN_HEIGHT_M = 2.5
MAX_HEIGHT_M = 300.0
SLENDERNESS_RANGE = (0.1, 8.0)


def from_osm_tags(tags: dict) -> tuple[float, HeightSource] | None:
    """OSM `height` is definitive when present; `building:levels` is next."""
    raw = (tags or {}).get("height")
    if raw is not None:
        try:
            # Values arrive as "20.7", "20.7 m", occasionally with junk.
            v = float(str(raw).strip().split()[0].replace("m", ""))
            if MIN_HEIGHT_M <= v <= MAX_HEIGHT_M:
                return v, HeightSource.OSM_HEIGHT
        except (ValueError, IndexError):
            pass

    raw = (tags or {}).get("building:levels")
    if raw is not None:
        try:
            levels = float(str(raw).strip().split(";")[0])
            v = levels * LEVEL_HEIGHT_M
            if MIN_HEIGHT_M <= v <= MAX_HEIGHT_M:
                return v, HeightSource.OSM_LEVELS
        except ValueError:
            pass
    return None


def from_microsoft(height_m: float | None) -> tuple[float, HeightSource] | None:
    """Microsoft GlobalMLBuildingFootprints carries height for 174M+ buildings.

    The property is -1 when unknown. Spec 7.1: prefer OSM when they disagree and
    OSM has an explicit `height`.
    """
    if height_m is None or height_m < 0:
        return None
    if MIN_HEIGHT_M <= height_m <= MAX_HEIGHT_M:
        return float(height_m), HeightSource.MICROSOFT
    return None


def proportional_fallback(mesh_height_units: float, footprint_scale: float,
                          footprint_area_m2: float) -> tuple[float, HeightSource]:
    """Preserve the mesh's intrinsic proportions under the uniform plan scale.

    Spec 7.4. The last resort, and the one most likely to be wrong on a
    generated mesh whose vertical proportions were hallucinated.
    """
    h = mesh_height_units * footprint_scale
    h = max(MIN_HEIGHT_M, min(h, MAX_HEIGHT_M))
    # Keep slenderness plausible even when the mesh proportions are absurd.
    lo, hi = SLENDERNESS_RANGE
    root_area = math.sqrt(max(footprint_area_m2, 1.0))
    h = max(lo * root_area, min(h, hi * root_area))
    return float(h), HeightSource.PROPORTIONAL_FALLBACK


def sanity_check(h: float, footprint_area_m2: float) -> tuple[bool, str]:
    """Spec 7.4: 2.5 m <= h <= 300 m, slenderness h/sqrt(area) in [0.1, 8]."""
    if not (MIN_HEIGHT_M <= h <= MAX_HEIGHT_M):
        return False, f"height {h:.1f} m outside [{MIN_HEIGHT_M}, {MAX_HEIGHT_M}]"
    slender = h / math.sqrt(max(footprint_area_m2, 1.0))
    lo, hi = SLENDERNESS_RANGE
    if not (lo <= slender <= hi):
        return False, f"slenderness {slender:.2f} outside [{lo}, {hi}]"
    return True, ""


def resolve_height(*,
                   osm_tags: dict | None = None,
                   microsoft_height: float | None = None,
                   depth_estimate: float | None = None,
                   mesh_height_units: float = 1.0,
                   footprint_scale: float = 1.0,
                   footprint_area_m2: float = 100.0) -> tuple[float, HeightSource, list[str]]:
    """Walk spec 7's priority chain. -> (height_m, source, notes).

    `depth_estimate` is addendum C's monocular metric depth. It never overrides
    an authoritative OSM tag — addendum C.3: "OSM wins." When both exist and
    disagree by more than 30%, the disagreement is logged, because a table of
    those is a good honest-results slide.

    Order: OSM tags -> monocular depth (if sane) -> Microsoft -> proportional.
    Microsoft sits BELOW depth, deviating from spec 7.1, because on this campus
    it was measured underestimating every building with an explicit OSM height
    by 44-67% (see HeightSource.is_authoritative). A known-biased value should
    not outrank an unbiased-but-noisy one.
    """
    notes: list[str] = []

    result = from_osm_tags(osm_tags or {})
    if result is not None:
        h, src = result
        for label, other in (("monocular depth", depth_estimate),
                             ("Microsoft", microsoft_height)):
            if other is not None and other > 0 and h > 0:
                rel = abs(other - h) / h
                if rel > 0.30:
                    notes.append(
                        f"{label} {other:.1f} m disagrees with {src.value} "
                        f"{h:.1f} m by {rel:.0%} — authoritative tag wins"
                    )
        return h, src, notes

    ms = from_microsoft(microsoft_height)

    if depth_estimate is not None:
        ok, why = sanity_check(depth_estimate, footprint_area_m2)
        if ok:
            if ms is not None:
                notes.append(f"Microsoft height {ms[0]:.1f} m available but "
                             "ranked below depth (known ~50% low on this campus)")
            return float(depth_estimate), HeightSource.MONOCULAR_DEPTH, notes
        notes.append(f"monocular depth rejected: {why}")

    if ms is not None:
        notes.append("Microsoft height used — NOT authoritative; measured ~50% "
                     "low against OSM on this campus, expect an underestimate")
        return ms[0], ms[1], notes

    h, src = proportional_fallback(mesh_height_units, footprint_scale,
                                   footprint_area_m2)
    notes.append("no authoritative height available — proportional fallback used")
    return h, src, notes
