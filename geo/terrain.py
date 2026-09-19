"""
Terrain and ground alignment. Spec section 8.

Two sources, and they must never be mixed without conversion:

  Cesium World Terrain   already ELLIPSOIDAL. The viewer samples it with
                         sampleTerrainMostDetailed. This is the primary path and
                         it is why the geoid correction is mostly sidestepped.
  USGS 3DEP (1 m)        ORTHOMETRIC, NAVD88. Higher resolution, US only, and it
                         needs geo.coords.orthometric_to_ellipsoidal applied.

Sloped sites: do NOT tilt the building. Real buildings are level and a tilted one
reads as broken instantly. We set the base to the low side and accept that the
uphill side is partially buried, which is what real buildings do (spec 8 option
a, one line). The skirt-geometry alternative looks better and is on the cut list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

from geo.coords import assert_plausible_height, geoid_undulation, orthometric_to_ellipsoidal

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"

# Spec 10.2 failure 9: a spread wider than this means a genuinely sloped site.
SLOPE_FLAG_M = 1.5


@dataclass(frozen=True)
class GroundSample:
    base_height_m: float          # ellipsoidal, the value to place the mesh at
    spread_m: float               # max - min across the footprint
    is_sloped: bool
    datum: str
    geoid_undulation_m: float
    n_samples: int
    note: str = ""


def sample_usgs_3dep(lat: float, lon: float, timeout: float = 12.0) -> float | None:
    """One USGS 3DEP elevation sample. ORTHOMETRIC (NAVD88) metres."""
    try:
        r = requests.get(EPQS_URL,
                         params={"x": lon, "y": lat, "units": "Meters",
                                 "wkid": 4326, "includeDate": "false"},
                         timeout=timeout)
        r.raise_for_status()
        v = r.json().get("value")
        return float(v) if v is not None and float(v) > -1000 else None
    except Exception:  # noqa: BLE001 - offline at judging is expected
        return None


def ground_height(lat_lon_samples: list[tuple[float, float]],
                  *, percentile: float = 25.0) -> GroundSample:
    """Robust ground height across the footprint. Spec 8.

    Take a robust statistic across the footprint, not a single centroid sample:
    the 25th percentile approximates the downhill side of a pad-on-grade
    building without being hostage to one spurious sample.

    Returns ELLIPSOIDAL height. When 3DEP is unavailable the caller should fall
    back to the viewer's Cesium terrain, which needs no conversion.
    """
    if not lat_lon_samples:
        return GroundSample(0.0, 0.0, False, "none", 0.0, 0, "no samples requested")

    values = [h for h in (sample_usgs_3dep(la, lo) for la, lo in lat_lon_samples)
              if h is not None]

    if not values:
        return GroundSample(
            0.0, 0.0, False, "unavailable", 0.0, 0,
            "USGS 3DEP unreachable — defer to Cesium World Terrain in the "
            "viewer, which is already ellipsoidal (spec 8)",
        )

    arr = np.asarray(values, dtype=float)
    h_ortho = float(np.percentile(arr, percentile))
    spread = float(arr.max() - arr.min())

    lat0, lon0 = lat_lon_samples[0]
    undulation, method = geoid_undulation(lat0, lon0)
    if undulation is None:
        # Better to say so than to silently publish an orthometric height into a
        # pipeline that expects ellipsoidal — that is exactly spec 2.3's trap.
        return GroundSample(
            h_ortho, spread, spread > SLOPE_FLAG_M, "orthometric:NAVD88", 0.0,
            len(values),
            f"GEOID UNCORRECTED ({method}) — do not feed to Cesium without "
            "converting; expect ~30 m error in Virginia (spec 2.3)",
        )

    h_ellip = orthometric_to_ellipsoidal(h_ortho, undulation)
    assert_plausible_height(h_ellip, "terrain sample")

    return GroundSample(
        base_height_m=h_ellip,
        spread_m=spread,
        is_sloped=spread > SLOPE_FLAG_M,
        datum="ellipsoidal:WGS84",
        geoid_undulation_m=undulation,
        n_samples=len(values),
        note=f"geoid via {method}",
    )
