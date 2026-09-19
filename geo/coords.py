"""
Coordinate systems. Spec section 2.

Getting this layer wrong invalidates everything downstream, and the errors are
silent: a building 40 cm off or 33 m underground looks like a bug in the fitting
code, not here. Hence the unit tests in tests/test_coords.py cross-check the
hand-rolled transforms against pyproj to 1e-9.

Why not UTM (spec 2.1): the transverse Mercator scale factor runs 0.9996 at the
central meridian to above 1.0 at the zone edges — about 1 part in 2500, so ~4 cm
on a 100 m building, which is tolerable — but zone boundaries create
discontinuities. For a single building a local tangent plane is exact enough and
has no seams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# WGS84 (spec 2.2)
WGS84_A = 6378137.0                    # semi-major axis, metres
WGS84_F = 1.0 / 298.257223563          # flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F**2    # first eccentricity squared

# Spec 10.2 failure 13: a geoid mix-up puts the building ~30 m underground in
# Virginia. Anything beyond this is a datum error, not a tall building.
MAX_ABS_Z_M = 2500.0


def prime_vertical_radius(lat_rad: float) -> float:
    """N(phi) = a / sqrt(1 - e^2 sin^2 phi)."""
    s = math.sin(lat_rad)
    return WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    """Geodetic (phi, lambda, h_ellipsoidal) -> ECEF XYZ. Spec 2.2.

    `h_m` must be ELLIPSOIDAL height. If you have an orthometric height from a
    DEM, run it through `orthometric_to_ellipsoidal` first.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    n = prime_vertical_radius(lat)
    cos_lat, sin_lat = math.cos(lat), math.sin(lat)
    return np.array(
        [
            (n + h_m) * cos_lat * math.cos(lon),
            (n + h_m) * cos_lat * math.sin(lon),
            (n * (1.0 - WGS84_E2) + h_m) * sin_lat,
        ]
    )


def enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """ECEF -> ENU rotation matrix at the tangent point. Spec 2.2."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lon), math.cos(lon)
    sp, cp = math.sin(lat), math.cos(lat)
    return np.array(
        [
            [-sl, cl, 0.0],
            [-sp * cl, -sp * sl, cp],
            [cp * cl, cp * sl, sp],
        ]
    )


@dataclass(frozen=True)
class ENUFrame:
    """A local East-North-Up tangent plane anchored at the footprint centroid.

    All footprint geometry, all mesh fitting, and all metric reasoning happen in
    this frame. Convert back only at render time.
    """

    lat0: float
    lon0: float
    h0: float = 0.0  # ellipsoidal

    @property
    def origin_ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.lat0, self.lon0, self.h0)

    @property
    def rotation(self) -> np.ndarray:
        return enu_rotation(self.lat0, self.lon0)

    def geodetic_to_enu(self, lat_deg, lon_deg, h_m=0.0) -> np.ndarray:
        """(lat, lon, h) -> (e, n, u) metres.

        Scalar in, shape-(3,) out; array in, shape-(N,3) out. The scalar check
        happens BEFORE atleast_1d, because afterwards everything looks 1-D.
        """
        scalar = np.ndim(lat_deg) == 0
        lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
        lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
        h = np.broadcast_to(np.asarray(h_m, dtype=float), lat.shape)

        ecef = np.stack(
            [geodetic_to_ecef(la, lo, hh) for la, lo, hh in zip(lat, lon, h)]
        )
        enu = (self.rotation @ (ecef - self.origin_ecef).T).T
        return enu[0] if scalar else enu

    def enu_to_geodetic(self, e, n, u=0.0) -> np.ndarray:
        """(e, n, u) -> (lat, lon, h_ellipsoidal). Inverse via Bowring."""
        scalar = np.ndim(e) == 0
        e = np.atleast_1d(np.asarray(e, dtype=float))
        n = np.atleast_1d(np.asarray(n, dtype=float))
        u = np.broadcast_to(np.asarray(u, dtype=float), e.shape)

        enu = np.stack([e, n, u], axis=-1)
        ecef = (self.rotation.T @ enu.T).T + self.origin_ecef
        out = np.stack([_ecef_to_geodetic(p) for p in ecef])
        return out[0] if scalar else out


def _ecef_to_geodetic(p: np.ndarray) -> np.ndarray:
    """ECEF -> (lat_deg, lon_deg, h_ellipsoidal). Bowring's closed-form method.

    Accurate to well under a millimetre for terrestrial heights, which is three
    orders of magnitude tighter than anything else in this pipeline.
    """
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    lon = math.atan2(y, x)

    b = WGS84_A * (1.0 - WGS84_F)
    ep2 = (WGS84_A**2 - b**2) / b**2
    r = math.hypot(x, y)
    theta = math.atan2(z * WGS84_A, r * b)

    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3,
        r - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )
    n = prime_vertical_radius(lat)
    h = r / math.cos(lat) - n if abs(math.cos(lat)) > 1e-12 else z / math.sin(lat) - n * (1 - WGS84_E2)
    return np.array([math.degrees(lat), math.degrees(lon), h])


# --------------------------------------------------------------------------
# The height datum trap. Spec 2.3.
# --------------------------------------------------------------------------


def orthometric_to_ellipsoidal(h_ortho: float, undulation: float) -> float:
    """h_ellipsoidal = H_orthometric + N_geoid.

    DEMs publish orthometric heights (NAVD88 in the US, EGM96/EGM2008 globally).
    CesiumJS, glTF and WGS84 all want ellipsoidal. In Virginia N_geoid is roughly
    -30 m; ignore it and every building in the demo is buried three storeys down.
    """
    return h_ortho + undulation


def geoid_undulation(lat_deg: float, lon_deg: float) -> tuple[float | None, str]:
    """N_geoid at a point, via pyproj's grids.  -> (undulation_m, method).

    Returns (None, reason) when the grids are unavailable — pyproj needs either a
    local grid file or PROJ_NETWORK=ON plus internet, and neither is guaranteed
    at judging.

    This is the FALLBACK path only. The demo routes terrain through Cesium World
    Terrain, which already serves ellipsoidal heights, so no conversion is needed
    there. Keep the two sources strictly separate and never mix them without
    converting — record which produced each height.
    """
    try:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:4979", "EPSG:5703", always_xy=True)
        _, _, h = tf.transform(lon_deg, lat_deg, 0.0)
        if h is None or not math.isfinite(h) or abs(h) > 200.0:
            return None, "pyproj_grid_unavailable"
        # transforming ellipsoidal 0 to orthometric gives -N
        return -float(h), "pyproj:EPSG:4979->5703"
    except Exception as exc:  # noqa: BLE001 - grids missing, offline, etc.
        return None, f"pyproj_failed:{type(exc).__name__}"


def assert_plausible_height(z_m: float, context: str = "") -> None:
    """Spec 10.2 failure 13. Fail loudly rather than render a buried building."""
    if not math.isfinite(z_m) or abs(z_m) > MAX_ABS_Z_M:
        raise ValueError(
            f"Implausible height {z_m:.1f} m{' for ' + context if context else ''}. "
            "Almost certainly a geoid/datum mix-up — see spec 2.3."
        )


# --------------------------------------------------------------------------
# Angle conventions. Spec 2.4.
# --------------------------------------------------------------------------
#
# Three conventions collide in this codebase. They are converted here and
# nowhere else.
#
#   mathematical theta   zero at +East,  CCW positive   <- everything in geo/
#   compass / Cesium     zero at North,  CW positive    <- the viewer boundary
#   glTF model space     +Z forward, +Y up, right-handed
#
# heading = pi/2 - theta, wrapped to [0, 2pi)


def theta_to_heading(theta_rad: float) -> float:
    """Mathematical theta (CCW from East) -> compass heading (CW from North)."""
    return float(np.mod(np.pi / 2.0 - theta_rad, 2.0 * np.pi))


def heading_to_theta(heading_rad: float) -> float:
    """Compass heading (CW from North) -> mathematical theta (CCW from East).

    Involutive with theta_to_heading, which tests/test_coords.py asserts.
    """
    return float(np.mod(np.pi / 2.0 - heading_rad, 2.0 * np.pi))


def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def wrap_half_pi(a: float) -> float:
    """Wrap to (-pi/4, pi/4] — the quotient that matters for spec 4.2.

    Building orientation is only defined modulo pi/2, so rotation error is
    reported both this way (did the axis alignment work) and absolutely (did
    disambiguation work). Spec 11 calls the gap between those two numbers the
    most informative single figure in the evaluation.
    """
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
