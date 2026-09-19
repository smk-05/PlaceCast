"""
Coordinate-frame tests. Spec section 2.

The errors this layer produces are silent — a building 40 cm off or 33 m
underground looks like a bug in the fitting code. So the hand-rolled transforms
are cross-checked against pyproj, which has no shared code with them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from geo import coords

# Burruss Hall, the primary demo address.
LAT, LON = 37.22906, -80.42372


def test_ecef_matches_pyproj():
    """Hand-rolled geodetic->ECEF vs pyproj, to sub-micrometre."""
    pyproj = pytest.importorskip("pyproj")
    tf = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)

    for lat, lon, h in [(LAT, LON, 0.0), (LAT, LON, 634.2),
                        (0.0, 0.0, 0.0), (-33.9, 151.2, 58.0),
                        (71.0, -8.0, 120.0)]:
        mine = coords.geodetic_to_ecef(lat, lon, h)
        theirs = np.array(tf.transform(lon, lat, h))
        assert np.allclose(mine, theirs, atol=1e-6), f"{lat},{lon},{h}"


def test_enu_roundtrip():
    """geodetic -> ENU -> geodetic recovers the input."""
    frame = coords.ENUFrame(lat0=LAT, lon0=LON, h0=634.2)

    for dlat, dlon, dh in [(0.0, 0.0, 0.0), (1e-4, -2e-4, 5.0), (-5e-4, 3e-4, -12.0)]:
        lat, lon, h = LAT + dlat, LON + dlon, 634.2 + dh
        enu = frame.geodetic_to_enu(lat, lon, h)
        back = frame.enu_to_geodetic(enu[0], enu[1], enu[2])
        assert abs(back[0] - lat) < 1e-9
        assert abs(back[1] - lon) < 1e-9
        assert abs(back[2] - h) < 1e-4


def test_enu_origin_is_zero():
    frame = coords.ENUFrame(lat0=LAT, lon0=LON, h0=100.0)
    assert np.allclose(frame.geodetic_to_enu(LAT, LON, 100.0), 0.0, atol=1e-6)


def test_enu_axes_point_the_right_way():
    """+East increases with longitude, +North with latitude."""
    frame = coords.ENUFrame(lat0=LAT, lon0=LON)

    east = frame.geodetic_to_enu(LAT, LON + 1e-4, 0.0)
    assert east[0] > 0 and abs(east[1]) < 1e-3

    north = frame.geodetic_to_enu(LAT + 1e-4, LON, 0.0)
    assert north[1] > 0 and abs(north[0]) < 1e-3


def test_enu_scale_is_metric():
    """1e-4 degrees of latitude is ~11.1 m anywhere on the ellipsoid."""
    frame = coords.ENUFrame(lat0=LAT, lon0=LON)
    d = frame.geodetic_to_enu(LAT + 1e-4, LON, 0.0)
    assert 11.0 < d[1] < 11.2


# --------------------------------------------------------------------------
# Spec 2.3 — the height datum trap
# --------------------------------------------------------------------------


def test_orthometric_to_ellipsoidal():
    """Virginia's undulation is about -30 m. Ignoring it buries the building."""
    assert coords.orthometric_to_ellipsoidal(634.2, -33.1) == pytest.approx(601.1)


def test_plausible_height_assertion():
    """Spec 10.2 failure 13. Fail loudly rather than render underground."""
    coords.assert_plausible_height(634.2)
    coords.assert_plausible_height(-30.0)

    with pytest.raises(ValueError, match="geoid"):
        coords.assert_plausible_height(6378137.0, "ECEF leaked into ENU")
    with pytest.raises(ValueError):
        coords.assert_plausible_height(float("nan"))


# --------------------------------------------------------------------------
# Spec 2.4 — angle conventions
# --------------------------------------------------------------------------


def test_theta_heading_known_values():
    """theta=0 is East, which is heading 90 degrees."""
    assert coords.theta_to_heading(0.0) == pytest.approx(math.pi / 2)
    assert coords.theta_to_heading(math.pi / 2) == pytest.approx(0.0)      # North
    assert coords.theta_to_heading(math.pi) == pytest.approx(3 * math.pi / 2)


def test_theta_heading_is_involutive():
    for t in np.linspace(0, 2 * math.pi, 37):
        back = coords.heading_to_theta(coords.theta_to_heading(t))
        assert math.isclose(back, t % (2 * math.pi), abs_tol=1e-12)


def test_wrap_half_pi_collapses_the_fourfold_symmetry():
    """Spec 4.2: building orientation is only defined modulo pi/2."""
    for k in range(4):
        assert coords.wrap_half_pi(0.1 + k * math.pi / 2) == pytest.approx(0.1)
    assert coords.wrap_half_pi(math.pi / 2 - 0.05) == pytest.approx(-0.05)
