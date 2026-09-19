"""
Address resolution, footprint retrieval, and footprint conditioning.
Spec sections 3 and 4.

Two things in here are load-bearing well beyond their line count:

  - `principal_orientation` (spec 4.2) returns the rectilinearity R alongside
    the principal axis, from the same single pass over the edges. R is the
    cheapest and best early-warning signal in the pipeline: R ~ 1 means a
    well-conditioned rotation estimate, R <~ 0.6 means the rotation is
    intrinsically ill-posed and the fit should be flagged regardless of IoU.

  - The disk cache. The public Overpass endpoint is a shared free service that
    hands out 429 and 504 under any load. Everything fetched here is cached to
    fixtures/footprints/ and committed to the repo, because the alternative is
    discovering the rate limit live at judging (spec 10.2 failure 12).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import requests
from shapely.geometry import Polygon, shape
from shapely.ops import orient

from contracts import OMBB, Footprint
from geo.coords import ENUFrame

CACHE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "footprints"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
DEFAULT_OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "vthacks14-procedura-placement/0.1 (academic hackathon project)"


# --------------------------------------------------------------------------
# 3.1 Geocoding
# --------------------------------------------------------------------------


def geocode(address: str, *, use_cache: bool = True) -> dict:
    """Address -> {lat, lon, location_type, provider, normalized}.

    The returned point is NOT the building. Geocoders hand back street-centreline
    interpolations, parcel centroids, or rooftop points depending on provider and
    address quality — so `location_type` is recorded and propagated into the
    confidence model rather than being quietly assumed accurate.
    """
    cache_path = CACHE_DIR / f"geocode_{_slug(address)}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    result = _geocode_google(address, key) if key else _geocode_nominatim(address)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _geocode_nominatim(address: str) -> dict:
    r = requests.get(
        NOMINATIM_URL,
        params={"q": address, "format": "jsonv2", "limit": 1, "polygon_geojson": 0},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    r.raise_for_status()
    hits = r.json()
    if not hits:
        # Spec 14: structured errors, never silent defaults. A pipeline that
        # quietly substitutes a default box produces a demo that appears to work
        # and is entirely fictitious.
        raise LookupError(f"Nominatim returned no result for {address!r}")
    h = hits[0]
    # Nominatim has no location_type. Approximate it from the OSM class: a way
    # or relation tagged building is effectively a rooftop match.
    is_building = h.get("category") == "building" or h.get("class") == "building"
    return {
        "provider": "nominatim",
        "lat": float(h["lat"]),
        "lon": float(h["lon"]),
        "location_type": "ROOFTOP" if is_building else "APPROXIMATE",
        "normalized": h.get("display_name", address),
    }


def _geocode_google(address: str, key: str) -> dict:
    r = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": address, "key": key},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "OK" or not payload.get("results"):
        raise LookupError(f"Google geocode failed for {address!r}: {payload.get('status')}")
    top = payload["results"][0]
    loc = top["geometry"]["location"]
    return {
        "provider": "google",
        "lat": float(loc["lat"]),
        "lon": float(loc["lng"]),
        "location_type": top["geometry"].get("location_type", "APPROXIMATE"),
        "normalized": top.get("formatted_address", address),
    }


# --------------------------------------------------------------------------
# 3.2 Footprint retrieval
# --------------------------------------------------------------------------


OVERPASS_QUERY = """[out:json][timeout:25];
(
  way["building"](around:{radius},{lat},{lon});
  relation["building"](around:{radius},{lat},{lon});
);
out geom;
way["highway"](around:{road_radius},{lat},{lon});
out geom;
"""


def fetch_osm(lat: float, lon: float, *, radius: int = 60, road_radius: int = 120,
              use_cache: bool = True, retries: int = 3) -> dict:
    """One Overpass call for buildings AND roads.

    Buildings at `radius` for the target plus its neighbours (spec 9.2 collision
    checking needs them anyway, so fetching them separately would be a second
    chance to get rate-limited). Roads at `road_radius` for spec 6.6 Filter 3's
    street-facing prior.
    """
    cache_path = CACHE_DIR / f"osm_{lat:.5f}_{lon:.5f}_r{radius}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = os.environ.get("OVERPASS_URL", DEFAULT_OVERPASS)
    query = OVERPASS_QUERY.format(
        radius=radius, road_radius=road_radius, lat=lat, lon=lon
    )

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(url, data={"data": query},
                              headers={"User-Agent": USER_AGENT}, timeout=40)
            if r.status_code in (429, 504):
                # The documented failure. Back off, then fall through to cache.
                time.sleep(2.0 * (attempt + 1))
                last_exc = RuntimeError(f"Overpass HTTP {r.status_code}")
                continue
            r.raise_for_status()
            payload = r.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        f"Overpass unavailable after {retries} attempts ({last_exc}). "
        f"No cache at {cache_path}. Run scripts/prefetch_footprints.py while "
        "the network is healthy and commit the result."
    ) from last_exc


def _element_to_polygon(el: dict) -> Polygon | None:
    """OSM way/relation -> shapely Polygon in (lon, lat), interior rings intact.

    Spec 3.3: courtyard buildings arrive as polygons with interior rings and OSM
    multipolygon relations encode them as outer/inner members. Every area, IoU
    and boundary distance downstream must respect them.
    """
    if el.get("type") == "way":
        geom = el.get("geometry") or []
        if len(geom) < 4:
            return None
        ring = [(p["lon"], p["lat"]) for p in geom]
        try:
            return Polygon(ring)
        except Exception:  # noqa: BLE001
            return None

    if el.get("type") == "relation":
        outers, inners = [], []
        for m in el.get("members", []):
            g = m.get("geometry") or []
            if len(g) < 4:
                continue
            ring = [(p["lon"], p["lat"]) for p in g]
            (outers if m.get("role") == "outer" else inners).append(ring)
        if not outers:
            return None
        try:
            return Polygon(outers[0], inners)
        except Exception:  # noqa: BLE001
            return None
    return None


def select_footprint(payload: dict, lat: float, lon: float,
                     address: str = "") -> tuple[dict, Polygon, list[Polygon]]:
    """Spec 3.2's selection rule. -> (element, polygon, neighbour_polygons).

    Explicitly NOT "take the nearest polygon":
      1. Point-in-polygon against the geocoded point; exactly one hit wins.
      2. Otherwise candidates within 50 m ranked by tag match, then centroid
         distance, then area plausibility.
      3. Zero candidates is an explicit failure. We do not synthesise a footprint.
    """
    from shapely.geometry import Point

    pt = Point(lon, lat)
    buildings: list[tuple[dict, Polygon]] = []
    for el in payload.get("elements", []):
        if "building" not in (el.get("tags") or {}):
            continue
        poly = _element_to_polygon(el)
        if poly is not None and poly.is_valid and poly.area > 0:
            buildings.append((el, poly))

    if not buildings:
        raise LookupError(
            f"No OSM building footprint near ({lat:.5f}, {lon:.5f}). "
            "Fall back to the Microsoft GlobalMLBuildingFootprints dataset, or "
            "fail — do not synthesise a footprint (spec 3.2, 14)."
        )

    containing = [(el, p) for el, p in buildings if p.contains(pt)]
    if len(containing) == 1:
        chosen = containing[0]
    else:
        pool = containing if containing else buildings
        housenumber = _extract_housenumber(address)

        def rank(item: tuple[dict, Polygon]) -> tuple:
            el, poly = item
            tags = el.get("tags") or {}
            tag_match = 0
            if housenumber and tags.get("addr:housenumber") == housenumber:
                tag_match = -2
            elif address and tags.get("name") and tags["name"].lower() in address.lower():
                tag_match = -1
            # metres, near enough at this latitude for ranking purposes
            dist = poly.centroid.distance(pt) * 111_000
            area_penalty = 0.0 if 20 < _approx_area_m2(poly) < 50_000 else 1.0
            return (tag_match, area_penalty, dist)

        chosen = min(pool, key=rank)

    neighbours = [p for el, p in buildings if el is not chosen[0]]
    return chosen[0], chosen[1], neighbours


def _extract_housenumber(address: str) -> str:
    token = address.strip().split(" ", 1)[0] if address.strip() else ""
    return token if token.isdigit() else ""


def _approx_area_m2(poly: Polygon) -> float:
    """Rough area for ranking only — degrees^2 scaled at the polygon's latitude."""
    lat = poly.centroid.y
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return poly.area * m_per_deg_lat * m_per_deg_lon


# --------------------------------------------------------------------------
# 4.1 Signed area and centroid
# --------------------------------------------------------------------------


def signed_area(pts: np.ndarray) -> float:
    """Shoelace. Sign gives winding order; positive is counter-clockwise."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def centroid(pts: np.ndarray) -> np.ndarray:
    """Area centroid via the shoelace weights (spec 4.1), not the vertex mean.

    The vertex mean is wrong whenever vertices are unevenly distributed along the
    boundary, which they always are after simplification.
    """
    a = signed_area(pts)
    if abs(a) < 1e-12:
        return pts.mean(axis=0)
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = float(np.sum((x + np.roll(x, -1)) * cross) / (6.0 * a))
    cy = float(np.sum((y + np.roll(y, -1)) * cross) / (6.0 * a))
    return np.array([cx, cy])


# --------------------------------------------------------------------------
# 4.2 Principal orientation via circular statistics
# --------------------------------------------------------------------------


def principal_orientation(pts: np.ndarray) -> tuple[float, float]:
    """-> (theta_star, rectilinearity R). Spec 4.2.

    The naive approach — take the longest edge — is fragile. Buildings are
    approximately Manhattan: their edges cluster around two perpendicular
    directions. Because directions are equivalent modulo pi/2, mapping each edge
    angle into a quadruple-angle representation collapses the four-fold symmetry
    to a single circular mean, weighted by edge length.

    R is the resultant length of that same sum — the von Mises concentration
    parameter of the edge-direction distribution in disguise. It falls out for
    free and it is the best cheap predictor of whether the rotation estimate is
    even well posed.
    """
    d = np.roll(pts, -1, axis=0) - pts
    lengths = np.hypot(d[:, 0], d[:, 1])
    total = float(lengths.sum())
    if total < 1e-9:
        return 0.0, 0.0

    angles = np.arctan2(d[:, 1], d[:, 0])
    c = float(np.sum(lengths * np.cos(4.0 * angles)))
    s = float(np.sum(lengths * np.sin(4.0 * angles)))

    theta_star = 0.25 * math.atan2(s, c)
    r = math.hypot(c, s) / total
    return theta_star, r


# --------------------------------------------------------------------------
# 4.3 Simplification / 4.4 Oriented minimum bounding box
# --------------------------------------------------------------------------


def simplify_ring(pts: np.ndarray, eps: float = 0.4) -> np.ndarray:
    """Douglas-Peucker at ~0.3-0.5 m: removes vertex noise, keeps real corners.

    Applied to the COPY used for fitting. The authoritative footprint stays
    untouched in the provenance record (spec 4.3).
    """
    poly = Polygon(pts).simplify(eps, preserve_topology=True)
    if poly.is_empty or not poly.exterior:
        return pts
    out = np.asarray(poly.exterior.coords)[:-1]
    return out if len(out) >= 3 else pts


def compute_ombb(pts: np.ndarray) -> OMBB:
    """Oriented minimum-AREA bounding box. Spec 4.4.

    Rotating calipers, via shapely's implementation of the Freeman-Shapira /
    Toussaint result: the minimum-area rectangle enclosing a convex polygon has
    a side collinear with one of the polygon's edges, which reduces a continuous
    search over orientations to n candidates.

    Minimum-area and minimum-perimeter boxes can differ in orientation by nearly
    45 degrees on some shapes. This project uses minimum-AREA everywhere.
    """
    rect = Polygon(pts).minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords)[:-1]  # 4 corners
    if len(coords) != 4:
        # Degenerate (collinear) input — fall back to an axis-aligned box.
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ext = np.maximum(hi - lo, 1e-6)
        a, b = float(max(ext)), float(min(ext))
        u = np.array([1.0, 0.0]) if ext[0] >= ext[1] else np.array([0.0, 1.0])
        v = np.array([-u[1], u[0]])
        return OMBB(centre=(lo + hi) / 2.0, u=u, v=v, a=a, b=b)

    e0 = coords[1] - coords[0]
    e1 = coords[2] - coords[1]
    l0, l1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))

    if l0 >= l1:
        long_vec, a, b = e0, l0, l1
    else:
        long_vec, a, b = e1, l1, l0

    u = long_vec / max(np.linalg.norm(long_vec), 1e-12)
    v = np.array([-u[1], u[0]])
    return OMBB(centre=coords.mean(axis=0), u=u, v=v, a=a, b=max(b, 1e-9))


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_footprint(polygon_lonlat: Polygon, frame: ENUFrame, *,
                    simplify_eps: float = 0.4,
                    source: str = "osm") -> Footprint:
    """Project an OSM polygon into the ENU frame and condition it. Spec 3-4."""
    poly = orient(polygon_lonlat, sign=1.0)  # CCW outer ring

    def ring_to_enu(ring) -> np.ndarray:
        arr = np.asarray(ring.coords)[:-1]
        enu = frame.geodetic_to_enu(arr[:, 1], arr[:, 0], 0.0)
        return np.asarray(enu)[:, :2]

    outer = simplify_ring(ring_to_enu(poly.exterior), simplify_eps)
    holes = tuple(ring_to_enu(r) for r in poly.interiors)

    theta_star, rect = principal_orientation(outer)
    shapely_poly = Polygon(outer, [h for h in holes])

    return Footprint(
        pts_enu=outer,
        holes_enu=holes,
        rectilinearity=rect,
        principal_angle=theta_star,
        ombb=compute_ombb(outer),
        area_m2=float(shapely_poly.area),
        source=source,
    )


def to_shapely(fp: Footprint) -> Polygon:
    """Footprint -> shapely Polygon, interior rings intact."""
    return Polygon(fp.pts_enu, [h for h in fp.holes_enu])


def _slug(s: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in s.lower())[:48]
    return f"{clean}_{hashlib.sha256(s.encode()).hexdigest()[:8]}"
