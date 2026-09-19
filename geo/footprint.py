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
from shapely.geometry import MultiPolygon, Polygon
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


NOMINATIM_MIN_INTERVAL_S = 1.0   # Nominatim usage policy: max 1 request/second
_last_nominatim_call = 0.0


def _throttle_nominatim() -> None:
    """Block until 1 s has passed since the previous Nominatim request.

    The public server's usage policy caps clients at one request per second and
    bans those that exceed it. Only live requests pass through here — cache hits
    in geocode() return before this is reached.
    """
    global _last_nominatim_call
    wait = NOMINATIM_MIN_INTERVAL_S - (time.monotonic() - _last_nominatim_call)
    if wait > 0:
        time.sleep(wait)
    _last_nominatim_call = time.monotonic()


def _geocode_nominatim(address: str) -> dict:
    _throttle_nominatim()
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


def _stitch_rings(members: list[dict], role: str) -> list[list[tuple]]:
    """Assemble OSM multipolygon member ways into closed rings.

    THIS IS THE STEP THAT IS EASY TO GET WRONG, AND GETTING IT WRONG IS SILENT.

    A single outer ring is routinely SPLIT across several member ways that must
    be stitched end-to-end. Burruss Hall (relation/1074686) is exactly this: four
    `outer` members, none of them closed, chaining A->B->C->D->A into one ring.

    So neither shortcut works:
      - taking members[0] yields a truncated building (what this code did before,
        and it silently shrank Burruss from 6350 to 4384 m2);
      - treating each member as its own ring yields four degenerate slivers.

    Lane Stadium (relation/2417911) shows both shapes at once: members 0 and 1
    are open segments that stitch into one ring, while 2, 3 and 4 are already
    closed rings. It is four disjoint parts, not five.

    Returns a list of closed coordinate rings.
    """
    segments: list[list[tuple]] = []
    for m in members:
        if m.get("role") != role:
            continue
        g = m.get("geometry") or []
        if len(g) < 2:
            continue
        segments.append([(p["lon"], p["lat"]) for p in g])

    rings: list[list[tuple]] = []
    pending = [s for s in segments]

    while pending:
        chain = pending.pop(0)

        # Already a closed way — nothing to stitch.
        if chain[0] == chain[-1]:
            if len(chain) >= 4:
                rings.append(chain)
            continue

        # Walk the open segments, joining whichever one continues the chain at
        # either end, flipping it if necessary, until the chain closes.
        progress = True
        while progress and chain[0] != chain[-1]:
            progress = False
            for i, seg in enumerate(pending):
                if seg[0] == chain[-1]:
                    chain = chain + seg[1:]
                elif seg[-1] == chain[-1]:
                    chain = chain + seg[-2::-1]
                elif seg[-1] == chain[0]:
                    chain = seg[:-1] + chain
                elif seg[0] == chain[0]:
                    chain = seg[::-1][:-1] + chain
                else:
                    continue
                pending.pop(i)
                progress = True
                break

        if chain[0] == chain[-1] and len(chain) >= 4:
            rings.append(chain)
        elif len(chain) >= 4:
            # Unclosed after exhausting candidates: a genuinely broken relation.
            # Close it explicitly rather than dropping the geometry, but this is
            # worth surfacing — see `ring_warnings` on the returned Footprint.
            rings.append(chain + [chain[0]])

    return rings


def _element_to_polygon(el: dict) -> Polygon | MultiPolygon | None:
    """OSM way/relation -> shapely geometry in (lon, lat), rings intact.

    Spec 3.3 covers polygons with interior rings (courtyards). Relations with
    multiple DISJOINT OUTER rings are a separate case the spec does not name —
    Lane Stadium's four stands, with the field as a genuine gap between them —
    and they need a MultiPolygon, not a Polygon. Every area computation, IoU and
    boundary distance downstream has to be MultiPolygon-aware, not merely
    hole-aware.
    """
    if el.get("type") == "way":
        geom = el.get("geometry") or []
        if len(geom) < 4:
            return None
        ring = [(p["lon"], p["lat"]) for p in geom]
        try:
            p = Polygon(ring)
            return p if p.is_valid else p.buffer(0)
        except Exception:  # noqa: BLE001
            return None

    if el.get("type") != "relation":
        return None

    members = el.get("members", [])
    outer_rings = _stitch_rings(members, "outer")
    inner_rings = _stitch_rings(members, "inner")
    if not outer_rings:
        return None

    try:
        outers = [Polygon(r) for r in outer_rings if len(r) >= 4]
        outers = [p for p in outers if p.is_valid and p.area > 0]
        if not outers:
            return None

        inners = [Polygon(r) for r in inner_rings if len(r) >= 4]
        inners = [p for p in inners if p.is_valid and p.area > 0]

        # Assign each interior ring to the outer ring that contains it, rather
        # than attaching every hole to every part.
        parts = []
        for o in outers:
            holes = [list(h.exterior.coords) for h in inners
                     if o.contains(h.representative_point())]
            parts.append(Polygon(list(o.exterior.coords), holes))

        geom = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        return geom if geom.is_valid else geom.buffer(0)
    except Exception:  # noqa: BLE001
        return None


def select_footprint(payload: dict, lat: float, lon: float,
                     address: str = "", *, max_distance_m: float = 50.0
                     ) -> tuple[dict, Polygon | MultiPolygon, list]:
    """Spec 3.2's selection rule. -> (element, geometry, neighbours).

    Explicitly NOT "take the nearest polygon", and — the part that is easy to
    omit — explicitly NOT "take the best-ranked polygon either".

    Spec 3.2 step 2 ranks the candidates but never says what to do when NOTHING
    matches. Ranking always returns something, so an unconstrained step 2 turns
    a query that should error into a confidently wrong polygon. Observed:
    Randolph Hall (demolished, now a construction site) selected the
    "Virginia Tech Stability Wind Tunnel" — 360 m2, 46 m away, no name match.
    A wind tunnel standing in for a classroom building produces a plausible IoU
    against the wrong footprint, which is precisely spec 14's warning that a
    pipeline quietly substituting a default "will produce a demo that appears
    to work and is entirely fictitious."

    So the rule here is:
      1. Containment + name match         -> take it.
      2. Containment, one candidate only  -> take it.
      3. Name match within range          -> take it.
      4. Exactly one building in range    -> take it, tagged as weak evidence.
      5. Anything else                    -> FAIL. Do not guess.

    The chosen element carries `_match_quality`, which propagates into the
    placement record and is a feature for the confidence model.
    """
    from shapely.geometry import Point

    pt = Point(lon, lat)
    buildings: list[tuple[dict, Polygon | MultiPolygon]] = []
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

    housenumber = _extract_housenumber(address)

    def names_match(el: dict) -> bool:
        tags = el.get("tags") or {}
        if housenumber and tags.get("addr:housenumber") == housenumber:
            return True
        name = (tags.get("name") or "").strip().lower()
        if not name or not address:
            return False
        addr = address.lower()
        if name in addr:
            return True
        # "Classroom Building" vs "New Classroom Building": accept when the
        # significant words of the OSM name are all present in the query.
        words = [w for w in name.replace(",", " ").split()
                 if len(w) > 3 and w not in ("hall", "building", "center", "centre")]
        return bool(words) and all(w in addr for w in words)

    # MultiPolygon-aware containment: inside any outer ring and not inside a
    # hole. shapely handles this correctly for MultiPolygon.contains().
    containing = [(el, p) for el, p in buildings if p.contains(pt)]
    named = [(el, p) for el, p in buildings if names_match(el)]
    in_range = [(el, p) for el, p in buildings
                if _distance_m(p, pt) <= max_distance_m]

    chosen = None
    quality = ""

    named_and_containing = [c for c in containing if names_match(c[0])]
    if named_and_containing:
        chosen, quality = named_and_containing[0], "contained_and_named"
    elif len(containing) == 1:
        chosen, quality = containing[0], "contained"
    elif named:
        named_in_range = [c for c in named if _distance_m(c[1], pt) <= max_distance_m]
        if named_in_range:
            chosen = min(named_in_range, key=lambda c: _distance_m(c[1], pt))
            quality = "name_match"
    elif len(in_range) == 1:
        # The one narrow exception. Weaker evidence, and labelled as such.
        chosen, quality = in_range[0], "unnamed_sole_candidate"

    if chosen is None:
        # Split buildings: "Hahn Hall" is two OSM buildings, "Hahn Hall North"
        # and "Hahn Hall South", and the geocoder lands on neither (a bus stop on
        # Drillfield Drive). Merging them would fit one photo's mesh to two
        # buildings, so this stays a failure — but it names the parts, so the
        # fix is one retyped address rather than an OSM investigation.
        siblings = _split_building_parts(buildings, address)
        if siblings:
            # The parts can straddle the search radius (Hahn Hall North is 55 m
            # out, just beyond it), and naming only one part implies there is
            # only one. Look wider — cached like every other Overpass call.
            try:
                wider = fetch_osm(lat, lon, radius=SPLIT_SEARCH_RADIUS_M)
                wide_buildings = [(el, _element_to_polygon(el))
                                  for el in wider.get("elements", [])
                                  if "building" in (el.get("tags") or {})]
                siblings = sorted(set(siblings) | set(_split_building_parts(wide_buildings, address)))
            except RuntimeError:
                pass  # Overpass down: report the parts we can see
            siblings = sorted(siblings)
            raise LookupError(
                f"{address!r} is split into separate OSM buildings: "
                + ", ".join(repr(s) for s in siblings)
                + ". A photo shows one of them — re-run with the specific name, "
                f"e.g. --address \"{siblings[0]}, {_address_tail(address)}\" "
                "(spec 10.2 failure 3)."
            )
        names = sorted({(el.get("tags") or {}).get("name", "?") for el, _ in in_range})
        raise LookupError(
            f"Ambiguous footprint for {address!r} at ({lat:.5f}, {lon:.5f}): "
            f"{len(containing)} polygons contain the point, {len(in_range)} are "
            f"within {max_distance_m:.0f} m, and none match by name. "
            f"Candidates: {names or 'none in range'}. "
            "Refusing to guess (spec 3.2 step 3, spec 14). Either the geocode is "
            "wrong, the building is absent from OSM, or the address is stale — "
            "check it by hand rather than letting the solver invent a result."
        )

    chosen[0]["_match_quality"] = quality
    neighbours = [p for el, p in buildings if el is not chosen[0]]
    return chosen[0], chosen[1], neighbours


SPLIT_SEARCH_RADIUS_M = 200


def _query_building_name(address: str) -> str:
    """'Hahn Hall, Blacksburg, VA' -> 'hahn hall'."""
    return address.split(",", 1)[0].strip().lower()


def _address_tail(address: str) -> str:
    """'Hahn Hall, Blacksburg, VA' -> 'Blacksburg, VA'."""
    parts = address.split(",", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _split_building_parts(buildings, address: str) -> list[str]:
    """Names of OSM buildings that are PARTS of the queried name.

    A part is a building whose name starts with the full query name followed by
    more words ("Hahn Hall" -> "Hahn Hall North"). Requiring the whole query as
    a prefix keeps "Hall" from matching every hall on campus.
    """
    query = _query_building_name(address)
    if len(query) < 4:
        return []
    parts = set()
    for el, _ in buildings:
        name = ((el.get("tags") or {}).get("name") or "").strip()
        low = name.lower()
        if low.startswith(query + " ") and len(low) > len(query) + 1:
            parts.add(name)
    return sorted(parts)


def _distance_m(geom, pt) -> float:
    """Boundary distance in metres. 0 when the point is inside."""
    lat = geom.centroid.y
    m_per_deg = 111_320.0
    return float(geom.distance(pt) * m_per_deg * math.cos(math.radians(lat)) ** 0.5)


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


def build_footprint(polygon_lonlat: Polygon | MultiPolygon, frame: ENUFrame, *,
                    simplify_eps: float = 0.4,
                    source: str = "osm",
                    match_quality: str = "",
                    geocode_location_type: str = "") -> Footprint:
    """Project an OSM geometry into the ENU frame and condition it. Spec 3-4.

    Multi-part footprints (Lane Stadium's four stands) keep every part. The
    LARGEST part drives the OMBB, the principal angle and the rectilinearity,
    because those are single-orientation quantities and the largest part is the
    best estimate of the building's axis — but `area_m2` and everything that
    goes through `to_shapely` cover all parts, so IoU and Hausdorff stay honest.
    """
    def ring_to_enu(ring) -> np.ndarray:
        arr = np.asarray(ring.coords)[:-1]
        enu = frame.geodetic_to_enu(arr[:, 1], arr[:, 0], 0.0)
        return np.asarray(enu)[:, :2]

    geoms = ([polygon_lonlat] if isinstance(polygon_lonlat, Polygon)
             else list(polygon_lonlat.geoms))
    geoms = [orient(g, sign=1.0) for g in geoms]           # CCW outer rings
    geoms.sort(key=lambda g: g.area, reverse=True)

    converted = []
    for g in geoms:
        outer = simplify_ring(ring_to_enu(g.exterior), simplify_eps)
        holes = tuple(ring_to_enu(r) for r in g.interiors)
        converted.append((outer, holes))

    primary_outer, primary_holes = converted[0]
    extra_parts = tuple(o for o, _ in converted[1:])

    theta_star, rect = principal_orientation(primary_outer)
    full = MultiPolygon([Polygon(o, list(h)) for o, h in converted]) \
        if len(converted) > 1 else Polygon(primary_outer, list(primary_holes))

    return Footprint(
        pts_enu=primary_outer,
        holes_enu=primary_holes,
        parts_enu=extra_parts,
        rectilinearity=rect,
        principal_angle=theta_star,
        ombb=compute_ombb(primary_outer),
        area_m2=float(full.area),
        source=source,
        match_quality=match_quality,
        geocode_location_type=geocode_location_type,
    )


def to_shapely(fp: Footprint) -> Polygon | MultiPolygon:
    """Footprint -> shapely geometry, interior rings AND extra parts intact.

    Everything that measures overlap must go through this, not through
    Polygon(fp.pts_enu), or multi-part buildings silently lose their other
    parts from the denominator.
    """
    primary = Polygon(fp.pts_enu, [h for h in fp.holes_enu])
    if not fp.parts_enu:
        return primary
    return MultiPolygon([primary] + [Polygon(p) for p in fp.parts_enu])


def _slug(s: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in s.lower())[:48]
    return f"{clean}_{hashlib.sha256(s.encode()).hexdigest()[:8]}"
