"""
Microsoft GlobalMLBuildingFootprints height check. Spec 3.2 / 7.1, addendum C.1.

Question it answers: for the buildings OSM leaves untagged, does Microsoft's
dataset supply a height? If it does, addendum section C (monocular depth) may be
unnecessary — a dataset lookup instead of a model.

How the dataset is laid out: one CSV index (dataset-links.csv) mapping
(country, zoom-9 quadkey) -> a URL. Each URL is gzipped line-delimited GeoJSON
with a misleading .csv.gz extension. Features carry `height` in metres (-1 when
unknown) and `confidence`.

Matching: Microsoft polygons are ML-detected and do not share IDs with OSM, so
each is matched to the OSM footprint by AREA OF OVERLAP, not by nearest point.
Large buildings are often split into several Microsoft polygons; every polygon
overlapping the footprint is reported, and the height is taken from the one
covering the most of it.

The relevant partition is cached under .cache/ (gitignored, large) and the
matched subset is written to fixtures/microsoft/ (small, committed).

    uv run python scripts/check_microsoft_heights.py
    uv run python scripts/check_microsoft_heights.py --all   # every benchmark building
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
from pathlib import Path

import requests
from shapely.geometry import shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geo.footprint import fetch_osm, geocode, select_footprint  # noqa: E402
from geo.height import from_microsoft, from_osm_tags  # noqa: E402
from scripts.demo_addresses import BENCHMARK, DEMO  # noqa: E402

INDEX_URL = ("https://minedbuildings.z5.web.core.windows.net/"
             "global-buildings/dataset-links.csv")
CACHE = ROOT / ".cache" / "microsoft"
FIXTURE = ROOT / "fixtures" / "microsoft"


def quadkey(lat: float, lon: float, zoom: int = 9) -> str:
    """Bing Maps tile quadkey — the dataset's partition key."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = (lon + 180.0) / 360.0
    s = math.sin(math.radians(lat))
    y = 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)
    n = 1 << zoom
    tx = min(max(int(x * n), 0), n - 1)
    ty = min(max(int(y * n), 0), n - 1)
    digits = []
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        d = (1 if tx & mask else 0) + (2 if ty & mask else 0)
        digits.append(str(d))
    return "".join(digits)


def partition_urls(qk: str) -> list[str]:
    CACHE.mkdir(parents=True, exist_ok=True)
    index = CACHE / "dataset-links.csv"
    if not index.exists():
        r = requests.get(INDEX_URL, timeout=60)
        r.raise_for_status()
        index.write_bytes(r.content)

    urls = []
    with open(index, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("QuadKey") == qk:
                urls.append(row["Url"])
    return urls


def load_partition(url: str) -> list[dict]:
    """Download (once) and parse one partition."""
    local = CACHE / url.rsplit("/", 1)[-1]
    if not local.exists():
        print(f"  downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        local.write_bytes(r.content)

    feats = []
    with gzip.open(local, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                feats.append(json.loads(line))
    return feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="check every building, not just the OSM-untagged ones")
    args = ap.parse_args()

    addresses = list(dict.fromkeys(DEMO + BENCHMARK))

    # Resolve every building through the SAME selection rule the pipeline uses,
    # so a building that fails selection is reported as such, not silently
    # matched to whatever Microsoft polygon happens to be nearby.
    targets = []
    for addr in addresses:
        try:
            g = geocode(addr)
            payload = fetch_osm(g["lat"], g["lon"])
            el, geom, _ = select_footprint(payload, g["lat"], g["lon"], addr)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {addr}: {type(exc).__name__}: {exc}")
            continue
        osm = from_osm_tags(el.get("tags") or {})
        if osm is None or args.all:
            targets.append((addr, geom, osm))

    if not targets:
        print("Nothing to check.")
        return 0

    qks = {quadkey(g.centroid.y, g.centroid.x) for _, g, _ in targets}
    print(f"Checking {len(targets)} buildings across quadkey(s) {sorted(qks)}\n")

    features = []
    for qk in qks:
        urls = partition_urls(qk)
        if not urls:
            print(f"  no Microsoft partition for quadkey {qk}")
        for u in urls:
            features.extend(load_partition(u))
    print(f"  {len(features):,} Microsoft footprints loaded\n")

    polys = [shape(f["geometry"]) for f in features]
    tree = STRtree(polys)

    hits = 0
    rows = []
    subset = []
    for addr, geom, osm in targets:
        idx = tree.query(geom)
        overlaps = []
        for i in idx:
            inter = polys[i].intersection(geom).area
            if inter <= 0:
                continue
            props = features[i].get("properties") or {}
            overlaps.append((inter / geom.area, props.get("height", -1),
                             props.get("confidence", -1), i))
        overlaps.sort(reverse=True)

        if not overlaps:
            rows.append((addr, "NO POLYGON", "-", "Microsoft has no footprint here"))
            continue

        cover = sum(o[0] for o in overlaps)
        best_frac, best_h, best_conf, best_i = overlaps[0]
        heights = [o[1] for o in overlaps if o[1] is not None and o[1] >= 0]
        subset.extend(features[o[3]] for o in overlaps)

        result = from_microsoft(best_h if best_h is not None else -1)
        note = (f"{len(overlaps)} polygon(s) cover {cover:.0%} of OSM footprint; "
                f"best covers {best_frac:.0%}, conf {best_conf}")
        if len(heights) > 1:
            note += f"; heights across parts {min(heights):.1f}-{max(heights):.1f} m"

        if result:
            hits += 1
            osm_note = f"  (OSM: {osm[0]:.1f} m)" if osm else ""
            rows.append((addr, "HIT", f"{result[0]:.1f} m", note + osm_note))
        else:
            rows.append((addr, "MISS", "-", f"height={best_h}; " + note))

    width = max(len(r[0]) for r in rows)
    for addr, status, val, note in rows:
        mark = {"HIT": "+", "MISS": "-", "NO POLYGON": "!"}[status]
        print(f"  {mark} {addr:<{width}}  {val:>8}  {note}")

    print(f"\nMicrosoft supplies a height for {hits}/{len(targets)} "
          f"{'buildings' if args.all else 'OSM-untagged buildings'}.")

    FIXTURE.mkdir(parents=True, exist_ok=True)
    out = FIXTURE / "matched_footprints.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": subset}),
                   encoding="utf-8")
    print(f"Matched subset written to {out.relative_to(ROOT)} (commit it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
