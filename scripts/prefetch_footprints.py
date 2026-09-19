"""
Warm the footprint cache and commit it. Spec 3.2, 10.2 failure 12, addendum F.6.

"The public Overpass endpoint is a shared free service and will hand you 429 and
504 under any load. For the demo, pre-fetch and cache every building you intend
to show, keyed by quadkey, and commit the cache to the repo."

Run this at hour 0 while the network is healthy, and again any time the demo set
changes. The cache lives in fixtures/footprints/ and is the ONE thing in data
terms that is deliberately NOT gitignored.

    uv run python scripts/prefetch_footprints.py
    uv run python scripts/prefetch_footprints.py --benchmark   # all 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geo.coords import ENUFrame  # noqa: E402
from geo.footprint import (  # noqa: E402
    CACHE_DIR,
    build_footprint,
    fetch_osm,
    geocode,
    select_footprint,
)
from scripts.demo_addresses import BENCHMARK, DEMO  # noqa: E402


def prefetch(addresses: list[str], *, refresh: bool = False) -> int:
    failures = 0
    print(f"Prefetching {len(addresses)} footprints into {CACHE_DIR}\n")

    for addr in addresses:
        try:
            g = geocode(addr, use_cache=not refresh)
            payload = fetch_osm(g["lat"], g["lon"], use_cache=not refresh)
            element, poly, neighbours = select_footprint(payload, g["lat"], g["lon"], addr)

            # Verify it actually conditions — spec 15 hours 0-2 says the exit
            # criterion is footprints "fetched, cached, and VERIFIED TO EXIST".
            # A cached 404 is worse than no cache.
            frame = ENUFrame(lat0=poly.centroid.y, lon0=poly.centroid.x)
            fp = build_footprint(poly, frame)

            tags = element.get("tags") or {}
            flag = "  <-- low R, expect review" if fp.is_ill_posed else ""
            print(f"  + {addr}")
            print(f"      {element['type']} {element['id']} "
                  f"{tags.get('name', '(unnamed)')}")
            print(f"      {fp.area_m2:>8.0f} m2   R={fp.rectilinearity:.3f}   "
                  f"aspect={fp.ombb.aspect:.2f}   "
                  f"{len(neighbours)} neighbours{flag}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ! {addr}\n      {type(exc).__name__}: {exc}")

    print(f"\n{len(addresses) - failures}/{len(addresses)} cached.")
    if failures:
        print("Fix the failures before judging — a missing footprint is a "
              "missing demo (spec 3.2).")
    else:
        print(f"Commit {CACHE_DIR.relative_to(Path.cwd())} — it is demo insurance.")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true",
                    help="all 20 spec 11 buildings, not just the 5 demo ones")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the cache and re-fetch")
    args = ap.parse_args()

    addresses = list(dict.fromkeys(DEMO + BENCHMARK)) if args.benchmark else DEMO
    return 1 if prefetch(addresses, refresh=args.refresh) else 0


if __name__ == "__main__":
    raise SystemExit(main())
