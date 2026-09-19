"""
Check the photo-walk photos before spending a cent on generation.

For every photos/<building>_<n>.jpg:
  - EXIF: is there a compass heading and a camera GPS fix? (spec 6.6 Filter 2)
  - Aim:  does the heading point AT the building? The bearing from the camera
          GPS to the footprint centroid should be within ~40 degrees of the
          heading. If not, the compass was wrong or the file is misnamed, and
          EXIF would confidently pick the wrong orientation.
  - Mask: is there a real segmentation mask for these exact bytes in
          .cache/perception/masks (Owen's cache files)? Without one,
          generation sees the raw photo and scenery becomes geometry.

    uv run python scripts/check_photos.py
    uv run python scripts/check_photos.py --import-masks path/to/owens/masks
    uv run python scripts/check_photos.py --import-openings path/to/owens/openings
    uv run python scripts/check_photos.py --export-openings path/to/hand/over

--import-masks copies Owen's <sha>_<config>.png/.json pairs into the cache.
The config hash must match this checkout's perception/segment.py settings; a
mismatch shows up as "no mask" and means we are on different commits.

--import-openings / --export-openings do the same for the door/window results
(perception/openings.py, .cache/perception/openings/<sha16>_<config>.json).
They load without torch, so a laptop without it can use what a GPU machine
computed. The config hash covers perception/openings.py's settings and
decision code; a mismatch shows up as "openings: none" for the same reason.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geo import exif  # noqa: E402
from geo.footprint import fetch_osm, geocode, select_footprint  # noqa: E402
from perception.openings import export_openings, import_openings, load_cached  # noqa: E402
from perception.segment import CACHE_DIR, cache_paths, image_sha256  # noqa: E402

PHOTOS = ROOT / "photos"
PREFIX = {
    "ncb": "Classroom Building, Blacksburg, VA",
    "classroom": "Classroom Building, Blacksburg, VA",
    "burruss": "Burruss Hall, Blacksburg, VA",
    "patton": "Patton Hall, Blacksburg, VA",
    "warmemorial": "War Memorial Hall, Blacksburg, VA",
    "war": "War Memorial Hall, Blacksburg, VA",
    "whittemore": "Whittemore Hall, Blacksburg, VA",
    "goodwin": "Goodwin Hall, Blacksburg, VA",
}
AIM_TOLERANCE_DEG = 40.0
EXTS = {".jpg", ".jpeg", ".png", ".heic"}


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def distance_m(lat1, lon1, lat2, lon2) -> float:
    k = 111_320.0
    return math.hypot((lat2 - lat1) * k, (lon2 - lon1) * k * math.cos(math.radians(lat1)))


def building_for(photo: Path) -> str | None:
    stem = photo.stem.lower().replace("-", "_")
    key = stem.split("_")[0]
    return PREFIX.get(key)


def centroid(address: str) -> tuple[float, float]:
    g = geocode(address)
    payload = fetch_osm(g["lat"], g["lon"])
    _, poly, _ = select_footprint(payload, g["lat"], g["lon"], address)
    return poly.centroid.y, poly.centroid.x


def check(photo: Path) -> bool:
    ok = True
    print(f"\n{photo.name}")
    if photo.suffix.lower() == ".heic":
        print("  ! HEIC — re-export as JPEG (iPhone: Settings > Camera > Formats > "
              "Most Compatible). The EXIF reader cannot open HEIC.")
        return False

    address = building_for(photo)
    print(f"  building: {address or '? (name it ncb_1.jpg, burruss_2.jpg, ...)'}")

    meta = exif.extract(photo)
    heading, gps = meta["heading_deg"], meta["gps"]
    print(f"  exif: heading={heading} gps={gps} pitch={meta['pitch_deg']}")
    if heading is None or gps is None:
        ok = False
        print("  ! no compass heading or no GPS — this photo cannot decide "
              "orientation. Was camera location on? Was it sent through a "
              "messaging app (which strips EXIF)?")
    elif address:
        lat, lon = centroid(address)
        to_bldg = bearing_deg(gps[0], gps[1], lat, lon)
        dist = distance_m(gps[0], gps[1], lat, lon)
        off = abs((heading - to_bldg + 180) % 360 - 180)
        flag = "ok" if off <= AIM_TOLERANCE_DEG else "! MISAIMED"
        print(f"  aim: camera faces {heading:.0f}°, building centre is at "
              f"{to_bldg:.0f}° and {dist:.0f} m -> off by {off:.0f}°  {flag}")
        if off > AIM_TOLERANCE_DEG:
            ok = False
        if dist > 1000:
            ok = False
            print("  ! camera is over 1 km away — GPS fix is wrong; EXIF will abstain")

    png, js = cache_paths(image_sha256(photo))
    if png.exists() and js.exists():
        print(f"  mask: real, cached ({png.name}) -> generation will be masked")
    else:
        print("  mask: none yet -> run with --perception real only after Owen's "
              "mask files are imported")
    res = load_cached(photo)
    if res is not None:
        print(f"  openings: cached, {len(res['openings'])} boxes")
    else:
        print("  openings: none cached for this checkout's config (import them, or compute with torch)")
    return ok


def import_masks(src: Path) -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.iterdir():
        if f.suffix.lower() in (".png", ".json"):
            shutil.copy2(f, CACHE_DIR / f.name)
            n += 1
    print(f"imported {n} files into {CACHE_DIR}")
    return n


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-masks", type=Path, default=None)
    ap.add_argument("--import-openings", type=Path, default=None)
    ap.add_argument("--export-openings", type=Path, default=None)
    ap.add_argument("photos", nargs="*", type=Path)
    args = ap.parse_args()

    if args.import_masks:
        import_masks(args.import_masks)
    if args.import_openings:
        import_openings(args.import_openings)
    if args.export_openings:
        export_openings(args.export_openings)

    photos = args.photos or sorted(p for p in PHOTOS.iterdir()
                                   if p.suffix.lower() in EXTS)
    results = [check(p) for p in photos]
    print(f"\n{sum(results)}/{len(results)} photos usable as orientation evidence")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
