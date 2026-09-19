"""
EXIF extraction. Spec 6.6 Filter 2, addendum C.2 gravity alignment, D.1 ingest.

This lives in geo/ rather than perception/ on purpose. EXIF is free arithmetic
with no model involved, and spec 6.6 Filter 2 — the strongest disambiguation cue
there is — depends on it. Nothing that strong is allowed to block on the
perception half arriving.

Addendum D.1: run this first at ingest, before anything else. It costs nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import ExifTags, Image

_GPS_TAG = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), 34853)
_GPS_NAMES = {v: k for k, v in ExifTags.GPSTAGS.items()}


def _ratio(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _dms_to_deg(dms, ref: str | None) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        deg = -deg
    return deg


def extract(image_path: Path) -> dict:
    """-> {heading_deg, pitch_deg, focal_mm, gps, sha256, width, height}.

    Every field is None when absent. Phone photos frequently carry
    GPSImgDirection; downloaded images never do, which is the normal case and
    not an error.
    """
    image_path = Path(image_path)
    out: dict = {
        "heading_deg": None,
        "pitch_deg": None,
        "focal_mm": None,
        "gps": None,
        "sha256": _sha256(image_path),
        "width": None,
        "height": None,
    }

    try:
        with Image.open(image_path) as im:
            out["width"], out["height"] = im.size
            exif = im.getexif()
            if not exif:
                return out

            named = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            out["focal_mm"] = _ratio(named.get("FocalLength"))

            gps = exif.get_ifd(_GPS_TAG) or {}
            g = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}

            heading = _ratio(g.get("GPSImgDirection"))
            if heading is not None:
                # GPSImgDirectionRef is "T" (true) or "M" (magnetic). We do not
                # correct for declination — it is ~10 deg in Virginia, well
                # inside the 90 deg bins this cue has to discriminate between.
                out["heading_deg"] = heading % 360.0

            pitch = _ratio(g.get("GPSPitch")) or _ratio(named.get("CameraElevationAngle"))
            if pitch is not None:
                out["pitch_deg"] = pitch

            lat = _dms_to_deg(g.get("GPSLatitude"), g.get("GPSLatitudeRef"))
            lon = _dms_to_deg(g.get("GPSLongitude"), g.get("GPSLongitudeRef"))
            if lat is not None and lon is not None:
                out["gps"] = (lat, lon)
    except Exception:  # noqa: BLE001 - a corrupt or EXIF-less image is normal
        pass

    return out


def _sha256(path: Path) -> str:
    """Content hash for the provenance record (spec 12.1 `inputs[].sha256`)."""
    if not Path(path).exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
