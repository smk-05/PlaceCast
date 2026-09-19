"""
Generate every demo building from the photo walk. Plan phase 3.2.

For each building in scripts/demo_addresses.DEMO, the primary photo is
photos/<key>_1.jpg (the corner shot); pipeline.run does EXIF, masking,
FLUX, TRELLIS, the solve and the gate. Asset ids go to demo_assets.json
(committed) so the viewer and the rehearsal use exactly these runs.

COSTS MONEY: one FLUX Kontext + one TRELLIS call per building on the
Replicate account in .env (~2 min, a few cents each). Without --yes it only
prints what it would do. Already-generated buildings are skipped unless
--regenerate; re-solving an existing asset is free:

    uv run python pipeline.py --address "..." --photo photos/ncb_1.jpg --asset-id <id>

    uv run python scripts/generate_demo.py                  # plan only
    uv run python scripts/generate_demo.py --yes            # spend
    uv run python scripts/generate_demo.py --yes --only ncb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.check_photos import PHOTOS  # noqa: E402

MANIFEST = ROOT / "demo_assets.json"
KEYS = {
    "ncb": "Classroom Building, Blacksburg, VA",
    "burruss": "Burruss Hall, Blacksburg, VA",
    "patton": "Patton Hall, Blacksburg, VA",
    "warmemorial": "War Memorial Hall, Blacksburg, VA",
    "whittemore": "Whittemore Hall, Blacksburg, VA",
    "goodwin": "Goodwin Hall, Blacksburg, VA",
}


# The shot that gets generated, chosen by eye from the 2026-09-19 walk: a
# corner view, whole building in frame, least in front of it. Default <key>_1.
#   ncb_5          ncb_1 is cropped on the left; _5 shows the NE end + long side
#   warmemorial_2  _2 shows two facades; _1 is nearly face-on
PRIMARY = {
    "ncb": "ncb_5.jpg",
    "warmemorial": "warmemorial_2.jpg",
}


def primary_photo(key: str) -> Path | None:
    if key in PRIMARY and (PHOTOS / PRIMARY[key]).exists():
        return PHOTOS / PRIMARY[key]
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png"):
        p = PHOTOS / f"{key}_1{ext}"
        if p.exists():
            return p
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually call Replicate")
    ap.add_argument("--only", action="append", default=[], choices=list(KEYS))
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--perception", choices=["auto", "real", "stub"], default="auto",
                    help="auto: real when Owen's mask for that exact photo is "
                         "cached (no torch needed on a cache hit), else stub")
    ap.add_argument("--prompt",
                    default="weathered concrete, ivy overgrowth, scorched upper floors")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    keys = args.only or list(KEYS)

    plan = []
    for k in keys:
        photo = primary_photo(k)
        if photo is None:
            print(f"  - {k:<12} no photos/{k}_1.jpg — skipped")
            continue
        if k in manifest and not args.regenerate:
            print(f"  = {k:<12} already generated (asset {manifest[k]['asset_id']})")
            continue
        plan.append((k, photo))
        print(f"  + {k:<12} {photo.name}")

    if not plan:
        print("\nnothing to generate")
        return 0
    if not args.yes:
        print(f"\n{len(plan)} building(s) would be generated (billed). Re-run with --yes.")
        return 0

    from perception.segment import cache_paths, image_sha256
    from pipeline import run

    def backend_for(photo: Path) -> str:
        if args.perception != "auto":
            return args.perception
        png, js = cache_paths(image_sha256(photo))
        return "real" if png.exists() and js.exists() else "stub"

    failures = 0
    for k, photo in plan:
        backend = backend_for(photo)
        print(f"\n=== {k} (perception: {backend}"
              f"{'' if backend == 'real' else ' - NOT masked'}) ===")
        try:
            rec = run(KEYS[k], photos=[photo], prompt=args.prompt,
                      perception=backend)
        except Exception as exc:  # noqa: BLE001 — one failure must not stop the batch
            failures += 1
            print(f"FAILED {k}: {type(exc).__name__}: {exc}")
            continue
        fit = rec.fit
        manifest[k] = {
            "address": KEYS[k],
            "asset_id": rec.asset_id,
            "photo": photo.name,
            "perception": backend,
            "decision": rec.decision.value,
            "iou": round(fit.iou, 3) if fit else None,
            "hausdorff_m": round(fit.hausdorff_m, 2) if fit else None,
            "disambiguated_by": fit.disambiguated_by.value if fit else None,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{k}: {rec.decision.value}  asset {rec.asset_id}")

    print("\n" + "\n".join(
        f"  {k:<12} {v['decision']:<12} IoU {v['iou']}  Hausdorff {v['hausdorff_m']} m  "
        f"via {v['disambiguated_by']}" for k, v in manifest.items()))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
