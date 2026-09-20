"""
Copy a run's GENERATED assets into a new run id, so the same mesh can be
re-solved a second way without paying for it again.

generate/edit.py and generate/lift.py both cache on their output path, so a
pipeline run whose directory already holds edited_*.png and mesh.glb calls
neither FLUX nor TRELLIS. That is what makes a side-by-side possible: the
honest uniform-scale record stays as it is, and the conformed one sits next to
it in the viewer's list, from the same mesh, for nothing.

    uv run python scripts/clone_run.py SRC_ID                 # -> new uuid
    uv run python scripts/clone_run.py SRC_ID --dst other-id
    uv run python scripts/clone_run.py SRC_ID --then-conform  # clone and solve

Record, overlay and log are NOT copied: the clone re-solves and writes its own.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "data" / "runs"
# The generated artefacts, i.e. everything that costs money or a GPU to remake.
PATTERNS = ("mesh.glb", "edited_*.png", "edited_clean_*.png", "masked_*.png",
            "mask.npy")


def clone(src_id: str, dst_id: str | None = None) -> str:
    src = RUNS / src_id
    if not src.is_dir():
        raise SystemExit(f"no run directory {src}")
    dst_id = dst_id or str(uuid.uuid4())
    dst = RUNS / dst_id
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for pattern in PATTERNS:
        for f in src.glob(pattern):
            shutil.copy2(f, dst / f.name)
            copied += 1
    if not copied:
        raise SystemExit(f"{src} holds no generated assets to clone")
    print(f"cloned {copied} file(s) {src_id[:8]} -> {dst_id}")
    return dst_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_id")
    ap.add_argument("--dst", default=None)
    ap.add_argument("--then-conform", action="store_true",
                    help="re-solve the clone with --conform, reusing the record's "
                         "own address, photos and perception backend")
    args = ap.parse_args()

    dst_id = clone(args.src_id, args.dst)
    if not args.then_conform:
        print(f"now: uv run python pipeline.py --address ... --asset-id {dst_id} --conform")
        return 0

    import json
    rec = json.loads((RUNS / args.src_id / "record.json").read_text(encoding="utf-8"))
    inputs = rec.get("inputs") or {}
    cmd = [sys.executable, "pipeline.py", "--address", rec["address_raw"],
           "--asset-id", dst_id, "--conform",
           "--perception", inputs.get("perception", "stub")]
    for p in inputs.get("photos", []):
        cmd += ["--photo", p]
    if inputs.get("dry_run"):
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
