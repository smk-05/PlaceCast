"""
Image -> mesh. Spec 5.2, TRELLIS via Replicate.

TRELLIS encodes assets into a Structured LATent representation — local latent
vectors anchored at sparse surface voxels on a 64^3 grid — and generates in two
rectified-flow stages: a sparse-structure stage producing the voxel scaffold,
then a structured-latent stage filling in geometry and appearance. The
image-conditioned variant substantially outperforms the text-conditioned one and
accepts multiple input images for multi-view conditioning without a separate
model variant.

Budget 10-60 seconds per generation. This is async by design (spec 14): a
synchronous HTTP handler that blocks on this will destroy the demo.

Multi-view (spec 5.1): if several photographs are supplied, they must be edited
JOINTLY or with a shared seed. Independently edited views disagree, and this
stage will either average them into mush or pick one and ignore the others.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

# A community model, so the version is PINNED: an unpinned run silently picks up
# whatever the owner publishes next, which breaks spec 12.1's reproducibility.
# Schema verified against this version on 2026-09-19.
MODEL = ("firtoz/trellis:"
         "e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c")

# Recorded in the placement record so the result is reproducible (spec 12.1).
DEFAULT_PARAMS = {
    "ss_sampling_steps": 12,
    "slat_sampling_steps": 12,
    "ss_guidance_strength": 7.5,
    "slat_guidance_strength": 3.0,
    "mesh_simplify": 0.95,      # quadric simplification retention
    "texture_size": 1024,       # 2048 only for hero assets (spec 13.3)
    "generate_model": True,     # the GLB — defaults to False on this deployment
    # This deployment defaults randomize_seed to TRUE, which silently discards
    # the seed we pass. Without this, spec 14's determinism is quietly broken.
    "randomize_seed": False,
    # Preview videos: extra GPU time we would pay for and never use.
    "generate_color": False,
    "generate_normal": False,
}


def lift_to_mesh(image_paths: list[Path], out_path: Path,
                 *, seed: int = 42, force: bool = False,
                 params: dict | None = None) -> tuple[Path, dict]:
    """-> (path to .glb, the params actually used). Cached by out_path."""
    out_path = Path(out_path)
    merged = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}

    if out_path.exists() and not force:
        return out_path, merged

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. Copy .env.example to .env — see the "
            "hour-0 account list."
        )

    import replicate

    handles = [open(p, "rb") for p in image_paths]
    try:
        payload = {**merged}
        if len(handles) == 1:
            payload["images"] = [handles[0]]
        else:
            payload["images"] = handles
        output = replicate.run(MODEL, input=payload)
    finally:
        for h in handles:
            h.close()

    url = _pick_glb(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path, merged


def _pick_glb(output) -> str:
    """TRELLIS returns a dict of artifacts; we want the glb."""
    if isinstance(output, dict):
        for key in ("model_file", "glb", "mesh", "model"):
            if output.get(key):
                return str(output[key])
        raise RuntimeError(f"No glb in TRELLIS output: {list(output)}")
    if isinstance(output, list):
        for item in output:
            if str(item).endswith(".glb"):
                return str(item)
        return str(output[0])
    return str(output)


def load_vertices(glb_path: Path):
    """Load a .glb and return its concatenated vertices as (N,3) float array.

    Generated meshes are routinely non-manifold, self-intersecting, and
    multi-component, so we take the vertex cloud rather than trusting the scene
    graph. geo/outline.py's rasterisation is indifferent to all of that.
    """
    import numpy as np
    import trimesh

    scene = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene, trimesh.Trimesh):
        return np.asarray(scene.vertices, dtype=float)

    chunks = []
    for name, geom in scene.geometry.items():
        v = np.asarray(geom.vertices, dtype=float)
        transform = scene.graph.get(name)[0] if name in scene.graph.nodes else None
        if transform is not None:
            v = trimesh.transformations.transform_points(v, transform)
        chunks.append(v)
    if not chunks:
        raise ValueError(f"No geometry in {glb_path}")
    return np.vstack(chunks)
