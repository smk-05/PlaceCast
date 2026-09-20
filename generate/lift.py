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

# A second lifter, because the first one is the demo's bottleneck. Single-view
# TRELLIS returns a shallow shell: NCB's plan came out 1.59:1 where the building
# is 2.22:1, Patton's filled 18% of its bounding box where the real footprint
# fills 81%. Hunyuan3D-2.1 is a different image-to-3D model with its own depth
# prior, so it is worth measuring rather than assuming. Same rules: version
# PINNED (verified live 2026-09-19), seed recorded, one image in, a glb out.
HUNYUAN_MODEL = ("ndreca/hunyuan3d-2.1:"
                 "895e514f953d39e8b5bfb859df9313481ad3fa3a8631e5c54c7e5c9c85a6aa9f")
HUNYUAN_PARAMS = {
    "steps": 50,
    "guidance_scale": 7.5,
    "octree_resolution": 256,
    "max_facenum": 20000,
    "generate_texture": True,
    # Our input is already a building cut out on white (generate/mask.py), so
    # the model's own background removal has nothing left to do — and running
    # it on a white field has been known to eat pale facades.
    "remove_background": False,
}

LIFTERS = {
    # name: (pinned model, default params, how the image goes into the payload)
    "trellis": (MODEL, DEFAULT_PARAMS, "images"),
    "hunyuan": (HUNYUAN_MODEL, HUNYUAN_PARAMS, "image"),
}


def lift_to_mesh(image_paths: list[Path], out_path: Path,
                 *, seed: int = 42, force: bool = False,
                 params: dict | None = None,
                 lifter: str = "trellis") -> tuple[Path, dict]:
    """-> (path to .glb, the params actually used). Cached by out_path.

    `lifter` picks the image-to-3D model: "trellis" (multi-view capable) or
    "hunyuan" (Hunyuan3D-2.1, single image only — extra views are ignored, and
    the caller is told).
    """
    out_path = Path(out_path)
    if lifter not in LIFTERS:
        raise ValueError(f"unknown lifter {lifter!r}; pick one of {sorted(LIFTERS)}")
    model, defaults, image_key = LIFTERS[lifter]
    merged = {**defaults, **(params or {}), "seed": seed}

    if out_path.exists() and not force:
        return out_path, merged

    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. Copy .env.example to .env — see the "
            "hour-0 account list."
        )

    from generate import throttle

    if image_key == "image" and len(image_paths) > 1:
        print(f"lift[{lifter}]: single-image model — using {Path(image_paths[0]).name} "
              f"and ignoring {len(image_paths) - 1} other view(s)")
        image_paths = image_paths[:1]

    handles = [open(p, "rb") for p in image_paths]
    try:
        # TRELLIS's PINNED version requires `images` (an array), verified
        # against the live API 2026-09-19; a 422 "image is required" means a
        # different, unpinned version is being called. Hunyuan takes a single
        # `image`, hence the per-lifter key rather than one hardcoded name.
        payload = {**merged, image_key: handles if image_key == "images" else handles[0]}
        output = throttle.run(model, input=payload)
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


def load_surface_points(glb_path: Path, n: int = 60_000):
    """Load a .glb and return points sampled over its SURFACE. -> (N,3).

    Prefer this to load_vertices for anything the outline is extracted from.
    A vertex cloud describes the mesh's topology, not its shape: a modelled
    asset puts four vertices on a whole wall, and spec 6.3's rasterisation then
    sees specks (a downloaded 48 x 27 m block gave a 39 m2 "plan"). Generated
    meshes are dense enough to hide the difference; bought ones are not.
    """
    import numpy as np
    import trimesh

    from geo.outline import sample_surface

    scene = trimesh.load(str(glb_path), force="scene")
    mesh = (scene if isinstance(scene, trimesh.Trimesh)
            else trimesh.util.concatenate(list(scene.geometry.values())))
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64) if hasattr(mesh, "faces") else None
    if f is None or len(f) == 0:
        return v
    return sample_surface(v, f, n=n)
