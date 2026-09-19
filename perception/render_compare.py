#!/usr/bin/env python
"""Render-and-compare orientation scoring (ML_ADDENDUM A.3 stages 2-3, SPEC 6.1(b) / 6.6 Filter 4).

    result = render_compare(mesh, mask)            # trimesh mesh + mask PNG path or bool array
    result.best, result.second, result.margin, result.candidates

Pipeline API (contracts.ScoreSilhouettes; what the smoke test and pipeline import):
    score_silhouettes(mesh, mask, [CandidateId]) -> {CandidateId: normalised IoU}
    margin_of(scores)                            -> top-two gap
    photographed_side(mesh, mask)                -> which side of the mesh the photo shows

score_silhouettes scores the UP-AXIS and is deliberately flat across azimuth_k: k is a rotation of the
PLACED mesh about the vertical, and the pipeline assumes the photographed facade is the mesh front, so
the silhouette cannot depend on k. Scoring by mesh side and passing those scores to geo as placement
candidates makes silhouette "decide" k=0 for every photo (tests/test_azimuth_contract.py). The
side-of-mesh answer is photographed_side(), for correcting MeshOutline.front_angle. PROCEDURA_PERCEPTION=stub
returns the abstaining stand-in (see perception/backend.py).

The mesh is rendered as an orthographic *silhouette* from every candidate
(up-axis x 4 azimuths) and each silhouette is scored against the photo's building
mask by normalised-shape IoU: both shapes are cropped to their bounding box,
uniformly scaled so the longer side fills a common square canvas, and centred
before the IoU. That discards position and scale (which a perspective photo
corrupts) and keeps aspect ratio and profile (which discriminate orientation).

Conventions
  * Mesh frame is glTF: +Y up, front facing +Z. Candidate up-axes default to the six
    signed coordinate axes; pass the top 2-3 from SPEC 6.1(a) to prune.
  * For up-axis u, azimuth phi places the camera at R_u(phi) . e1 looking at the mesh
    (right-handed rotation about u), where e1 is the coordinate axis after u's dominant
    axis in X->Y->Z->X order, made perpendicular to u. For u=+Y, phi=0 the camera sits on
    +Z (sees the glTF front facade) with +X to the image right; increasing phi swings the
    camera counter-clockwise seen from above. Candidate k is phi = theta0 + k*90 deg.
  * A candidate scores how well the mesh, viewed from that side with that up-axis,
    matches the photo. Mapping it to a footprint rotation is the placement solver's job.

margin = score(best) - score(second best), as in SPEC 6.6.
Silhouettes are mirror-symmetric under a 180 deg change of view, so a left-right
symmetric mesh ties with its own back view and a box has margin 0; `ties_with_best`
counts those exact ties.

Review and symmetry flags
  * needs_review: margin < 0.05 (SPEC 6.6). The silhouette cannot pick a side.
  * label == "arbitrary_symmetric" (A.4): margin < 0.05 AND the footprint aspect
    a_F/b_F < 1.1, i.e. a near-square footprint whose four facades are genuinely
    indistinguishable. Any candidate is acceptable (best is just the first tie); route to
    review. This is correct behaviour, not a bug. It needs `footprint_aspect`; without
    it the label is never set (needs_review still is), because a low margin on an
    elongated footprint is a different failure that must not be excused as symmetry.

Usage: python perception/render_compare.py MESH.glb MASK.png [--top N]
"""
import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `python perception/render_compare.py` puts perception/ first, not the repo root
    sys.path.insert(0, str(ROOT))

from perception.backend import STUB, resolve_backend  # noqa: E402

CANVAS_PX = 128
PAD_PX = 2
SUBPIXEL_BITS = 4  # cv2.fillConvexPoly fixed-point precision
TIE_TOL = 1e-9
REVIEW_MARGIN = 0.05  # SPEC 6.6: below this the silhouette cannot pick a side
SYMMETRIC_ASPECT = 1.1  # A.4: footprint a_F/b_F below this is "square"
ARBITRARY_SYMMETRIC = "arbitrary_symmetric"

# Y-up (glTF) first so exact ties resolve to the glTF-canonical orientation.
SIGNED_AXES = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]


@dataclass(frozen=True)
class Candidate:
    up_axis: tuple
    azimuth_deg: float
    score: float


@dataclass(frozen=True)
class Comparison:
    candidates: list  # every Candidate, best first
    best: Candidate
    second: Candidate
    margin: float
    ties_with_best: int  # other candidates scoring identically to best (symmetry, not evidence)
    label: Optional[str] = None  # "arbitrary_symmetric" (A.4) or None
    needs_review: bool = False  # margin < REVIEW_MARGIN (SPEC 6.6); always True when label is set

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ inputs


def _as_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())  # dump() bakes scene-graph transforms in
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a trimesh.Trimesh or Scene, got {type(mesh).__name__}")
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")
    return mesh


def _as_mask(mask):
    if isinstance(mask, (str, Path)):
        mask = np.array(Image.open(mask).convert("L")) > 127
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if not mask.any():
        raise ValueError("mask is empty")
    return mask


# ---------------------------------------------------------- normalisation


def normalise_mask(mask, size=CANVAS_PX):
    """Crop to the bounding box, scale uniformly so the longer side fits, centre on a size x size canvas."""
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    scale = (size - 2 * PAD_PX) / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) >= 0.5
    canvas = np.zeros((size, size), bool)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = small
    return canvas


def camera_basis(up_axis, azimuth_rad):
    """Returns (right, up, toward_camera) unit vectors for the convention in the module docstring."""
    up = np.asarray(up_axis, dtype=float)
    norm = np.linalg.norm(up)
    if norm == 0:
        raise ValueError("up axis must be non-zero")
    up = up / norm
    ref = np.eye(3)[(int(np.argmax(np.abs(up))) + 1) % 3]
    e1 = ref - ref.dot(up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    toward = np.cos(azimuth_rad) * e1 + np.sin(azimuth_rad) * e2  # camera sits here, looking back at the mesh
    right = np.cross(-toward, up)
    return right, up, toward


def render_silhouette(mesh, up_axis, azimuth_rad, size=CANVAS_PX):
    """Orthographic silhouette of `mesh`, already normalised to the common canvas (bool array)."""
    right, up, _ = camera_basis(up_axis, azimuth_rad)
    xy = mesh.vertices @ np.stack([right, up], axis=1)  # (V, 2): image-x, image-up
    tris = xy[mesh.faces]  # (F, 3, 2)
    lo, hi = tris.reshape(-1, 2).min(axis=0), tris.reshape(-1, 2).max(axis=0)
    extent = float((hi - lo).max())
    canvas = np.zeros((size, size), np.uint8)
    if extent == 0:
        return canvas.astype(bool)
    scale = (size - 2 * PAD_PX) / extent
    off = (size - (hi - lo) * scale) / 2  # centre the shorter side
    px = (tris[..., 0] - lo[0]) * scale + off[0]
    py = (hi[1] - tris[..., 1]) * scale + off[1]  # image y grows downward
    pts = np.stack([px, py], axis=-1) - 0.5  # edge convention -> pixel-centre convention
    # Triangles are filled one at a time: cv2.fillPoly fills all its polygons in one even-odd
    # pass, so overlapping triangles (front and back faces of any closed mesh) would cancel out.
    # Sub-pixel triangles dominate dense meshes, so mark the pixels under their vertices in one
    # vectorised step and only rasterise the rest.
    tiny = (pts.max(axis=1) - pts.min(axis=1)).max(axis=1) <= 1.0
    corners = np.clip(np.round(pts[tiny].reshape(-1, 2)).astype(int), 0, size - 1)
    canvas[corners[:, 1], corners[:, 0]] = 1
    for tri in np.round(pts[~tiny] * (1 << SUBPIXEL_BITS)).astype(np.int32):
        cv2.fillConvexPoly(canvas, tri, 1, shift=SUBPIXEL_BITS)
    return canvas.astype(bool)


def normalised_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


# ------------------------------------------------------------------- scoring


def symmetry_label(margin, footprint_aspect):
    """A.4 'truly symmetric building': margin < 0.05 AND footprint aspect < 1.1 -> "arbitrary_symmetric".

    `footprint_aspect` is a_F/b_F of the footprint's oriented bounding box; a ratio below 1 is
    read as its reciprocal (which extent is called `a` is arbitrary). None -> cannot say -> None.
    """
    if footprint_aspect is None:
        return None
    aspect = float(footprint_aspect)
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError(f"footprint_aspect must be a positive finite ratio, got {footprint_aspect!r}")
    aspect = max(aspect, 1 / aspect)
    return ARBITRARY_SYMMETRIC if margin < REVIEW_MARGIN and aspect < SYMMETRIC_ASPECT else None


def render_compare(mesh, mask, up_axes=None, theta0=0.0, size=CANVAS_PX, footprint_aspect=None):
    """Score every (up-axis, azimuth) candidate against the photo mask. See the module docstring.

    `footprint_aspect` (a_F/b_F of the footprint OMBB) enables the A.4 arbitrary_symmetric label.
    """
    mesh = _as_mesh(mesh)
    target = normalise_mask(_as_mask(mask), size)
    axes = SIGNED_AXES if up_axes is None else [tuple(float(v) for v in a) for a in up_axes]
    if not axes:
        raise ValueError("up_axes is empty")
    scored = []
    for axis in axes:
        for k in range(4):
            phi = theta0 + k * np.pi / 2
            sil = render_silhouette(mesh, axis, phi, size)
            scored.append(Candidate(tuple(axis), float(np.degrees(phi) % 360), normalised_iou(sil, target)))
    ranked = sorted(scored, key=lambda c: -c.score)  # stable: ties keep enumeration order
    best, second = ranked[0], ranked[1]
    ties = sum(abs(c.score - best.score) <= TIE_TOL for c in ranked[1:])
    margin = best.score - second.score
    return Comparison(
        ranked, best, second, margin, ties,
        label=symmetry_label(margin, footprint_aspect), needs_review=margin < REVIEW_MARGIN,
    )


# ------------------------------------------------------------ contracts adapters

# The six signed up-axes, in the order CandidateId.up_axis_idx indexes them (contracts.py, and
# geo.outline.UP_AXIS_CANDIDATES; tests/test_azimuth_contract.py asserts they agree). Duplicated rather
# than imported: perception must not import geo/.
UP_AXIS_CANDIDATES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


@dataclass(frozen=True)
class PhotographedSide:
    """Which side of the mesh the photo shows: camera azimuth 90 * index degrees, CCW from the mesh front seen
    from above (the canonical-frame angle MeshOutline.front_angle + 90 * index deg)."""

    index: int  # 0..3
    score: float
    margin: float  # best minus second-best side; ~0 when the silhouette cannot tell (a symmetric mesh)
    scores: tuple


def side_scores(mesh, mask, up_axis=(0, 1, 0), size=CANVAS_PX):
    """Normalised IoU of the mask against the mesh seen from each of its four sides (camera azimuth 0, 90, 180,
    270 about `up_axis`; 0 is the mesh front)."""
    mesh, target = _as_mesh(mesh), normalise_mask(_as_mask(mask), size)
    return tuple(normalised_iou(render_silhouette(mesh, up_axis, k * np.pi / 2, size), target) for k in range(4))


def photographed_side(mesh, mask, up_axis=(0, 1, 0), size=CANVAS_PX):
    scores = side_scores(mesh, mask, up_axis, size)
    order = sorted(range(4), key=lambda k: -scores[k])  # stable: ties keep the lower index
    return PhotographedSide(order[0], scores[order[0]], scores[order[0]] - scores[order[1]], scores)


def margin_of(scores):
    """Top-two gap of a {CandidateId: score} dict; 0.0 with fewer than two entries. Addendum A.3 feeds this to
    the confidence gate; keep it separate from the footprint-IoU margin in FitResult."""
    top = sorted(scores.values(), reverse=True)
    return float(top[0] - top[1]) if len(top) > 1 else 0.0


def score_silhouettes(mesh, mask, candidates, *, backend=None, size=CANVAS_PX):
    """contracts.ScoreSilhouettes: normalised silhouette IoU per CandidateId(up_axis_idx, azimuth_k).

    Depends on the up-axis only and is flat across azimuth_k (see the module docstring): margin_of() is 0 for
    candidates that share an up-axis, so PhotoEvidence.has_silhouette_evidence is False and geo.disambiguate
    abstains on this cue, which is correct. A None mesh, or the stub backend, abstains for every candidate.
    """
    if mesh is None or resolve_backend(backend) == STUB:
        return {c: 0.5 for c in candidates}
    mesh, target = _as_mesh(mesh), normalise_mask(_as_mask(mask), size)
    by_up = {}
    for c in candidates:
        if c.up_axis_idx not in by_up:
            axis = UP_AXIS_CANDIDATES[c.up_axis_idx]
            by_up[c.up_axis_idx] = normalised_iou(render_silhouette(mesh, axis, 0.0, size), target)
    return {c: by_up[c.up_axis_idx] for c in candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mesh")
    parser.add_argument("mask")
    parser.add_argument("--top", type=int, default=8, help="how many candidates to print")
    parser.add_argument("--footprint-aspect", type=float, help="footprint a_F/b_F; enables the arbitrary_symmetric label")
    args = parser.parse_args()
    result = render_compare(trimesh.load(args.mesh, force="mesh"), args.mask, footprint_aspect=args.footprint_aspect)
    for c in result.candidates[: args.top]:
        print(f"up={c.up_axis!s:<14} azimuth={c.azimuth_deg:5.1f}  score={c.score:.3f}")
    print(f"margin={result.margin:.3f}  ties_with_best={result.ties_with_best}  label={result.label}  needs_review={result.needs_review}")
    if result.needs_review:
        why = "arbitrary_symmetric (A.4)" if result.label else f"margin < {REVIEW_MARGIN} (SPEC 6.6)"
        print(f"route to review: {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
