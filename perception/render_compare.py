"""
Silhouette render-and-compare. Addendum A.3 Stages 2-3.

CURRENT: a non-ML stand-in returning uniform scores and a zero margin, which
forces geo/disambiguate.py to carry the decision on Filters 1-3. That is the
correct default: a zero margin means "this cue has nothing to say", not "all
orientations are equally good".

TO REPLACE (perception contributor). Two things matter more than the model:

1. ORDERING. Addendum D.1 makes this a hard constraint: geo 6.4's bas-relief
   rejection runs BEFORE this. A flat mesh's silhouette matches well from its one
   good view and is garbage from all others — it will PASS this test while being
   catastrophically wrong. Read MeshOutline.is_bas_relief and refuse to score it.

2. NORMALISED comparison, not raw masks. A street-level photograph is
   perspective, taken from roughly eye height and usually tilted up. An
   orthographic render from horizontal is a different projection of the same
   object and the silhouettes will not align even when the orientation is
   correct. Normalise each silhouette by its bounding box — centre and scale to
   a common box — then IoU. That discards absolute position and scale, which are
   exactly what the projection mismatch corrupts, and keeps aspect and profile,
   which are what discriminate orientation.

No pyrender. Rasterise triangles into a numpy array directly; offscreen OpenGL
on Windows has no OSMesa and no EGL and will cost an hour.
"""

from __future__ import annotations

import numpy as np

from contracts import CandidateId


def score_silhouettes(mesh, mask: np.ndarray,
                      candidates: list[CandidateId]) -> dict[CandidateId, float]:
    """-> normalised silhouette IoU per candidate. Non-ML stand-in.

    Returns a flat distribution so `PhotoEvidence.has_silhouette_evidence` is
    False (margin 0.0 < the 0.08 threshold) and the fusion in
    geo.disambiguate.choose_orientation skips this cue entirely.
    """
    return {c: 0.5 for c in candidates}


def margin_of(scores: dict[CandidateId, float]) -> float:
    """Top-two gap. Addendum A.3 feeds this to the confidence gate as a feature.

    Kept SEPARATE from the footprint-IoU margin in FitResult. They are
    independent evidence, and disagreement between them is itself a strong
    signal that something is wrong.
    """
    if len(scores) < 2:
        return 0.0
    ranked = sorted(scores.values(), reverse=True)
    return float(ranked[0] - ranked[1])


def normalised_iou(sil_a: np.ndarray, sil_b: np.ndarray,
                   size: int = 128) -> float:
    """Shape-descriptor IoU: crop to bounding box, rescale, compare.

    Provided now so the real implementation has the comparison already written
    and tested — this part needs no GPU and no model.
    """
    a, b = _normalise(sil_a, size), _normalise(sil_b, size)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _normalise(sil: np.ndarray, size: int) -> np.ndarray:
    """Crop a binary silhouette to its bounding box and resample to size x size."""
    ys, xs = np.nonzero(sil)
    if len(ys) == 0:
        return np.zeros((size, size), dtype=bool)
    crop = sil[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    yi = (np.arange(size) * h // size).clip(0, h - 1)
    xi = (np.arange(size) * w // size).clip(0, w - 1)
    return crop[np.ix_(yi, xi)].astype(bool)
