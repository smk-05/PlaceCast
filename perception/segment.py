"""
Building segmentation. Addendum A.3 Stage 1.

CURRENT: a non-ML stand-in. The input photographs are photos *of a building*, so
the building is centred and dominant by construction — the same observation that
justifies addendum A.2's own fallback ("SAM 2 alone with a box prompt covering
the central 80% of the image"). This stub takes that one step further and uses
the box directly, with no model at all.

TO REPLACE (perception contributor): Grounding DINO with the text prompt
"building. house. facade." supplies a box; SAM 2 refines it to a pixel-accurate
mask. Both are in HuggingFace transformers. Keep this signature:

    segment_building(image_path) -> (mask HxW bool, mask_area_frac, occluded)

Then clean() the result: largest connected component, morphological close with a
~5 px kernel to seal window gaps, fill interior holes. A facade mask with
windows punched out of it will not match a render.

Do NOT fine-tune. These models are used zero-shot; fine-tuning a segmentation
model on building facades inside the hackathon window loses Saturday.
Do NOT use pyrender or any OpenGL — offscreen GL on Windows is a trap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Addendum A.4 detection thresholds, applied to whatever produced the mask.
MIN_AREA_FRAC = 0.04       # tiny mask -> occluded or wrong object
MAX_AREA_FRAC = 0.90       # covers the frame -> segmented the streetscape
CENTRAL_BOX_FRAC = 0.80    # addendum A.2 fallback


def segment_building(image_path: Path) -> tuple[np.ndarray, float, bool]:
    """-> (binary mask, mask_area_frac, occlusion_flag). Non-ML stand-in."""
    with Image.open(image_path) as im:
        w, h = im.size

    mask = np.zeros((h, w), dtype=bool)
    mx = int(w * (1.0 - CENTRAL_BOX_FRAC) / 2.0)
    my = int(h * (1.0 - CENTRAL_BOX_FRAC) / 2.0)
    mask[my:h - my, mx:w - mx] = True

    frac = float(mask.mean())
    return mask, frac, False


def clean(mask: np.ndarray, close_px: int = 5) -> np.ndarray:
    """Largest component, morphological close, fill holes. Addendum A.3.

    Kept here rather than inside the model call so the real implementation and
    the stand-in are post-processed identically.
    """
    m = ndimage.binary_closing(mask, structure=np.ones((close_px, close_px)))
    m = ndimage.binary_fill_holes(m)
    labels, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, labels, range(1, n + 1))
        m = labels == (int(np.argmax(sizes)) + 1)
    return m.astype(bool)


def assess_mask(mask: np.ndarray) -> tuple[float, bool, str]:
    """Addendum A.4's failure detection. -> (area_frac, occluded, note).

    Runs on any mask, whichever component produced it — that is the point.
    """
    frac = float(mask.mean())

    if frac > MAX_AREA_FRAC:
        return frac, True, "mask covers >90% of frame — segmented the streetscape"
    if frac < MIN_AREA_FRAC:
        return frac, True, "mask tiny relative to frame — occluded or wrong object"

    # Touching more than three image edges means the subject is not bounded.
    edges = sum([bool(mask[0].any()), bool(mask[-1].any()),
                 bool(mask[:, 0].any()), bool(mask[:, -1].any())])
    if edges > 3:
        return frac, True, "mask touches >3 image edges"

    # High boundary complexity relative to area indicates tree/vehicle occlusion
    # chewing holes in the silhouette (addendum A.4).
    filled = ndimage.binary_fill_holes(mask)
    perimeter = float(np.abs(np.diff(filled.astype(int), axis=0)).sum()
                      + np.abs(np.diff(filled.astype(int), axis=1)).sum())
    area = max(float(filled.sum()), 1.0)
    complexity = perimeter / np.sqrt(area)
    if complexity > 12.0:
        return frac, True, f"boundary complexity {complexity:.1f} — likely occlusion"

    return frac, False, ""
