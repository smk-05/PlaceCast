"""
A hardcoded PhotoEvidence so geo/ can be exercised before segmentation exists.
Addendum F.4, reversed direction.

Three presets, matching the three cases geo/disambiguate.py has to handle:

  no_evidence()        margin 0.0 — the cue abstains, Filters 1-3 must decide
  confident(k)         margin 0.22 — silhouette decides, above the 0.08 gate
  with_exif(heading)   EXIF present — the arithmetic path wins outright
"""

from __future__ import annotations

import numpy as np

from contracts import CandidateId, PhotoEvidence


def _box_mask(h: int = 480, w: int = 640, frac: float = 0.55) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    mh, mw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    m[mh:h - mh, mw:w - mw] = True
    return m


def no_evidence(up_axis_idx: int = 4) -> PhotoEvidence:
    """The default. A flat distribution means "this cue has nothing to say"."""
    return PhotoEvidence(
        mask=_box_mask(),
        mask_area_frac=0.30,
        occlusion_flag=False,
        silhouette_scores={CandidateId(up_axis_idx, k): 0.5 for k in range(4)},
        silhouette_margin=0.0,
        segmentation_model="fixture:none",
    )


def confident(correct_k: int = 0, up_axis_idx: int = 4,
              margin: float = 0.22) -> PhotoEvidence:
    """Silhouette strongly prefers `correct_k`. Above the 0.08 fusion gate."""
    scores = {CandidateId(up_axis_idx, k): 0.55 for k in range(4)}
    scores[CandidateId(up_axis_idx, correct_k)] = 0.55 + margin
    return PhotoEvidence(
        mask=_box_mask(),
        mask_area_frac=0.30,
        silhouette_scores=scores,
        silhouette_margin=margin,
        segmentation_model="fixture:confident",
    )


def with_exif(heading_deg: float = 142.5,
              gps: tuple[float, float] | None = None,
              pitch_deg: float | None = 8.0) -> PhotoEvidence:
    """EXIF present — spec 6.6 Filter 2 turns inference into arithmetic."""
    ev = no_evidence()
    return PhotoEvidence(
        mask=ev.mask,
        mask_area_frac=ev.mask_area_frac,
        exif_heading_deg=heading_deg,
        exif_pitch_deg=pitch_deg,
        exif_gps=gps,
        silhouette_scores=ev.silhouette_scores,
        silhouette_margin=0.0,
        segmentation_model="fixture:exif",
    )


def occluded() -> PhotoEvidence:
    """Addendum A.4: trees/vehicles. Mask is a fragment, not the building."""
    m = _box_mask(frac=0.55)
    m[:, 200:280] = False   # a tree trunk through the facade
    m[300:, 400:520] = False  # a parked car
    return PhotoEvidence(
        mask=m,
        mask_area_frac=float(m.mean()),
        occlusion_flag=True,
        silhouette_scores={CandidateId(4, k): 0.42 for k in range(4)},
        silhouette_margin=0.01,
        segmentation_model="fixture:occluded",
    )
