"""Feature vector for the learned confidence gate (ML_ADDENDUM B.3).

    x = feature_vector(fit, photo, fp)

Inputs are the three contracts.py objects and nothing else: everything B.3 asks for is on them
(contracts.py v1.0 added `Footprint.geocode_rooftop` and `FitResult.height_source_authoritative` for
exactly this). Nothing is defaulted: a missing attribute raises AttributeError, and a non-finite value or
a non-positive area ratio raises ValueError.

Feature table (B.3), with the expected sign the fitted coefficient is compared against:

  footprint_iou                +   fit.iou                                  SPEC 9.1
  hausdorff_m                  -   fit.hausdorff_m                          SPEC 9.1
  abs_area_ratio_log           -   |log(fit.area_ratio)|                    SPEC 9.1
  rotation_margin_footprint    +   fit.rotation_margin_footprint            SPEC 6.6
  rotation_margin_silhouette   +   photo.silhouette_margin                  A.3
  rectilinearity               +   fp.rectilinearity                        SPEC 4.2
  abs_anisotropy_log_ratio     -   |fit.anisotropy_log_ratio|               SPEC 6.9
  max_neighbor_overlap         -   fit.max_neighbor_overlap                 SPEC 9.2
  geocode_rooftop              +   fp.geocode_rooftop                        SPEC 3.1
  height_source_authoritative  +   fit.height_source_authoritative          SPEC 7 (OSM tags only)

The table's `area_ratio_log` is used as |log(ratio)|: log makes it symmetric about 1 (0.5x and 2x
are equally wrong) and the "||.|| -" sign means the magnitude is what hurts, which a linear model
can only express on the absolute value. The same goes for the sign-arbitrary anisotropy log-ratio.

Deliberately NOT features: footprint match quality (`fp.is_weak_match`), multi-part footprints,
mirrored fits and `fit.exif_silhouette_disagree` are hard rules in confidence/gate.py. They are rare and
categorical, a 10-feature model on ~100 rows cannot learn them, and a wrong-building match must never
depend on a fitted coefficient.
"""
from __future__ import annotations

import math

import numpy as np

FEATURE_NAMES = (
    "footprint_iou",
    "hausdorff_m",
    "abs_area_ratio_log",
    "rotation_margin_footprint",
    "rotation_margin_silhouette",
    "rectilinearity",
    "abs_anisotropy_log_ratio",
    "max_neighbor_overlap",
    "geocode_rooftop",
    "height_source_authoritative",
)

EXPECTED_SIGN = {
    "footprint_iou": +1,
    "hausdorff_m": -1,
    "abs_area_ratio_log": -1,
    "rotation_margin_footprint": +1,
    "rotation_margin_silhouette": +1,
    "rectilinearity": +1,
    "abs_anisotropy_log_ratio": -1,
    "max_neighbor_overlap": -1,
    "geocode_rooftop": +1,
    "height_source_authoritative": +1,
}

# B.4 "too few rows" response: 4 features on 60 rows is defensible, 10 is not.
REDUCED_FEATURES = ("footprint_iou", "hausdorff_m", "rectilinearity", "rotation_margin_footprint")

assert set(EXPECTED_SIGN) == set(FEATURE_NAMES) and set(REDUCED_FEATURES) <= set(FEATURE_NAMES)


def _finite(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"feature {name} is not finite: {value!r}")
    return value


def feature_dict(fit, photo, fp):
    """All ten B.3 features, by name."""
    area_ratio = _finite("area_ratio", fit.area_ratio)
    if area_ratio <= 0:
        raise ValueError(f"fit.area_ratio must be positive to take its log, got {area_ratio!r}")
    return {
        "footprint_iou": _finite("footprint_iou", fit.iou),
        "hausdorff_m": _finite("hausdorff_m", fit.hausdorff_m),
        "abs_area_ratio_log": abs(math.log(area_ratio)),
        "rotation_margin_footprint": _finite("rotation_margin_footprint", fit.rotation_margin_footprint),
        "rotation_margin_silhouette": _finite("rotation_margin_silhouette", photo.silhouette_margin),
        "rectilinearity": _finite("rectilinearity", fp.rectilinearity),
        "abs_anisotropy_log_ratio": abs(_finite("anisotropy_log_ratio", fit.anisotropy_log_ratio)),
        "max_neighbor_overlap": _finite("max_neighbor_overlap", fit.max_neighbor_overlap),
        "geocode_rooftop": float(bool(fp.geocode_rooftop)),
        "height_source_authoritative": float(bool(fit.height_source_authoritative)),
    }


def _select(values, names):
    unknown = [n for n in names if n not in values]
    if unknown:
        raise KeyError(f"unknown feature(s) {unknown}; known: {list(FEATURE_NAMES)}")
    return np.array([values[n] for n in names], dtype=float)


def feature_vector(fit, photo, fp, *, names=FEATURE_NAMES):
    """Features as a 1-D float array in `names` order."""
    return _select(feature_dict(fit, photo, fp), names)


def feature_matrix(rows, names=FEATURE_NAMES):
    """(n_rows, len(names)) matrix from labelled rows exposing .fit .photo .footprint
    (e.g. fixtures.fake_fits.LabelledFit)."""
    return np.vstack([feature_vector(r.fit, r.photo, r.footprint, names=names) for r in rows])
