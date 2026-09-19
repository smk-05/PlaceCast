"""
Synthetic FitResult rows with hand-set labels. Addendum B.2 / F.4.

Purpose: let confidence/features.py and confidence/train.py be written, run and
tested before a single real benchmark row exists. When scripts/run_benchmark.py
produces the real 20-building sweep at hour 30, the data source swaps and the
training code is already correct.

These are NOT training data for the shipped model. Addendum B.2's real set is
20 buildings x 4 ablation configurations = 80 labelled placements, plus every
logged debug run, plus deliberate corruptions as free negatives. Do not report
numbers fitted on this file.

The generators below deliberately reproduce the interaction addendum B.1 uses to
justify the model over the table:
  - IoU 0.72 on a highly rectilinear footprint with a large rotation margin is a
    GOOD fit that the table sends to review.
  - IoU 0.78 on a round building with a 0.02 rotation margin is a coin-flip the
    table auto-accepts.
"""

from __future__ import annotations

import numpy as np

from contracts import (
    OMBB,
    CandidateId,
    Disambiguator,
    FitResult,
    Footprint,
    HeightSource,
)


def _ombb(a: float, b: float, angle: float = 0.0) -> OMBB:
    u = np.array([np.cos(angle), np.sin(angle)])
    return OMBB(centre=np.zeros(2), u=u, v=np.array([-u[1], u[0]]), a=a, b=b)


def make_footprint(rectilinearity: float, aspect: float = 1.6,
                   area: float = 800.0) -> Footprint:
    a = float(np.sqrt(area * aspect))
    b = float(area / a)
    pts = np.array([[0, 0], [a, 0], [a, b], [0, b]], dtype=float)
    return Footprint(
        pts_enu=pts,
        rectilinearity=rectilinearity,
        principal_angle=0.0,
        ombb=_ombb(max(a, b), min(a, b)),
        area_m2=area,
    )


def make_fit(*, iou: float, hausdorff: float, area_ratio: float = 1.0,
             margin: float = 0.15, aniso: float = 0.0,
             neighbour: float = 0.0, mirrored: bool = False,
             authoritative_height: bool = True) -> FitResult:
    return FitResult(
        theta=0.4,
        scale_x=-1.0 if mirrored else 1.0,
        scale_y=1.0,
        scale_z=1.0,
        tx=0.0, ty=0.0, z_offset=0.0,
        iou=iou,
        hausdorff_m=hausdorff,
        area_ratio=area_ratio,
        rotation_margin_footprint=margin,
        anisotropy_log_ratio=aniso,
        max_neighbor_overlap=neighbour,
        height_m=14.0,
        height_source=(HeightSource.OSM_HEIGHT if authoritative_height
                       else HeightSource.PROPORTIONAL_FALLBACK),
        disambiguated_by=Disambiguator.EXIF_HEADING,
        solver="ombb+icp+nelder-mead",
        candidate_ious=tuple(
            (CandidateId(4, k), max(iou - k * margin, 0.0)) for k in range(4)
        ),
    )


def generate(n: int = 140, seed: int = 42) -> list[tuple[FitResult, Footprint, int]]:
    """-> [(fit, footprint, label)] with label 1 = a human would accept.

    Addendum B.2's labelling protocol is BINARY, collapsing "review" and
    "reject" into "not acceptable": the three-way decision is recovered from the
    probability threshold, and a three-class model on ~120 rows is
    over-parameterised.

    Class balance is deliberately imbalanced toward accept (~65%), because
    addendum B.4 warns that a working solver produces exactly that and the model
    will otherwise learn "always accept". Use class_weight='balanced'.
    """
    rng = np.random.default_rng(seed)
    rows: list[tuple[FitResult, Footprint, int]] = []

    for _ in range(int(n * 0.55)):  # clean accepts
        rect = float(rng.uniform(0.80, 0.98))
        fit = make_fit(
            iou=float(rng.uniform(0.78, 0.95)),
            hausdorff=float(rng.uniform(0.3, 1.9)),
            area_ratio=float(rng.normal(1.0, 0.04)),
            margin=float(rng.uniform(0.10, 0.35)),
        )
        rows.append((fit, make_footprint(rect), 1))

    for _ in range(int(n * 0.12)):  # good fits the TABLE sends to review
        rect = float(rng.uniform(0.88, 0.98))
        fit = make_fit(
            iou=float(rng.uniform(0.68, 0.75)),
            hausdorff=float(rng.uniform(1.8, 2.6)),
            area_ratio=float(rng.normal(1.0, 0.05)),
            margin=float(rng.uniform(0.22, 0.40)),
        )
        rows.append((fit, make_footprint(rect), 1))

    for _ in range(int(n * 0.12)):  # coin-flips the TABLE auto-accepts
        rect = float(rng.uniform(0.40, 0.58))
        fit = make_fit(
            iou=float(rng.uniform(0.76, 0.84)),
            hausdorff=float(rng.uniform(1.0, 2.0)),
            margin=float(rng.uniform(0.005, 0.03)),
        )
        rows.append((fit, make_footprint(rect, aspect=1.03), 0))

    for _ in range(int(n * 0.16)):  # honest failures
        rect = float(rng.uniform(0.45, 0.85))
        fit = make_fit(
            iou=float(rng.uniform(0.15, 0.48)),
            hausdorff=float(rng.uniform(4.0, 14.0)),
            area_ratio=float(rng.choice([rng.uniform(0.4, 0.68),
                                         rng.uniform(1.32, 2.1)])),
            margin=float(rng.uniform(0.0, 0.08)),
            neighbour=float(rng.uniform(0.0, 0.25)),
            authoritative_height=False,
        )
        rows.append((fit, make_footprint(rect), 0))

    # Addendum B.2's free negatives: deliberate corruptions. A mirrored mesh, a
    # 2x scale error, a known-wrong orientation. These populate the reject class
    # the benchmark otherwise under-samples.
    for _ in range(max(n - len(rows), 0)):
        kind = rng.integers(0, 3)
        if kind == 0:
            fit = make_fit(iou=0.55, hausdorff=3.0, mirrored=True, margin=0.2)
        elif kind == 1:
            fit = make_fit(iou=0.31, hausdorff=9.0, area_ratio=2.0, margin=0.2)
        else:
            fit = make_fit(iou=0.22, hausdorff=11.0, margin=0.001)
        rows.append((fit, make_footprint(float(rng.uniform(0.5, 0.95))), 0))

    rng.shuffle(rows)
    return rows
