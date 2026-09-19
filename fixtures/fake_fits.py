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

Two generators live here. `generate()` (above all: `make_fit`, `make_footprint`) is the original, used by
the geometry half's tests and the smoke test; it returns (fit, footprint, label) tuples with a ~65% accept
rate. `generate_labelled()` (bottom of the file) is what confidence/train.py trains on: full B.2 layout
(20 buildings x 4 solver configs + deliberate corruptions), a PhotoEvidence per row, building strata, and
labels drawn BEFORE the metrics so they are not contaminated by the features (B.4).

The generators below deliberately reproduce the interaction addendum B.1 uses to
justify the model over the table:
  - IoU 0.72 on a highly rectilinear footprint with a large rotation margin is a
    GOOD fit that the table sends to review.
  - IoU 0.78 on a round building with a 0.02 rotation margin is a coin-flip the
    table auto-accepts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from contracts import (
    OMBB,
    CandidateId,
    Disambiguator,
    FitResult,
    Footprint,
    HeightSource,
    PhotoEvidence,
)


def _ombb(a: float, b: float, angle: float = 0.0) -> OMBB:
    u = np.array([np.cos(angle), np.sin(angle)])
    return OMBB(centre=np.zeros(2), u=u, v=np.array([-u[1], u[0]]), a=a, b=b)


def make_footprint(rectilinearity: float, aspect: float = 1.6,
                   area: float = 800.0,
                   match_quality: str = "contained_and_named",
                   geocode_location_type: str = "ROOFTOP") -> Footprint:
    """A synthetic footprint.

    `match_quality` defaults to the strong case. It is NOT cosmetic: an empty
    value means "unverified" and routes to review on its own, because a footprint
    accepted on thin evidence taints every metric computed against it. A high IoU
    against the wrong building is the most dangerous output this pipeline can
    produce, since it looks exactly like success.
    """
    a = float(np.sqrt(area * aspect))
    b = float(area / a)
    pts = np.array([[0, 0], [a, 0], [a, b], [0, b]], dtype=float)
    return Footprint(
        pts_enu=pts,
        rectilinearity=rectilinearity,
        principal_angle=0.0,
        ombb=_ombb(max(a, b), min(a, b)),
        area_m2=area,
        match_quality=match_quality,
        geocode_location_type=geocode_location_type,
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

    # The dangerous class: a GOOD-LOOKING fit against a footprint that was
    # matched on thin evidence. Every metric reads healthy and the building may
    # simply be the wrong one — observed live, where a demolished Randolph Hall
    # selected a 360 m2 wind tunnel 46 m away. `match_quality` is the only
    # feature that separates these from genuine accepts, so the training set has
    # to contain them or the model cannot learn to distrust them.
    for _ in range(int(n * 0.08)):
        fit = make_fit(
            iou=float(rng.uniform(0.74, 0.90)),
            hausdorff=float(rng.uniform(0.8, 2.4)),
            area_ratio=float(rng.normal(1.0, 0.06)),
            margin=float(rng.uniform(0.10, 0.30)),
        )
        rows.append((fit, make_footprint(float(rng.uniform(0.75, 0.98)),
                                         match_quality="unnamed_sole_candidate"), 0))

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


# ==========================================================================
# generate_labelled(): the training set for confidence/train.py
# ==========================================================================

@dataclass(frozen=True)
class LabelledFit:
    fit: FitResult
    photo: PhotoEvidence
    footprint: Footprint
    label: int  # 1 = a human would accept this placement into the game world, 0 = not
    kind: str  # what the generator drew before label noise
    building_id: int
    config_index: int | None  # ablation config, None for a deliberate corruption
    label_flipped: bool


STRATA = ("rectangular", "complex", "near_square", "sloped")
CONFIG_QUALITY = (0.0, 0.35, 0.7, 0.9)  # ablation: OMBB only, +Umeyama, +IoU refinement, +silhouette
CORRUPTIONS = ("corrupt_wrong_orientation", "corrupt_mirrored", "corrupt_scale_2x")
BAD_BENCHMARK_KINDS = (("poor_fit", 0.35), ("wrong_orientation", 0.35), ("borderline", 0.30))


def _clip(x, lo, hi):
    return float(min(max(x, lo), hi))


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _lognormal(rng, median, sigma):
    return float(rng.lognormal(math.log(median), sigma))


@dataclass(frozen=True)
class _Building:
    building_id: int
    stratum: str
    rectilinearity: float
    aspect: float
    area_m2: float
    has_holes: bool
    geocode_rooftop: bool
    height_authoritative: bool
    rot_margin_base: float  # how decisive the footprint's four OMBB candidates are
    sil_margin_base: float  # how decisive the photo silhouette is


def _make_building(rng, building_id):
    stratum = STRATA[building_id % len(STRATA)]
    r_lo, r_hi = {"rectangular": (0.92, 0.99), "complex": (0.72, 0.90), "near_square": (0.90, 0.98), "sloped": (0.85, 0.97)}[stratum]
    near_square = stratum == "near_square"
    return _Building(
        building_id=building_id,
        stratum=stratum,
        rectilinearity=float(rng.uniform(r_lo, r_hi)),
        aspect=float(rng.uniform(1.0, 1.09) if near_square else rng.uniform(1.3, 3.0)),
        area_m2=float(rng.uniform(400, 4000)),
        has_holes=bool(rng.random() < (0.3 if stratum == "complex" else 0.05)),
        geocode_rooftop=bool(rng.random() < 0.75),
        height_authoritative=bool(rng.random() < 0.65),
        rot_margin_base=float(rng.uniform(0.0, 0.05) if near_square else rng.uniform(0.12, 0.5)),
        sil_margin_base=float(rng.uniform(0.0, 0.04) if near_square else rng.uniform(0.05, 0.35)),
    )


def _p_acceptable(b, quality):
    """Chance a placement of this building at this solver quality is one a human accepts."""
    z = -1.4 + 3.2 * quality + 5.0 * (b.rectilinearity - 0.9) + 0.5 * b.geocode_rooftop
    z += 0.3 * b.height_authoritative - 0.8 * (b.stratum == "near_square")
    return _sigmoid(z)


def _metrics(rng, kind, b):
    """Metric draws for one placement of `kind` on building `b` (signed log quantities)."""
    c = 0.95 - b.rectilinearity  # complexity: hurts even good fits a little
    sign = 1.0 if rng.random() < 0.5 else -1.0
    rot, sil = b.rot_margin_base, b.sil_margin_base
    if kind == "good":
        m = dict(iou=rng.normal(0.87 - 0.5 * c, 0.045), haus=_lognormal(rng, 1.5 + 4 * c, 0.3), arl=rng.normal(0, 0.05),
                 aniso=rng.normal(0, 0.035), overlap=rng.beta(1, 80), rot=rot * rng.uniform(0.8, 1.2), sil=sil * rng.uniform(0.7, 1.3))
    elif kind == "borderline":
        m = dict(iou=rng.normal(0.79, 0.05), haus=_lognormal(rng, 2.6, 0.3), arl=rng.normal(0, 0.10),
                 aniso=rng.normal(0, 0.07), overlap=rng.beta(1, 40), rot=rot * rng.uniform(0.5, 1.0), sil=sil * rng.uniform(0.4, 1.0))
    elif kind == "poor_fit":
        m = dict(iou=rng.normal(0.68, 0.07), haus=_lognormal(rng, 3.6, 0.3), arl=sign * rng.normal(0.28, 0.10),
                 aniso=rng.normal(0, 0.14), overlap=rng.beta(1, 15), rot=rot * rng.uniform(0.3, 0.9), sil=sil * rng.uniform(0.2, 0.8))
    elif kind == "wrong_orientation":
        m = dict(iou=rng.normal(0.60, 0.08), haus=_lognormal(rng, 4.5, 0.3), arl=rng.normal(0, 0.15),
                 aniso=rng.normal(0, 0.08), overlap=rng.beta(1, 25), rot=rng.uniform(0, 0.05), sil=rng.uniform(0, 0.05))
    elif kind == "corrupt_wrong_orientation":
        m = dict(iou=rng.normal(0.50, 0.08), haus=_lognormal(rng, 6.0, 0.3), arl=rng.normal(0, 0.10),
                 aniso=rng.normal(0, 0.05), overlap=rng.beta(1, 20), rot=rng.uniform(0, 0.06), sil=rng.uniform(0, 0.06))
    elif kind == "corrupt_mirrored":
        m = dict(iou=rng.normal(0.45, 0.09), haus=_lognormal(rng, 7.0, 0.3), arl=rng.normal(0, 0.12),
                 aniso=rng.normal(0, 0.06), overlap=rng.beta(1, 15), rot=rng.uniform(0, 0.10), sil=rng.uniform(0, 0.10))
    elif kind == "corrupt_scale_2x":
        m = dict(iou=rng.normal(0.32, 0.07), haus=_lognormal(rng, 8.0, 0.3), arl=sign * (math.log(2) + rng.normal(0, 0.05)),
                 aniso=rng.normal(0, 0.03), overlap=rng.beta(1, 8), rot=rot, sil=sil)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return dict(
        iou=_clip(m["iou"], 0.05, 0.99), haus=max(0.05, m["haus"]), arl=float(m["arl"]), aniso=float(m["aniso"]),
        overlap=_clip(m["overlap"], 0.0, 1.0), rot=_clip(m["rot"], 0.0, 1.0), sil=_clip(m["sil"], 0.0, 1.0),
    )


def _disambiguated_by(rng, kind):
    probs = {"good": (0.25, 0.55, 0.20), "borderline": (0.2, 0.4, 0.4), "poor_fit": (0.2, 0.4, 0.4)}.get(kind, (0.1, 0.3, 0.6))
    return Disambiguator(str(rng.choice(["exif_heading", "silhouette", "road_normal"], p=probs)))


# The next three draw no random numbers, so adding contract fields did not change the random stream.
def _height_source(b):
    if b.height_authoritative:
        return HeightSource.OSM_HEIGHT if b.building_id % 2 == 0 else HeightSource.OSM_LEVELS
    return HeightSource.MICROSOFT if b.building_id % 3 == 0 else HeightSource.PROPORTIONAL_FALLBACK


def _location_type(b):
    return "ROOFTOP" if b.geocode_rooftop else ("RANGE_INTERPOLATED" if b.building_id % 2 == 0 else "APPROXIMATE")


def _match_quality(b):
    return "contained_and_named" if b.building_id % 2 == 0 else "contained"


def _build_row(rng, b, kind, label, config_index, label_flipped):
    m = _metrics(rng, kind, b)
    # Scales coherent with the metrics: area_ratio = s_x * s_y and anisotropy = log(s_x / s_y).
    log_sx, log_sy = m["arl"] / 2 + m["aniso"] / 2, m["arl"] / 2 - m["aniso"] / 2
    theta = float(rng.uniform(0, 2 * math.pi))
    tx, ty, z_offset = float(rng.normal(0, 0.8)), float(rng.normal(0, 0.8)), float(rng.uniform(600, 660))
    disambiguated_by = _disambiguated_by(rng, kind)
    mirrored = kind == "corrupt_mirrored"  # a reflection: FitResult.is_mirrored, which the gate rejects outright
    fit = FitResult(
        theta=theta, scale_x=-math.exp(log_sx) if mirrored else math.exp(log_sx), scale_y=math.exp(log_sy), scale_z=1.0,
        tx=tx, ty=ty, z_offset=z_offset,
        iou=m["iou"], hausdorff_m=m["haus"], area_ratio=math.exp(m["arl"]), rotation_margin_footprint=m["rot"],
        anisotropy_log_ratio=m["aniso"], max_neighbor_overlap=m["overlap"],
        height_m=14.0, height_source=_height_source(b), disambiguated_by=disambiguated_by, solver="ombb+icp+nelder-mead",
        candidate_ious=tuple((CandidateId(2, k), max(m["iou"] - k * m["rot"], 0.0)) for k in range(4)),
    )
    # Hardcoded-style mask: a centred rectangle covering roughly a third to two thirds of the frame.
    side = int(round(32 * math.sqrt(rng.uniform(0.25, 0.7))))
    mask = np.zeros((32, 32), bool)
    lo = (32 - side) // 2
    mask[lo : lo + side, lo : lo + side] = True
    best = float(rng.uniform(0.65, 0.95))
    second = max(0.0, best - m["sil"])
    scores = {CandidateId(2, k): max(0.0, second - float(rng.uniform(0, 0.2))) for k in range(4)}
    best_k, second_k = rng.choice(4, size=2, replace=False)
    scores[CandidateId(2, int(best_k))], scores[CandidateId(2, int(second_k))] = best, second
    has_exif = rng.random() < 0.4
    photo = PhotoEvidence(
        mask=mask, mask_area_frac=float(mask.mean()), occlusion_flag=bool(rng.random() < 0.1),
        exif_heading_deg=float(rng.uniform(0, 360)) if has_exif else None,
        exif_pitch_deg=float(rng.normal(8, 6)) if has_exif or rng.random() < 0.4 else None,
        silhouette_scores=scores, silhouette_margin=best - second,
    )
    short = math.sqrt(b.area_m2 / b.aspect)
    a, s = b.aspect * short, short
    pts = np.array([[-a / 2, -s / 2], [a / 2, -s / 2], [a / 2, s / 2], [-a / 2, s / 2]])
    courtyard = np.array([[-a / 6, -s / 6], [a / 6, -s / 6], [a / 6, s / 6], [-a / 6, s / 6]])
    footprint = Footprint(
        pts_enu=pts, holes_enu=(courtyard,) if b.has_holes else (), rectilinearity=b.rectilinearity,
        ombb=OMBB(centre=np.zeros(2), u=np.array([1.0, 0.0]), v=np.array([0.0, 1.0]), a=a, b=s),
        area_m2=b.area_m2, match_quality=_match_quality(b), geocode_location_type=_location_type(b),
    )
    return LabelledFit(fit, photo, footprint, int(label), kind, b.building_id, config_index, label_flipped)


def generate_labelled(n_buildings=20, n_configs=4, n_corruptions=40, seed=0, label_noise=0.05):
    """Deterministic for a given seed. n_buildings * n_configs + n_corruptions LabelledFit rows (B.2).

    Buildings cycle the four strata (rectangular, complex, near-square, sloped); each is placed under
    `n_configs` ablation configs of rising solver quality; `n_corruptions` deliberate corruptions (a
    known-wrong orientation, a MIRRORED mesh, a 2x scale error) are always labelled 0. About 5% of the
    benchmark labels are flipped to mimic annotator disagreement, so the classes are not cleanly separable.
    """
    if not 0 <= label_noise < 0.5:
        raise ValueError("label_noise must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    buildings = [_make_building(rng, i) for i in range(n_buildings)]
    rows = []
    for b in buildings:
        for c in range(n_configs):
            acceptable = rng.random() < _p_acceptable(b, CONFIG_QUALITY[c % len(CONFIG_QUALITY)])
            if acceptable:
                kind = "good"
            else:
                names, probs = zip(*BAD_BENCHMARK_KINDS)
                kind = str(rng.choice(names, p=probs))
            flipped = bool(rng.random() < label_noise)  # annotator disagreement
            rows.append(_build_row(rng, b, kind, int(acceptable) ^ int(flipped), c, flipped))
    for i in range(n_corruptions):
        b = buildings[i % n_buildings]
        rows.append(_build_row(rng, b, CORRUPTIONS[i % len(CORRUPTIONS)], 0, None, False))
    return rows
