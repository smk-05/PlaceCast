"""
contracts.py — the data types every stage passes across.

This is the interface between the geometry half (geo/, generate/, pipeline.py)
and the perception half (perception/, confidence/). Perception is developed on
another machine and incorporated later, so these types are the handoff: if the
three functions at the bottom of this file satisfy their signatures against a
fixture, integration is an import change.

Design rules, from spec section 14:
  - Every stage consumes and returns an immutable dataclass.
  - No stage reaches backwards.
  - The whole pipeline is therefore replayable from any intermediate state.

Deviations from addendum F.1, all deliberate:
  - OMBB is a real dataclass, not a bare `tuple`. F.1's `(centre, (u,v), (a,b))`
    is positional and unlabelled; it is touched in six modules and indexing it
    wrongly is a silent 90-degree error.
  - Footprint/MeshOutline carry `holes_enu` explicitly. F.1 kept only a
    `has_holes` bool, which is lossy — spec 3.3 requires every area, IoU and
    boundary computation downstream to respect interior rings, so the rings have
    to actually travel with the polygon. `has_holes` survives as a property so it
    cannot desync from the data.
  - `Decision` is an enum. F.1 returned a bare str; a typo becomes a silent
    wrong branch rather than an error.
  - FitResult gains height/scale_z (spec 7) and candidate_ious (spec 9.3's review
    payload). Without height_source, confidence/features.py cannot build the
    `height_source_authoritative` feature that addendum B.3 asks for.
"""

from __future__ import annotations

# FROZEN. Changing any field name, type or meaning below needs both
# contributors to agree, a bump here, and a new git tag (contracts-vN).
# Adding a defaulted field is backwards-compatible; still bump the minor.
CONTRACT_VERSION = "1.0"

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

# --------------------------------------------------------------------------
# Enums and small value types
# --------------------------------------------------------------------------


class Decision(str, Enum):
    """Outcome of the confidence gate (spec 9.1 table, or addendum B's model).

    str-valued so it serialises to JSON as "auto_accept" with no encoder hook.
    """

    AUTO_ACCEPT = "auto_accept"
    REVIEW = "review"
    REJECT = "reject"


class HeightSource(str, Enum):
    """Which tier of spec 7's priority chain produced the height.

    Recorded always — spec 7.4: "Always record which of the four sources
    produced the height." The first two are authoritative; that distinction is
    a feature in the learned gate (addendum B.3).
    """

    OSM_HEIGHT = "osm:height"
    OSM_LEVELS = "osm:building:levels"
    MICROSOFT = "microsoft:height"
    SINGLE_VIEW_METROLOGY = "single_view_metrology"
    MONOCULAR_DEPTH = "monocular_depth"
    PROPORTIONAL_FALLBACK = "proportional_fallback"

    @property
    def is_authoritative(self) -> bool:
        """OSM tags only. Microsoft is deliberately NOT authoritative.

        Spec 7.1 treats Microsoft's `height` as authoritative. Measured on the
        five VT buildings that carry an explicit OSM `height` in metres
        (scripts/check_microsoft_heights.py, 2026-09-19), Microsoft
        underestimated every one by 44-67% (median 51%) and compressed all 20
        campus buildings it covered into a 7-16 m band. Calling that
        authoritative would make the confidence model trust a value that is
        wrong by half.
        """
        return self in (HeightSource.OSM_HEIGHT, HeightSource.OSM_LEVELS)


class Disambiguator(str, Enum):
    """Which filter resolved the four-fold azimuth ambiguity (spec 6.6).

    Ordered by strength. EXIF is arithmetic; silhouette is evidence; road normal
    is a prior and addendum A.3 says anything decided by it is flagged for review
    regardless of IoU, because at that point we are guessing.
    """

    EXIF_HEADING = "exif_heading"        # spec 6.6 Filter 2 — arithmetic
    SILHOUETTE = "silhouette"            # addendum A.3 — evidence, needs perception
    FACADE_DETAIL = "facade_detail"      # spec 6.6 Filter 3b — mesh-only, no ML
    ROAD_NORMAL = "road_normal"          # spec 6.6 Filter 3a — prior; always flag
    ASPECT_RATIO = "aspect_ratio"        # spec 6.6 Filter 1 — only excludes 90deg
    ARBITRARY_SYMMETRIC = "arbitrary_symmetric"  # addendum A.4 — genuinely ambiguous


class CandidateId(NamedTuple):
    """Identifies one (up-axis, azimuth) orientation candidate.

    Addendum F.1 says to agree on this scheme but does not put it in the file.
    It keys PhotoEvidence.silhouette_scores, so it has to live here or that dict
    is untyped across the machine boundary.

    up_axis_idx: 0..5, indexing the six signed axes (+X,-X,+Y,-Y,+Z,-Z) in the
                 order produced by geo.outline.UP_AXIS_CANDIDATES.
    azimuth_k:   0..3, the rotation theta_0 + k*pi/2 from spec 6.5.
    """

    up_axis_idx: int
    azimuth_k: int

    def __str__(self) -> str:  # stable key for JSON dicts
        return f"u{self.up_axis_idx}a{self.azimuth_k}"


@dataclass(frozen=True, eq=False)
class OMBB:
    """Oriented minimum bounding box (spec 4.4), via rotating calipers.

    Invariant: a >= b. Callers rely on this for the extent pairing in spec 6.5.
    """

    centre: np.ndarray  # (2,) metres or model units
    u: np.ndarray       # (2,) unit vector along the long axis
    v: np.ndarray       # (2,) unit vector along the short axis
    a: float            # extent along u  (the long one)
    b: float            # extent along v

    def __post_init__(self) -> None:
        if self.a < self.b:
            raise ValueError(f"OMBB invariant violated: a={self.a} < b={self.b}")

    @property
    def aspect(self) -> float:
        """a/b. Spec 6.6 Filter 1 excludes the 90deg candidates when this > 1.1."""
        return self.a / self.b if self.b > 1e-9 else float("inf")

    @property
    def angle(self) -> float:
        """Orientation of the long axis, radians CCW from +East (spec 2.4)."""
        return float(np.arctan2(self.u[1], self.u[0]))


# --------------------------------------------------------------------------
# Produced by geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Footprint:
    """The authoritative building footprint, in the local ENU frame (spec 2.2).

    This is the *conditioned copy* used for fitting. Spec 4.3: the authoritative
    footprint stays untouched in the provenance record — that raw geometry lives
    in PlacementRecord.footprint_geojson, not here.
    """

    pts_enu: np.ndarray                          # (N,2) metres, LARGEST outer ring
    holes_enu: tuple[np.ndarray, ...] = ()       # interior rings (spec 3.3)
    # Additional DISJOINT outer rings, largest-first. Spec 3.3 covers holes but
    # not multi-part buildings; Lane Stadium's four stands, with the field as a
    # genuine gap between them, need this. Use geo.footprint.to_shapely() for
    # anything that measures area or overlap — reading pts_enu alone silently
    # drops these parts.
    parts_enu: tuple[np.ndarray, ...] = ()
    rectilinearity: float = 0.0                  # spec 4.2, in [0,1]
    principal_angle: float = 0.0                 # spec 4.2 theta*, mod pi/2
    ombb: OMBB | None = None                     # spec 4.4, of the largest part
    area_m2: float = 0.0                         # ALL parts
    source: str = "osm"                          # "osm" | "microsoft"
    # How confidently this polygon was matched to the address (spec 3.2).
    # "contained_and_named" | "contained" | "name_match" | "unnamed_sole_candidate"
    # A feature for the confidence model, alongside geocode_rooftop.
    match_quality: str = ""
    # The geocoder's own precision claim (spec 3.1): ROOFTOP, RANGE_INTERPOLATED,
    # GEOMETRIC_CENTER or APPROXIMATE. Lives here because it is evidence about
    # whether THIS polygon is the right building, and score_confidence receives
    # the Footprint but never the raw geocode.
    geocode_location_type: str = ""

    @property
    def has_holes(self) -> bool:
        return len(self.holes_enu) > 0

    @property
    def is_multipart(self) -> bool:
        return len(self.parts_enu) > 0

    @property
    def is_weak_match(self) -> bool:
        """True when the footprint was accepted on thin evidence (spec 3.2)."""
        return self.match_quality in ("unnamed_sole_candidate", "")

    @property
    def geocode_rooftop(self) -> bool:
        """Addendum B.3 feature `geocode_rooftop`."""
        return self.geocode_location_type == "ROOFTOP"

    @property
    def is_ill_posed(self) -> bool:
        """Spec 4.2: R <= 0.6 means rotation is intrinsically meaningless.

        The cheapest early-warning signal in the whole pipeline, and it costs one
        pass over the edges. Flag for review regardless of IoU.
        """
        return self.rectilinearity < 0.6


@dataclass(frozen=True, eq=False)
class MeshOutline:
    """The mesh's ground-plane outline M_0 (spec 6.3), in model units.

    is_bas_relief MUST be set before any silhouette comparison runs. Addendum
    D.1 makes this a hard ordering constraint: a flat mesh scores well on
    silhouette from its one good view and would otherwise win the orientation
    vote while being catastrophically wrong in 3D.
    """

    pts_enu: np.ndarray                          # (M,2) model units
    holes_enu: tuple[np.ndarray, ...] = ()       # courtyards; raster extraction
    ombb: OMBB | None = None
    is_bas_relief: bool = False                  # spec 6.4 extent-ratio test
    extent_ratio: float = 1.0                    # min/max horizontal extent
    up_axis_idx: int = 4                         # spec 6.1, default +Z
    mesh_height_units: float = 1.0               # model-unit height, for scale_z
    # Direction the mesh's FRONT facade faces in the canonical frame, radians CCW
    # from canonical +X. For a glTF-conforming (+Y up, front +Z) mesh this is
    # -pi/2. None when unknowable — a box has no front, and a Z-up mesh broke the
    # glTF convention — in which case every facade-to-bearing filter abstains.
    front_angle: float | None = None

    @property
    def has_holes(self) -> bool:
        return len(self.holes_enu) > 0


@dataclass(frozen=True, eq=False)
class FitResult:
    """The planar similarity that maps M_0 onto F, plus every quality metric.

    All quantities are in the ENU frame (spec 2.2) and all angles are
    mathematical theta — CCW from +East. Convert to compass heading only at the
    Cesium boundary, using spec 2.4's `heading = pi/2 - theta`.
    """

    # --- the transform (spec 1.2) ---
    theta: float
    scale_x: float
    scale_y: float
    scale_z: float
    tx: float
    ty: float
    z_offset: float                  # ENU-frame vertical translation of the base

    # --- quality metrics (spec 9.1) ---
    iou: float = 0.0
    hausdorff_m: float = float("inf")
    area_ratio: float = 0.0
    rotation_margin_footprint: float = 0.0   # spec 6.6, IoU_1 - IoU_2
    anisotropy_log_ratio: float = 0.0        # spec 6.9, |log(sx/sy)|
    max_neighbor_overlap: float = 0.0        # spec 9.2

    # --- height (spec 7) ---
    height_m: float = 0.0
    height_source: HeightSource = HeightSource.PROPORTIONAL_FALLBACK

    # --- provenance of the decision ---
    disambiguated_by: Disambiguator = Disambiguator.ASPECT_RATIO
    solver: str = "stub"                     # e.g. "ombb+icp+nelder-mead"
    candidate_ious: tuple[tuple[CandidateId, float], ...] = ()  # spec 9.3 payload

    # Did EXIF heading and the silhouette score pick DIFFERENT candidates?
    # None when either cue was absent, so "no disagreement" and "nothing to
    # compare" stay distinguishable. Addendum A.3: disagreement between
    # independent orientation evidence is itself a strong signal.
    #
    # Lives on FitResult, not PhotoEvidence, on purpose: PhotoEvidence is
    # perception's output and is frozen BEFORE geometry runs, but deciding which
    # candidate an EXIF bearing selects needs facade_heading(), which depends on
    # the fitted mesh outline. It can only be computed during disambiguation.
    exif_silhouette_disagree: bool | None = None

    @property
    def height_source_authoritative(self) -> bool:
        """Addendum B.3 feature `height_source_authoritative`."""
        return self.height_source.is_authoritative

    @property
    def is_mirrored(self) -> bool:
        """Spec 1.2 / 6.8: we optimise over SO(2), never O(2).

        Umeyama's S correction is what stops the SVD silently returning a
        reflection on badly corrupted data. A mirrored building passes every
        numeric check, so this is tested explicitly rather than assumed.
        """
        return self.scale_x < 0 or self.scale_y < 0

    @property
    def heading_rad(self) -> float:
        """Compass heading for Cesium (spec 2.4): from North, clockwise."""
        return float(np.mod(np.pi / 2 - self.theta, 2 * np.pi))


# --------------------------------------------------------------------------
# Produced by perception (friend's half)
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PhotoEvidence:
    """Everything the photograph tells us that the footprint cannot.

    Addendum A.0's boundary: "Geometry handles everything determined by the
    footprint. ML handles everything determined by the photograph, plus the
    accept/reject judgement."

    Every field here has a working non-ML default (see fixtures and the
    perception stubs), so the pipeline never blocks on this arriving.
    """

    mask: np.ndarray                            # binary HxW building mask (A.3)
    mask_area_frac: float = 0.0
    occlusion_flag: bool = False
    photo_path: Path | None = None
    photo_sha256: str = ""

    # EXIF — free, extracted at ingest, no model involved (spec 6.6 Filter 2)
    exif_heading_deg: float | None = None       # GPSImgDirection
    exif_pitch_deg: float | None = None         # for addendum C.2 gravity alignment
    exif_focal_mm: float | None = None
    exif_gps: tuple[float, float] | None = None  # (lat, lon) of the camera

    # Silhouette render-and-compare (addendum A.3) — requires segmentation.
    # silhouette_margin = top-two gap of silhouette_scores. The EXIF-vs-
    # silhouette disagreement flag is on FitResult; see the note there.
    silhouette_scores: dict[CandidateId, float] = field(default_factory=dict)
    silhouette_margin: float = 0.0
    segmentation_model: str = "stub"

    @property
    def has_silhouette_evidence(self) -> bool:
        """Addendum A.3 fusion rule: silhouette decides only if margin >= 0.08.

        Below that we fall through to the road-normal prior and flag for review
        regardless of IoU, because at that point we are guessing.
        """
        return bool(self.silhouette_scores) and self.silhouette_margin >= 0.08


# --------------------------------------------------------------------------
# The provenance record (spec 12.1)
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PlacementRecord:
    """The durable artifact. Spec 12.1.

    Store the transform, never the transformed mesh: the mesh is a large binary,
    the transform is ten floats, and re-deriving the placement from this record
    must be bit-identical. That is why model versions, seeds, and the footprint's
    OSM *version* are all pinned here — OSM footprints get edited, and a re-fetch
    six months later may silently return a different polygon.
    """

    asset_id: str
    created_at: str
    address_raw: str
    address_normalized: str = ""

    # geocode (spec 3.1)
    geocode_provider: str = "nominatim"
    lat: float = 0.0
    lon: float = 0.0
    location_type: str = "APPROXIMATE"   # ROOFTOP / RANGE_INTERPOLATED / ...

    # footprint (spec 3.2) — the UNTOUCHED authoritative geometry
    footprint_source: str = "osm"
    osm_type: str = ""
    osm_id: int = 0
    osm_version: int = 0
    footprint_geojson: dict = field(default_factory=dict)
    rectilinearity: float = 0.0
    # OMBB extents (a, b) in metres. The viewer sizes the hour-6 milestone box
    # from these when no mesh exists yet.
    footprint_ombb_m: tuple[float, float] = (0.0, 0.0)

    # crs (spec 2.2, 2.3)
    enu_origin_geodetic: tuple[float, float, float] = (0.0, 0.0, 0.0)
    geoid_undulation_m: float = 0.0
    height_datum: str = "cesium_ion_ellipsoidal"

    # generation (spec 5)
    prompt: str = ""
    models: tuple[dict, ...] = ()

    # the answer
    fit: FitResult | None = None
    decision: Decision = Decision.REVIEW
    confidence_p: float | None = None
    confidence_method: str = "threshold_table"
    review_reasons: tuple[str, ...] = ()

    def to_json_dict(self) -> dict:
        """One serialisation point, so provenance cannot drift between writers."""
        d = asdict(self)
        if self.fit is not None:
            f = self.fit
            d["fit"] = {
                "transform": {
                    "frame": "ENU",
                    "translation_m": [f.tx, f.ty, f.z_offset],
                    "rotation_z_rad": f.theta,
                    "heading_rad": f.heading_rad,
                    "scale": [f.scale_x, f.scale_y, f.scale_z],
                },
                "iou": f.iou,
                "hausdorff_m": f.hausdorff_m,
                "area_ratio": f.area_ratio,
                "rotation_margin_footprint": f.rotation_margin_footprint,
                "anisotropy_log_ratio": f.anisotropy_log_ratio,
                "max_neighbor_overlap": f.max_neighbor_overlap,
                "disambiguated_by": f.disambiguated_by.value,
                "exif_silhouette_disagree": f.exif_silhouette_disagree,
                "solver": f.solver,
                "candidate_ious": {str(c): v for c, v in f.candidate_ious},
            }
            d["height"] = {
                "value_m": f.height_m,
                "source": f.height_source.value,
                "authoritative": f.height_source.is_authoritative,
            }
        d["decision"] = self.decision.value
        return d


# --------------------------------------------------------------------------
# The three functions perception must satisfy
# --------------------------------------------------------------------------
#
# These are Protocols rather than abstract bases so the perception half can be
# plain modules with plain functions — no inheritance, no imports back into geo/.
# Each has a working non-ML implementation in perception/ already; the friend's
# versions replace them behind a flag.


class SegmentBuilding(Protocol):
    def __call__(self, image_path: Path) -> tuple[np.ndarray, float, bool]:
        """Addendum A.3 Stage 1.  -> (binary HxW mask, mask_area_frac, occluded)"""
        ...


class ScoreSilhouettes(Protocol):
    def __call__(
        self,
        mesh,  # trimesh.Trimesh — untyped to keep trimesh out of this module
        mask: np.ndarray,
        candidates: list[CandidateId],
    ) -> dict[CandidateId, float]:
        """Addendum A.3 Stages 2-3.  -> normalised silhouette IoU per candidate.

        "Normalised" matters: compare shape descriptors after centring and
        scaling to a common box, not raw masks. A street photo is perspective
        and the render is orthographic, and that mismatch corrupts exactly the
        absolute position and scale that normalisation discards.
        """
        ...


class FacadeHeading(Protocol):
    def __call__(self, fp: Footprint, mo: MeshOutline,
                 candidate: CandidateId) -> float:
        """Provided BY geometry, FOR perception: geo.disambiguate.facade_heading.

        -> compass bearing in degrees [0, 360) that the mesh's front facade
        faces when placed with `candidate`. 0 = North, 90 = East (spec 2.4).

        Raises ValueError when the mesh's front is unknown (MeshOutline.
        front_angle is None). Callers must treat that as "no bearing", never
        substitute a default.
        """
        ...


class ScoreConfidence(Protocol):
    def __call__(
        self,
        fit: FitResult,
        photo: PhotoEvidence,
        fp: Footprint,
    ) -> tuple[float, Decision]:
        """Addendum B.  -> (probability, decision).

        The spec 9.1 threshold table in geo/validate.py is the primary path and
        stays in the codebase permanently. The learned logistic regression is a
        flag-selected upgrade, never a dependency.
        """
        ...
