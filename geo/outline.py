"""
Mesh canonicalisation: up-axis, grounding, the ground outline M_0, bas-relief.
Spec sections 6.1 - 6.4.

Ordering constraint (addendum D.1, hard): 6.4 bas-relief rejection runs BEFORE
any silhouette comparison. A flattened single-view relief scores well on
silhouette IoU from its one good view and would win the orientation vote while
being catastrophically wrong in 3D. `MeshOutline.is_bas_relief` is therefore set
here, and perception reads it rather than recomputing it.

No OpenGL. Spec 6.3's recommended orthographic rasterisation is a numpy
occupancy grid plus skimage morphology and marching squares — indifferent to
non-manifold geometry, self-intersections and disconnected components, all of
which generated meshes produce, and it yields interior rings for courtyards for
free.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure

from contracts import OMBB, MeshOutline
from geo.footprint import compute_ombb, simplify_ring

# The six signed axis candidates, in the order CandidateId.up_axis_idx indexes.
UP_AXIS_CANDIDATES = np.array(
    [
        [1.0, 0.0, 0.0],   # 0: +X
        [-1.0, 0.0, 0.0],  # 1: -X
        [0.0, 1.0, 0.0],   # 2: +Y   <- glTF's nominal up
        [0.0, -1.0, 0.0],  # 3: -Y
        [0.0, 0.0, 1.0],   # 4: +Z
        [0.0, 0.0, -1.0],  # 5: -Z
    ]
)


# --------------------------------------------------------------------------
# 6.1(a) Prismatic-consistency score
# --------------------------------------------------------------------------


def prismatic_score(vertices: np.ndarray, axis: np.ndarray, k: int = 20) -> float:
    """Score a candidate up-axis by cross-sectional constancy. Spec 6.1(a).

    Buildings are, to first order, extruded footprints: their cross-section is
    nearly constant along the vertical. Slice into k slabs along the candidate
    axis, take each slab's projected area, and score by the negative coefficient
    of variation. The true vertical maximises this for any prismatic object.

    Unlike the naive face-normal-area heuristic, this does not flip sign between
    tall and squat buildings.
    """
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    t = vertices @ axis
    lo, hi = float(t.min()), float(t.max())
    if hi - lo < 1e-9:
        return -np.inf

    # Two vectors spanning the plane perpendicular to the axis.
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(axis, e1)

    proj = np.stack([vertices @ e1, vertices @ e2], axis=1)
    edges = np.linspace(lo, hi, k + 1)
    areas = []
    for j in range(k):
        m = (t >= edges[j]) & (t <= edges[j + 1])
        if m.sum() < 3:
            continue
        try:
            areas.append(Polygon(proj[m]).convex_hull.area)
        except Exception:  # noqa: BLE001
            continue

    if len(areas) < max(3, k // 3):
        return -np.inf
    a = np.asarray(areas, dtype=float)
    mean = float(a.mean())
    if mean < 1e-12:
        return -np.inf
    return -float(a.std() / mean)


GLTF_UP_IDX = 2          # +Y, the glTF 2.0 convention (spec 6.1, 13.1)
UP_OVERRIDE_MARGIN = 0.15


def choose_up_axis(vertices: np.ndarray) -> tuple[int, str]:
    """-> (up_axis_idx, reason). glTF +Y is the PRIOR; prismatic score must beat
    it clearly to override.

    Why not just take the best prismatic score (spec 6.1(a))? Because a box has
    a near-constant cross-section along ALL THREE axes, so for boxy buildings —
    most buildings — the score is a near-tie and the argmax is noise. Observed
    on the first real TRELLIS mesh (NCB): +Z scored -0.3625, +Y -0.3882, and
    +Z was wrong; Y was the shortest extent and held 21% of vertices in its
    bottom tenth vs 4% in its top. Spec 6.1 itself says generators mostly
    respect glTF's +Y up, so that is the default.

    The score is also sign-blind (+Y and -Y score identically), so the sign is
    set separately: the base is the end with more vertex mass — ground-floor
    detail, plinths and any ground plane concentrate there, the roof is sparse.
    """
    scores = np.array([prismatic_score(vertices, ax) for ax in UP_AXIS_CANDIDATES])
    best = int(np.argmax(scores))
    gltf_score = scores[GLTF_UP_IDX]

    if best // 2 == GLTF_UP_IDX // 2 or scores[best] - gltf_score < UP_OVERRIDE_MARGIN:
        axis_pair = GLTF_UP_IDX // 2
        reason = (f"glTF +Y prior kept (best prismatic {scores[best]:.3f} vs "
                  f"+Y {gltf_score:.3f}; override needs +{UP_OVERRIDE_MARGIN})")
    else:
        axis_pair = best // 2
        reason = (f"prismatic override: axis {axis_pair} beats +Y by "
                  f"{scores[best] - gltf_score:.3f}")

    t = vertices[:, axis_pair]
    lo, hi = float(t.min()), float(t.max())
    if hi - lo < 1e-12:
        return axis_pair * 2, reason
    u = (t - lo) / (hi - lo)
    bottom, top = float(np.mean(u < 0.1)), float(np.mean(u > 0.9))
    # base at the min end -> the positive axis points up
    idx = axis_pair * 2 if bottom >= top else axis_pair * 2 + 1
    return idx, reason + f"; sign from mass (bottom {bottom:.2f} / top {top:.2f})"


def rank_up_axes(vertices: np.ndarray, top_n: int = 3) -> list[int]:
    """-> indices into UP_AXIS_CANDIDATES, best first.

    Addendum A.3 Stage 2: prune to the top 2-3 rather than rendering all six.
    The prismatic score is reliable enough to prune on, and pruning halves the
    render count for the silhouette comparison downstream.
    """
    scores = [(prismatic_score(vertices, ax), i)
              for i, ax in enumerate(UP_AXIS_CANDIDATES)]
    scores.sort(reverse=True)
    return [i for _, i in scores[:top_n]]


# --------------------------------------------------------------------------
# 6.2 Pivot normalisation and grounding
# --------------------------------------------------------------------------


def robust_base(heights: np.ndarray, percentile: float = 1.0) -> float:
    """The mesh's true base height. Spec 6.2.

    Use a robust minimum, never min(). Generated meshes routinely carry a
    handful of stray vertices or thin spikes below the main body, and a single
    outlier 2 m down floats the entire building 2 m into the air — a failure
    that is nearly invisible until someone looks at the model from the side.
    """
    return float(np.percentile(heights, percentile))


def canonical_rotation(up_axis_idx: int) -> np.ndarray:
    """The 3x3 rotation taking candidate up-axis `up_axis_idx` to +Z."""
    axis = UP_AXIS_CANDIDATES[up_axis_idx]
    z = np.array([0.0, 0.0, 1.0])

    if np.allclose(axis, z):
        return np.eye(3)
    if np.allclose(axis, -z):
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(axis, z)
    c = float(np.dot(axis, z))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# glTF 2.0: the asset front faces +Z (spec 6.1, 13.1).
GLTF_FRONT = np.array([0.0, 0.0, 1.0])


def front_angle_canonical(up_axis_idx: int) -> float | None:
    """Direction the glTF front (+Z) points after canonicalisation, radians CCW
    from canonical +X. None when it cannot be known.

    For the normal case — a glTF-conforming mesh, +Y up (idx 2) — the front
    lands on canonical -Y, i.e. -pi/2. NOT +X.

    If the mesh turned out to be Z-up (idx 4/5), the generator did not follow the
    glTF convention, so +Z is the roof rather than the front and the front is
    genuinely unknown. Returning None there is deliberate: every filter that
    matches a facade to a bearing (EXIF, road normal) must then abstain rather
    than guess.

    ASSUMPTION TO VERIFY on the first real TRELLIS mesh: that the photographed
    facade is the glTF front. Render it once and check.
    """
    f = canonical_rotation(up_axis_idx) @ GLTF_FRONT
    if float(np.hypot(f[0], f[1])) < 0.5:
        return None
    return float(np.arctan2(f[1], f[0]))


def canonical_offsets(vertices: np.ndarray, up_axis_idx: int
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """-> (R, o, height_units): canonical = R @ v - o.

    The single definition of the mesh -> canonical frame. canonicalise() applies
    it, and geo.placement records it so the viewer can reproduce the placement:
    the solver's transform is defined on CANONICAL coordinates, so a viewer
    given only that transform and the raw glb cannot place the mesh.
    """
    rot = canonical_rotation(up_axis_idx)
    rotated = np.asarray(vertices, dtype=float) @ rot.T
    base = robust_base(rotated[:, 2])
    top = float(np.percentile(rotated[:, 2], 99.0))
    o = np.array([rotated[:, 0].mean(), rotated[:, 1].mean(), base])
    return rot, o, max(top - base, 1e-6)


def canonicalise(vertices: np.ndarray, up_axis_idx: int) -> tuple[np.ndarray, float]:
    """Rotate so the chosen axis is +Z, drop the base to z=0, centre in XY.

    -> (vertices in canonical frame, height in model units)
    """
    rot, o, height = canonical_offsets(vertices, up_axis_idx)
    return np.asarray(vertices, dtype=float) @ rot.T - o, height


# --------------------------------------------------------------------------
# 6.4 Bas-relief detection
# --------------------------------------------------------------------------


def bas_relief_check(vertices_canonical: np.ndarray,
                     threshold: float = 0.25) -> tuple[bool, float]:
    """-> (is_bas_relief, extent_ratio). Spec 6.4.

    Some image-to-3D pipelines, given a single frontal photograph, produce a
    flattened relief rather than a closed volume. It will "fit" the footprint
    with a plausible IoU while looking catastrophically wrong in 3D, so it is
    detected before fitting, not after.

    Caller must compare against the real footprint's own aspect: a genuinely
    elongated building is not a relief just because its extents differ.
    """
    xy = vertices_canonical[:, :2]
    ext = xy.max(axis=0) - xy.min(axis=0)
    lo, hi = float(min(ext)), float(max(ext))
    ratio = lo / hi if hi > 1e-9 else 1.0
    return ratio < threshold, ratio


# --------------------------------------------------------------------------
# 6.3 Extracting the ground outline M_0
# --------------------------------------------------------------------------


def ground_outline(vertices_canonical: np.ndarray,
                   faces: np.ndarray | None = None,
                   *,
                   band: tuple[float, float] = (0.03, 0.50),
                   grid_cells: int = 300,
                   close_frac: float = 0.02,
                   open_frac: float = 0.025,
                   simplify_frac: float = 0.004) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Orthographic rasterisation into an occupancy grid. Spec 6.3 (recommended).

    -> (outer ring, interior rings) in model units.

    Preferred over alpha shapes because a single global alpha handles uniformly
    dense convex point sets well but degrades on exactly our cases: non-uniform
    density, and the concave corners of L-shaped and courtyard buildings.

    EVERY length here is a fraction of the mesh's own size. This runs BEFORE
    scaling, on a mesh ~1 model unit across. An earlier version took its grid
    cell (0.10) and simplification tolerance (0.25) in metres, which on the
    first real TRELLIS mesh meant a ~10-cell grid and a simplifier that erased
    the stepped NCB plan down to a 4-corner box.

    Height band, not the bottom slab. Spec 10.1 resolution 2 says fit the base,
    and the base is still what we read — but not the very bottom: TRELLIS
    reconstructs the scenery in the photo (lawn patches, bushes, lamp posts,
    passers-by) as geometry at ground level, outside the walls. Starting the
    band at 3% of height clears a ground sheet; the morphological opening then
    severs thin attachments (posts, poles) so that keeping the largest component
    discards them.
    """
    z = vertices_canonical[:, 2]
    height = float(z.max()) if z.size else 1.0
    m = (z >= height * band[0]) & (z <= height * band[1])
    slab = vertices_canonical[m] if m.sum() >= 50 else vertices_canonical

    xy = slab[:, :2]
    # Robust size: ignore the extreme 1% so one stray vertex cannot set the scale.
    size = float(max(np.ptp(np.percentile(xy, [1, 99], axis=0), axis=0).max(), 1e-9))
    # Cell size must track POINT SPACING, not just mesh size. With a fixed
    # size/300 grid, a sparsely sampled mesh leaves gaps wider than the closing
    # radius, the plan shatters into fragments, and "keep the largest component"
    # keeps a sliver (observed: 11% of an L-shaped plan). Cells no smaller than
    # ~0.75 of the median neighbour spacing keep closing able to fuse neighbours.
    from scipy.spatial import cKDTree
    sample = xy if len(xy) <= 5000 else xy[np.random.default_rng(0).choice(len(xy), 5000, replace=False)]
    nn = cKDTree(sample).query(sample, k=2)[0][:, 1]
    spacing = float(np.median(nn[nn > 0])) if np.any(nn > 0) else 0.0
    pixel = max(size / grid_cells, 0.75 * spacing)
    close_px = max(3, int(round(close_frac * grid_cells)))
    open_px = max(3, int(round(open_frac * grid_cells)))
    simplify_eps = simplify_frac * size

    pad = pixel * (close_px + open_px + 4)
    lo = xy.min(axis=0) - pad
    hi = xy.max(axis=0) + pad
    dims = np.maximum(((hi - lo) / pixel).astype(int) + 1, 4)

    grid = np.zeros((dims[1], dims[0]), dtype=bool)
    idx = ((xy - lo) / pixel).astype(int)
    grid[np.clip(idx[:, 1], 0, dims[1] - 1), np.clip(idx[:, 0], 0, dims[0] - 1)] = True

    # Seal the gaps between sampled wall vertices, then fill the interior so the
    # outer contour is a real boundary rather than a band of surface points.
    disk = lambda r: (lambda yy, xx: xx * xx + yy * yy <= r * r)(*np.ogrid[-r:r + 1, -r:r + 1])  # noqa: E731
    grid = ndimage.binary_closing(grid, structure=disk(close_px // 2 + 1))
    filled = ndimage.binary_fill_holes(grid)

    # Opening severs thin appendages — lamp posts, poles, a bush touching a wall
    # — so the largest-component step below can drop them.
    opened = ndimage.binary_opening(filled, structure=disk(open_px // 2 + 1))
    if opened.sum() > 0.5 * filled.sum():
        filled = opened

    labels, n = ndimage.label(filled)
    if n > 1:
        sizes = ndimage.sum(filled, labels, range(1, n + 1))
        filled = labels == (int(np.argmax(sizes)) + 1)
    pixel_m = pixel  # model units per cell; name kept for to_model below

    padded = np.pad(filled.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        # Degenerate: fall back to the convex hull of the slab.
        hull = Polygon(xy).convex_hull
        return np.asarray(hull.exterior.coords)[:-1], ()

    def to_model(c: np.ndarray) -> np.ndarray:
        # find_contours returns (row, col) on the padded grid
        return np.stack([(c[:, 1] - 1) * pixel_m + lo[0],
                         (c[:, 0] - 1) * pixel_m + lo[1]], axis=1)

    rings = sorted((to_model(c) for c in contours),
                   key=lambda r: Polygon(r).area if len(r) > 3 else 0.0,
                   reverse=True)

    outer = simplify_ring(rings[0], simplify_eps)
    # Interior rings = courtyards. Anything under 0.1% of the plan is noise.
    min_hole = 1e-3 * size * size
    holes = tuple(simplify_ring(r, simplify_eps) for r in rings[1:]
                  if len(r) > 3 and Polygon(r).area > min_hole)
    return outer, holes


def build_mesh_outline(vertices: np.ndarray,
                       *,
                       up_axis_idx: int | None = None,
                       footprint_aspect: float = 1.0) -> MeshOutline:
    """Full 6.1 -> 6.4 chain. -> MeshOutline, ready for the solve."""
    if up_axis_idx is None:
        up_axis_idx, _ = choose_up_axis(np.asarray(vertices, dtype=float))

    canon, height_units = canonicalise(np.asarray(vertices, dtype=float), up_axis_idx)
    is_relief, ratio = bas_relief_check(canon)

    # A genuinely elongated building is not a relief. Only call it one if the
    # mesh is far flatter than the real footprint is elongated.
    if is_relief and footprint_aspect > 1.0 / max(ratio, 1e-6) * 0.5:
        is_relief = False

    outer, holes = ground_outline(canon)
    return MeshOutline(
        pts_enu=outer,
        holes_enu=holes,
        ombb=compute_ombb(outer),
        is_bas_relief=is_relief,
        extent_ratio=ratio,
        up_axis_idx=up_axis_idx,
        mesh_height_units=height_units,
        front_angle=front_angle_canonical(up_axis_idx),
    )


def outline_from_footprint(fp) -> MeshOutline:
    """Stub path for --dry-run: use the footprint's own OMBB as M_0.

    A plain box, which is the spec 15 hour-2-6 milestone object. On a rectangular
    building it fits perfectly; on anything with wings it does not, and the
    Hausdorff metric says so loudly while IoU stays deceptively high. That
    contrast is the point — it is spec 9.1's argument for carrying both metrics,
    demonstrated on real data before the mesh path exists.

    Note the box is symmetric under 180 degrees, so the footprint-IoU margin
    between candidates k and k+2 is structurally zero. That is not a bug: it is
    precisely why spec 6.6 needs filters beyond IoU, and the run will correctly
    report a zero margin and route to review.

    A box has no front, so `front_angle` stays None and the EXIF and road-normal
    filters abstain on this path.
    """
    box = fp.ombb
    corners = np.array([
        box.centre + box.u * box.a / 2 + box.v * box.b / 2,
        box.centre - box.u * box.a / 2 + box.v * box.b / 2,
        box.centre - box.u * box.a / 2 - box.v * box.b / 2,
        box.centre + box.u * box.a / 2 - box.v * box.b / 2,
    ])
    return MeshOutline(
        pts_enu=corners - box.centre,
        ombb=compute_ombb(corners - box.centre),
        is_bas_relief=False,
        extent_ratio=box.b / box.a,
        up_axis_idx=4,
        mesh_height_units=1.0,
    )
