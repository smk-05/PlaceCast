"""
The placement solve. Spec sections 6.5 - 6.9.

The objective (spec 1.2) is Jaccard overlap over a planar similarity:

    J(s, theta, t) = area(T(M_0) ^ F) / area(T(M_0) v F)

It is non-convex, non-differentiable, and piecewise-smooth with kinks at every
vertex-edge incidence event. Gradient descent from a random start fails. The
strategy is therefore: closed-form geometric initialisation (6.5) -> discrete
disambiguation (6.6, in disambiguate.py) -> local continuous refinement (6.8).

Reflections are excluded throughout. We optimise over SO(2), not O(2) — a
building is a physical object, not a shape-matching exercise. Generated meshes
do occasionally come out mirrored, and Umeyama's S correction below is what
stops the SVD silently returning one.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon

from contracts import CandidateId, Disambiguator, FitResult, Footprint, MeshOutline


# --------------------------------------------------------------------------
# Transform helpers
# --------------------------------------------------------------------------


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def apply_similarity(pts: np.ndarray, theta: float, sx: float, sy: float,
                     t: np.ndarray) -> np.ndarray:
    """Anisotropic scale in the mesh's own axes, then rotate, then translate."""
    scaled = pts * np.array([sx, sy])
    return scaled @ rot2(theta).T + t


def _poly(pts: np.ndarray, holes: tuple[np.ndarray, ...] = ()) -> Polygon:
    p = Polygon(pts, [h for h in holes])
    return p if p.is_valid else p.buffer(0)


def _target_geom(fp: Footprint):
    """The full footprint geometry, all parts and holes. Never Polygon(pts_enu).

    Multi-part footprints lose their other parts from the IoU denominator if
    you build the target from pts_enu alone.
    """
    from geo.footprint import to_shapely
    g = to_shapely(fp)
    return g if g.is_valid else g.buffer(0)


def iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    union = a.union(b).area
    return float(a.intersection(b).area / union) if union > 1e-12 else 0.0


# --------------------------------------------------------------------------
# 6.5 Closed-form initialisation via OMBB alignment
# --------------------------------------------------------------------------


def ombb_candidates(fp: Footprint, mo: MeshOutline) -> list[tuple[CandidateId, dict]]:
    """The four OMBB-derived placements. Spec 6.5.

    Exact when both shapes are rectangles and near-optimal when both are
    approximately rectilinear — which the footprint's rectilinearity R tells you
    in advance, before you try to solve.

    Because the OMBB is defined up to a relabelling of its axes, this produces
    four candidates theta_k = theta_0 + k*pi/2, with k in {1,3} additionally
    swapping the extent pairing in the scale.
    """
    bf, bm = fp.ombb, mo.ombb
    if bf is None or bm is None:
        raise ValueError("Both Footprint and MeshOutline need an OMBB (spec 4.4).")

    out = []
    for k in range(4):
        theta = bf.angle - bm.angle + k * math.pi / 2.0

        if k % 2 == 0:
            sx = bf.a / max(bm.a, 1e-9)
            sy = bf.b / max(bm.b, 1e-9)
        else:
            # 90-degree candidates pair the long mesh extent to the short
            # footprint extent.
            sx = bf.b / max(bm.a, 1e-9)
            sy = bf.a / max(bm.b, 1e-9)

        s_iso = 0.5 * (sx + sy)
        t = bf.centre - s_iso * (rot2(theta) @ bm.centre)

        out.append((
            CandidateId(up_axis_idx=mo.up_axis_idx, azimuth_k=k),
            {"theta": theta, "sx": s_iso, "sy": s_iso, "t": t},
        ))
    return out


def score_candidates(fp: Footprint, mo: MeshOutline,
                     candidates: list[tuple[CandidateId, dict]]
                     ) -> list[tuple[CandidateId, float]]:
    """Footprint IoU for each candidate. The margin between the top two is the
    spec 6.6 ambiguity signal and goes straight into the confidence model."""
    target = _target_geom(fp)
    scored = []
    for cid, p in candidates:
        moved = apply_similarity(mo.pts_enu, p["theta"], p["sx"], p["sy"], p["t"])
        holes = tuple(apply_similarity(h, p["theta"], p["sx"], p["sy"], p["t"])
                      for h in mo.holes_enu)
        scored.append((cid, iou(_poly(moved, holes), target)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# --------------------------------------------------------------------------
# 6.8 Continuous refinement — correspondence-based (Kabsch-Umeyama)
# --------------------------------------------------------------------------


def _resample(ring: np.ndarray, n: int = 400) -> np.ndarray:
    """Uniform arc-length resampling of a closed ring."""
    closed = np.vstack([ring, ring[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(ring[:1], n, axis=0)
    targets = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.stack([np.interp(targets, cum, closed[:, 0]),
                     np.interp(targets, cum, closed[:, 1])], axis=1)


def umeyama_similarity(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form similarity Procrustes in 2D. Umeyama, TPAMI 1991.

    -> (scale, R, t) minimising sum ||y_i - (s R x_i + t)||^2.

    The S correction is Umeyama's specific contribution over Arun et al. and
    Horn: without it the SVD solution silently returns a REFLECTION instead of a
    rotation when the data is badly corrupted. For this pipeline that means a
    mirrored building that passes every numeric check. Do not omit it.
    """
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mu_x, y - mu_y
    var_x = float((xc ** 2).sum() / len(x))

    cov = (yc.T @ xc) / len(x)
    u, d, vt = np.linalg.svd(cov)

    s_mat = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[1, 1] = -1.0

    r = u @ s_mat @ vt
    scale = float(np.trace(np.diag(d) @ s_mat) / var_x) if var_x > 1e-12 else 1.0
    t = mu_y - scale * (r @ mu_x)
    return scale, r, t


def icp_refine(fp: Footprint, mo: MeshOutline, params: dict,
               iters: int = 12, n: int = 400) -> dict:
    """2-D ICP: alternate correspondence and closed-form solve. Spec 6.8.

    Converges locally and quickly, which is exactly why 6.5's initialisation
    matters. Run this first (fast, gets close), then IoU refinement (slower,
    optimises the metric we are actually judged on).
    """
    target_ring = _resample(fp.pts_enu, n)
    src = _resample(mo.pts_enu, n)
    p = dict(params)

    from shapely.geometry import LinearRing, Point
    boundary = LinearRing(fp.pts_enu)

    for _ in range(iters):
        moved = apply_similarity(src, p["theta"], p["sx"], p["sy"], p["t"])
        # nearest point on the footprint boundary for each moved sample
        corr = np.array([
            np.asarray(boundary.interpolate(boundary.project(Point(q))).coords[0])
            for q in moved
        ])
        scale, r, t = umeyama_similarity(moved, corr)
        if scale <= 0 or not np.isfinite(scale):
            break

        delta = math.atan2(r[1, 0], r[0, 0])
        p["theta"] += delta
        p["sx"] *= scale
        p["sy"] *= scale
        p["t"] = (scale * (r @ p["t"])) + t

        if abs(delta) < 1e-6 and abs(scale - 1.0) < 1e-6:
            break
    return p


# --------------------------------------------------------------------------
# 6.8 / 6.9 Direct IoU maximisation with anisotropy regularisation
# --------------------------------------------------------------------------


ANISO_LAMBDA = 0.5      # spec 6.9 penalty weight
ANISO_EPS = 0.15        # spec 6.9 hard cap: |log(sx/sy)| <= log(1+eps)


def iou_refine(fp: Footprint, mo: MeshOutline, params: dict,
               *, allow_anisotropy: bool = False,
               max_evals: int = 400) -> dict:
    """Derived-free local maximisation of polygon IoU. Spec 6.8, 6.9.

    Polygon IoU is piecewise-smooth, so Nelder-Mead works where gradients do
    not. Scale is parameterised logarithmically: it makes the step size
    scale-invariant and keeps s > 0 without a constraint.

    Anisotropy is off by default. The challenge prefers proportion-preserving
    uniform scaling; when enabled, spec 6.9's squared-log-ratio penalty applies
    — symmetric under swapping the axes, scale-invariant, and zero exactly at
    isotropy.
    """
    target = _target_geom(fp)
    cap = math.log(1.0 + ANISO_EPS)

    def unpack(v):
        if allow_anisotropy:
            theta, tx, ty, log_sx, log_sy = v
            return theta, math.exp(log_sx), math.exp(log_sy), np.array([tx, ty])
        theta, tx, ty, log_s = v
        s = math.exp(log_s)
        return theta, s, s, np.array([tx, ty])

    def objective(v):
        theta, sx, sy, t = unpack(v)
        moved = apply_similarity(mo.pts_enu, theta, sx, sy, t)
        holes = tuple(apply_similarity(h, theta, sx, sy, t) for h in mo.holes_enu)
        j = iou(_poly(moved, holes), target)

        penalty = 0.0
        if allow_anisotropy:
            lr = math.log(sx / sy)
            if abs(lr) > cap:
                penalty += 1e3 * (abs(lr) - cap) ** 2   # the hard constraint
            penalty += ANISO_LAMBDA * lr ** 2           # the soft one
        return -j + penalty

    if allow_anisotropy:
        x0 = [params["theta"], params["t"][0], params["t"][1],
              math.log(params["sx"]), math.log(params["sy"])]
    else:
        x0 = [params["theta"], params["t"][0], params["t"][1],
              math.log(0.5 * (params["sx"] + params["sy"]))]

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxfev": max_evals, "xatol": 1e-4, "fatol": 1e-5})

    theta, sx, sy, t = unpack(res.x)
    return {"theta": theta, "sx": sx, "sy": sy, "t": t}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def solve(fp: Footprint, mo: MeshOutline,
          *,
          chosen: CandidateId | None = None,
          disambiguated_by: Disambiguator = Disambiguator.ASPECT_RATIO,
          use_icp: bool = True,
          use_iou_refine: bool = True,
          allow_anisotropy: bool = False) -> FitResult:
    """Full solve. -> FitResult in the ENU frame.

    `chosen` comes from geo.disambiguate; when None, the best footprint IoU
    wins, which is correct for asymmetric footprints and meaningless for square
    ones — hence the margin, which is recorded either way.
    """
    cands = ombb_candidates(fp, mo)
    scored = score_candidates(fp, mo, cands)
    lookup = dict(cands)

    if chosen is None:
        chosen = scored[0][0]
    params = dict(lookup[chosen])

    margin = scored[0][1] - scored[1][1] if len(scored) > 1 else 0.0
    stages = ["ombb"]

    if use_icp:
        params = icp_refine(fp, mo, params)
        stages.append("icp")
    if use_iou_refine:
        params = iou_refine(fp, mo, params, allow_anisotropy=allow_anisotropy)
        stages.append("nelder-mead")

    from geo.validate import compute_metrics
    metrics = compute_metrics(fp, mo, params)

    sx, sy = params["sx"], params["sy"]
    return FitResult(
        theta=float(params["theta"]),
        scale_x=float(sx),
        scale_y=float(sy),
        scale_z=1.0,               # set by geo.height once the height is known
        tx=float(params["t"][0]),
        ty=float(params["t"][1]),
        z_offset=0.0,              # set by geo.terrain
        iou=metrics["iou"],
        hausdorff_m=metrics["hausdorff_m"],
        area_ratio=metrics["area_ratio"],
        rotation_margin_footprint=float(margin),
        anisotropy_log_ratio=abs(math.log(sx / sy)) if sy > 0 else 0.0,
        max_neighbor_overlap=0.0,  # set by geo.validate.neighbour_overlap
        disambiguated_by=disambiguated_by,
        solver="+".join(stages),
        candidate_ious=tuple(scored),
    )
