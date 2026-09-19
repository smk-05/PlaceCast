"""
The spec 11 benchmark: synthetic perturbation on the 20 REAL footprints.

Why synthetic: a generated mesh has no ground-truth pose, so measuring the
solver on TRELLIS output would measure TRELLIS. Instead each real OSM footprint
F is turned into a fake "mesh outline" with a KNOWN pose,

    M = R(-theta) (F - c_F) / s + c_M          (model units, ~1 across)

then deliberately damaged the way generated meshes are damaged, and handed to
fit.solve. The truth is exact, so every error below is the solver's.

Corruptions (each models an observed failure, not a hypothetical one):
    clean          the pose alone
    noise          boundary jitter, sigma 0.5 m (raster outline extraction)
    lobe           an extra blob on one side (NCB's 12x15 m scenery lobe)
    missing_wing   a corner block cut away (a wing TRELLIS never saw)
    anisotropy     the mesh stretched 15% along its own x axis
    mirror         a reflected mesh; has no correct placement, must not accept
    mislabel_90    the orientation filter picked a 90-degree-wrong candidate

Ablations, via fit.solve's existing flags (spec 11):
    ombb           closed-form OMBB alignment only
    +icp           + Umeyama/ICP boundary refinement
    +nm            + Nelder-Mead IoU refinement      (the shipped solver)
    +nm_aniso      + per-axis scale allowed

Scoring is against the TRUTH, not against what the solver reports:
    gt_iou        IoU of the placed outline against F. For shape corruptions
                  the UNDAMAGED outline is placed with the recovered pose, so a
                  lobe does not count against the pose; for anisotropy/mirror
                  the damaged outline is placed (the right answer undoes the
                  stretch; a mirror has no right answer).
    rot_err_abs   |theta_hat - theta_true|, wrapped to [0, 180]
    rot_err_mod90 the same, modulo 90 degrees. The gap between the two is the
                  price of the four-fold OMBB ambiguity, which IoU cannot see.
    label         1 iff gt_iou >= 0.80 and rot_err_abs < 10 deg, and not a
                  mirror that is visible in plan (IoU(mirror(F), F) < 0.95).

The gate decision is Owen's confidence.gate.evaluate_gate on the threshold
table, with disambiguated_by=ASPECT_RATIO (the footprint-IoU pick; no photo in
a synthetic run) — so a small rotation margin routes to review, as in
production. The headline safety number is the FALSE-ACCEPT rate: auto-accepted
rows whose label is 0.

    uv run python scripts/run_benchmark.py              # ~minutes, all 20
    uv run python scripts/run_benchmark.py --reps 3     # more poses per cell
    uv run python scripts/run_benchmark.py --quick      # 5 buildings, smoke

Also emits addendum B training rows (fixtures.fake_fits.LabelledFit) to
outputs/benchmark/labelled_fits.pkl; `benchmark_labelled_rows()` rebuilds them.
Strata are MEASURED per footprint (measured_stratum), never taken from names.
There is no "sloped" stratum: slope matters only to terrain height, which this
2D benchmark does not exercise.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from contracts import (  # noqa: E402
    CandidateId,
    Disambiguator,
    Footprint,
    MeshOutline,
    PhotoEvidence,
)
from geo.coords import ENUFrame  # noqa: E402
from geo.fit import apply_similarity, iou, ombb_candidates, solve  # noqa: E402
from geo.footprint import (  # noqa: E402
    build_footprint,
    compute_ombb,
    fetch_osm,
    geocode,
    select_footprint,
)
from geo.validate import symmetric_hausdorff  # noqa: E402
from scripts.demo_addresses import BENCHMARK  # noqa: E402

OUT_DIR = ROOT / "outputs" / "benchmark"

CONFIGS = {
    "ombb":      dict(use_icp=False, use_iou_refine=False, allow_anisotropy=False),
    "+icp":      dict(use_icp=True,  use_iou_refine=False, allow_anisotropy=False),
    "+nm":       dict(use_icp=True,  use_iou_refine=True,  allow_anisotropy=False),
    "+nm_aniso": dict(use_icp=True,  use_iou_refine=True,  allow_anisotropy=True),
}
CORRUPTIONS = ("clean", "noise", "lobe", "missing_wing", "anisotropy", "mirror",
               "mislabel_90")

LABEL_IOU = 0.80
LABEL_ROT_DEG = 10.0


# --------------------------------------------------------------------------
# Footprints
# --------------------------------------------------------------------------


def measured_stratum(fp: Footprint) -> str:
    """The same rule as demo_addresses' comment: shape, not building name."""
    if fp.rectilinearity < 0.85:
        return "complex"
    if fp.ombb.aspect < 1.15:
        return "near_square"
    return "rectangular"


def load_footprints(addresses: list[str]) -> list[tuple[str, str, Footprint]]:
    """-> [(address, stratum, footprint)] from the committed cache.

    Multi-part footprints are reduced to their largest part: a MeshOutline is a
    single ring, so the other parts would sit in the IoU denominator with no
    way to be matched, and that would be a benchmark artefact, not a solver
    error. (Lane Stadium is the one affected.)
    """
    out = []
    for addr in addresses:
        try:
            g = geocode(addr)
            payload = fetch_osm(g["lat"], g["lon"])
            element, poly, _ = select_footprint(payload, g["lat"], g["lon"], addr)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skip {addr}: {type(exc).__name__}: {exc}")
            continue
        frame = ENUFrame(lat0=poly.centroid.y, lon0=poly.centroid.x)
        fp = build_footprint(poly, frame,
                             match_quality=element.get("_match_quality", ""),
                             geocode_location_type=g["location_type"])
        if fp.parts_enu:
            primary = Polygon(fp.pts_enu, list(fp.holes_enu))
            fp = dataclasses.replace(fp, parts_enu=(), area_m2=float(primary.area))
        out.append((addr, measured_stratum(fp), fp))
    return out


# --------------------------------------------------------------------------
# Synthetic meshes
# --------------------------------------------------------------------------


def _largest(g) -> Polygon:
    if isinstance(g, MultiPolygon):
        g = max(g.geoms, key=lambda p: p.area)
    return g if g.is_valid else g.buffer(0)


def _to_model(poly: Polygon, theta: float, s: float, c_f: np.ndarray,
              c_m: np.ndarray) -> Polygon:
    """F (metres, ENU) -> M (model units) under the known pose."""
    g = affinity.translate(poly, -c_f[0], -c_f[1])
    g = affinity.rotate(g, -theta, origin=(0, 0), use_radians=True)
    g = affinity.scale(g, 1 / s, 1 / s, origin=(0, 0))
    return affinity.translate(g, c_m[0], c_m[1])


def _jitter(poly: Polygon, sigma: float, rng) -> Polygon:
    dense = poly.segmentize(max(sigma, 1e-6))
    ext = np.asarray(dense.exterior.coords)[:-1]
    ext = ext + rng.normal(0, sigma, ext.shape)
    holes = [np.asarray(h.coords)[:-1] + rng.normal(0, sigma, (len(h.coords) - 1, 2))
             for h in dense.interiors]
    return _largest(Polygon(ext, holes).buffer(0))


def corrupt(F: Polygon, kind: str, rng) -> tuple[Polygon, bool]:
    """Damage the footprint in METRES (so noise/lobe sizes are physical).

    -> (damaged polygon, damage_is_geometric). Geometric damage (stretch,
    mirror) is something the placement is expected to undo or refuse, so the
    damaged outline is what gets scored; shape damage is not the solver's to
    undo, so the clean outline is scored with the recovered pose.
    """
    ob = compute_ombb(np.asarray(F.exterior.coords)[:-1])
    u, v = ob.u, ob.v
    if kind in ("clean", "mislabel_90"):
        return F, False
    if kind == "noise":
        return _jitter(F, 0.5, rng), False
    if kind == "lobe":
        # A blob centred on the boundary, ~NCB's lobe relative to its building.
        pt = F.exterior.interpolate(rng.uniform(0, F.exterior.length))
        r = 0.18 * math.sqrt(F.area)
        return _largest(F.union(Point(pt.x, pt.y).buffer(r, 16))), False
    if kind == "missing_wing":
        sx, sy = rng.choice([-1, 1]), rng.choice([-1, 1])
        corner = ob.centre + sx * ob.a / 2 * u + sy * ob.b / 2 * v
        w, h = 0.4 * ob.a, 0.4 * ob.b
        cut = Polygon([corner, corner - sx * w * u, corner - sx * w * u - sy * h * v,
                       corner - sy * h * v])
        return _largest(F.difference(cut)), False
    if kind == "anisotropy":
        # 15% along the building's own long axis: rotate to the OMBB frame,
        # stretch, rotate back.
        ang = math.atan2(u[1], u[0])
        g = affinity.rotate(F, -ang, origin=tuple(ob.centre), use_radians=True)
        g = affinity.scale(g, 1.15, 1.0, origin=tuple(ob.centre))
        return affinity.rotate(g, ang, origin=tuple(ob.centre), use_radians=True), True
    if kind == "mirror":
        ang = math.atan2(u[1], u[0])
        g = affinity.rotate(F, -ang, origin=tuple(ob.centre), use_radians=True)
        # Reflect across the SHORT axis: a building mirrored end-to-end. An
        # exactly symmetric footprint makes this a no-op; that is honest.
        g = affinity.scale(g, -1.0, 1.0, origin=tuple(ob.centre))
        return affinity.rotate(g, ang, origin=tuple(ob.centre), use_radians=True), True
    raise ValueError(kind)


def mesh_outline(poly: Polygon) -> MeshOutline:
    pts = np.asarray(poly.exterior.coords)[:-1]
    holes = tuple(np.asarray(h.coords)[:-1] for h in poly.interiors)
    return MeshOutline(pts_enu=pts, holes_enu=holes, ombb=compute_ombb(pts),
                       up_axis_idx=4)


def _wrap_deg(d: float, period: float) -> float:
    d = (d + period / 2) % period - period / 2
    return abs(d)


def _true_candidate(fp: Footprint, mo: MeshOutline, theta_true: float) -> CandidateId:
    best, best_d = None, 1e9
    for cid, p in ombb_candidates(fp, mo):
        d = _wrap_deg(math.degrees(p["theta"] - theta_true), 360)
        if d < best_d:
            best, best_d = cid, d
    return best


# --------------------------------------------------------------------------
# One trial
# --------------------------------------------------------------------------


def run_trial(fp: Footprint, kind: str, rng) -> list[dict]:
    """One random pose + one corruption, solved under every config."""
    from confidence.gate import evaluate_gate

    F = Polygon(fp.pts_enu, list(fp.holes_enu))
    c_f = np.array(F.centroid.coords[0])
    theta_true = float(rng.uniform(0, 2 * math.pi))
    s_true = float(fp.ombb.a) * float(np.exp(rng.uniform(-0.3, 0.3)))  # mesh ~1 unit
    c_m = rng.normal(0, 0.1, 2)

    damaged, geometric = corrupt(F, kind, rng)
    M_seen = _to_model(damaged, theta_true, s_true, c_f, c_m)
    M_eval = M_seen if geometric else _to_model(F, theta_true, s_true, c_f, c_m)
    mo = mesh_outline(M_seen)
    # A mirror of a plan-symmetric footprint is the same outline; in 2D it is
    # not a defect, and labelling it bad would invent false accepts.
    visible_mirror = kind == "mirror" and iou(damaged, F) < 0.95
    eval_pts = np.asarray(M_eval.exterior.coords)[:-1]
    eval_holes = [np.asarray(h.coords)[:-1] for h in M_eval.interiors]

    chosen = None
    if kind == "mislabel_90":
        k = _true_candidate(fp, mo, theta_true)
        chosen = CandidateId(k.up_axis_idx, (k.azimuth_k + 1) % 4)

    rows = []
    for cfg_i, (cfg, flags) in enumerate(CONFIGS.items()):
        t0 = time.perf_counter()
        fit = solve(fp, mo, chosen=chosen,
                    disambiguated_by=Disambiguator.ASPECT_RATIO, **flags)
        ms = 1000 * (time.perf_counter() - t0)
        # A mirror: the solver works in SO(2) and cannot express a reflection,
        # so fit.is_mirrored never fires here. What is measured is whether the
        # 2D geometry alone would let a mirrored building through the gate.

        placed = Polygon(
            apply_similarity(eval_pts, fit.theta, fit.scale_x, fit.scale_y,
                             np.array([fit.tx, fit.ty])),
            [apply_similarity(h, fit.theta, fit.scale_x, fit.scale_y,
                              np.array([fit.tx, fit.ty])) for h in eval_holes])
        placed = placed if placed.is_valid else placed.buffer(0)
        gt_iou = iou(placed, F)
        gt_haus = symmetric_hausdorff(placed, F)
        rot_abs = _wrap_deg(math.degrees(fit.theta - theta_true), 360)
        rot_90 = _wrap_deg(math.degrees(fit.theta - theta_true), 90)
        label = int(gt_iou >= LABEL_IOU and rot_abs < LABEL_ROT_DEG
                    and not visible_mirror)

        gate = evaluate_gate(fit, None, fp, method="threshold_table")
        rows.append(dict(
            corruption=kind, config=cfg, config_index=cfg_i,
            gt_iou=gt_iou, gt_hausdorff_m=gt_haus,
            rot_err_abs_deg=rot_abs, rot_err_mod90_deg=rot_90,
            scale_err=abs(math.sqrt(abs(fit.scale_x * fit.scale_y)) / s_true - 1),
            fit_iou=fit.iou, fit_hausdorff_m=fit.hausdorff_m,
            rotation_margin=fit.rotation_margin_footprint,
            decision=gate.decision.value, label=label, ms=ms,
            _fit=fit,
        ))
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _q(xs, q):
    xs = sorted(x for x in xs if math.isfinite(x))
    if not xs:
        return float("nan")
    return float(np.quantile(xs, q))


def summarise(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    out = []
    for name, rs in groups.items():
        acc = [r for r in rs if r["decision"] == "auto_accept"]
        out.append(dict(
            name=name, n=len(rs),
            iou_med=_q([r["gt_iou"] for r in rs], 0.5),
            iou_p10=_q([r["gt_iou"] for r in rs], 0.1),
            haus_p90=_q([r["gt_hausdorff_m"] for r in rs], 0.9),
            rot90_p90=_q([r["rot_err_mod90_deg"] for r in rs], 0.9),
            rot_abs_med=_q([r["rot_err_abs_deg"] for r in rs], 0.5),
            orient_ok=sum(r["rot_err_abs_deg"] < LABEL_ROT_DEG for r in rs) / len(rs),
            good=sum(r["label"] for r in rs) / len(rs),
            accept=len(acc) / len(rs),
            false_accept=(sum(1 - r["label"] for r in acc) / len(acc)) if acc else 0.0,
            ms_med=_q([r["ms"] for r in rs], 0.5),
        ))
    return out


HEADER = ("| {k} | n | median IoU | p10 IoU | p90 Hausdorff | p90 rot err mod 90° "
          "| orientation correct (<10°) | good placements | auto-accepted "
          "| false accepts | median ms |")


def markdown_table(summary: list[dict], key: str) -> str:
    lines = [HEADER.format(k=key), "|" + "---|" * 11]
    for s in summary:
        lines.append(
            f"| {s['name']} | {s['n']} | {s['iou_med']:.3f} | {s['iou_p10']:.3f} "
            f"| {s['haus_p90']:.2f} m | {s['rot90_p90']:.2f}° | {s['orient_ok']:.0%} "
            f"| {s['good']:.0%} | {s['accept']:.0%} | {s['false_accept']:.0%} "
            f"| {s['ms_med']:.0f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Addendum B rows
# --------------------------------------------------------------------------


def _no_photo() -> PhotoEvidence:
    return PhotoEvidence(mask=np.zeros((32, 32), bool),
                         segmentation_model="none:synthetic_benchmark")


def to_labelled(rows: list[dict]) -> list:
    """-> [fixtures.fake_fits.LabelledFit], source = synthetic ground truth."""
    from fixtures.fake_fits import LabelledFit
    photo = _no_photo()
    return [LabelledFit(fit=r["_fit"], photo=photo, footprint=r["_fp"],
                        label=r["label"], kind=f"synthetic_gt:{r['corruption']}",
                        building_id=r["building_id"], config_index=r["config_index"],
                        label_flipped=False)
            for r in rows]


def benchmark_labelled_rows(seed: int = 0, reps: int = 1) -> list:
    """For confidence/train.py's load_rows: the real-footprint benchmark rows."""
    return to_labelled(run(seed=seed, reps=reps, quiet=True)[0])


# --------------------------------------------------------------------------


def run(*, seed: int = 0, reps: int = 1, quick: bool = False, quiet: bool = False):
    addresses = list(BENCHMARK)
    if quick:
        addresses = addresses[::4]
    fps = load_footprints(addresses)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for b_id, (addr, stratum, fp) in enumerate(fps):
        t0 = time.perf_counter()
        for kind in CORRUPTIONS:
            for _ in range(reps):
                for r in run_trial(fp, kind, rng):
                    r.update(address=addr, stratum=stratum, building_id=b_id, _fp=fp)
                    rows.append(r)
        if not quiet:
            print(f"  {addr:<42} {stratum:<12} R={fp.rectilinearity:.3f} "
                  f"aspect={fp.ombb.aspect:.2f}  {time.perf_counter() - t0:5.1f} s")
    return rows, fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1, help="random poses per cell")
    ap.add_argument("--quick", action="store_true", help="5 buildings")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--markdown", default=str(ROOT / "BENCHMARK.md"),
                    help="where to write the committed summary ('' to skip)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # the Windows console is cp1252

    print("Benchmark: synthetic perturbation on the real footprints\n")
    rows, fps = run(seed=args.seed, reps=args.reps, quick=args.quick)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = [k for k in rows[0] if not k.startswith("_")]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    shipped = [r for r in rows if r["config"] == "+nm"]
    by_config = markdown_table(summarise(rows, "config"), "config")
    by_corruption = markdown_table(summarise(shipped, "corruption"), "corruption (+nm)")
    by_stratum = markdown_table(summarise(shipped, "stratum"), "stratum (+nm)")

    false_accepts = [r for r in rows
                     if r["decision"] == "auto_accept" and r["label"] == 0]
    fa_lines = "\n".join(
        f"- {r['address'].split(',')[0]} · {r['corruption']} · {r['config']}: "
        f"IoU vs truth {r['gt_iou']:.3f}, rotation off by {r['rot_err_abs_deg']:.0f}°, "
        f"solver saw IoU {r['fit_iou']:.3f}, margin {r['rotation_margin']:.3f}"
        for r in false_accepts) or "- none"
    per_building = "\n".join(
        f"| {addr.split(',')[0]} | {stratum} | {fp.rectilinearity:.3f} "
        f"| {fp.ombb.aspect:.2f} | {fp.area_m2:.0f} |"
        for addr, stratum, fp in fps)

    labelled = to_labelled(rows)
    with open(out / "labelled_fits.pkl", "wb") as f:
        pickle.dump(labelled, f)

    report = f"""# Benchmark — synthetic perturbation on {len(fps)} real footprints

Generated by `uv run python scripts/run_benchmark.py --seed {args.seed} --reps {args.reps}`.
{len(rows)} solves: {len(fps)} buildings × {len(CORRUPTIONS)} corruptions × {args.reps} pose(s) × {len(CONFIGS)} solver configs.

Each real OSM footprint is turned into a fake mesh outline with a known random
pose (rotation, scale, offset), damaged, and solved. Errors are measured against
the known truth. "Good" = IoU vs truth ≥ {LABEL_IOU} **and** rotation error < {LABEL_ROT_DEG:.0f}°.
A mirrored mesh is never good unless its plan is symmetric (then the mirror is invisible in 2D). "False accepts" = auto-accepted placements that are not good,
as a share of all auto-accepts: the safety number.

The gap between *rotation error mod 90°* and *orientation correct* is the
four-fold OMBB ambiguity: the outline fits, but the building may face the wrong
way, and IoU cannot tell. That gap is what EXIF headings and silhouettes close.

## Ablation (all corruptions)

{by_config}

## By corruption (shipped solver, +nm)

{by_corruption}

## By stratum (shipped solver, +nm)

{by_stratum}

## Every false accept ({len(false_accepts)} of {len(rows)} solves)

{fa_lines}

## Buildings (stratum measured, not named)

| building | stratum | R | aspect | area m² |
|---|---|---|---|---|
{per_building}
"""
    (out / "summary.md").write_text(report, encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"rows: {out / 'results.csv'}\nlabelled fits ({len(labelled)}): "
          f"{out / 'labelled_fits.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
