"""
The debug overlay. Spec 9.3.

"The overlay image is the highest-value artifact in your entire debugging
workflow — build it in hour three, not hour thirty."

Taken literally. This module exists before the solver is finished, because the
failure this project is really about is an IoU of 0.91 on a building that is
rotated 90 degrees, and no scalar catches that. A human looking at this PNG
catches it in under a second.

Matplotlib only. No OpenGL anywhere in this codebase.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from contracts import Decision, FitResult, Footprint, MeshOutline  # noqa: E402
from geo.fit import apply_similarity, ombb_candidates  # noqa: E402

FOOTPRINT_COLOUR = "#1b6ca8"
PLACED_COLOUR = "#e8590c"
NEIGHBOUR_COLOUR = "#adb5bd"
DECISION_COLOUR = {
    Decision.AUTO_ACCEPT: "#2f9e44",
    Decision.REVIEW: "#f08c00",
    Decision.REJECT: "#c92a2a",
}


def _ring(ax, pts, colour, label=None, lw=2.0, ls="-", fill=False, alpha=0.18):
    closed = np.vstack([pts, pts[:1]])
    if fill:
        ax.fill(closed[:, 0], closed[:, 1], color=colour, alpha=alpha, zorder=1)
    ax.plot(closed[:, 0], closed[:, 1], color=colour, lw=lw, ls=ls,
            label=label, zorder=3)


def render_overlay(out_path: Path,
                   fp: Footprint,
                   mo: MeshOutline,
                   fit: FitResult,
                   *,
                   neighbours: list[Polygon] | None = None,
                   decision: Decision = Decision.REVIEW,
                   reasons: list[str] | None = None,
                   address: str = "",
                   show_candidates: bool = True) -> Path:
    """Top-down overlay of F and T(M_0), plus the four candidate orientations.

    Spec 9.3 wants the candidates and their scores in the review payload so a
    human can pick the right one in a single click rather than re-running the
    solver. They are drawn faintly on the main axes and listed with scores.
    """
    reasons = reasons or []
    fig = plt.figure(figsize=(13, 7.5), dpi=110)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect("equal")

    for n in (neighbours or []):
        if n.is_empty:
            continue
        xs, ys = n.exterior.xy
        ax.fill(xs, ys, color=NEIGHBOUR_COLOUR, alpha=0.30, zorder=0)
        ax.plot(xs, ys, color=NEIGHBOUR_COLOUR, lw=1.0, zorder=1)

    # The four OMBB candidates, faint — so a wrong pick is visually obvious.
    if show_candidates:
        try:
            for cid, p in ombb_candidates(fp, mo):
                cand = apply_similarity(mo.pts_enu, p["theta"], p["sx"], p["sy"], p["t"])
                _ring(ax, cand, "#868e96", lw=0.8, ls=":")
        except Exception:  # noqa: BLE001 - never let the debug view crash the run
            pass

    _ring(ax, fp.pts_enu, FOOTPRINT_COLOUR, "authoritative footprint F",
          lw=2.4, fill=True)
    for h in fp.holes_enu:
        _ring(ax, h, FOOTPRINT_COLOUR, None, lw=1.6, ls="--")
    # Disjoint outer parts (Lane Stadium's stands). Drawn, because a multi-part
    # footprint that renders as one part looks like a solver bug.
    for part in fp.parts_enu:
        _ring(ax, part, FOOTPRINT_COLOUR, None, lw=2.0, fill=True)

    placed = apply_similarity(mo.pts_enu, fit.theta, fit.scale_x, fit.scale_y,
                              np.array([fit.tx, fit.ty]))
    _ring(ax, placed, PLACED_COLOUR, "placed mesh T(M0)", lw=2.4, fill=True, alpha=0.22)
    for h in mo.holes_enu:
        moved_h = apply_similarity(h, fit.theta, fit.scale_x, fit.scale_y,
                                   np.array([fit.tx, fit.ty]))
        _ring(ax, moved_h, PLACED_COLOUR, None, lw=1.6, ls="--")

    # North arrow, pinned to the axes corner. Orientation bugs are the whole
    # point of this image, so the reader must never have to infer which way is
    # up: ENU means +y is North by construction, and this says so.
    ax.annotate("", xy=(0.055, 0.96), xytext=(0.055, 0.86),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#343a40", lw=1.8))
    ax.text(0.055, 0.825, "N", transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color="#343a40", fontweight="bold")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(address or "placement overlay", fontsize=12, pad=10)
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ---- the metrics panel -------------------------------------------------
    tx = fig.add_subplot(gs[0, 1])
    tx.axis("off")

    colour = DECISION_COLOUR.get(decision, "#495057")
    tx.text(0.0, 1.0, decision.value.replace("_", " ").upper(),
            fontsize=17, fontweight="bold", color=colour,
            va="top", transform=tx.transAxes)

    rows = [
        ("footprint IoU", f"{fit.iou:.3f}", 0.75),
        ("Hausdorff (m)", f"{fit.hausdorff_m:.2f}", None),
        ("area ratio", f"{fit.area_ratio:.3f}", None),
        ("rotation margin", f"{fit.rotation_margin_footprint:.3f}", 0.05),
        ("rectilinearity R", f"{fp.rectilinearity:.3f}", 0.75),
        ("footprint parts", f"{1 + len(fp.parts_enu)}", None),
        ("match quality", fp.match_quality or "unknown", None),
        ("anisotropy |log|", f"{fit.anisotropy_log_ratio:.4f}", None),
        ("neighbour overlap", f"{fit.max_neighbor_overlap:.4f}", None),
        ("", "", None),
        ("theta (deg)", f"{np.degrees(fit.theta) % 360:.2f}", None),
        ("heading (deg)", f"{np.degrees(fit.heading_rad):.2f}", None),
        ("scale x, y", f"{fit.scale_x:.4f}, {fit.scale_y:.4f}", None),
        ("height (m)", f"{fit.height_m:.2f}", None),
        ("height source", fit.height_source.value, None),
        ("disambiguated by", fit.disambiguated_by.value, None),
        ("solver", fit.solver, None),
    ]

    y = 0.92
    for name, value, _ in rows:
        if name:
            tx.text(0.0, y, name, fontsize=9.5, color="#495057",
                    va="top", transform=tx.transAxes)
            tx.text(0.62, y, value, fontsize=9.5, color="#212529",
                    va="top", fontfamily="monospace", transform=tx.transAxes)
        y -= 0.042

    if fit.candidate_ious:
        y -= 0.015
        tx.text(0.0, y, "candidate orientations (spec 9.3)", fontsize=9.5,
                fontweight="bold", color="#495057", va="top", transform=tx.transAxes)
        y -= 0.040
        for cid, score in fit.candidate_ious:
            tx.text(0.02, y, f"k={cid.azimuth_k}  up={cid.up_axis_idx}",
                    fontsize=9, color="#495057", va="top",
                    fontfamily="monospace", transform=tx.transAxes)
            tx.text(0.62, y, f"{score:.3f}", fontsize=9, color="#212529",
                    va="top", fontfamily="monospace", transform=tx.transAxes)
            y -= 0.036

    if reasons:
        y -= 0.015
        tx.text(0.0, y, "flags", fontsize=9.5, fontweight="bold",
                color=colour, va="top", transform=tx.transAxes)
        y -= 0.038
        # matplotlib's wrap=True does not respect transAxes widths, so wrap by
        # hand. This panel is read more than any other output in the project.
        for r in reasons[:6]:
            for i, line in enumerate(textwrap.wrap(r, width=58)):
                tx.text(0.02 if i == 0 else 0.035, y,
                        f"- {line}" if i == 0 else line,
                        fontsize=8.0, color="#495057", va="top",
                        transform=tx.transAxes)
                y -= 0.028
                if y < 0.02:
                    return _finish(fig, out_path)
            y -= 0.008

    return _finish(fig, out_path)


def _finish(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
