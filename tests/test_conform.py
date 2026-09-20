"""--conform: stretch a shallow mesh onto the footprint, and say so.

Single-view generators return a shell that is too shallow — NCB came back
1.59:1 where the building is 2.22:1 — and a uniform scale cannot fix a wrong
aspect. Conform allows a per-axis scale; these tests pin down that it only
happens when asked, that it stays inside its cap, and that the default solve is
untouched.
"""

from __future__ import annotations

import math

import numpy as np

from contracts import Disambiguator
from geo.fit import iou_refine, ombb_candidates, solve
from pipeline import CONFORM_ANISO_CAP
from tests.test_fit_synthetic import RECT, _fp, _mo


def _squashed(ratio: float) -> np.ndarray:
    """RECT (30x18) squashed along its long axis: the generated-mesh failure."""
    pts = RECT.astype(float).copy()
    pts[:, 0] /= ratio
    return pts / 30.0          # model units, ~1 across


def _first(fp, mo):
    return ombb_candidates(fp, mo)[0][0]


def test_conform_recovers_a_squashed_mesh():
    fp = _fp(RECT)
    mo = _mo(_squashed(1.4))
    chosen = _first(fp, mo)

    uniform = solve(fp, mo, chosen=chosen, disambiguated_by=Disambiguator.EXIF_HEADING)
    conformed = solve(fp, mo, chosen=chosen, disambiguated_by=Disambiguator.EXIF_HEADING,
                      allow_anisotropy=True, aniso_cap=CONFORM_ANISO_CAP)

    assert conformed.iou > uniform.iou + 0.1
    # 0.94, not 0.99: Nelder-Mead stops at its tolerance, and the squashed
    # outline's corners never line up exactly. Measured 0.949 at this ratio.
    assert conformed.iou >= 0.94
    # The stretch is ~the squash that was applied, and it is visible in the fit.
    assert 1.2 <= math.exp(abs(conformed.anisotropy_log_ratio)) <= 1.6
    assert conformed.solver.endswith("conform")


def test_the_cap_is_respected():
    fp = _fp(RECT)
    mo = _mo(_squashed(4.0))            # far more squashed than the cap allows
    conformed = solve(fp, mo, chosen=_first(fp, mo),
                      disambiguated_by=Disambiguator.EXIF_HEADING,
                      allow_anisotropy=True, aniso_cap=CONFORM_ANISO_CAP)
    assert abs(conformed.anisotropy_log_ratio) <= CONFORM_ANISO_CAP + 1e-3


def test_default_solve_is_unchanged_and_stays_uniform():
    fp = _fp(RECT)
    mo = _mo(_squashed(1.4))
    r = solve(fp, mo, chosen=_first(fp, mo), disambiguated_by=Disambiguator.EXIF_HEADING)
    assert r.scale_x == r.scale_y            # uniform: no stretch without asking
    assert r.anisotropy_log_ratio == 0.0
    assert r.solver == "ombb+icp+nelder-mead"


def test_iou_refine_keeps_the_spec_cap_by_default():
    """Anisotropy without a cap override obeys spec 6.9's 15%."""
    fp = _fp(RECT)
    mo = _mo(_squashed(1.4))
    params = dict(ombb_candidates(fp, mo)[0][1])
    out = iou_refine(fp, mo, params, allow_anisotropy=True)
    assert abs(math.log(out["sx"] / out["sy"])) <= math.log(1.15) + 1e-3
