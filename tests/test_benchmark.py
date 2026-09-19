"""The benchmark's truth must be exact, or every number it prints is noise."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from scripts.run_benchmark import CORRUPTIONS, corrupt, run_trial, to_labelled
from tests.test_fit_synthetic import LSHAPE, _fp


def test_clean_trial_recovers_the_known_pose():
    fp = _fp(LSHAPE)
    rows = run_trial(fp, "clean", np.random.default_rng(3))
    shipped = next(r for r in rows if r["config"] == "+nm")
    # An L-shape resolves uniquely, so the ABSOLUTE rotation must come back.
    assert shipped["rot_err_abs_deg"] < 0.5
    assert shipped["gt_iou"] > 0.99
    assert shipped["scale_err"] < 0.01
    assert shipped["label"] == 1


def test_mislabel_is_scored_as_wrong_orientation():
    fp = _fp(LSHAPE)
    rows = run_trial(fp, "mislabel_90", np.random.default_rng(3))
    ombb = next(r for r in rows if r["config"] == "ombb")
    assert abs(ombb["rot_err_abs_deg"] - 90) < 1
    assert ombb["label"] == 0


def test_every_corruption_produces_a_valid_polygon():
    F = Polygon(LSHAPE)
    rng = np.random.default_rng(0)
    for kind in CORRUPTIONS:
        g, _ = corrupt(F, kind, rng)
        assert g.is_valid and g.area > 0.3 * F.area, kind


def test_rows_convert_to_labelled_fits():
    fp = _fp(LSHAPE)
    rows = run_trial(fp, "noise", np.random.default_rng(1))
    for r in rows:
        r.update(_fp=fp, building_id=0)
    lf = to_labelled(rows)
    assert len(lf) == len(rows)
    assert all(x.kind.startswith("synthetic_gt:") for x in lf)
    assert all(math.isfinite(x.fit.iou) for x in lf)
