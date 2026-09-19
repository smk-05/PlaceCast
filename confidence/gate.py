"""Confidence gate: SPEC 9.1 threshold table (default) with the learned model behind a flag (ML_ADDENDUM B.4).

    probability, decision = score_confidence(fit, photo, fp)            # ML_ADDENDUM F.1 signature
    result = evaluate_gate(fit, photo, fp, ...)                          # the inspectable version

decision is one of "auto_accept", "review", "reject".

The table is the default and works standalone: pure Python, no numpy / scikit-learn, no model file, no
training. It never returns a probability (the table has none; probability is NaN from score_confidence,
None on the result), because a made-up number would look like evidence.

Table (SPEC 9.1). Each metric lands in a band; the decision is the WORST band across metrics, so a fit
auto-accepts only if every metric does, and IoU and Hausdorff must both pass:

  footprint IoU            accept >= 0.75    review 0.50-0.75   reject < 0.50
  symmetric Hausdorff (m)  accept <= 2.0     review 2-5         reject > 5
  area ratio               accept 0.85-1.15  review 0.7-1.3     reject outside
  rotation margin          accept >= 0.05    review < 0.05      (never rejects)
  rectilinearity           accept >= 0.75    review < 0.75      (never rejects)
  anisotropy |log(sx/sy)|  accept <= 0.05    review 0.05-0.15   reject >= 0.15
  neighbour overlap        accept <= 0.02    review 0.02-0.10   reject > 0.10

Boundaries follow the spec's symbols (>= / <= belong to accept; a value equal to a review/reject edge
takes the milder band, except anisotropy 0.15 which the spec rejects). The spec leaves 0.14-0.15 of
anisotropy unassigned; it is review, since reject starts at 0.15. Metrics must be finite: NaN would
compare False against every edge and silently pick a band, so it raises.

Learned model (B.4): a second code path, selected by PROCEDURA_LEARNED_GATE=1 or use_learned=True. It
reads the model.json written by confidence/train.py: p >= p_accept -> auto_accept, p <= p_reject ->
reject, else review; a threshold that training could not set (None) is never applied, so a model that
cannot reach the precision target never auto-accepts. If the flag is on and the model is missing,
malformed, or trained on synthetic data, this RAISES: silently using the table would leave someone
believing the learned gate is live. To fall back, unset the flag. The learned path needs the two
binaries geocode_rooftop / height_source_authoritative when the model uses them (they are not in
the F.1 objects). The table's own decision is always computed and returned as `table_decision`, so
disagreements between model and table can be reported as findings (B.4).

Hard rules, applied on BOTH paths (they can only turn auto_accept into review, never upgrade):
  * orientation was guessed (fit.disambiguated_by in road_normal / arbitrary_symmetric / unresolved):
    A.3 and A.4 say flag for review regardless of IoU.
  * rectilinearity < 0.6 (SPEC 10.2 #8): round or organic building, rotation is meaningless.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

AUTO_ACCEPT, REVIEW, REJECT = "auto_accept", "review", "reject"
_RANK = {AUTO_ACCEPT: 0, REVIEW: 1, REJECT: 2}

ENV_FLAG = "PROCEDURA_LEARNED_GATE"
ENV_MODEL = "PROCEDURA_GATE_MODEL"
DEFAULT_MODEL_PATH = ROOT / "outputs" / "confidence" / "model.json"
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"", "0", "false", "no", "off"}

GUESSED_ORIENTATION = frozenset({"road_normal", "arbitrary_symmetric", "unresolved"})
ROUND_BUILDING_RECTILINEARITY = 0.6


# ---------------------------------------------------------------- data types


@dataclass(frozen=True)
class MetricCheck:
    name: str
    value: float
    band: str  # auto_accept | review | reject
    rule: str  # the table row, for the review payload (SPEC 9.3)


@dataclass(frozen=True)
class GateDecision:
    decision: str
    probability: Optional[float]  # None on the table path
    path: str  # "table" | "learned"
    table_decision: str  # what the SPEC 9.1 table says, whichever path decided
    checks: tuple  # every table metric with its band (always computed)
    reasons: tuple  # why the decision is not a clean auto_accept
    forced_review: tuple  # hard rules that fired


# ------------------------------------------------------------------ the table


def _band_iou(v):
    return AUTO_ACCEPT if v >= 0.75 else REVIEW if v >= 0.50 else REJECT


def _band_hausdorff(v):
    return AUTO_ACCEPT if v <= 2.0 else REVIEW if v <= 5.0 else REJECT


def _band_area_ratio(v):
    return AUTO_ACCEPT if 0.85 <= v <= 1.15 else REVIEW if 0.70 <= v <= 1.30 else REJECT


def _band_rotation_margin(v):
    return AUTO_ACCEPT if v >= 0.05 else REVIEW


def _band_rectilinearity(v):
    return AUTO_ACCEPT if v >= 0.75 else REVIEW


def _band_anisotropy(v):
    return AUTO_ACCEPT if v <= 0.05 else REJECT if v >= 0.15 else REVIEW


def _band_overlap(v):
    return AUTO_ACCEPT if v <= 0.02 else REVIEW if v <= 0.10 else REJECT


# (name, band function, rule text)
TABLE = (
    ("footprint_iou", _band_iou, "accept >= 0.75, review 0.50-0.75, reject < 0.50"),
    ("hausdorff_m", _band_hausdorff, "accept <= 2.0 m, review 2-5 m, reject > 5 m"),
    ("area_ratio", _band_area_ratio, "accept 0.85-1.15, review 0.7-1.3, reject outside"),
    ("rotation_margin_footprint", _band_rotation_margin, "accept >= 0.05, review < 0.05"),
    ("rectilinearity", _band_rectilinearity, "accept >= 0.75, review < 0.75"),
    ("anisotropy_log_ratio", _band_anisotropy, "accept <= 0.05, review 0.05-0.15, reject >= 0.15 (|log(sx/sy)|)"),
    ("max_neighbor_overlap", _band_overlap, "accept <= 0.02, review 0.02-0.10, reject > 0.10"),
)


def _finite(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"gate metric {name} is not finite: {value!r}")
    return value


def _metrics(fit, fp):
    area_ratio = _finite("area_ratio", fit.area_ratio)
    if area_ratio < 0:
        raise ValueError(f"gate metric area_ratio cannot be negative: {area_ratio!r}")
    return {
        "footprint_iou": _finite("footprint_iou", fit.iou),
        "hausdorff_m": _finite("hausdorff_m", fit.hausdorff_m),
        "area_ratio": area_ratio,
        "rotation_margin_footprint": _finite("rotation_margin_footprint", fit.rotation_margin_footprint),
        "rectilinearity": _finite("rectilinearity", fp.rectilinearity),
        "anisotropy_log_ratio": abs(_finite("anisotropy_log_ratio", fit.anisotropy_log_ratio)),
        "max_neighbor_overlap": _finite("max_neighbor_overlap", fit.max_neighbor_overlap),
    }


def table_checks(fit, fp):
    """SPEC 9.1: every metric with its band."""
    values = _metrics(fit, fp)
    return tuple(MetricCheck(name, values[name], band(values[name]), rule) for name, band, rule in TABLE)


def worst(bands):
    return max(bands, key=_RANK.__getitem__)


def hard_review_reasons(fit, fp):
    reasons = []
    if fit.disambiguated_by in GUESSED_ORIENTATION:
        reasons.append(f"orientation was guessed (disambiguated_by={fit.disambiguated_by}); review regardless of IoU (A.3/A.4)")
    if _finite("rectilinearity", fp.rectilinearity) < ROUND_BUILDING_RECTILINEARITY:
        reasons.append(f"rectilinearity {fp.rectilinearity:.2f} < {ROUND_BUILDING_RECTILINEARITY}: round/organic building, rotation is meaningless (SPEC 10.2 #8)")
    return tuple(reasons)


# --------------------------------------------------------------- learned model


def learned_gate_enabled():
    raw = os.environ.get(ENV_FLAG, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{ENV_FLAG}={raw!r} is not a recognised value; use 1/0")


@dataclass(frozen=True)
class LearnedModel:
    feature_names: tuple
    mean: tuple
    scale: tuple
    coef: tuple
    intercept: float
    p_accept: Optional[float]
    p_reject: Optional[float]
    trained_on_synthetic: bool

    def probability(self, values):
        z = self.intercept + sum(c * (values[n] - m) / s for n, c, m, s in zip(self.feature_names, self.coef, self.mean, self.scale))
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def load_model(path=None, *, allow_synthetic=False):
    """Read the model.json written by confidence/train.py. Raises rather than guessing."""
    import json

    path = Path(path or os.environ.get(ENV_MODEL) or DEFAULT_MODEL_PATH)
    hint = f"train one with confidence/train.py, or unset {ENV_FLAG} to use the SPEC 9.1 table"
    if not path.is_file():
        raise FileNotFoundError(f"learned gate is enabled but there is no model at {path}; {hint}")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"learned gate model {path} is not valid JSON: {exc}") from exc
    needed = ("feature_names", "scaler_mean", "scaler_scale", "coef_standardised", "intercept", "thresholds", "trained_on_synthetic")
    missing = [k for k in needed if k not in spec]
    if missing:
        raise ValueError(f"learned gate model {path} is missing {missing}")
    if spec["trained_on_synthetic"] and not allow_synthetic:
        raise ValueError(f"learned gate model {path} was trained on SYNTHETIC data (fixtures/fake_fits.py) and must not gate real placements; {hint}")
    names, n = tuple(spec["feature_names"]), len(spec["feature_names"])
    if not (len(spec["scaler_mean"]) == len(spec["scaler_scale"]) == len(spec["coef_standardised"]) == n) or n == 0:
        raise ValueError(f"learned gate model {path}: feature/scaler/coefficient lengths disagree")
    if any(s == 0 for s in spec["scaler_scale"]):
        raise ValueError(f"learned gate model {path}: a scaler scale is zero")
    return LearnedModel(
        names, tuple(spec["scaler_mean"]), tuple(spec["scaler_scale"]), tuple(spec["coef_standardised"]), float(spec["intercept"]),
        spec["thresholds"].get("p_accept"), spec["thresholds"].get("p_reject"), bool(spec["trained_on_synthetic"]),
    )


def _learned(fit, photo, fp, model, geocode_rooftop, height_source_authoritative):
    import features  # lazy: the table path must not need numpy

    supplied = {"geocode_rooftop": geocode_rooftop, "height_source_authoritative": height_source_authoritative}
    absent = [n for n in supplied if n in model.feature_names and supplied[n] is None]
    if absent:
        raise ValueError(f"the learned model uses {absent}, which are not in FitResult/PhotoEvidence/Footprint; pass them to the gate")
    values = features.feature_dict(
        fit, photo, fp,
        geocode_rooftop=bool(geocode_rooftop) if geocode_rooftop is not None else False,  # unused by this model
        height_source_authoritative=bool(height_source_authoritative) if height_source_authoritative is not None else False,
    )
    p = model.probability(values)
    if model.p_accept is not None and p >= model.p_accept:
        decision = AUTO_ACCEPT
    elif model.p_reject is not None and p <= model.p_reject:
        decision = REJECT
    else:
        decision = REVIEW
    accept = "none (model cannot reach the precision target)" if model.p_accept is None else f"{model.p_accept:.3f}"
    reject = "none" if model.p_reject is None else f"{model.p_reject:.3f}"
    return p, decision, f"learned p={p:.3f} (p_accept={accept}, p_reject={reject}) -> {decision}"


# --------------------------------------------------------------------- the gate


def evaluate_gate(
    fit, photo, fp, *, geocode_rooftop=None, height_source_authoritative=None, use_learned=None, model=None, allow_synthetic=False
) -> GateDecision:
    """The inspectable gate. `use_learned=None` reads PROCEDURA_LEARNED_GATE (default off -> table)."""
    checks = table_checks(fit, fp)
    table_decision = worst(c.band for c in checks)
    forced = hard_review_reasons(fit, fp)

    learned = learned_gate_enabled() if use_learned is None else bool(use_learned)
    if learned:
        model = model or load_model(allow_synthetic=allow_synthetic)
        probability, decision, why = _learned(fit, photo, fp, model, geocode_rooftop, height_source_authoritative)
        reasons = [why]
    else:
        probability, decision, reasons = None, table_decision, []
    reasons += [f"{c.name}={c.value:.4g} -> {c.band} ({c.rule})" for c in checks if c.band != AUTO_ACCEPT]
    if forced and decision == AUTO_ACCEPT:
        decision = REVIEW
    return GateDecision(decision, probability, "learned" if learned else "table", table_decision, checks, tuple(reasons), forced)


def score_confidence(fit, photo, fp, **kwargs):
    """ML_ADDENDUM F.1: -> (probability, decision). probability is NaN on the table path (the table has none).

    Keyword arguments are those of evaluate_gate; the F.1 three-argument call runs the table.
    """
    result = evaluate_gate(fit, photo, fp, **kwargs)
    return (math.nan if result.probability is None else result.probability), result.decision
