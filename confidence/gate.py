"""Confidence gate: SPEC 9.1 threshold table (default) with the learned model behind a flag (ML_ADDENDUM B.4).

    score, decision = score_confidence(fit, photo, fp, method=None)            # contracts.ScoreConfidence
    score, decision, reasons = score_confidence_verbose(fit, photo, fp, ...)   # what pipeline.py calls
    result = evaluate_gate(fit, photo, fp, ...)                                 # the inspectable version

`decision` is a contracts.Decision. `method` is "threshold_table" (default) or "learned"; these are the
strings PlacementRecord.confidence_method and the pipeline's --confidence flag already use. With
method=None the environment picks: PROCEDURA_LEARNED_GATE=1 means "learned", otherwise the table. An explicit
`method=` always wins over the environment.

The table works with no trained model and no ML stack: it needs only contracts.py (which imports numpy).
It never invents a probability: GateDecision.probability is None. The float that score_confidence returns on
the table path is a crude monotone `pseudo_score` for ordering and logging, NOT a calibrated probability
(pipeline.py writes it into record.json, which cannot hold NaN).

Table (SPEC 9.1). Each metric lands in a band; the decision is the WORST band across metrics, so a fit
auto-accepts only if every metric does, and IoU and Hausdorff must both pass:

  footprint IoU            accept >= 0.75    review 0.50-0.75   reject < 0.50
  symmetric Hausdorff (m)  accept <= 2.0     review 2-5         reject > 5
  area ratio               accept 0.85-1.15  review 0.7-1.3     reject outside
  rotation margin          accept >= 0.05    review < 0.05      (never rejects)
  rectilinearity           accept >= 0.75    review < 0.75      (never rejects)
  anisotropy |log(sx/sy)|  accept <= 0.05    review 0.05-0.15   reject >= 0.15
  neighbour overlap        accept <= 0.02    review 0.02-0.10   reject > 0.10

Boundaries follow the spec's symbols (>= / <= belong to accept; a value equal to a review/reject edge takes the
milder band, except anisotropy 0.15 which the spec rejects). The spec leaves 0.14-0.15 of anisotropy
unassigned; it is review, since reject starts at 0.15. Metrics must be finite: NaN would compare False
against every edge and silently pick a band, so it raises. These numbers duplicate geo.validate.THRESHOLDS on
purpose (this module must not import geo/); tests/test_gate.py asserts the two agree.

Hard rules, applied on BOTH methods. Each sets a floor under the decision; they never lower it:
  reject  the fit is a reflection (fit.is_mirrored, SPEC 1.2 / 6.8 / 10.2 #5): rejected, never accepted
  review  orientation was guessed: fit.disambiguated_by is ROAD_NORMAL, FACADE_DETAIL or ARBITRARY_SYMMETRIC
          (A.3 / A.4: flag regardless of IoU; ROAD_NORMAL and FACADE_DETAIL come from the street-facing prior),
          or ASPECT_RATIO (the no-cue fallback) while the rotation margin is below the 9.1 accept threshold. At or
          above it the footprint IoU decided the orientation, which is geometric evidence.
  review  round or organic building (fp.is_ill_posed, R < 0.6; SPEC 10.2 #8): rotation is meaningless
  review  footprint matched on thin evidence (fp.match_quality is "unnamed_sole_candidate" or unrecorded): a
          high IoU against the WRONG building is the most dangerous output this pipeline can produce
  review  multi-part footprint (fp.is_multipart): the OMBB and rotation come from the largest part only
  review  EXIF and the silhouette picked different candidates (fit.exif_silhouette_disagree is True)

Learned model (B.4): reads the model.json written by confidence/train.py: p >= p_accept -> auto_accept,
p <= p_reject -> reject, else review; a threshold that training could not set (None) is never applied, so a
model that cannot reach the precision target never auto-accepts. If the model FILE is absent the gate falls
back to the table and says so: `method` on the result is "threshold_table", `fallback_reason` names the
missing path, and the reason is the first review reason. Anything else wrong with a model that IS there
(malformed, trained on synthetic data, mismatched features) raises: that is a bug or a mistake, not an
absence, and hiding it behind the table would leave someone believing the learned gate is live. The table's
own decision is always computed and returned as `table_decision`; when the model disagrees a
"DISAGREEMENT" reason records both (B.4: that disagreement is the finding, not a bug).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from contracts import Decision, Disambiguator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

AUTO_ACCEPT, REVIEW, REJECT = Decision.AUTO_ACCEPT, Decision.REVIEW, Decision.REJECT
_RANK = {AUTO_ACCEPT: 0, REVIEW: 1, REJECT: 2}

METHOD_TABLE, METHOD_LEARNED = "threshold_table", "learned"
METHODS = (METHOD_TABLE, METHOD_LEARNED)
ENV_FLAG = "PROCEDURA_LEARNED_GATE"
ENV_MODEL = "PROCEDURA_GATE_MODEL"
DEFAULT_MODEL_PATH = ROOT / "outputs" / "confidence" / "model.json"
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"", "0", "false", "no", "off"}

ROTATION_MARGIN_ACCEPT = 0.05  # SPEC 9.1: the footprint-IoU margin between the top two candidates, accept at or above

# Every Disambiguator is evidence (EXIF_HEADING, SILHOUETTE), a guess, or evidence-when-decisive.
GUESSED_ORIENTATION = frozenset({Disambiguator.ROAD_NORMAL, Disambiguator.FACADE_DETAIL, Disambiguator.ARBITRARY_SYMMETRIC})
# ASPECT_RATIO is geo.disambiguate's final fallback: it only excludes the 90-degree candidates and then takes the
# best footprint IoU. That is a guess when the top two candidates tie (rotation margin below the 9.1 accept
# threshold) and geometric evidence when one clearly wins.
GUESSED_BELOW_MARGIN = frozenset({Disambiguator.ASPECT_RATIO})


def orientation_guessed(fit):
    """True when fit.disambiguated_by is a guess. A non-finite margin is a guess, never evidence."""
    if fit.disambiguated_by in GUESSED_ORIENTATION:
        return True
    return fit.disambiguated_by in GUESSED_BELOW_MARGIN and not fit.rotation_margin_footprint >= ROTATION_MARGIN_ACCEPT


# ---------------------------------------------------------------- data types


@dataclass(frozen=True)
class MetricCheck:
    name: str
    value: float
    band: Decision
    rule: str  # the table row, for the review payload (SPEC 9.3)


@dataclass(frozen=True)
class GateDecision:
    decision: Decision
    probability: Optional[float]  # the learned model's p; None on the table path
    score: float  # what score_confidence returns: p when learned, else the table's pseudo_score
    method: str  # the method that RAN (may differ from `requested_method` on a fallback)
    requested_method: str
    fallback_reason: Optional[str]  # set when "learned" was requested but the table ran
    table_decision: Decision  # what the SPEC 9.1 table says, whichever method decided
    checks: tuple  # every table metric with its band (always computed)
    reasons: tuple  # why the decision is not a clean auto_accept, most important first
    forced: tuple  # the hard rules that fired, as (floor Decision, reason)


# ------------------------------------------------------------------ the table


def _band_iou(v):
    return AUTO_ACCEPT if v >= 0.75 else REVIEW if v >= 0.50 else REJECT


def _band_hausdorff(v):
    return AUTO_ACCEPT if v <= 2.0 else REVIEW if v <= 5.0 else REJECT


def _band_area_ratio(v):
    return AUTO_ACCEPT if 0.85 <= v <= 1.15 else REVIEW if 0.70 <= v <= 1.30 else REJECT


def _band_rotation_margin(v):
    return AUTO_ACCEPT if v >= ROTATION_MARGIN_ACCEPT else REVIEW


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


def worst(decisions):
    return max(decisions, key=_RANK.__getitem__)


def _clip01(x):
    return min(max(x, 0.0), 1.0)


def pseudo_score(fit, fp):
    """A crude monotone 0-1 summary so the placement record has one comparable number under either method.
    NOT a probability and not calibrated (ported from geo.validate._pseudo_score, without numpy)."""
    terms = (
        _clip01(fit.iou / 0.85),
        _clip01(1.0 - fit.hausdorff_m / 6.0),
        _clip01(1.0 - abs(math.log(max(fit.area_ratio, 1e-6))) / 0.35),
        _clip01(fit.rotation_margin_footprint / 0.20),
        _clip01(fp.rectilinearity),
    )
    return float(sum(terms) / len(terms))


def hard_rules(fit, fp):
    """Floors under the decision: [(Decision.REVIEW or REJECT, reason)]. They apply to both methods."""
    rules = []
    if fit.is_mirrored:
        rules.append((REJECT, "mirrored mesh - det(R) < 0; reflections are rejected, never accepted (SPEC 6.8)"))
    if orientation_guessed(fit):
        why = (f"; rotation margin {fit.rotation_margin_footprint:.3g} < {ROTATION_MARGIN_ACCEPT}, the top candidates tie"
               if fit.disambiguated_by in GUESSED_BELOW_MARGIN else "")
        rules.append((REVIEW, f"orientation was guessed (disambiguated_by={fit.disambiguated_by.value}{why}); review regardless of IoU (A.3/A.4)"))
    if fp.is_ill_posed:
        rules.append((REVIEW, f"rectilinearity {fp.rectilinearity:.2f} < 0.6: round/organic building, rotation is intrinsically meaningless (SPEC 10.2 #8)"))
    if fp.match_quality == "unnamed_sole_candidate":
        rules.append((REVIEW, "footprint matched only as the sole nearby candidate, with no name match - verify it is the right building before accepting (SPEC 3.2)"))
    elif not fp.match_quality:
        rules.append((REVIEW, "footprint match quality unrecorded - treat as unverified (SPEC 3.2)"))
    if fp.is_multipart:
        rules.append((REVIEW, f"multi-part footprint ({1 + len(fp.parts_enu)} disjoint outer rings): the OMBB and rotation come from the largest part only"))
    if fit.exif_silhouette_disagree is True:
        rules.append((REVIEW, "EXIF and the silhouette picked different candidates - independent evidence conflicts (A.3)"))
    return tuple(rules)


# --------------------------------------------------------------- learned model


def learned_gate_enabled():
    raw = os.environ.get(ENV_FLAG, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{ENV_FLAG}={raw!r} is not a recognised value; use 1/0")


def resolve_method(method=None):
    if method is None:
        method = METHOD_LEARNED if learned_gate_enabled() else METHOD_TABLE
    if method not in METHODS:
        raise ValueError(f"unknown confidence method {method!r}; use one of {METHODS}")
    return method


class ModelAbsentError(FileNotFoundError):
    """No model file at `path`: the one condition under which method="learned" falls back to the table."""

    def __init__(self, path, hint):
        super().__init__(f"no learned-gate model at {path}; {hint}")
        self.path = Path(path)


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
    """Read the model.json written by confidence/train.py. Absent -> ModelAbsentError; anything else wrong raises."""
    import json

    path = Path(path or os.environ.get(ENV_MODEL) or DEFAULT_MODEL_PATH)
    hint = "train one with confidence/train.py"
    if not path.is_file():
        raise ModelAbsentError(path, hint)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"learned gate model {path} is not valid JSON: {exc}") from exc
    needed = ("feature_names", "scaler_mean", "scaler_scale", "coef_standardised", "intercept", "thresholds", "trained_on_synthetic")
    missing = [k for k in needed if k not in spec]
    if missing:
        raise ValueError(f"learned gate model {path} is missing {missing}")
    if spec["trained_on_synthetic"] and not allow_synthetic:
        raise ValueError(f"learned gate model {path} was trained on SYNTHETIC data (fixtures/fake_fits.py) and must not gate real placements; {hint} on real benchmark rows")
    names, n = tuple(spec["feature_names"]), len(spec["feature_names"])
    if not (len(spec["scaler_mean"]) == len(spec["scaler_scale"]) == len(spec["coef_standardised"]) == n) or n == 0:
        raise ValueError(f"learned gate model {path}: feature/scaler/coefficient lengths disagree")
    if any(s == 0 for s in spec["scaler_scale"]):
        raise ValueError(f"learned gate model {path}: a scaler scale is zero")
    return LearnedModel(
        names, tuple(spec["scaler_mean"]), tuple(spec["scaler_scale"]), tuple(spec["coef_standardised"]), float(spec["intercept"]),
        spec["thresholds"].get("p_accept"), spec["thresholds"].get("p_reject"), bool(spec["trained_on_synthetic"]),
    )


def _learned(fit, photo, fp, model):
    from confidence import features  # lazy: the table must not need the feature code

    p = model.probability(features.feature_dict(fit, photo, fp))
    if model.p_accept is not None and p >= model.p_accept:
        decision = AUTO_ACCEPT
    elif model.p_reject is not None and p <= model.p_reject:
        decision = REJECT
    else:
        decision = REVIEW
    accept = "none (model cannot reach the precision target)" if model.p_accept is None else f"{model.p_accept:.3f}"
    reject = "none" if model.p_reject is None else f"{model.p_reject:.3f}"
    return p, decision, f"learned p={p:.3f} (p_accept={accept}, p_reject={reject}) -> {decision.value}"


# --------------------------------------------------------------------- the gate


def evaluate_gate(fit, photo, fp, *, method=None, model=None, allow_synthetic=False) -> GateDecision:
    """The inspectable gate. `method=None` reads PROCEDURA_LEARNED_GATE (default off -> the table)."""
    requested = resolve_method(method)
    checks = table_checks(fit, fp)
    table_decision = worst(c.band for c in checks)
    table_reasons = [f"{c.name}={c.value:.4g} -> {c.band.value} ({c.rule})" for c in checks if c.band != AUTO_ACCEPT]
    forced = hard_rules(fit, fp)

    ran, decision, probability, fallback = METHOD_TABLE, table_decision, None, None
    reasons = []
    if requested == METHOD_LEARNED:
        try:
            model = model or load_model(allow_synthetic=allow_synthetic)
        except ModelAbsentError as exc:
            fallback = f"learned gate requested but its model file is absent ({exc.path}); using the SPEC 9.1 threshold table"
            reasons.append(fallback)
        else:
            ran = METHOD_LEARNED
            probability, decision, why = _learned(fit, photo, fp, model)
            reasons.append(why)
            if decision != table_decision:
                reasons.append(f"DISAGREEMENT: threshold table says {table_decision.value}, learned gate says {decision.value}")
                reasons.extend(table_reasons[:3])
    reasons.extend(r for r in table_reasons if r not in reasons)
    reasons.extend(why for _, why in forced)
    decision = worst([decision, *(floor for floor, _ in forced)])
    score = probability if probability is not None else pseudo_score(fit, fp)
    return GateDecision(decision, probability, score, ran, requested, fallback, table_decision, checks, tuple(reasons), forced)


def score_confidence(fit, photo, fp, *, method=None, **kwargs):
    """contracts.ScoreConfidence: -> (score, Decision). `score` is the model's probability on the learned path
    and the table's pseudo_score otherwise (see the module docstring)."""
    result = evaluate_gate(fit, photo, fp, method=method, **kwargs)
    return result.score, result.decision


def score_confidence_verbose(fit, photo, fp, *, method=None, **kwargs):
    """As score_confidence, plus the reasons for the review queue (what pipeline.py calls)."""
    result = evaluate_gate(fit, photo, fp, method=method, **kwargs)
    return result.score, result.decision, list(result.reasons)
