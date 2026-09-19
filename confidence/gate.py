"""
score_confidence() — the accept/review/reject judgement. Addendum B.

Two code paths, selected by a flag:

  method="threshold_table"  spec 9.1's table, in geo/validate.py.  ALWAYS WORKS.
  method="learned"          addendum B's logistic regression.       Upgrade.

The table is not a placeholder. Addendum B.1's argument for the learned model is
that the thresholds are guesses and that the metrics interact in ways a table
cannot express — both true — but B.4 is equally clear that the table ships first
and stays as the fallback. If the model is not ready, flip the flag and demo the
table.
"""

from __future__ import annotations

from pathlib import Path

from contracts import Decision, FitResult, Footprint, PhotoEvidence
from geo.validate import threshold_decision

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

# Addendum B.3: two operating points chosen from the leave-one-out ROC, not by
# hand. p_accept sits at HIGH PRECISION deliberately — the cost of a wrong
# auto-accept (a broken building permanently in the game world) is much higher
# than the cost of an unnecessary review. State the asymmetry in the pitch.
P_ACCEPT = 0.75
P_REJECT = 0.25


def score_confidence(fit: FitResult,
                     photo: PhotoEvidence,
                     fp: Footprint,
                     *,
                     method: str = "threshold_table") -> tuple[float, Decision]:
    """-> (score, decision). See contracts.ScoreConfidence."""
    score, decision, _ = score_confidence_verbose(fit, photo, fp, method=method)
    return score, decision


def score_confidence_verbose(fit: FitResult,
                             photo: PhotoEvidence,
                             fp: Footprint,
                             *,
                             method: str = "threshold_table"
                             ) -> tuple[float, Decision, list[str]]:
    """As above, plus the human-readable reasons for the review queue."""
    if method == "learned":
        result = _try_learned(fit, photo, fp)
        if result is not None:
            return result
        # Falling back is a normal outcome, not an error: the model does not
        # exist until the spec 11 benchmark has been run.

    return threshold_decision(fit, fp)


def _try_learned(fit: FitResult, photo: PhotoEvidence,
                 fp: Footprint) -> tuple[float, Decision, list[str]] | None:
    """Load and apply the fitted logistic regression, or return None."""
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib

        from confidence.features import feature_vector

        bundle = joblib.load(MODEL_PATH)
        model, columns = bundle["model"], bundle["columns"]
        x = feature_vector(fit, photo, fp, columns=columns)
        p = float(model.predict_proba([x])[0][1])

        if p >= P_ACCEPT:
            decision = Decision.AUTO_ACCEPT
        elif p <= P_REJECT:
            decision = Decision.REJECT
        else:
            decision = Decision.REVIEW

        reasons = [f"learned gate p={p:.3f} (n_train={bundle.get('n_train', '?')})"]

        # Addendum B.4: when the model and the table disagree, that is the
        # finding, not a bug. Record both so the disagreement can be shown with
        # its overlay at judging.
        _, table_decision, table_reasons = threshold_decision(fit, fp)
        if table_decision != decision:
            reasons.append(
                f"DISAGREEMENT: threshold table says {table_decision.value}, "
                f"learned gate says {decision.value}"
            )
            reasons.extend(table_reasons[:3])
        return p, decision, reasons
    except Exception as exc:  # noqa: BLE001
        return None
