#!/usr/bin/env python
"""Train the learned confidence gate (ML_ADDENDUM B.3): L2 logistic regression, LOO CV, coefficients,
reliability diagram. No calibration layer.

    python confidence/train.py [--source fake] [--seed 0] [--out-dir DIR]
                               [--accept-precision 0.95] [--reject-precision 0.95]

Rows are anything exposing .fit .photo .footprint (the contracts.py objects) and .label (1 = a human
would accept the placement). Two sources: "fake" (fixtures/fake_fits.py, demonstrates the pipeline only) and
"benchmark" (the spec 11 benchmark: synthetic corruptions of real footprints, scored against known truth;
scripts/run_benchmark.py). A model trained on the fake source is flagged `trained_on_synthetic` in model.json
and must never be shipped.

`--source benchmark` does not run the row-level leave-one-out below: the benchmark has 84 correlated rows per
building, so that would leak. It runs leave-one-building-out (confidence/benchmark_eval.py) and reports it.

Method (B.3)
  * standardised features -> LogisticRegression, L2 (scikit-learn's default penalty; passing
    penalty='l2' is deprecated since 1.8), C chosen by inner stratified CV on log-loss.
  * Validation is leave-one-out with C re-chosen inside every fold (nested), so the reported
    accuracy does not benefit from tuning C on the row being predicted.
  * NO calibration layer. Logistic regression fitted under log-loss is calibrated by construction
    on its training distribution, and Platt/isotonic on ~100 rows would overfit (B.3). The
    reliability diagram (4 bins, LOO predictions) is a validation artifact, not a correction.
  * B.4: fewer than 60 rows -> the 4-feature model; a minority class under 25% -> class_weight=
    'balanced' (which pulls probabilities away from the base rate, so it is printed and stored).
  * Thresholds p_accept / p_reject come from the LOO probabilities, not by hand: p_accept is the
    lowest probability whose accepted set still has precision >= --accept-precision (a wrong
    auto-accept is a broken building permanently in the game world, so precision comes first);
    p_reject is the highest whose rejected set is >= --reject-precision negatives. Between them,
    review. Both are chosen on the same LOO predictions they are then reported on, so they are
    optimistic; with ~100 rows treat them as indicative.

Artifacts (default outputs/confidence/): coefficients.csv, reliability.csv, reliability.png, model.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from confidence import features as feat  # noqa: E402

MIN_ROWS_FULL_MODEL = 60  # B.4
IMBALANCE_THRESHOLD = 0.25  # B.4: a class below this share -> class_weight='balanced'
C_GRID = np.logspace(-3, 2, 11)
INNER_FOLDS = 5
RELIABILITY_BINS = 4  # B.3: ~4 bins
ACCEPT_PRECISION = 0.95
REJECT_PRECISION = 0.95
MIN_SUPPORT = 5  # a threshold must select at least this many rows to count
DEFAULT_OUT_DIR = ROOT / "outputs" / "confidence"
BENCHMARK_PKL = ROOT / "outputs" / "benchmark" / "labelled_fits.pkl"
BENCHMARK_REPS = 3  # 7 corruptions x 4 solver configs x 3 poses = 84 correlated rows per building


# ----------------------------------------------------------------------- data


def load_rows(source, seed=0):
    if source == "fake":
        from fixtures.fake_fits import generate_labelled

        return generate_labelled(seed=seed)
    if source == "benchmark":
        # outputs/benchmark/labelled_fits.pkl is what scripts/run_benchmark.py writes (seed 0); anything else, or
        # no file, rebuilds the rows. Delete the pickle after changing the solver: it holds the solver's old answers.
        if seed == 0 and BENCHMARK_PKL.is_file():
            import pickle

            with open(BENCHMARK_PKL, "rb") as f:
                return pickle.load(f)
        from scripts.run_benchmark import benchmark_labelled_rows

        return benchmark_labelled_rows(seed=seed, reps=BENCHMARK_REPS)
    raise ValueError(f"unknown source {source!r}: use 'fake' or 'benchmark'")


def choose_features(n_rows):
    if n_rows < MIN_ROWS_FULL_MODEL:
        return feat.REDUCED_FEATURES, (
            f"{n_rows} rows < {MIN_ROWS_FULL_MODEL}: using the {len(feat.REDUCED_FEATURES)}-feature model (B.4)"
        )
    return feat.FEATURE_NAMES, f"{n_rows} rows: full {len(feat.FEATURE_NAMES)}-feature model"


def choose_class_weight(y):
    return "balanced" if min(np.mean(y), 1 - np.mean(y)) < IMBALANCE_THRESHOLD else None


def check_labels(y):
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError(f"labels must be 0/1, got {sorted(set(np.unique(y)))}")
    counts = np.bincount(y, minlength=2)
    if counts.min() < 3:
        raise ValueError(
            f"need at least 3 examples of each class for nested leave-one-out CV, got {counts[0]} negative / {counts[1]} positive"
        )


# ---------------------------------------------------------------------- model


def make_search(class_weight, n_splits, seed):
    """Standardise -> L2 logistic regression, with C chosen by stratified CV on log-loss."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [("scale", StandardScaler()), ("lr", LogisticRegression(class_weight=class_weight, max_iter=2000))]
    )  # default penalty is L2
    return GridSearchCV(
        pipeline,
        {"lr__C": C_GRID},
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
        scoring="neg_log_loss",
        refit=True,
    )


def loo_probabilities(X, y, class_weight, seed, n_splits):
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    # n_jobs=-1: the outer folds are independent, and each one runs its own inner C search.
    search = make_search(class_weight, n_splits, seed)
    return cross_val_predict(search, X, y, cv=LeaveOneOut(), method="predict_proba", n_jobs=-1)[:, 1]


def coefficient_table(lr, scaler, names):
    """Standardised coefficients, largest magnitude first, with the expected-sign check."""
    rows = []
    for name, coef, mean, scale in zip(names, lr.coef_[0], scaler.mean_, scaler.scale_):
        expected = feat.EXPECTED_SIGN[name]
        rows.append(
            {
                "feature": name,
                "coef_per_sd": float(coef),
                "odds_ratio_per_sd": float(math.exp(coef)),
                "expected_sign": expected,
                "sign_matches": bool(np.sign(coef) == expected),
                "mean": float(mean),
                "sd": float(scale),
            }
        )
    return sorted(rows, key=lambda r: -abs(r["coef_per_sd"]))


# ------------------------------------------------------------------ operating points


def choose_thresholds(y, p, accept_precision=ACCEPT_PRECISION, reject_precision=REJECT_PRECISION, min_support=MIN_SUPPORT):
    """p_accept / p_reject from (LOO) probabilities. Either is None if no threshold meets its target."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    candidates = np.unique(p)
    p_accept = next((float(t) for t in candidates if (p >= t).sum() >= min_support and y[p >= t].mean() >= accept_precision), None)
    p_reject = next(
        (
            float(t)
            for t in candidates[::-1]
            if (p <= t).sum() >= min_support
            and (1 - y[p <= t]).mean() >= reject_precision
            and (p_accept is None or t < p_accept)
        ),
        None,
    )
    accept = p >= p_accept if p_accept is not None else np.zeros(len(p), bool)
    reject = p <= p_reject if p_reject is not None else np.zeros(len(p), bool)
    return {
        "p_accept": p_accept,
        "p_reject": p_reject,
        "accept_target_precision": accept_precision,
        "reject_target_precision": reject_precision,
        "n_accept": int(accept.sum()),
        "accept_precision": float(y[accept].mean()) if accept.any() else None,
        "false_accepts": int((y[accept] == 0).sum()),  # the expensive error
        "auto_accept_coverage": float(accept.mean()),
        "n_reject": int(reject.sum()),
        "reject_precision": float((1 - y[reject]).mean()) if reject.any() else None,
        "good_placements_rejected": int((y[reject] == 1).sum()),
        "n_review": int(len(p) - accept.sum() - reject.sum()),
    }


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_bins(y, p, n_bins=RELIABILITY_BINS):
    """Equal-width bins on the predicted probability. Empty bins are kept (n=0) so they show."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        k = int(y[sel].sum())
        lo, hi = wilson(k, n)
        bins.append(
            {
                "bin_lo": b / n_bins,
                "bin_hi": (b + 1) / n_bins,
                "n": n,
                "mean_predicted": float(p[sel].mean()) if n else float("nan"),
                "observed_frequency": k / n if n else float("nan"),
                "wilson_lo": lo,
                "wilson_hi": hi,
            }
        )
    filled = [b for b in bins if b["n"]]
    ece = sum(b["n"] * abs(b["observed_frequency"] - b["mean_predicted"]) for b in filled) / len(y)
    return {"bins": bins, "ece": float(ece)}


# ------------------------------------------------------------------------ train


@dataclass
class TrainResult:
    names: tuple
    feature_note: str
    n_rows: int
    n_positive: int
    class_weight: str | None
    C: float
    y: np.ndarray
    loo_proba: np.ndarray
    metrics: dict
    coefficients: list
    intercept: float
    scaler_mean: list
    scaler_scale: list
    thresholds: dict
    reliability: dict
    model: object  # the fitted sklearn Pipeline


def train(rows, seed=0, accept_precision=ACCEPT_PRECISION, reject_precision=REJECT_PRECISION):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y = np.array([int(r.label) for r in rows])
    check_labels(y)
    names, feature_note = choose_features(len(rows))
    X = feat.feature_matrix(rows, names)
    class_weight = choose_class_weight(y)
    min_count = int(np.bincount(y).min())

    proba = loo_probabilities(X, y, class_weight, seed, n_splits=min(INNER_FOLDS, min_count - 1))
    search = make_search(class_weight, min(INNER_FOLDS, min_count), seed).fit(X, y)
    model = search.best_estimator_
    lr, scaler = model.named_steps["lr"], model.named_steps["scale"]

    metrics = {
        "loo_accuracy": float(np.mean((proba >= 0.5) == y)),
        "loo_auc": float(roc_auc_score(y, proba)),
        "loo_log_loss": float(log_loss(y, proba)),
        "loo_brier": float(brier_score_loss(y, proba)),
        "base_rate": float(y.mean()),
        "majority_class_accuracy": float(max(y.mean(), 1 - y.mean())),
    }
    return TrainResult(
        names=tuple(names), feature_note=feature_note, n_rows=len(rows), n_positive=int(y.sum()),
        class_weight=class_weight, C=float(search.best_params_["lr__C"]), y=y, loo_proba=proba, metrics=metrics,
        coefficients=coefficient_table(lr, scaler, names), intercept=float(lr.intercept_[0]),
        scaler_mean=[float(v) for v in scaler.mean_], scaler_scale=[float(v) for v in scaler.scale_],
        thresholds=choose_thresholds(y, proba, accept_precision, reject_precision),
        reliability=reliability_bins(y, proba), model=model,
    )


# --------------------------------------------------------------------- reporting


def _fmt(v, spec=".3f"):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def format_report(r, source):
    m, t = r.metrics, r.thresholds
    lines = []
    if source == "fake":
        lines += ["*** SYNTHETIC DATA (fixtures/fake_fits.py): these numbers demonstrate the pipeline, not its accuracy. ***", ""]
    lines += [
        f"rows: {r.n_rows}  acceptable: {r.n_positive} ({m['base_rate']:.1%} base rate; always-predict-majority = {m['majority_class_accuracy']:.1%})",
        f"features: {r.feature_note}",
        f"class_weight: {r.class_weight}" + ("  (probabilities are shifted away from the base rate)" if r.class_weight else ""),
        f"chosen C: {r.C:.4g}  (L2, chosen by 5-fold CV on log-loss)",
        "",
        "Leave-one-out validation (C re-chosen inside every fold):",
        f"  accuracy @0.5 : {m['loo_accuracy']:.3f}",
        f"  ROC AUC       : {m['loo_auc']:.3f}",
        f"  log-loss      : {m['loo_log_loss']:.3f}    Brier: {m['loo_brier']:.3f}",
        "",
        "Operating points (from LOO probabilities; optimistic, chosen on the data they are reported on):",
    ]
    if t["p_accept"] is None:
        lines.append(f"  p_accept: NONE - no threshold reaches {t['accept_target_precision']:.0%} precision on >= {MIN_SUPPORT} rows; nothing is auto-accepted")
    else:
        lines.append(
            f"  p_accept = {t['p_accept']:.3f}: auto-accepts {t['n_accept']} ({t['auto_accept_coverage']:.0%}), "
            f"precision {t['accept_precision']:.3f}, false accepts {t['false_accepts']}"
        )
    if t["p_reject"] is None:
        lines.append(f"  p_reject: NONE - no threshold reaches {t['reject_target_precision']:.0%} rejection precision on >= {MIN_SUPPORT} rows")
    else:
        lines.append(
            f"  p_reject = {t['p_reject']:.3f}: rejects {t['n_reject']}, rejection precision {t['reject_precision']:.3f}, "
            f"good placements rejected {t['good_placements_rejected']}"
        )
    lines += [f"  review queue: {t['n_review']}", "", "Standardised coefficients (largest first; odds multiply by exp(coef) per +1 SD):"]
    lines.append(f"  {'feature':<28}{'coef/SD':>9}{'odds x':>9}  {'expected':>8}  sign")
    for c in r.coefficients:
        lines.append(
            f"  {c['feature']:<28}{c['coef_per_sd']:>+9.3f}{c['odds_ratio_per_sd']:>9.2f}  "
            f"{'+' if c['expected_sign'] > 0 else '-':>8}  {'ok' if c['sign_matches'] else 'FLIPPED'}"
        )
    lines += ["", f"Reliability (LOO predictions, {RELIABILITY_BINS} equal-width bins; ECE {r.reliability['ece']:.3f}):"]
    lines.append(f"  {'bin':<12}{'n':>4}{'mean pred':>11}{'observed':>10}   95% Wilson")
    for b in r.reliability["bins"]:
        lines.append(
            f"  {b['bin_lo']:.2f}-{b['bin_hi']:.2f}  {b['n']:>4}{_fmt(b['mean_predicted']):>11}{_fmt(b['observed_frequency']):>10}"
            f"   [{_fmt(b['wilson_lo'])}, {_fmt(b['wilson_hi'])}]"
        )
    lines += ["", "No calibration layer: probabilities are the logistic regression's own (B.3)."]
    return "\n".join(lines)


def plot_reliability(rel, path, subtitle):
    """Reliability diagram: one series (no legend), labelled reference diagonal, recessive grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, ref, series = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3de", "#b9b8b1", "#2a78d6"
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200, facecolor=surface)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.83, bottom=0.17)
    ax.set_facecolor(surface)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    ax.grid(True, color=grid, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)

    ax.plot([0, 1], [0, 1], color=ref, linewidth=1.0, zorder=1)

    filled = [b for b in rel["bins"] if b["n"]]
    xs, ys = [b["mean_predicted"] for b in filled], [b["observed_frequency"] for b in filled]
    # Direct label for the reference line, placed where the data (points and interval bars) is farthest.
    def clearance(t):
        return min([math.hypot(t - b["mean_predicted"], t - min(max(t, b["wilson_lo"]), b["wilson_hi"])) for b in filled] or [1.0])

    t = max(np.linspace(0.14, 0.86, 37), key=clearance)
    ax.text(t - 0.02, t + 0.02, "perfect calibration", color=ink2, fontsize=8, rotation=45, rotation_mode="anchor", ha="center", va="bottom")
    ax.vlines(xs, [b["wilson_lo"] for b in filled], [b["wilson_hi"] for b in filled], color=series, linewidth=1.5, alpha=0.4, zorder=2)
    ax.plot(xs, ys, color=series, linewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.plot(xs, ys, linestyle="none", marker="o", markersize=7, markerfacecolor=series, markeredgecolor=surface, markeredgewidth=1.5, zorder=4)

    edges = [rel["bins"][0]["bin_lo"]] + [b["bin_hi"] for b in rel["bins"]]
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:g}" for e in edges])
    ax.set_yticks(edges)
    ax.set_yticklabels([f"{e:g}" for e in edges])
    ax.set_xticks([(b["bin_lo"] + b["bin_hi"]) / 2 for b in rel["bins"]], minor=True)
    ax.set_xticklabels([f"n={b['n']}" for b in rel["bins"]], minor=True)
    ax.tick_params(which="major", colors=ink2, labelsize=8.5, length=3, color=grid)
    ax.tick_params(which="minor", axis="x", length=0, pad=17, labelsize=8, labelcolor=muted)
    ax.set_xlabel("Mean predicted probability of an acceptable placement", color=ink2, fontsize=9, labelpad=22)
    ax.set_ylabel("Observed fraction acceptable", color=ink2, fontsize=9)
    fig.text(0.13, 0.955, "Confidence gate reliability (leave-one-out)", color=ink, fontsize=12.5, fontweight="bold", ha="left")
    for i, line in enumerate(subtitle):
        fig.text(0.13, 0.918 - 0.03 * i, line, color=ink2, fontsize=8.5, ha="left")
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def write_artifacts(r, out_dir, source, seed):
    import sklearn

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "coefficients.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(r.coefficients[0]))
        writer.writeheader()
        writer.writerows(r.coefficients)
    with open(out_dir / "reliability.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(r.reliability["bins"][0]))
        writer.writeheader()
        writer.writerows(r.reliability["bins"])
    plot_reliability(
        r.reliability, out_dir / "reliability.png",
        [
            f"Observed vs predicted, {RELIABILITY_BINS} equal-width bins, {r.n_rows} rows, ECE {r.reliability['ece']:.3f}",
            "Bars: 95% Wilson intervals" + ("   |   SYNTHETIC DATA" if source == "fake" else ""),
        ],
    )
    model = {
        "trained_on_synthetic": source == "fake",
        "source": source,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_rows": r.n_rows,
        "n_positive": r.n_positive,
        "feature_names": list(r.names),
        "scaler_mean": r.scaler_mean,
        "scaler_scale": r.scaler_scale,
        "coef_standardised": [next(c["coef_per_sd"] for c in r.coefficients if c["feature"] == n) for n in r.names],
        "intercept": r.intercept,
        "C": r.C,
        "class_weight": r.class_weight,
        "predict": "p = sigmoid(intercept + sum_i coef_i * (x_i - mean_i) / scale_i)",
        "thresholds": r.thresholds,
        "metrics": r.metrics,
        "reliability_ece": r.reliability["ece"],
        "calibration_layer": None,
    }
    (out_dir / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default="fake")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None,
                        help=f"default {DEFAULT_OUT_DIR}, or {DEFAULT_OUT_DIR / 'benchmark'} for --source benchmark")
    parser.add_argument("--accept-precision", type=float, default=ACCEPT_PRECISION)
    parser.add_argument("--reject-precision", type=float, default=REJECT_PRECISION)
    args = parser.parse_args(argv)
    try:
        rows = load_rows(args.source, args.seed)
        if args.source == "benchmark":
            from confidence import benchmark_eval

            report = benchmark_eval.evaluate(rows, args.seed, args.accept_precision, args.reject_precision)
            print(benchmark_eval.format_report(report))
            out = benchmark_eval.write_artifacts(report, args.out_dir or DEFAULT_OUT_DIR / "benchmark")
            print(f"\nartifacts: {out}")
            return 0
        result = train(rows, args.seed, args.accept_precision, args.reject_precision)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(result, args.source))
    out = write_artifacts(result, args.out_dir or DEFAULT_OUT_DIR, args.source, args.seed)
    print(f"\nartifacts: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
