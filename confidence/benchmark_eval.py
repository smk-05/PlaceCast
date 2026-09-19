"""Leave-one-building-out evaluation of the learned confidence gate on the spec 11 benchmark.

    python confidence/train.py --source benchmark [--out-dir DIR]

Data: synthetic corruptions of real footprints, scored against known truth. Each of the 20 real OSM
footprints is turned into a fake mesh outline with a KNOWN pose, damaged the way generated meshes are damaged,
and solved; `label` says whether the recovered pose is right (scripts/run_benchmark.py). It is not a set of
human-judged placements of generated meshes, so it measures the gate against known truth on the solver's
failure modes, not against how a person would judge a real TRELLIS building.

Why leave-one-BUILDING-out: the 84 rows of one building share a footprint, so they are strongly correlated.
Row-level cross-validation would put near-copies of the held-out row in the training folds and report a number
the gate cannot reach on a building it has never seen. Every choice below is made inside the folds:
  * the 19 training buildings pick C (grouped 5-fold CV on log-loss) and the accept / reject thresholds (from
    grouped out-of-fold probabilities, train.choose_thresholds), then score the held-out building;
  * features constant across the benchmark are dropped (a synthetic run has no photo, no rooftop geocode and
    a fixed height source, so those columns carry nothing and would only show up as a "flipped" coefficient).

Decisions. Both methods are read the way production reads them: the method's own decision, floored by
confidence.gate.hard_rules (called, not modified). The table baseline is recomputed here on the same rows with
the current gate, so its number is comparable rather than quoted.

Height. This benchmark is 2D: no row has a height, so every fit carries the non-authoritative
`proportional_fallback` source, and the gate's height rule (non-authoritative -> REVIEW) then floors every row to
REVIEW for BOTH methods; nothing could be auto-accepted and there would be nothing to compare. By default both
methods are therefore scored as if the height were authoritative (`authoritative_height=True`), which isolates
the rest of the gate. The as-is table result is reported alongside so the assumption is visible, not hidden.
"""
from __future__ import annotations

import collections
import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

DATA_DESCRIPTION = "synthetic corruptions of real footprints, scored against known truth"


def _kind(row):
    return row.kind.split(":", 1)[-1]  # "synthetic_gt:lobe" -> "lobe"


def _constant_columns(X):
    return [i for i in range(X.shape[1]) if np.ptp(X[:, i]) == 0.0]


def _group_search(class_weight, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from confidence.train import C_GRID, INNER_FOLDS

    pipeline = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(class_weight=class_weight, max_iter=2000))])
    return GridSearchCV(pipeline, {"lr__C": C_GRID}, cv=GroupKFold(INNER_FOLDS), scoring="neg_log_loss", refit=True)


def _decide(p, p_accept, p_reject, forced):
    """accept / review / reject from a probability and the two thresholds, floored by the hard rules."""
    d = "accept" if (p_accept is not None and p >= p_accept) else "reject" if (p_reject is not None and p <= p_reject) else "review"
    if forced == "reject":
        return "reject"
    return "review" if forced == "review" and d == "accept" else d


def _gate_fit(fit, authoritative_height):
    """The fit as the gate should see it: with an authoritative height source if that is being assumed."""
    from contracts import HeightSource

    if authoritative_height and not fit.height_source_authoritative:
        return replace(fit, height_source=HeightSource.OSM_HEIGHT)
    return fit


def table_accepts_n(table):
    return sum(d == "accept" for d in table)


def _address(building_id):
    try:
        from scripts.demo_addresses import BENCHMARK

        return list(BENCHMARK)[building_id].split(",")[0]
    except Exception:  # the label is a convenience; an unknown building is just its id
        return f"building {building_id}"


def _forced(fit, fp):
    from confidence import gate

    rules = gate.hard_rules(fit, fp)
    floors = {floor for floor, _ in rules}
    return "reject" if gate.REJECT in floors else "review" if floors else None


def _table_decision(fit, photo, fp):
    from confidence import gate

    return {gate.AUTO_ACCEPT: "accept", gate.REVIEW: "review", gate.REJECT: "reject"}[
        gate.evaluate_gate(fit, photo, fp, method="threshold_table").decision
    ]


def evaluate(rows, seed=0, accept_precision=0.95, reject_precision=0.95, authoritative_height=True):
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    from confidence import features as feat
    from confidence.train import INNER_FOLDS, check_labels, choose_class_weight, choose_thresholds, coefficient_table

    y = np.array([int(r.label) for r in rows])
    check_labels(y)
    groups = np.array([r.building_id for r in rows])
    kinds = np.array([_kind(r) for r in rows])
    buildings = sorted(set(groups.tolist()))
    if len(buildings) < INNER_FOLDS + 2:
        raise ValueError(f"need at least {INNER_FOLDS + 2} buildings for nested grouped CV, got {len(buildings)}")

    X_all = feat.feature_matrix(rows, feat.FEATURE_NAMES)
    keep = [i for i in range(X_all.shape[1]) if i not in _constant_columns(X_all)]
    names = tuple(feat.FEATURE_NAMES[i] for i in keep)
    dropped = tuple(n for n in feat.FEATURE_NAMES if n not in names)
    X = X_all[:, keep]

    p = np.zeros(len(rows))
    fold_of = np.zeros(len(rows), dtype=int)
    cut = {}  # building -> (p_accept, p_reject) chosen from the training buildings only
    folds = []
    for b in buildings:
        te, tr = groups == b, groups != b
        weight = choose_class_weight(y[tr])
        search = _group_search(weight, seed).fit(X[tr], y[tr], groups=groups[tr])
        p[te] = search.predict_proba(X[te])[:, 1]
        inner = cross_val_predict(search.best_estimator_, X[tr], y[tr], groups=groups[tr], cv=GroupKFold(INNER_FOLDS),
                                  method="predict_proba")[:, 1]
        t = choose_thresholds(y[tr], inner, accept_precision, reject_precision)
        cut[b] = (t["p_accept"], t["p_reject"])
        fold_of[te] = b
        folds.append({"building_id": int(b), "n": int(te.sum()), "positives": int(y[te].sum()),
                      "accuracy": float(np.mean((p[te] >= 0.5) == y[te])), "C": float(search.best_params_["lr__C"]),
                      "p_accept": cut[b][0], "p_reject": cut[b][1]})

    gate_fits = [_gate_fit(r.fit, authoritative_height) for r in rows]
    forced = [_forced(f, r.footprint) for f, r in zip(gate_fits, rows)]
    learned = [_decide(p[i], *cut[groups[i]], forced[i]) for i in range(len(rows))]
    learned_raw = [_decide(p[i], *cut[groups[i]], None) for i in range(len(rows))]
    table = [_table_decision(f, r.photo, r.footprint) for f, r in zip(gate_fits, rows)]
    table_as_is = [_table_decision(r.fit, r.photo, r.footprint) for r in rows]  # the gate exactly as it is today
    wrong_table = collections.Counter(
        (_address(int(groups[i])), kinds[i], round(float(rows[i].fit.rotation_margin_footprint), 3))
        for i in range(len(rows)) if table[i] == "accept" and y[i] == 0)

    def tally(decisions, mask=None):
        m = np.ones(len(rows), bool) if mask is None else mask
        d = np.array(decisions)
        return {"n": int(m.sum()), "accepts": int(((d == "accept") & m).sum()),
                "wrong_accepts": int(((d == "accept") & (y == 0) & m).sum()),
                "rejects": int(((d == "reject") & m).sum()),
                "wrong_rejects": int(((d == "reject") & (y == 1) & m).sum())}

    per_corruption = []
    for k in sorted(set(kinds.tolist())):
        m = kinds == k
        per_corruption.append({
            "corruption": k, "n": int(m.sum()), "positive_rate": float(y[m].mean()),
            "learned_error_at_0.5": float(np.mean((p[m] >= 0.5) != y[m])),
            "learned": tally(learned, m), "table": tally(table, m),
        })

    # One global threshold on the pooled out-of-fold probabilities, hard rules applied. Chosen on the predictions it
    # is scored on, so both numbers below are optimistic; they are for comparing coverage, not for shipping.
    eligible = np.array([f is None for f in forced])
    order = np.argsort(-p[eligible])
    p_e, y_e = p[eligible][order], y[eligible][order]

    def accepted_at(threshold):
        m = eligible & (p >= threshold)
        return {"threshold": float(threshold), "accepts": int(m.sum()), "wrong_accepts": int((m & (y == 0)).sum()),
                "coverage": float(m.sum() / len(rows))}

    equal_coverage = accepted_at(p_e[min(table_accepts_n(table), len(p_e)) - 1]) if len(p_e) else None
    first_wrong = int(np.argmax(y_e == 0)) if (y_e == 0).any() else len(p_e)
    distinct = [t for t in np.unique(p_e)[::-1] if (p_e >= t).sum() <= first_wrong] if first_wrong else []
    zero_wrong = accepted_at(min(distinct)) if distinct else None

    mirror_m = kinds == "mirror"
    mirror = {"n": int(mirror_m.sum()), "positive": int(y[mirror_m].sum()),
              "flagged_mirrored_by_the_fit": int(sum(bool(rows[i].fit.is_mirrored) for i in np.where(mirror_m)[0])),
              "floored_by_a_hard_rule": int(sum(forced[i] is not None for i in np.where(mirror_m)[0])),
              "learned_accepts": int(sum(learned[i] == "accept" for i in np.where(mirror_m)[0])),
              "learned_wrong_accepts": int(sum(learned[i] == "accept" and y[i] == 0 for i in np.where(mirror_m)[0])),
              "table_accepts": int(sum(table[i] == "accept" for i in np.where(mirror_m)[0])),
              "table_wrong_accepts": int(sum(table[i] == "accept" and y[i] == 0 for i in np.where(mirror_m)[0]))}

    pooled = choose_thresholds(y, p, accept_precision, reject_precision)  # optimistic reference: chosen on what it scores
    final = _group_search(choose_class_weight(y), seed).fit(X, y, groups=groups)
    return {
        "names": names, "dropped_constant": dropped, "n_rows": len(rows), "n_buildings": len(buildings),
        "base_rate": float(y.mean()), "y": y, "p": p, "groups": groups, "kinds": kinds,
        "learned_decision": learned, "table_decision": table, "forced": forced,
        "folds": folds, "auc": float(roc_auc_score(y, p)), "accuracy_at_0.5": float(np.mean((p >= 0.5) == y)),
        "log_loss": float(log_loss(y, p)), "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
        "learned": tally(learned), "learned_without_hard_rules": tally(learned_raw), "table": tally(table),
        "table_as_is": tally(table_as_is), "table_wrong_accepts_detail": dict(wrong_table),
        "equal_coverage": equal_coverage, "zero_wrong": zero_wrong, "mirror": mirror,
        "authoritative_height": bool(authoritative_height),
        "height_sources": dict(collections.Counter(r.fit.height_source.value for r in rows)),
        "per_corruption": per_corruption, "pooled_thresholds": pooled,
        "coefficients": coefficient_table(final.best_estimator_.named_steps["lr"],
                                          final.best_estimator_.named_steps["scale"], names),
    }


def format_report(r):
    L, T = r["learned"], r["table"]
    lines = [
        f"Data: {DATA_DESCRIPTION}.",
        f"{r['n_rows']} rows from {r['n_buildings']} real buildings ({r['base_rate']:.1%} acceptable; "
        f"always-predict-majority = {r['majority_accuracy']:.1%}). Not human-labelled placements of generated meshes.",
        f"Validation: leave-one-building-out (every fold holds out one building's rows; C and both thresholds are "
        f"chosen on the other {r['n_buildings'] - 1}).",
        f"Features: {', '.join(r['names'])}" + (f"   (dropped, constant in the benchmark: {', '.join(r['dropped_constant'])})"
                                                 if r["dropped_constant"] else ""),
        "",
        f"Overall: AUC {r['auc']:.3f}   accuracy@0.5 {r['accuracy_at_0.5']:.3f}   log-loss {r['log_loss']:.3f}",
        "",
        "Per-fold accuracy @0.5 (held-out building):",
        f"  {'bldg':>4} {'n':>4} {'pos':>4} {'acc':>6} {'C':>8} {'p_accept':>9} {'p_reject':>9}",
    ]
    for f in r["folds"]:
        pa = "none" if f["p_accept"] is None else f"{f['p_accept']:.3f}"
        pr = "none" if f["p_reject"] is None else f"{f['p_reject']:.3f}"
        lines.append(f"  {f['building_id']:>4} {f['n']:>4} {f['positives']:>4} {f['accuracy']:>6.3f} {f['C']:>8.3g} {pa:>9} {pr:>9}")
    accs = [f["accuracy"] for f in r["folds"]]
    lines += [f"  fold accuracy: mean {np.mean(accs):.3f}, min {min(accs):.3f}, max {max(accs):.3f}", "",
              "Wrong auto-accepts (label 0 but accepted) at the thresholds each fold chose from its training buildings:"]
    lines.append(f"  learned gate (hard rules applied): {L['wrong_accepts']} of {L['accepts']} accepted "
                 f"({L['accepts'] / r['n_rows']:.0%} coverage);  wrong rejects {L['wrong_rejects']} of {L['rejects']}")
    W = r["learned_without_hard_rules"]
    lines.append(f"  learned gate (model alone):        {W['wrong_accepts']} of {W['accepts']} accepted")
    lines.append(f"  table method (current gate, recomputed): {T['wrong_accepts']} of {T['accepts']} accepted "
                 f"({T['accepts'] / r['n_rows']:.0%} coverage);  wrong rejects {T['wrong_rejects']} of {T['rejects']}")
    pt = r["pooled_thresholds"]
    if pt["p_accept"] is not None:
        lines.append(f"  (reference, optimistic: one threshold chosen on all out-of-fold predictions, p_accept "
                     f"{pt['p_accept']:.3f}: {pt['false_accepts']} wrong of {pt['n_accept']} accepted)")
    lines += ["", "Table method on these rows (the reproduction of BENCHMARK.md):"]
    a = r["table_as_is"]
    lines.append(f"  gate exactly as it is today:            {a['accepts']} auto-accepts, {a['wrong_accepts']} wrong"
                 f"   (height sources on the rows: {r['height_sources']})")
    lines.append(f"  with an authoritative height (used here): {T['accepts']} auto-accepts, {T['wrong_accepts']} wrong")
    for (where, kind, margin), n in sorted(r["table_wrong_accepts_detail"].items()):
        lines.append(f"      {n} x {where} / {kind} / rotation margin {margin}")
    if r["authoritative_height"]:
        lines.append("  This benchmark is 2D and has no height, so the gate's height rule (non-authoritative -> REVIEW) "
                     "would floor every row for both methods. Both are scored as if the height were authoritative.")
    lines += ["", "Per-corruption (error@0.5 = learned model's misclassification rate; A = accepted, wA = wrong accepts, wR = wrong rejects):",
              f"  {'corruption':<14}{'n':>5}{'pos%':>6}{'err@.5':>8} | {'learned A':>9}{'wA':>4}{'wR':>4} | {'table A':>8}{'wA':>4}{'wR':>4}"]
    for c in r["per_corruption"]:
        a, t = c["learned"], c["table"]
        lines.append(f"  {c['corruption']:<14}{c['n']:>5}{c['positive_rate']:>6.0%}{c['learned_error_at_0.5']:>8.1%} | "
                     f"{a['accepts']:>9}{a['wrong_accepts']:>4}{a['wrong_rejects']:>4} | {t['accepts']:>8}{t['wrong_accepts']:>4}{t['wrong_rejects']:>4}")
    lines += ["", "Standardised coefficients of a final fit on all rows (largest first):"]
    for c in r["coefficients"]:
        lines.append(f"  {c['feature']:<28}{c['coef_per_sd']:>+8.3f}  expected {'+' if c['expected_sign'] > 0 else '-'}  "
                     f"{'ok' if c['sign_matches'] else 'FLIPPED'}")
    return "\n".join(lines)


MD_START, MD_END = "<!-- learned-gate:start -->", "<!-- learned-gate:end -->"


def markdown_section(r):
    """The "Learned gate" section of BENCHMARK.md. Regenerated by `train.py --source benchmark --markdown`."""
    L, T, W = r["learned"], r["table"], r["learned_without_hard_rules"]
    eq, zw, mir = r["equal_coverage"], r["zero_wrong"], r["mirror"]
    folds = sorted(r["folds"], key=lambda f: f["accuracy"])
    accs = [f["accuracy"] for f in r["folds"]]
    flipped = [c for c in r["coefficients"] if not c["sign_matches"] and abs(c["coef_per_sd"]) >= 0.1]
    near_zero = [c for c in r["coefficients"] if not c["sign_matches"] and abs(c["coef_per_sd"]) < 0.1]
    a = r["table_as_is"]
    n = r["n_rows"]
    out = [MD_START, "", "## Learned gate (leave-one-building-out)", "",
           f"Data: {DATA_DESCRIPTION}. The {r['n_buildings']} buildings and {n} rows above, {r['base_rate']:.1%} of them "
           "acceptable. This measures the gate against known truth on the solver's failure modes; it is not a set of "
           "human-judged placements of generated meshes.", "",
           "**The threshold table remains the default confidence method.** The learned gate stays behind "
           "`PROCEDURA_LEARNED_GATE` (or `--confidence learned`), no benchmark-trained model is shipped, and "
           "this evaluation calls `hard_rules()` without changing it. The numbers below are why.", "",
           "**Method.** Each fold holds out one whole building (the 84 rows of a building share a footprint, so row-level "
           "cross-validation would leak). C and both thresholds are chosen from the other "
           f"{r['n_buildings'] - 1} buildings. Constant columns are dropped ({', '.join(r['dropped_constant'])}). "
           "Both methods are read as production reads them: the method's decision floored by `hard_rules()`.", "",
           "**Height.** The benchmark is 2D, so every row carries `proportional_fallback`, which the gate's height rule "
           f"floors to REVIEW: the table as it stands makes {a['accepts']} auto-accepts on these rows, not the 12 listed "
           "above (this document was generated at commit e16228e; the height rule, 40b15c0, came after it). Both methods "
           "here are scored as if the "
           f"height were authoritative, which gives the table {T['accepts']} auto-accepts and {T['wrong_accepts']} wrong, "
           "the same 12 as above.", "",
           f"**Discrimination.** Pooled out-of-fold AUC {r['auc']:.3f}; accuracy {r['accuracy_at_0.5']:.3f} at 0.5 "
           f"(always-predict-majority: {r['majority_accuracy']:.3f}). Per-fold accuracy: mean {np.mean(accs):.3f}, "
           f"min {min(accs):.3f}, max {max(accs):.3f}; weakest "
           + ", ".join(f"{_address(f['building_id'])} {f['accuracy']:.3f}" for f in folds[:3]) + ".", "",
           "### Wrong auto-accepts", "",
           "| method | accepted | wrong | coverage |", "|---|---|---|---|",
           f"| table | {T['accepts']} | **{T['wrong_accepts']}** | {T['accepts'] / n:.0%} |",
           f"| learned, thresholds chosen inside each fold | {L['accepts']} | **{L['wrong_accepts']}** | {L['accepts'] / n:.0%} |",
           f"| learned model alone (no hard rules) | {W['accepts']} | {W['wrong_accepts']} | {W['accepts'] / n:.0%} |"]
    if eq:
        out.append(f"| learned at the table's coverage (p >= {eq['threshold']:.3f}) | {eq['accepts']} | "
                   f"**{eq['wrong_accepts']}** | {eq['coverage']:.0%} |")
    if zw:
        out.append(f"| learned, the threshold with 0 wrong (p >= {zw['threshold']:.3f}) | {zw['accepts']} | "
                   f"**{zw['wrong_accepts']}** | {zw['coverage']:.0%} |")
    out += ["",
            (f"At the table's coverage the learned gate makes {eq['wrong_accepts']} wrong accepts to the table's "
             f"{T['wrong_accepts']}" + (", so it is not safer. " if eq['wrong_accepts'] > T['wrong_accepts'] else ". ")
             if eq else "") +
            f"Its advantage is that it accepts more good placements "
            f"({L['accepts'] - L['wrong_accepts']} vs {T['accepts'] - T['wrong_accepts']}) and wrongly rejects far fewer "
            f"({L['wrong_rejects']} vs {T['wrong_rejects']}). The last two rows use one threshold chosen on the "
            "out-of-fold predictions they are scored on, so they are optimistic; the thresholds each fold chose from its "
            f"own training buildings gave realised precision {1 - L['wrong_accepts'] / max(1, L['accepts']):.1%} against a "
            "95% target.", "",
            "### Per corruption", "",
            "| corruption | n | good | learned error @0.5 | learned accepted | learned wrong | table accepted | table wrong | "
            "learned wrong rejects | table wrong rejects |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in r["per_corruption"]:
        out.append(f"| {c['corruption']} | {c['n']} | {c['positive_rate']:.0%} | {c['learned_error_at_0.5']:.1%} | "
                   f"{c['learned']['accepts']} | {c['learned']['wrong_accepts']} | {c['table']['accepts']} | "
                   f"{c['table']['wrong_accepts']} | {c['learned']['wrong_rejects']} | {c['table']['wrong_rejects']} |")
    out += ["", "### Coefficients", "",
            "Standardised, largest first: " + ", ".join(f"{c['feature']} {c['coef_per_sd']:+.2f}" for c in r["coefficients"]) + ".", ""]
    if flipped:
        out += [f"{', '.join(c['feature'] for c in flipped)} come out with the opposite sign to the one expected. "
                "The reason is the label, not a bug: it measures **pose correctness** (the undamaged outline placed with "
                "the recovered pose must reach IoU 0.80 against truth, and the rotation must be within 10 degrees), not "
                "shape. A lobe or a missing wing inflates Hausdorff distance and area ratio while leaving the pose right, so "
                "in this data a larger Hausdorff or area error does not mean a worse label; with footprint IoU in the model "
                "these features act as corrections and take that sign. That is a property of this benchmark's labels, not "
                "evidence that Hausdorff distance stops mattering for real generated meshes, and it is also why the table's "
                f"{T['wrong_rejects']} wrong rejects (mostly lobe and missing_wing, which Hausdorff rejects) look worse here "
                "than they would to a person.", ""]
    if near_zero:
        out += [", ".join(f"{c['feature']} ({c['coef_per_sd']:+.2f})" for c in near_zero) + " also miss their expected "
                "sign but are indistinguishable from zero, so there is nothing to explain.", ""]
    out += ["### Mirrors", "",
            f"Mirrors cannot be caught from the geometry features. Of {mir['n']} mirror rows, {mir['positive']} are good "
            "(a symmetric plan makes the reflection invisible in 2D). The fit never flags a mirror: "
            f"{mir['flagged_mirrored_by_the_fit']} of them carry a negative scale, because the reflection is baked into "
            "the outline and the solver returns an ordinary positive-scale fit, so the mirror hard rule has nothing to "
            "fire on. The features then only see how well the reflected outline happens to fit the footprint, and where "
            "it fits well (near-symmetric plans, e.g. Cassell Coliseum at IoU 0.994 in the false accepts above) a wrong "
            "pose is indistinguishable from a right one. Nothing in the feature vector encodes handedness. The learned "
            f"gate accepts {mir['learned_accepts']} mirror rows, {mir['learned_wrong_accepts']} of them wrong (the table: "
            f"{mir['table_accepts']} accepted, {mir['table_wrong_accepts']} wrong); mislabel_90 shows the same pattern "
            "at a smaller scale. Catching these needs evidence beyond the footprint fit (a facade or silhouette cue, or a "
            "handedness check on the mesh itself), not a better model on these features.", "",
            "Regenerate this section with `python confidence/train.py --source benchmark --markdown BENCHMARK.md` "
            "(`scripts/run_benchmark.py` rewrites the whole file and drops it).", "", MD_END, ""]
    return "\n".join(out)


def write_markdown(r, path):
    """Insert or replace the Learned gate section in `path` (between its markers; appended if absent)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    section = markdown_section(r)
    if MD_START in text and MD_END in text:
        head, rest = text.split(MD_START, 1)
        tail = rest.split(MD_END, 1)[1]
        text = head + section.rstrip("\n") + tail
    else:
        text = text.rstrip("\n") + "\n\n" + section
    path.write_text(text, encoding="utf-8")
    return path


def write_artifacts(r, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_report.txt").write_text(format_report(r) + "\n", encoding="utf-8")
    with open(out / "benchmark_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["building_id", "corruption", "label", "p", "learned_decision", "table_decision", "hard_rule_floor"])
        for i in range(r["n_rows"]):
            w.writerow([int(r["groups"][i]), r["kinds"][i], int(r["y"][i]), f"{r['p'][i]:.5f}",
                        r["learned_decision"][i], r["table_decision"][i], r["forced"][i] or ""])
    return out
