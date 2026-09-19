"""Leave-one-building-out evaluation of the learned gate (confidence/benchmark_eval.py) and the "benchmark"
source of confidence.train.load_rows.

The fake rows stand in for the benchmark here: same LabelledFit shape, 20 buildings, no need to run the solver.
"""
import pickle

import numpy as np
import pytest

from confidence import benchmark_eval, train
from fixtures.fake_fits import generate_labelled


@pytest.fixture(scope="module")
def rows():
    return generate_labelled(seed=0)


@pytest.fixture(scope="module")
def report(rows):
    return benchmark_eval.evaluate(rows, seed=0)


def test_every_building_is_held_out_once_and_never_seen_in_its_own_training_set(rows, monkeypatch):
    seen = []
    real = benchmark_eval._group_search

    def spy(class_weight, seed):
        search = real(class_weight, seed)
        fit = search.fit

        def recording_fit(X, y, groups=None, **kw):
            seen.append(set(groups.tolist()))
            return fit(X, y, groups=groups, **kw)

        search.fit = recording_fit
        return search

    monkeypatch.setattr(benchmark_eval, "_group_search", spy)
    report = benchmark_eval.evaluate(rows, seed=0)
    held_out = [f["building_id"] for f in report["folds"]]
    assert sorted(held_out) == sorted({r.building_id for r in rows})
    for building, training in zip(held_out, seen):  # the last fit is the final model on everything
        assert building not in training and len(training) == len(held_out) - 1
    assert seen[-1] == set(held_out)


def test_folds_account_for_every_row_and_the_metrics_are_probabilities(rows, report):
    assert sum(f["n"] for f in report["folds"]) == len(rows) == report["n_rows"]
    assert 0.0 <= report["auc"] <= 1.0 and np.all((report["p"] >= 0) & (report["p"] <= 1))
    assert sum(c["n"] for c in report["per_corruption"]) == len(rows)


def test_the_table_baseline_is_recomputed_on_the_same_rows(rows, report):
    assert report["table"]["n"] == len(rows)
    assert report["table"]["wrong_accepts"] <= report["table"]["accepts"]
    assert report["learned"]["wrong_accepts"] <= report["learned"]["accepts"]


def test_hard_rules_only_ever_lower_a_learned_accept(report):
    for raw, floored in zip(report["learned_decision"], report["learned_decision"]):
        assert floored in ("accept", "review", "reject")
    assert report["learned"]["accepts"] <= report["learned_without_hard_rules"]["accepts"]


def test_the_report_states_what_the_data_is(report):
    text = benchmark_eval.format_report(report)
    assert "synthetic corruptions of real footprints, scored against known truth" in text
    assert "leave-one-building-out" in text and "Per-fold accuracy" in text and "Per-corruption" in text


def test_load_rows_reads_the_benchmark_pickle_and_still_rejects_unknown_sources(tmp_path, monkeypatch, rows):
    pkl = tmp_path / "labelled_fits.pkl"
    pkl.write_bytes(pickle.dumps(rows[:5]))
    monkeypatch.setattr(train, "BENCHMARK_PKL", pkl)
    assert len(train.load_rows("benchmark")) == 5
    with pytest.raises(ValueError, match="unknown source"):
        train.load_rows("real")


# ----------------------------------------------------------- the height assumption


@pytest.fixture(scope="module")
def no_height_rows(rows):
    """The benchmark's situation: no row has a height, so every source is non-authoritative."""
    import dataclasses

    from contracts import HeightSource

    return [dataclasses.replace(r, fit=dataclasses.replace(r.fit, height_source=HeightSource.PROPORTIONAL_FALLBACK)) for r in rows]


def test_without_a_height_the_gate_floors_every_row_for_both_methods(no_height_rows):
    report = benchmark_eval.evaluate(no_height_rows, seed=0, authoritative_height=False)
    assert report["table"]["accepts"] == 0 and report["learned"]["accepts"] == 0


def test_assuming_an_authoritative_height_lets_the_rest_of_the_gate_speak(no_height_rows):
    report = benchmark_eval.evaluate(no_height_rows, seed=0, authoritative_height=True)
    assert report["table_as_is"]["accepts"] == 0                 # what the gate does today, reported alongside
    assert report["table"]["accepts"] > 0                        # what it does once the height is not the blocker
    assert set(report["height_sources"]) == {"proportional_fallback"}
    text = benchmark_eval.format_report(report)
    assert "gate exactly as it is today" in text and "scored as if the height were authoritative" in text


def test_the_assumption_never_touches_the_hard_rules_or_the_features(no_height_rows):
    import inspect

    from confidence import gate

    src = inspect.getsource(benchmark_eval)
    assert "hard_rules(" in src and "gate.hard_rules = " not in src and "monkeypatch" not in src
    before = inspect.getsource(gate.hard_rules)
    benchmark_eval.evaluate(no_height_rows, seed=0, authoritative_height=True)
    assert inspect.getsource(gate.hard_rules) == before
