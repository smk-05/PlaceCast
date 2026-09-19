"""The height hard rule in confidence/gate.py: a non-authoritative height caps the decision at REVIEW (SPEC 7).

hard_rules() reads only a handful of attributes, so plain namespaces stand in for FitResult and Footprint, and
orientation_guessed is pinned to False so that no other rule can fire and the height rule is the only signal.
"""
from types import SimpleNamespace

import pytest

from confidence import gate
from contracts import Decision


def _fit(authoritative):
    return SimpleNamespace(is_mirrored=False, exif_silhouette_disagree=None, height_source_authoritative=authoritative)


def _fp():
    return SimpleNamespace(is_ill_posed=False, match_quality="contained_and_named", is_multipart=False)


def _height_rules(rules):
    return [(floor, why) for floor, why in rules if "height" in why]


@pytest.fixture(autouse=True)
def no_orientation_guess(monkeypatch):
    monkeypatch.setattr(gate, "orientation_guessed", lambda fit: False)


def test_a_non_authoritative_height_forces_review_and_says_why():
    rules = gate.hard_rules(_fit(authoritative=False), _fp())
    assert [floor for floor, _ in rules] == [Decision.REVIEW]  # the only rule that fired
    (floor, why), = _height_rules(rules)
    assert floor is Decision.REVIEW
    assert "height is not authoritative" in why


def test_an_authoritative_height_adds_no_height_rule():
    rules = gate.hard_rules(_fit(authoritative=True), _fp())
    assert _height_rules(rules) == []
    assert rules == ()
