"""perception/openings.py's last two decision rules: occlusion by a distractor, and window columns."""
import numpy as np
import pytest

from perception import openings as O

SIZE = 1000
MASK = np.ones((SIZE, SIZE), bool)
GRAY = np.full((SIZE, SIZE), 120, np.uint8)  # flat: every edge factor is the same for every class
GEOM = (0, 0, SIZE, SIZE)


def det(box, **scores):
    """A detection whose ALL_CLASSES score vector is zero except the named classes."""
    v = np.zeros(len(O.ALL_CLASSES))
    for name, s in scores.items():
        v[O.ALL_CLASSES.index(name)] = s
    return {"box": box, "dino": v, "source": "dino"}


def classify(*dets):
    return O.classify(list(dets), MASK, GRAY, GEOM)


DOOR_AT_GROUND = (400, 700, 500, 995)  # aspect ~2.9, touching the ground line


@pytest.mark.parametrize("lamp,replaces", [(0.70, False), (0.77, False), (0.90, True)])
def test_a_distractor_replaces_the_opening_type_only_at_1_3x_the_best_opening_class(lamp, replaces):
    [o] = classify(det(DOOR_AT_GROUND, door=0.60, lamp=lamp))
    if replaces:
        assert (o["type"], o["other_label"], o["occluded_by"]) == ("other", "lamp", None)
    else:
        assert o["type"] == "door" and o["occluded_by"] == "lamp"
        assert o["decision"] == "REVIEW" and "partly occluded by lamp" in o["reasons"]


def test_only_lamp_column_and_sign_count_as_occluders():
    [o] = classify(det(DOOR_AT_GROUND, door=0.60, balcony=0.70))  # a balcony that wins is still a balcony
    assert o["type"] == "other" and o["other_label"] == "balcony" and o["occluded_by"] is None


def test_a_confident_opening_with_no_distractor_is_untouched():
    [o] = classify(det(DOOR_AT_GROUND, door=0.60))
    assert o["type"] == "door" and o["decision"] == "ACCEPT" and o["occluded_by"] is None


# ---------------------------------------------------------------- window columns


def test_window_above_needs_a_same_width_window_overlapping_and_higher():
    d = det((400, 700, 500, 900), door=0.5)
    above = det((405, 400, 495, 600), window=0.5)  # width 90 vs 100, overlap 90%, well above
    assert O.window_above(d, [d, above])
    assert not O.window_above(d, [d, det((405, 400, 495, 600), door=0.5)])   # what is above is not a window
    assert not O.window_above(d, [d, det((400, 400, 560, 600), window=0.5)])  # 60% wider: not the same width
    assert not O.window_above(d, [d, det((470, 400, 570, 600), window=0.5)])  # only 30% horizontal overlap
    assert not O.window_above(d, [d, det((405, 800, 495, 990), window=0.5)])  # below, not above
    assert not O.window_above(d, [d, det((405, 650, 495, 850), window=0.5)])  # overlaps the box vertically


def test_a_window_column_raises_the_window_prior_and_halves_door_and_entrance():
    plain = O.priors(True, 2.0, 0.1, peers=0)
    column = O.priors(True, 2.0, 0.1, peers=0, column=True)
    assert plain["window"] == pytest.approx(0.45 * 0.8) and column["window"] == 1.0
    assert column["door"] == pytest.approx(plain["door"] * 0.5)
    assert column["entrance"] == pytest.approx(plain["entrance"] * 0.5)
    assert O.priors(True, 2.0, 0.1, peers=0)["door"] == plain["door"]  # column defaults to off


def test_the_column_rule_turns_a_door_like_window_into_a_window_when_dino_allows():
    below = det((400, 700, 500, 995), door=0.30, window=0.25)  # DINO barely prefers door
    above = det((405, 300, 495, 500), window=0.6)
    with_column = classify(below, above)
    assert [o["type"] for o in with_column if o["box_px"][1] == 700] == ["window"]
    assert classify(below)[0]["type"] == "door"  # no window above: still a door
