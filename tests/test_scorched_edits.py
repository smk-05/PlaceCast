"""scripts/make_scorched_edits.py: the parts that never touch Replicate (the IoU, the call cap, putting an edit back
in the photo's frame, and the prompts against generate/edit.py's guardrail)."""
import numpy as np
import pytest
from PIL import Image

from generate.edit import validate_prompt
from scripts import make_scorched_edits as S


def test_iou_of_identical_disjoint_and_half_overlapping_masks():
    a = np.zeros((10, 10), bool)
    a[:, :5] = True
    b = np.zeros((10, 10), bool)
    b[:, 5:] = True
    assert S.mask_iou(a, a) == 1.0 and S.mask_iou(a, b) == 0.0
    c = np.zeros((10, 10), bool)
    c[:, :10] = True
    assert S.mask_iou(a, c) == pytest.approx(0.5)
    assert S.mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 0.0  # two empty masks: no overlap, not a crash
    with pytest.raises(ValueError, match="differ in size"):
        S.mask_iou(a, np.zeros((5, 5), bool))


def test_the_call_cap_is_hard_and_cannot_be_raised_above_six():
    budget = S.Budget(100)
    assert budget.cap == S.MAX_CALLS == 6
    for _ in range(6):
        budget.spend()
    with pytest.raises(S.BudgetExceeded):
        budget.spend()
    small = S.Budget(2)
    small.spend(), small.spend()
    with pytest.raises(S.BudgetExceeded):
        small.spend()


def test_both_prompts_pass_edit_pys_silhouette_guardrail_and_keep_the_constraints():
    for prompt in (S.PROMPT, S.PROMPT_STRONG):
        assert validate_prompt(prompt) == (True, "")
        for phrase in ("same camera angle", "silhouette", "every window and door", "sepia"):
            assert phrase in prompt
    assert "Do not change the shape" in S.PROMPT_STRONG and "Do not change the shape" not in S.PROMPT


def test_the_retry_prompt_asks_for_wall_surface_features_only():
    retry = S.PROMPT_STRONG.lower()
    for word in S.ROOF_WORDS:
        assert word not in retry, f"the retry prompt still mentions {word!r}"
    for feature in ("plating", "rust", "pipes", "scorch", "soot"):
        assert feature in retry
    assert any(w in S.PROMPT.lower() for w in S.ROOF_WORDS)  # the first try is unchanged


def test_relative_iou_is_measured_against_the_noise_floor():
    assert S.relative(0.90, 0.95) == (0.947, 0.05)
    assert S.relative(None, 0.95) == (None, None) and S.relative(0.9, 0) == (None, None)


def test_an_edit_is_put_back_at_its_crop_position_in_a_photo_sized_frame(tmp_path):
    (tmp_path / "raw").mkdir()
    masked = tmp_path / "masked.png"
    edited = tmp_path / "edited.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(masked)
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    img.paste((200, 30, 30), (16, 16, 48, 48))  # a red "building" in the middle of the square edit
    img.save(edited)
    size, crop = (100, 80), (-20, 10, 60, 90)  # an 80 px crop that starts left of the frame and ends below the top
    out, border_white, flux_size = S.frame_edit(edited, masked, crop, size, tmp_path / "out.png", tmp_path)
    framed = np.asarray(Image.open(out))
    assert framed.shape == (80, 100, 3) and flux_size == (64, 64) and border_white == 1.0
    red = np.argwhere((framed[..., 0] == 200) & (framed[..., 1] == 30))
    # the 32 px red square is 40 px after the 64 -> 80 resize, centred in the crop: x in [0, 40), y in [30, 70)
    assert red[:, 1].min() >= 0 and red[:, 1].max() < 40 and red[:, 0].min() >= 28 and red[:, 0].max() < 72
    assert (framed[:, 60:] == 255).all()  # everything right of the crop stays white


# ------------------------------------------------------------------ did the openings stay in place?


def _res(*boxes, kinds=None, width=1000, decision="ACCEPT"):
    return {"mask_bbox_px": [0, 0, width, 500],
            "openings": [{"type": (kinds or ["window"] * len(boxes))[i], "decision": decision, "box_px": list(b)}
                         for i, b in enumerate(boxes)]}


def test_openings_that_did_not_move_have_zero_shift_and_pass():
    boxes = [(100, 100, 140, 160), (300, 100, 340, 160), (500, 100, 540, 160)]
    st = S.shift_stats(_res(*boxes), _res(*boxes))
    assert (st["originals"], st["matched"], st["unmatched"]) == (3, 3, 0)
    assert st["median_shift"] == 0.0 and st["p90_shift"] == 0.0 and st["passes"]


def test_a_shift_is_a_fraction_of_the_building_width_and_the_criterion_is_applied():
    before = _res((100, 100, 140, 160), (300, 100, 340, 160), (500, 100, 540, 160), width=1000)
    small = _res((110, 100, 150, 160), (310, 100, 350, 160), (510, 100, 550, 160), width=1000)   # 10 px = 1%
    st = S.shift_stats(before, small)
    assert st["median_shift"] == pytest.approx(0.01) and st["passes"]
    big = _res((130, 100, 170, 160), (330, 100, 370, 160), (530, 100, 570, 160), width=1000)     # 30 px = 3%
    assert S.shift_stats(before, big)["median_shift"] == pytest.approx(0.03) and not S.shift_stats(before, big)["passes"]
    wide = _res((100, 100, 200, 160), (300, 100, 400, 160), (500, 100, 600, 160), width=1000)
    tail = _res((100, 100, 200, 160), (300, 100, 400, 160), (600, 100, 700, 160), width=1000)     # one moved 100 px
    st = S.shift_stats(wide, tail)
    assert st["median_shift"] == 0.0 and st["p90_shift"] > 0.05 and not st["passes"]  # the p90 catches the tail


def test_an_original_with_no_admissible_partner_is_counted_not_dropped():
    before = _res((100, 100, 140, 160), (300, 100, 340, 160))
    st = S.shift_stats(before, _res((100, 100, 140, 160), (800, 300, 840, 360)))  # the second one is gone
    assert (st["originals"], st["matched"], st["unmatched"]) == (2, 1, 1)
    assert S.shift_stats(before, _res((100, 100, 300, 400)))["matched"] == 0  # a box 20x the size is not the same opening


def test_only_accepted_openings_count_as_originals_and_matching_ignores_the_edits_type():
    before = _res((100, 100, 140, 160), decision="REVIEW")
    assert S.shift_stats(before, _res((100, 100, 140, 160)))["originals"] == 0
    accepted = _res((100, 100, 140, 160), kinds=["door"])
    st = S.shift_stats(accepted, _res((102, 100, 142, 160), kinds=["other"]))  # re-typed by the detector, same place
    assert st["matched"] == 1 and st["by_kind"]["doors"]["matched"] == 1


# -------------------------------------------------------------------------- the bold NCB variant


def test_the_bold_prompt_is_wall_only_stronger_and_passes_the_guardrail():
    bold = S.PROMPT_BOLD.lower()
    assert validate_prompt(S.PROMPT_BOLD) == (True, "")
    for word in S.ROOF_WORDS:
        assert word not in bold
    for phrase in ("heavy riveted steel plates", "bright orange rust streaks", "thick pipes",
                   "black scorch marks around every window", "do not change the shape", "same camera angle"):
        assert phrase in bold


def test_the_bold_run_refuses_to_spend_a_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT_DIR", tmp_path)
    (tmp_path / f"{S.BOLD_STEM}_edit_bold.png").write_bytes(b"already here")
    budget = S.Budget(1)
    with pytest.raises(FileExistsError):
        S.run_bold(budget)
    assert budget.used == 0  # it stopped before any call


def test_the_three_panel_compare_is_the_original_then_each_edit_at_the_originals_aspect(tmp_path):
    photo = Image.new("RGB", (400, 200), (10, 20, 30))
    edits = []
    for i, colour in enumerate(((200, 0, 0), (0, 200, 0))):
        path = tmp_path / f"e{i}.png"
        Image.new("RGB", (400, 200), colour).save(path)
        edits.append(path)
    out = S.compare_all_png(photo, edits, tmp_path / "all.png", width=100)
    img = np.asarray(Image.open(out))
    assert img.shape == (50, 3 * 100 + 2 * 24, 3)
    assert tuple(img[25, 50]) == (10, 20, 30) and tuple(img[25, 174]) == (200, 0, 0) and tuple(img[25, 298]) == (0, 200, 0)
    assert tuple(img[25, 112]) == (255, 255, 255)  # a white gutter between the panels
