"""Plan 3.1: generation sees only the building, and only with a real mask."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from contracts import PhotoEvidence
from generate.mask import MaskError, mask_photo, prepare_mask, square_crop_box


def _photo(tmp_path, w=400, h=300, orientation=None):
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:] = (40, 160, 40)                    # "scenery": green
    arr[100:250, 120:320] = (200, 50, 50)     # "building": red
    im = Image.fromarray(arr)
    p = tmp_path / "photo.jpg"
    if orientation:
        exif = im.getexif()
        exif[0x0112] = orientation
        im.save(p, exif=exif, quality=95)
    else:
        im.save(p, quality=95)
    return p


def _building_mask(h=300, w=400):
    m = np.zeros((h, w), bool)
    m[100:250, 120:320] = True
    return m


def test_background_goes_white_and_building_survives(tmp_path):
    out = mask_photo(_photo(tmp_path), _building_mask(), tmp_path / "m.png")
    im = np.asarray(Image.open(out).convert("RGB")).astype(int)
    assert im.shape[0] == im.shape[1]                       # square for TRELLIS
    assert (im[0, 0] == 255).all()                          # corner is background
    c = im[im.shape[0] // 2, im.shape[1] // 2]
    assert c[0] > 150 and c[1] < 100                        # centre is building
    green = (im[..., 1] > 130) & (im[..., 0] < 90)
    # Raw photo is 75% scenery; what is left is the deliberate dilation ring.
    assert green.mean() < 0.05


def test_mask_is_applied_in_the_exif_transposed_frame(tmp_path):
    # Orientation 6 = rotate 90 CW on display: the displayed image is 300x400,
    # and that is the frame perception's mask lives in.
    p = _photo(tmp_path, orientation=6)
    with pytest.raises(MaskError, match="EXIF-transposed"):
        mask_photo(p, _building_mask(), tmp_path / "bad.png")
    mask_photo(p, np.rot90(_building_mask(), k=-1), tmp_path / "ok.png")


def test_holes_are_filled_so_an_occluded_window_is_kept():
    m = _building_mask()
    m[150:170, 200:220] = False
    assert prepare_mask(m, m.shape)[160, 210]


def test_a_tiny_mask_is_refused():
    m = np.zeros((300, 400), bool)
    m[:5, :5] = True
    with pytest.raises(MaskError):
        prepare_mask(m, m.shape)


def test_crop_box_is_square_and_contains_the_mask():
    l, t, r, b = square_crop_box(_building_mask())
    assert r - l == b - t
    assert l <= 120 and t <= 100 and r >= 320 and b >= 250


# -- pipeline wiring ---------------------------------------------------------


def _ev(model, mask):
    return PhotoEvidence(mask=mask, segmentation_model=model)


def test_stub_mask_is_not_used_for_generation(tmp_path):
    from pipeline import _generation_input
    p = _photo(tmp_path)
    img, prov = _generation_input(p, _ev("stub:central-box", _building_mask()),
                                  tmp_path, lambda *_: None, use_mask=True)
    assert img == p and prov is None


def test_real_mask_produces_masked_input_with_provenance(tmp_path):
    from pipeline import _generation_input
    p = _photo(tmp_path)
    img, prov = _generation_input(p, _ev("real:dino+sam2", _building_mask()),
                                  tmp_path, lambda *_: None, use_mask=True)
    assert img.name == "masked_0.png" and img.exists()
    assert prov["stage"] == "mask" and prov["name"] == "real:dino+sam2"


def test_no_mask_flag_and_bad_mask_fall_back_to_the_raw_photo(tmp_path):
    from pipeline import _generation_input
    p = _photo(tmp_path)
    ev = _ev("real:dino+sam2", _building_mask())
    assert _generation_input(p, ev, tmp_path, lambda *_: None, use_mask=False)[0] == p
    bad = _ev("real:dino+sam2", np.zeros((300, 400), bool))
    assert _generation_input(p, bad, tmp_path, lambda *_: None, use_mask=True)[0] == p


def test_unmasked_cached_run_is_not_relabelled_as_masked(tmp_path):
    from pipeline import _generation_input
    p = _photo(tmp_path)
    (tmp_path / "edited_0.png").write_bytes(b"cached")
    img, prov = _generation_input(p, _ev("real:dino+sam2", _building_mask()),
                                  tmp_path, lambda *_: None, use_mask=True)
    assert img == p and prov is None
