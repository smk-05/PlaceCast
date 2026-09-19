"""perception/openings.py's result cache: .cache/perception/openings/<sha16>_<config_fingerprint>.json.

The point of the cache is a machine without torch (or OpenCV) using what a GPU machine computed, so the load
path is tested with those imports made impossible.
"""
import sys

import pytest
from PIL import Image

from perception import openings

RESULT = {"photo": "p.jpg", "openings": [{"id": "window_0", "type": "window"}], "passes": {"template_boxes": 0}}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(openings, "OPENINGS_CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def _photo(tmp_path, colour=(90, 120, 150)):
    path = tmp_path / "p.jpg"
    Image.new("RGB", (64, 48), colour).save(path)
    return path


def _block_ml(monkeypatch):
    for name in ("torch", "cv2", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)  # any import of these now raises ImportError


def test_a_cached_result_loads_without_torch_or_opencv(tmp_path, cache, monkeypatch):
    photo = _photo(tmp_path)
    openings.save_cached(photo, RESULT)
    _block_ml(monkeypatch)
    res, from_cache = openings.openings_for(photo)
    assert from_cache and res == RESULT
    assert openings.cache_path(photo).name.endswith(".json")
    assert openings.cache_path(photo).name.startswith(openings.seg.image_sha256(photo)[:16] + "_")


def test_a_miss_without_torch_says_what_to_do(tmp_path, cache, monkeypatch):
    _block_ml(monkeypatch)
    with pytest.raises(RuntimeError, match="no cached openings.*--import-openings"):
        openings.openings_for(_photo(tmp_path))


def test_the_key_changes_with_the_image_the_model_the_passes_and_the_tuning(tmp_path, cache, monkeypatch):
    a, b = _photo(tmp_path), _photo(tmp_path / "..", colour=(1, 2, 3))
    assert openings.cache_path(a) != openings.cache_path(b)                                 # different bytes
    base = openings.cache_path(a)
    assert openings.cache_path(a) == base                                                   # stable
    assert openings.cache_path(a, fine=False) != base                                       # pass flags
    assert openings.cache_path(a, template=False) != base
    assert openings.cache_path(a, model_id="IDEA-Research/grounding-dino-base") != base     # model
    monkeypatch.setattr(openings, "ACCEPT_SCORE", 0.30)
    assert openings.cache_path(a) != base                                                   # a tuning constant


def test_an_edit_to_a_decision_function_invalidates_old_results(tmp_path, cache, monkeypatch):
    photo = _photo(tmp_path)
    before = openings.config_fingerprint()

    def priors(touches, aspect, width_frac, peers):  # a different priors(): the fingerprint hashes the source
        return {}

    monkeypatch.setattr(openings, "priors", priors)
    assert openings.config_fingerprint() != before


def test_export_and_import_move_only_the_json_results(tmp_path, cache, monkeypatch):
    photo = _photo(tmp_path)
    openings.save_cached(photo, RESULT)
    (cache / "0123456789abcdef_deadbeef.pkl").write_bytes(b"machine-local DINO outputs")
    out = tmp_path / "handover"
    assert openings.export_openings(out) == 1
    assert [f.name for f in out.iterdir()] == [openings.cache_path(photo).name]

    monkeypatch.setattr(openings, "OPENINGS_CACHE_DIR", tmp_path / "other_machine")
    assert openings.load_cached(photo) is None
    assert openings.import_openings(out) == 1
    assert openings.load_cached(photo) == RESULT
