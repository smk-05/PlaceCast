"""perception/prism_ai_fill.py: tiling, the 60% rule, the call budget, the Replicate request shape and caching.

Nothing here touches the network: the backend is a fake, and the Replicate runner is injected.
"""
import io
import json

import numpy as np
import pytest
from PIL import Image

from perception import prism_ai_fill as F


class FakeBackend:
    """Fills the masked pixels with one colour, and checks every rule about what may be sent."""

    def __init__(self, colour=(0, 255, 0), fail=None):
        self.colour, self.fail, self.calls, self.cache_hits = colour, fail, [], 0

    def fill(self, image, mask, prompt):
        assert image.shape[:2] == mask.shape and max(image.shape[:2]) <= F.TILE_MAX  # a tile, at most 1024 px
        assert mask.mean() <= F.MAX_EMPTY_FRACTION  # NEVER a tile more than 60% empty
        self.calls.append({"shape": image.shape[:2], "empty": float(mask.mean()), "prompt": prompt})
        if self.fail:
            raise self.fail
        out = image.copy()
        out[mask] = self.colour
        return out


def _square_mask(h, w, empty):
    """A (h, w) mask whose leftmost `empty` fraction of columns is True."""
    m = np.zeros((h, w), bool)
    m[:, : int(round(empty * w))] = True
    return m


# ------------------------------------------------------------------- tiles


def test_tile_starts_cover_the_length_with_the_required_overlap():
    assert F.tile_starts(900) == [0] and F.tile_starts(1024) == [0]
    starts = F.tile_starts(2000)
    assert starts[0] == 0 and starts[-1] + F.TILE_MAX == 2000  # the last tile ends exactly at the edge
    assert all(b - a <= F.TILE_MAX - F.OVERLAP for a, b in zip(starts, starts[1:]))  # consecutive tiles overlap >= 128
    long_starts = F.tile_starts(4000)
    assert long_starts[-1] + F.TILE_MAX == 4000 and all(b - a <= 896 for a, b in zip(long_starts, long_starts[1:]))


def test_no_tile_is_ever_larger_than_1024_px():
    fill = np.zeros((1500, 2600), bool)
    fill[::3, ::3] = True
    fill[:, :700] = True
    send, empty = F.plan_wall_tiles(0, fill)
    assert send or empty
    for t in send + empty:
        assert t.x1 - t.x0 <= F.TILE_MAX and t.y1 - t.y0 <= F.TILE_MAX


def test_a_tile_more_than_60_percent_empty_is_never_planned_for_sending():
    fill = _square_mask(400, 800, 0.65)  # one tile (800 px), 65% to fill
    send, too_empty = F.plan_wall_tiles(0, fill)
    assert send == [] and len(too_empty) == 1 and too_empty[0].empty == pytest.approx(0.65)
    send, too_empty = F.plan_wall_tiles(0, _square_mask(400, 800, 0.55))
    assert len(send) == 1 and too_empty == []


def test_a_tile_with_almost_nothing_to_fill_is_not_worth_a_call():
    fill = np.zeros((400, 800), bool)
    fill[:10, :10] = True  # 100 px
    assert F.plan_wall_tiles(0, fill) == ([], [])


def test_overlapping_tiles_blend_to_one():
    shape = (300, 1920)
    a, b = (F.Tile(0, x, 0, x + 1024, 300, 1, 1.0) for x in (0, 896))  # the tiler's own spacing: a 128 px overlap
    wa, wb = F.tile_weights(a, shape), F.tile_weights(b, shape)
    total = np.zeros(shape, np.float32)
    total[:, 0:1024] += wa
    total[:, 896:1920] += wb
    overlap = total[:, 896:1024]
    assert overlap.min() > 0.95 and overlap.max() < 1.1  # complementary linear ramps: a seam-free mix
    assert wa[:, 100:800].min() == 1.0 and total[:, :890].min() == 1.0  # tile interiors are full weight
    assert wa[:, 0].min() == 1.0 and wb[:, -1].min() == 1.0  # a border at the wall's own edge does not ramp


def test_run_normalises_the_blend_so_two_tiles_meet_without_a_jump():
    class Constant(FakeBackend):
        def fill(self, image, mask, prompt):
            self.calls.append(1)
            out = image.copy()
            out[mask] = (0, 60 if len(self.calls) == 1 else 220, 0)  # tile 1 paints 60, tile 2 paints 220
            return out

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (300, 1920, 3), dtype=np.uint8)
    fill = np.zeros((300, 1920), bool)
    fill[:, 700:1300] = True  # spans the overlap of tiles [0, 1024) and [896, 1920)
    ai = F.AIFill(Constant(), "x", max_calls=5)
    out = ai.run({0: rgb}, {0: fill})
    green = out[0][0][:, :, 1]
    row = green[150, 700:1300]
    assert ai.stats["sent"] == 2
    assert np.abs(np.diff(row)).max() < 15  # ramps meet smoothly: no step where one tile hands over to the other
    assert abs(float(row[0]) - float(row[-1])) > 100  # ... and the answer really does move from one tile's colour to the other's


# ------------------------------------------------------------------ budget


def test_the_budget_counts_attempts_and_stops_at_its_cap():
    b = F.Budget(2)
    b.spend(), b.spend()
    with pytest.raises(F.BudgetExceeded):
        b.spend()
    assert b.used == 2 and b.left == 0
    assert F.Budget(1000).cap == F.MAX_TOTAL_CALLS == 40  # nobody can raise the hard ceiling


def test_the_persistent_budget_holds_across_instances(tmp_path):
    path = tmp_path / "budget.json"
    a = F.PersistentBudget(path, cap=3)
    a.spend(), a.spend()
    b = F.PersistentBudget(path, cap=3)  # a second process
    assert b.used == 2
    b.spend()
    with pytest.raises(F.BudgetExceeded):
        F.PersistentBudget(path, cap=3).spend()
    assert json.loads(path.read_text()) == {"cap": 3, "used": 3}
    assert F.PersistentBudget(tmp_path / "other.json", cap=99).cap == 40


# ------------------------------------------------------------------ AIFill


def _walls_and_fills(empties):
    rng = np.random.default_rng(0)
    walls = {i: rng.integers(0, 255, (400, 800, 3), dtype=np.uint8) for i in range(len(empties))}
    return walls, {i: _square_mask(400, 800, e) for i, e in enumerate(empties)}


def test_run_sends_the_biggest_tiles_first_up_to_the_cap_and_skips_over_full_ones():
    walls, fills = _walls_and_fills([0.10, 0.50, 0.30, 0.70])  # wall 3 is >60% empty; wall 1 is the biggest fill
    backend = FakeBackend()
    ai = F.AIFill(backend, "a stone facade", max_calls=2)
    out = ai.run(walls, fills)
    assert ai.stats["candidate_tiles"] == 3 and ai.stats["skipped_too_empty"] == 1 and ai.stats["sent"] == 2
    assert sorted(out) == [1, 2]  # walls 1 and 2 (the two biggest fills); wall 0 was ranked third, wall 3 skipped
    assert [round(c["empty"], 2) for c in backend.calls][0] > [round(c["empty"], 2) for c in backend.calls][1]
    assert all(c["prompt"] == "a stone facade" for c in backend.calls)
    rgb, cover = out[1]
    assert rgb.shape == (400, 800, 3) and cover.all()
    assert (rgb[fills[1]][:, 1] > 240).mean() > 0.99  # the fake's green went where the mask was


def test_run_with_no_backend_only_plans():
    walls, fills = _walls_and_fills([0.2, 0.4])
    ai = F.AIFill(None, "x")
    assert ai.run(walls, fills) == {} and ai.stats["candidate_tiles"] == 2 and ai.stats["sent"] == 0


def test_the_mask_that_is_sent_is_grown_but_a_tile_pushed_over_60_percent_by_growing_is_skipped():
    walls, fills = _walls_and_fills([0.598])
    backend = FakeBackend()
    ai = F.AIFill(backend, "x")
    ai.run(walls, fills)
    assert ai.stats["sent"] == 0 and ai.stats["skipped_too_empty"] == 1 and backend.calls == []  # 0.598 + growth > 0.60


def test_a_failing_tile_is_recorded_and_the_others_continue():
    walls, fills = _walls_and_fills([0.3, 0.4])
    backend = FakeBackend(fail=RuntimeError("boom"))
    ai = F.AIFill(backend, "x")
    assert ai.run(walls, fills) == {}
    assert ai.stats["failed"] == 2 and ai.stats["sent"] == 0 and "boom" in ai.stats["errors"][0]


def test_an_exhausted_budget_stops_the_run_cleanly():
    class Broke(FakeBackend):
        def fill(self, image, mask, prompt):
            raise F.BudgetExceeded("none left")

    walls, fills = _walls_and_fills([0.3, 0.4])
    ai = F.AIFill(Broke(), "x")
    ai.run(walls, fills)
    assert ai.stats["budget_stopped"] is True and ai.stats["sent"] == 0


# ---------------------------------------------------------------- backend


def _png_bytes(array, mode="RGB"):
    buf = io.BytesIO()
    Image.fromarray(array, mode).save(buf, format="PNG")
    buf.seek(0)
    return buf


class FakeRunner:
    """Stands in for generate.throttle.run: records the request, answers with the image with masked pixels inverted."""

    def __init__(self, out_size=None):
        self.requests, self.out_size = [], out_size

    def __call__(self, model, *, input):
        image = np.array(Image.open(io.BytesIO(input["image"].getvalue())).convert("RGB"))
        mask = np.array(Image.open(io.BytesIO(input["mask"].getvalue())).convert("L")) > 127
        self.requests.append({"model": model, "input": dict(input), "image_shape": image.shape, "mask": mask})
        out = image.copy()
        out[mask] = 255 - out[mask]
        if self.out_size:
            out = np.array(Image.fromarray(out).resize(self.out_size))
        return _png_bytes(out)


def test_the_backend_sends_a_padded_png_pair_and_crops_the_result_back(tmp_path):
    runner, budget = FakeRunner(), F.Budget(5)
    backend = F.FluxFillBackend(budget, tmp_path / "cache", runner=runner)
    image = np.random.default_rng(1).integers(0, 255, (100, 300, 3), dtype=np.uint8)  # smaller than 256 on one side
    mask = np.zeros((100, 300), bool)
    mask[:, :120] = True
    out = backend.fill(image, mask, "a stone facade")
    req = runner.requests[0]
    assert req["model"] == "black-forest-labs/flux-fill-pro"
    assert set(req["input"]) == {"image", "mask", "prompt", "steps", "guidance", "seed", "output_format"}
    assert req["input"]["prompt"] == "a stone facade" and req["input"]["output_format"] == "png"
    assert req["image_shape"][:2] == (256, 320)  # padded up to whole multiples of 32, at least 256 a side
    assert req["mask"][:100, :120].all() and not req["mask"][:100, 120:300].any() and not req["mask"][100:].any()
    assert out.shape == (100, 300, 3)  # cropped back to the tile
    assert (out[:, 120:] == image[:, 120:]).all()  # the kept area comes back as it went in
    assert (out[:, :120] == 255 - image[:, :120]).all()  # the masked area is the model's answer
    assert budget.used == 1


def test_an_identical_tile_is_served_from_the_cache_and_costs_nothing(tmp_path):
    runner, budget = FakeRunner(), F.Budget(5)
    backend = F.FluxFillBackend(budget, tmp_path / "cache", runner=runner)
    image = np.random.default_rng(2).integers(0, 255, (300, 500, 3), dtype=np.uint8)
    mask = _square_mask(300, 500, 0.4)
    first = backend.fill(image, mask, "p")
    second = backend.fill(image, mask, "p")
    assert (first == second).all() and len(runner.requests) == 1 and budget.used == 1 and backend.cache_hits == 1
    backend.fill(image, mask, "a different prompt")  # anything that changes the request is a new call
    assert len(runner.requests) == 2 and budget.used == 2
    # ... and the cache survives a new backend (a re-bake in another process)
    again = F.FluxFillBackend(F.Budget(0), tmp_path / "cache", runner=runner)
    assert (again.fill(image, mask, "p") == first).all()


def test_a_result_of_another_size_is_put_back_on_the_tiles_grid(tmp_path):
    runner = FakeRunner(out_size=(640, 512))  # the model returned a different resolution
    backend = F.FluxFillBackend(F.Budget(1), tmp_path / "cache", runner=runner)
    out = backend.fill(np.zeros((300, 500, 3), np.uint8), _square_mask(300, 500, 0.3), "p")
    assert out.shape == (300, 500, 3)


def test_the_call_is_counted_before_it_is_made_so_a_failure_gets_no_free_retry(tmp_path):
    def broken(model, *, input):
        raise RuntimeError("replicate is down")

    budget = F.Budget(3)
    backend = F.FluxFillBackend(budget, tmp_path / "cache", runner=broken)
    with pytest.raises(RuntimeError):
        backend.fill(np.zeros((300, 500, 3), np.uint8), _square_mask(300, 500, 0.3), "p")
    assert budget.used == 1
    exhausted = F.FluxFillBackend(F.Budget(0), tmp_path / "cache2", runner=FakeRunner())
    with pytest.raises(F.BudgetExceeded):
        exhausted.fill(np.zeros((300, 500, 3), np.uint8), _square_mask(300, 500, 0.3), "p")
