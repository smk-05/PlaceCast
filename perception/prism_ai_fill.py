#!/usr/bin/env python
"""AI hole fill for the photo-textured prism walls: FLUX Fill Pro (Replicate) inpaints what the camera did not see.

perception/prism_texture.py bakes a photo onto the prism's walls; texels the camera could not see (occluder holes, and
bands above and below the photographed coverage) are otherwise a plain procedural facade. On a wall that IS well
covered, this module asks FLUX Fill to continue the facade instead: same material, same window rows.

Rules (all enforced here, and tested against a fake backend, never the network):
  * only walls with at least MIN_COVERAGE (30%) of their texels from the photo are eligible;
  * a wall texture is worked in TILES of at most TILE_MAX (1024) px with an OVERLAP (128) px overlap, blended with
    linear ramps in the overlap; the mask sent is the unseen texels grown by MASK_GROW_PX (white = inpaint);
  * a tile whose mask is more than MAX_EMPTY_FRACTION (60%) of it is NEVER sent (the model would be inventing a wall,
    not continuing one): it stays procedural; a tile with almost nothing to fill is not sent either;
  * every real call is counted against a Budget. PersistentBudget keeps the count in a file, so a hard cap of
    MAX_TOTAL_CALLS (40) holds across runs and across buildings. A cache hit costs nothing and is not counted;
  * calls go through generate/throttle.py (paced for a low-credit account, retried on 429).

The filled pixels are recorded (WallTexture.ai_weight in perception/prism_texture.py) so the record can say how much of
each wall is AI (`ai_filled_fraction`) and the viewer can tint it.
"""
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = "black-forest-labs/flux-fill-pro"
TILE_MAX = 1024
OVERLAP = 128
MIN_COVERAGE = 0.30
MAX_EMPTY_FRACTION = 0.60
MIN_FILL_PX = 2000  # a tile with less than this to fill (about 5 m2 at 20 px/m) is not worth a call
MASK_GROW_PX = 6
MULTIPLE = 32  # the model wants sizes in whole multiples of this
MIN_SIDE = 256
MAX_TOTAL_CALLS = 40
DEFAULT_BUDGET_FILE = ROOT / "data" / "replicate_budget_prism_fill.json"
STEPS, GUIDANCE = 50, 60

BASE_PROMPT = (
    "{style}. Continue the existing wall, its material and its rows of windows seamlessly into the masked area, "
    "keeping the window sizes, spacing and alignment of the visible rows. Flat frontal view, even natural daylight, "
    "photographic detail. No people, no cars, no trees, no sky, no text."
)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Counts Replicate predictions. A call is counted when it is attempted, so a failure never gets a free retry."""

    def __init__(self, cap):
        self.cap, self.used = min(int(cap), MAX_TOTAL_CALLS), 0

    def spend(self):
        if self.used >= self.cap:
            raise BudgetExceeded(f"{self.used} Replicate calls used; the cap is {self.cap}")
        self.used += 1

    @property
    def left(self):
        return self.cap - self.used


class PersistentBudget(Budget):
    """A Budget whose count lives in a JSON file, so the cap holds across processes: {"cap": 40, "used": n}."""

    def __init__(self, path=DEFAULT_BUDGET_FILE, cap=MAX_TOTAL_CALLS):
        self.path = Path(path)
        self.cap = min(int(cap), MAX_TOTAL_CALLS)
        self.used = json.loads(self.path.read_text())["used"] if self.path.is_file() else 0

    def spend(self):
        if self.used >= self.cap:
            raise BudgetExceeded(f"{self.used} Replicate calls used; the cap is {self.cap}")
        self.used += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"cap": self.cap, "used": self.used}))  # written BEFORE the call is made


# ------------------------------------------------------------------- tiles


@dataclass(frozen=True)
class Tile:
    wall: int
    x0: int
    y0: int
    x1: int
    y1: int
    fill_px: int
    empty: float  # fraction of the tile that is to be filled

    @property
    def size(self):
        return self.x1 - self.x0, self.y1 - self.y0


def tile_starts(length, tile=TILE_MAX, overlap=OVERLAP):
    """Start offsets of tiles of at most `tile` px covering `length`, consecutive tiles overlapping by `overlap`."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] + tile < length:
        starts.append(length - tile)  # the last tile is pulled back to end exactly at the edge
    return starts


def plan_wall_tiles(wall, fill):
    """-> (tiles worth sending, tiles skipped as too empty). `fill` is the (h, w) bool mask of texels to inpaint."""
    h, w = fill.shape
    send, too_empty = [], []
    for y0 in tile_starts(h):
        for x0 in tile_starts(w):
            x1, y1 = min(w, x0 + TILE_MAX), min(h, y0 + TILE_MAX)
            n = int(fill[y0:y1, x0:x1].sum())
            empty = n / float((x1 - x0) * (y1 - y0))
            tile = Tile(wall, x0, y0, x1, y1, n, empty)
            if n < MIN_FILL_PX:
                continue
            (too_empty if empty > MAX_EMPTY_FRACTION else send).append(tile)
    return send, too_empty


def tile_weights(tile, wall_shape):
    """Blend weights for a tile: 1 in its interior, ramping linearly to ~0 across the overlap at each border that is
    shared with a neighbouring tile (borders at the wall's own edge do not ramp)."""
    h, w = wall_shape
    tw, th = tile.x1 - tile.x0, tile.y1 - tile.y0

    def ramp(n, lo_shared, hi_shared):
        r = np.ones(n, np.float32)
        k = min(OVERLAP, n // 2)
        if lo_shared:
            r[:k] = np.linspace(0.02, 1.0, k, dtype=np.float32)
        if hi_shared:
            r[n - k :] = np.minimum(r[n - k :], np.linspace(1.0, 0.02, k, dtype=np.float32))
        return r

    return ramp(th, tile.y0 > 0, tile.y1 < h)[:, None] * ramp(tw, tile.x0 > 0, tile.x1 < w)[None, :]


# ---------------------------------------------------------------- backends


def _png(array, mode):
    buf = io.BytesIO()
    Image.fromarray(array, mode).save(buf, format="PNG")
    buf.name = "tile.png"
    return buf


def _round_up(n, multiple=MULTIPLE, minimum=MIN_SIDE):
    return max(minimum, -(-n // multiple) * multiple)


def _read_output(output):
    """Replicate returns a URL, a file-like object, or a list of either."""
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, "read"):
        return output.read()
    import requests

    r = requests.get(str(output), timeout=180)
    r.raise_for_status()
    return r.content


class FluxFillBackend:
    """FLUX Fill Pro on Replicate. `fill(image, mask, prompt)` -> the tile with the masked area inpainted.

    The tile is padded (edge-reflected, mask 0 = kept) up to whole multiples of 32 and at least 256 px, sent as PNGs
    (mask white = inpaint), and the result is cropped back. Results are cached on disk by a hash of everything that
    determines them, so re-baking the same tile costs nothing; only a real call is counted against the budget."""

    def __init__(self, budget, cache_dir, *, seed=7, steps=STEPS, guidance=GUIDANCE, runner=None):
        self.budget, self.cache_dir = budget, Path(cache_dir)
        self.seed, self.steps, self.guidance = seed, steps, guidance
        self.runner = runner  # (model, input=...) -> output; default generate.throttle.run
        self.cache_hits = 0

    def _key(self, image_png, mask_png, prompt):
        h = hashlib.sha256()
        for part in (MODEL, prompt, str(self.seed), str(self.steps), str(self.guidance), image_png, mask_png):
            h.update(part if isinstance(part, bytes) else part.encode())
        return h.hexdigest()[:24]

    def fill(self, image, mask, prompt):
        h, w = image.shape[:2]
        ph, pw = _round_up(h), _round_up(w)
        img = np.pad(image, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect" if ph - h < h and pw - w < w else "edge")
        msk = np.zeros((ph, pw), np.uint8)
        msk[:h, :w] = np.where(mask, 255, 0)
        image_png, mask_png = _png(img, "RGB"), _png(msk, "L")
        key = self._key(image_png.getvalue(), mask_png.getvalue(), prompt)
        cached = self.cache_dir / f"{key}.png"
        if cached.is_file():
            self.cache_hits += 1
            out = np.array(Image.open(cached).convert("RGB"))
        else:
            self.budget.spend()
            run = self.runner
            if run is None:
                from generate import throttle

                run = throttle.run
            output = run(MODEL, input={
                "image": image_png, "mask": mask_png, "prompt": prompt, "steps": self.steps,
                "guidance": self.guidance, "seed": self.seed, "output_format": "png",
            })
            data = _read_output(output)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
            out = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        if out.shape[:2] != (ph, pw):  # the model may return another size: put it back on the tile's grid
            out = cv2.resize(out, (pw, ph), interpolation=cv2.INTER_LANCZOS4)
        return out[:h, :w]


# --------------------------------------------------------------- orchestration


@dataclass
class AIFill:
    """The AI-fill settings for ONE bake (one variant). backend=None plans without spending anything."""

    backend: object = None
    prompt: str = "a stone masonry facade"
    max_calls: int = MAX_TOTAL_CALLS
    min_coverage: float = MIN_COVERAGE
    stats: dict = field(default_factory=dict)

    def plan(self, walls_fill):
        """walls_fill: {wall index: (h, w) bool mask to inpaint} for the ELIGIBLE walls. -> ranked tiles to send,
        biggest fill first, and how many were skipped as too empty."""
        send, empty = [], []
        for i, fill in walls_fill.items():
            s, e = plan_wall_tiles(i, fill)
            send += s
            empty += e
        send.sort(key=lambda t: -t.fill_px)
        return send, empty

    def run(self, walls, eligible_fill):
        """walls: {index: (rgb (h, w, 3) uint8 context image)}; eligible_fill: {index: fill mask}.
        -> {wall index: (ai_rgb float32 (h, w, 3), cover bool (h, w))} for walls where any tile was filled."""
        send, too_empty = self.plan(eligible_fill)
        self.stats = {"candidate_tiles": len(send), "skipped_too_empty": len(too_empty), "candidate_fill_px":
                      int(sum(t.fill_px for t in send)), "eligible_walls": sorted(eligible_fill), "sent": 0,
                      "failed": 0, "cache_hits": 0, "budget_stopped": False, "errors": []}
        if self.backend is None:
            return {}
        acc, wsum = {}, {}
        sent_or_tried = 0
        for tile in send:
            if sent_or_tried >= self.max_calls:
                break
            rgb = walls[tile.wall]
            fill = eligible_fill[tile.wall]
            region = (slice(tile.y0, tile.y1), slice(tile.x0, tile.x1))
            grown = cv2.dilate(fill[region].astype(np.uint8), np.ones((2 * MASK_GROW_PX + 1,) * 2, np.uint8)).astype(bool)
            if grown.mean() > MAX_EMPTY_FRACTION:  # the mask actually SENT is what the 60% rule is about
                self.stats["skipped_too_empty"] += 1
                continue
            sent_or_tried += 1
            try:
                out = self.backend.fill(rgb[region], grown, self.prompt)
            except BudgetExceeded:
                self.stats["budget_stopped"] = True
                break
            except Exception as exc:  # noqa: BLE001 - one failed tile leaves that region procedural
                self.stats["failed"] += 1
                self.stats["errors"].append(f"wall {tile.wall} tile ({tile.x0},{tile.y0}): {type(exc).__name__}: {exc}")
                continue
            self.stats["sent"] += 1
            weights = tile_weights(tile, fill.shape)
            a = acc.setdefault(tile.wall, np.zeros(rgb.shape, np.float32))
            s = wsum.setdefault(tile.wall, np.zeros(fill.shape, np.float32))
            a[region] += out.astype(np.float32) * weights[..., None]
            s[region] += weights
        self.stats["cache_hits"] = getattr(self.backend, "cache_hits", 0)
        return {i: (a / np.maximum(wsum[i], 1e-6)[..., None], wsum[i] > 0) for i, a in acc.items()}
