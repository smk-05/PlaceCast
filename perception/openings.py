"""
Openings: doors, windows, garage doors and entrances on the photographed facade.

Sponsor directive (day 2): doors must be separable from windows and oriented, so
each opening becomes its own interactable element in the game.

Per photo:
  1. Building mask from perception.segment (cache hit; the building segmentation
     defaults are NOT touched here, so mask keys still match Soumik's).
  2. Grounding DINO on the mask's bbox with outside-mask pixels greyed out (no car
     doors / car windows), tiled so windows on wide facades keep enough pixels.
     Two tiling passes: the base tiles and a finer pass at half the tile size
     (small or foreshortened windows), merged with NMS.
  3. Per-class scores read from DINO's token logits, so every box carries a score
     for every phrase: the four opening classes plus distractors (vent, sign, lamp,
     column, chimney, balcony). A winning distractor makes the box type "other".
  4. Geometric priors: ground contact (relative to the opening's own height),
     aspect ratio, repetition in rows (2+ row peers is strong window evidence).
  5. Template gap-fill: ACCEPT windows become templates; normalised cross-correlation
     inside the building mask, in horizontal bands around the existing window rows,
     proposes missed windows, each verified by re-running DINO on a 2.5x-context crop.
  6. OpenCV edge features (frame, mullions, interior contrast) -> a soft factor in
     [0.8, 1.1] on every class score, stored under edge_features for training.
  7. Per-opening decision: REVIEW when the passable / window / other choice is
     ambiguous or weak.

Writes outputs/openings/<stem>.json and <stem>.png (overlay). Positions are also
given normalised to the building-mask bbox (u right, v down); that is what the
2D -> 3D projection onto the front_angle face consumes.

    python -m perception.openings photos\\burruss_1.jpg [--ablate]
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception import segment as seg  # noqa: E402

# --------------------------------------------------------------------- config
CLASSES = ("door", "window", "garage_door", "entrance")          # the opening classes
DISTRACTORS = ("vent", "sign", "lamp", "column", "chimney", "balcony")
ALL_CLASSES = CLASSES + DISTRACTORS                               # prompt order
PHRASES = {"door": "door", "window": "window",
           "garage_door": "garage door", "entrance": "entrance",
           **{d: d for d in DISTRACTORS}}
PROMPT_CORE = " . ".join(PHRASES[c] for c in CLASSES) + " ."          # the original prompt: opening scores come from it
PROMPT = " . ".join(PHRASES[c] for c in ALL_CLASSES) + " ."           # + distractors: read ONLY for the distractor scores
DISTRACTOR_MIN = 0.10       # lowest distractor-pass score worth fusing onto a box
PASSABLE = {"door", "garage_door", "entrance"}                    # vs "window" vs "other"
OTHER_PRIOR = 1.0

BOX_THRESHOLD = 0.25        # min best-class DINO score to keep a query
NMS_IOU = 0.5
TILE_ASPECT = 1.4           # tile long side = short side * this
TILE_OVERLAP = 0.20
FINE_TILE_SCALE = 0.5       # second pass: tiles at half the base tile size
EDGE_PX = 4                 # drop boxes cut by an interior tile edge
CROP_MARGIN = 0.03
GREY = 124
MIN_IN_MASK = 0.40          # fraction of box inside the building mask
MAX_BOX_AREA_FRAC = 0.06    # bigger boxes (of the building-mask bbox) are facade-level, not openings
MIN_BOX_PX = 12
FINE_MAX_TILE_AREA_FRAC = 0.5   # fine pass: a box filling most of its own tile is a tile-sized artifact, not an opening
GROUND_TOL = 0.05           # legacy: gap to local ground line / FACADE height (still reported)
GROUND_TOL_BOX = 0.15       # ground contact: gap below the box < this * the box's OWN height

# decision (tunable; first pass)
ACCEPT_SCORE = 0.20         # min combined score of the chosen class
GROUP_MARGIN_ACCEPT = 0.35  # passable / window / other relative margin
TYPE_MARGIN_FLAG = 0.25     # door/entrance/garage subtype margin -> flag only

# containment and garage evidence
CONTAIN_INNER_IN_OUTER = 0.70   # inner box fraction inside the outer one ...
CONTAIN_AREA_RATIO = 0.50       # ... and inner area under this * outer area: the outer box is a container
GARAGE_MIN_SCORE = 0.35         # a garage door needs a raw DINO garage score of at least this ...
GARAGE_MIN_LEAD = 0.10          # ... and this much above the raw door score, else it is a "recess"

# template gap-fill
TEMPLATE_SCALES = tuple(round(float(s), 2) for s in np.arange(0.4, 1.101, 0.1))
TEMPLATE_SQUEEZE = (1.0, 0.6)   # extra width factors: windows seen at an angle are narrower, not smaller
TEMPLATE_MAX_PER_ROW = 4
TEMPLATE_NCC_MIN = 0.55
TEMPLATE_MAX_CANDIDATES = 40
TEMPLATE_CONTEXT = 2.5          # DINO re-check crop = candidate box * this
TEMPLATE_BAND_PAD = 1.0         # band = window-row extent +/- this * median window height

OUT_DIR = ROOT / "outputs" / "openings"
COLORS = {"door": (40, 200, 70), "entrance": (0, 180, 180),
          "garage_door": (255, 150, 0), "window": (60, 120, 255),
          "other": (150, 150, 150)}


def group_of(c):
    return "passable" if c in PASSABLE else "window" if c == "window" else "other"


# --------------------------------------------------------------------- detector
def phrase_spans(tokenizer, ids, classes=ALL_CLASSES):
    """Token positions of each class phrase inside the tokenised prompt."""
    spans, start = {}, 0
    for c in classes:
        toks = tokenizer(PHRASES[c], add_special_tokens=False)["input_ids"]
        n = len(toks)
        for i in range(start, len(ids) - n + 1):
            if ids[i:i + n] == toks:
                spans[c] = list(range(i, i + n))
                start = i + n
                break
        else:
            raise RuntimeError(f"phrase {PHRASES[c]!r} not found in tokenised prompt")
    return spans


class Detector:
    def __init__(self, model_id):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (AutoModelForZeroShotObjectDetection
                      .from_pretrained(model_id).to(self.device).eval())

    def __call__(self, tile, classes=CLASSES, threshold=BOX_THRESHOLD):
        """-> list of (xyxy in tile px, per-phrase score array in `classes` order). The prompt is built from
        `classes`, so CLASSES reproduces the original four-phrase prompt exactly."""
        prompt = " . ".join(PHRASES[c] for c in classes) + " ."
        with self.torch.no_grad():
            inputs = self.processor(images=tile, text=prompt,
                                    return_tensors="pt").to(self.device)
            out = self.model(**inputs)
        spans = phrase_spans(self.processor.tokenizer, inputs["input_ids"][0].tolist(), classes)
        probs = out.logits[0].sigmoid().float().cpu().numpy()        # (Q, T)
        boxes = out.pred_boxes[0].float().cpu().numpy()              # cx cy w h, 0-1
        cls = np.stack([probs[:, spans[c]].mean(axis=1) for c in classes], axis=1)
        W, H = tile.size
        res = []
        for q in np.nonzero(cls.max(axis=1) >= threshold)[0]:
            cx, cy, w, h = boxes[q]
            res.append(((float((cx - w / 2) * W), float((cy - h / 2) * H),
                         float((cx + w / 2) * W), float((cy + h / 2) * H)), cls[q]))
        return res


# --------------------------------------------------------------------- geometry
def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("empty building mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def tiles_for(x0, y0, x1, y1):
    w, h = x1 - x0, y1 - y0

    def splits(length, other):
        side = int(other * TILE_ASPECT)
        if length <= side * 1.1:
            return [(0, length)]
        ov = int(side * TILE_OVERLAP)
        n = max(2, math.ceil((length - ov) / (side - ov)))
        step = (length - side) / (n - 1)
        return [(int(round(i * step)), int(round(i * step)) + side) for i in range(n)]

    xs = splits(w, h) if w >= h else [(0, w)]
    ys = splits(h, w) if h > w else [(0, h)]
    return [(x0 + a, y0 + b, x0 + c, y0 + d) for a, c in xs for b, d in ys]


def tile_grid(x0, y0, x1, y1, tw, th):
    """A grid of tw x th tiles (overlapping TILE_OVERLAP) covering the box; splits both axes."""
    def spans(length, size):
        size = max(1, min(int(size), length))
        if length <= size * 1.1:
            return [(0, length)]
        ov = int(size * TILE_OVERLAP)
        n = max(2, math.ceil((length - ov) / (size - ov)))
        step = (length - size) / (n - 1)
        return [(int(round(i * step)), int(round(i * step)) + size) for i in range(n)]

    return [(x0 + a, y0 + b, x0 + c, y0 + d)
            for a, c in spans(x1 - x0, tw) for b, d in spans(y1 - y0, th)]


def fine_tiles_for(x0, y0, x1, y1):
    """Tiles at FINE_TILE_SCALE of the base tile size, in both dimensions."""
    bx0, by0, bx1, by1 = tiles_for(x0, y0, x1, y1)[0]
    return tile_grid(x0, y0, x1, y1, (bx1 - bx0) * FINE_TILE_SCALE, (by1 - by0) * FINE_TILE_SCALE)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets):
    """Greedy NMS on the best opening-class score, then MERGE: the survivor keeps its box but takes the per-class
    maximum over every box it suppressed. The same window seen at two tile scales is often window 0.30 at one and
    "garage door" 0.31 at the other (zoomed in, a window looks like a big door); keeping one box's scores whole
    let the wrong scale win, so each class now keeps its best evidence and the priors pick between them."""
    dets = sorted(dets, key=lambda d: -float(d["dino"][:len(CLASSES)].max()))
    kept, merged = [], []
    for d in dets:
        for i, k in enumerate(kept):
            if iou(d["box"], k["box"]) >= NMS_IOU:
                merged[i] = np.maximum(merged[i], d["dino"])
                break
        else:
            kept.append(d)
            merged.append(np.array(d["dino"], dtype=float))
    return [{**k, "dino": m} for k, m in zip(kept, merged)]  # new dicts: the DINO memo is shared across variants


def local_ground_y(mask, x0, x1):
    """Median lowest mask row under the box's columns (the facade's ground line there)."""
    cols = mask[:, max(0, int(x0)):max(int(x0) + 1, int(math.ceil(x1)))]
    has = cols.any(axis=0)
    if not has.any():
        return None
    last = cols.shape[0] - 1 - np.argmax(cols[::-1, :], axis=0)
    return float(np.median(last[has]))


def row_peers(d, dets):
    x0, y0, x1, y1 = d["box"]
    h, cy = y1 - y0, (y0 + y1) / 2
    n = 0
    for o in dets:
        if o is d:
            continue
        ox0, oy0, ox1, oy1 = o["box"]
        oh = oy1 - oy0
        if abs((oy0 + oy1) / 2 - cy) < 0.35 * h and 0.7 < oh / h < 1.4:
            n += 1
    return n


def priors(touches, aspect, width_frac, peers):
    """Soft geometric plausibility per class, in (0, 1].

    `touches` is ground contact measured against the opening's own height. Two or more row peers are strong
    evidence of a window: window prior 1.0 even at ground level, door / entrance x0.5, and a door additionally
    needs ground contact and h/w >= 1.5 (else it all but drops out).
    """
    pri = {
        "door":        (1.0 if touches else 0.3) * (1.0 if 1.5 <= aspect <= 3.5 else 0.5),
        "entrance":    (1.0 if touches else 0.3) * (1.0 if 0.6 <= aspect <= 2.2 else 0.6),
        "garage_door": ((1.0 if touches else 0.2) * (1.0 if 0.45 <= aspect <= 1.2 else 0.4)
                        * (1.0 if width_frac >= 0.06 else 0.5)),
        "window":      (0.45 if touches else 1.0) * (1.0 if peers >= 1 else 0.8),
    }
    if peers >= 2:
        pri["window"] = 1.0
        pri["door"] *= 0.5 if (touches and aspect >= 1.5) else 0.05
        pri["entrance"] *= 0.5
    for c in DISTRACTORS:
        pri[c] = OTHER_PRIOR
    return pri


# --------------------------------------------------------------------- edge features
def _rect(shape, box, grow):
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    m[max(0, int(y0 - grow)):max(0, int(y1 + grow)), max(0, int(x0 - grow)):max(0, int(x1 + grow))] = True
    return m


def _cluster_count(vals, tol):
    n, last = 0, None
    for v in sorted(vals):
        if last is None or v - last > tol:
            n += 1
        last = v
    return n


def edge_features(gray, mask, box):
    """Frame strength, mullion lines and interior contrast of one opening (OpenCV, on the aligned photo)."""
    import cv2
    H, W = gray.shape
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    t = max(2, int(round(0.08 * min(bw, bh))))
    pad = 3 * t + 2
    rx0, ry0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    rx1, ry1 = min(W, int(math.ceil(x1)) + pad), min(H, int(math.ceil(y1)) + pad)
    g = gray[ry0:ry1, rx0:rx1].astype(np.float32)
    m = mask[ry0:ry1, rx0:rx1]
    b = (x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

    # frame: gradient in a thin band straddling the border vs a ring just outside it
    band = _rect(g.shape, b, t) & ~_rect(g.shape, (b[0] + t, b[1] + t, b[2] - t, b[3] - t), 0)
    ring = _rect(g.shape, b, 2 * t) & ~_rect(g.shape, b, t)
    ring = ring & m if (ring & m).sum() >= 20 else ring
    border = float(gm[band].mean()) if band.any() else 0.0
    outer = float(gm[ring].mean()) if ring.any() else 0.0
    frame_score = border / (border + outer + 1e-6)

    # mullions: long near-vertical / near-horizontal lines strictly inside the box
    ix0, iy0 = b[0] + 0.1 * bw, b[1] + 0.1 * bh
    ix1, iy1 = b[2] - 0.1 * bw, b[3] - 0.1 * bh
    inner = g[int(iy0):int(iy1), int(ix0):int(ix1)]
    n_v = n_h = 0
    if inner.shape[0] >= 16 and inner.shape[1] >= 16:
        ih, iw = inner.shape
        edges = cv2.Canny(cv2.GaussianBlur(inner, (3, 3), 0).astype(np.uint8), 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(8, int(0.2 * min(ih, iw))),
                                minLineLength=int(0.5 * min(ih, iw)), maxLineGap=max(3, int(0.05 * min(ih, iw))))
        vs, hs = [], []
        for x_a, y_a, x_b, y_b in (np.asarray(lines).reshape(-1, 4) if lines is not None else []):  # (N,1,4) or (N,4)
            ang = abs(math.degrees(math.atan2(y_b - y_a, x_b - x_a)))
            length = math.hypot(x_b - x_a, y_b - y_a)
            if abs(ang - 90) < 12 and length >= 0.5 * ih:
                vs.append((x_a + x_b) / 2)
            elif (ang < 12 or ang > 168) and length >= 0.5 * iw:
                hs.append((y_a + y_b) / 2)
        n_v, n_h = _cluster_count(vs, max(3, 0.06 * iw)), _cluster_count(hs, max(3, 0.06 * ih))

    # interior vs the surrounding facade
    around = _rect(g.shape, b, 3 * t) & ~_rect(g.shape, b, t)
    around = around & m if (around & m).sum() >= 20 else around
    inside_mean = float(inner.mean()) if inner.size else 0.0
    around_mean = float(g[around].mean()) if around.any() else inside_mean
    return {"frame_score": round(frame_score, 4), "frame_border_grad": round(border, 2),
            "frame_outer_grad": round(outer, 2), "mullion_count": int(n_v + n_h),
            "mullion_v": int(n_v), "mullion_h": int(n_h),
            "interior_contrast": round((inside_mean - around_mean) / 255.0, 4)}


def edge_factors(feat):
    """Soft per-class multiplier in [0.8, 1.1]: a strong frame raises it, many mullions favour window over door,
    and a window is usually darker than the wall around it."""
    frame = 0.9 + 0.2 * min(1.0, max(0.0, (feat["frame_score"] - 0.45) / 0.25))
    mull = min(1.0, feat["mullion_count"] / 4.0)
    dark = min(1.0, max(-1.0, -feat["interior_contrast"] / 0.2))
    fac = {}
    for c in ALL_CLASSES:
        f = frame
        if c == "window":
            f *= (1.0 + 0.1 * mull) * (1.0 + 0.05 * dark)
        elif c in PASSABLE:
            f *= 1.0 - 0.2 * mull
        fac[c] = min(1.1, max(0.8, f))
    return fac


# --------------------------------------------------------------------- main step
def load_aligned(photo, mask):
    img = Image.open(photo)
    img.load()
    rgb = img.convert("RGB")
    if mask.shape == (rgb.height, rgb.width):
        return rgb
    t = ImageOps.exif_transpose(img).convert("RGB")
    if mask.shape == (t.height, t.width):
        return t
    raise ValueError(f"mask {mask.shape} does not match image {rgb.size} (or EXIF-rotated)")


def _filter_dets(raw, mask, mw, mh):
    kept = []
    for d in raw:
        x0, y0, x1, y1 = d["box"]
        bw, bh = x1 - x0, y1 - y0
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            continue
        if bw * bh > MAX_BOX_AREA_FRAC * mw * mh:
            continue
        sub = mask[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))]
        if sub.size == 0 or sub.mean() < MIN_IN_MASK:
            continue
        kept.append(d)
    return kept


def _dino(detector, arr, box, memo):
    """Fused detections on arr[box] -> [(xyxy in region px, scores in ALL_CLASSES order)], memoised so ablation
    variants never re-run DINO on the same pixels.

    Two prompts, because extending the prompt changes the scores of the original phrases (measured on burruss_1:
    window 0.30 -> 0.15, garage door 0.05 -> 0.30), which breaks every threshold tuned on them. Boxes and the
    four opening scores come from the original prompt, unchanged; the six distractor scores are read from the
    extended prompt and fused onto each box (max over extended-prompt boxes with IoU > 0.5, else 0).
    """
    key = tuple(int(v) for v in box)
    if memo is not None and key in memo:
        return memo[key]
    x0, y0, x1, y1 = key
    tile = Image.fromarray(arr[y0:y1, x0:x1])
    core = detector(tile, CLASSES)
    ext = detector(tile, ALL_CLASSES, threshold=DISTRACTOR_MIN)
    out = []
    for b, sc in core:
        dist = np.zeros(len(DISTRACTORS))
        for eb, es in ext:
            if iou(b, eb) > 0.5:
                dist = np.maximum(dist, es[len(CLASSES):])
        out.append((b, np.concatenate([sc, dist])))
    if memo is not None:
        memo[key] = out
    return out


def _run_tiles(detector, arr, tiles, crop, size, memo, source, max_tile_area_frac=None):
    cx0, cy0, cx1, cy1 = crop
    W, H = size
    raw = []
    for tx0, ty0, tx1, ty1 in tiles:
        for (bx0, by0, bx1, by1), scores in _dino(detector, arr, (tx0, ty0, tx1, ty1), memo):
            interior_cut = ((tx0 > cx0 and bx0 < EDGE_PX) or
                            (tx1 < cx1 and bx1 > (tx1 - tx0) - EDGE_PX) or
                            (ty0 > cy0 and by0 < EDGE_PX) or
                            (ty1 < cy1 and by1 > (ty1 - ty0) - EDGE_PX))
            if interior_cut:
                continue
            if max_tile_area_frac and (bx1 - bx0) * (by1 - by0) > max_tile_area_frac * (tx1 - tx0) * (ty1 - ty0):
                continue
            box = (max(0.0, tx0 + bx0), max(0.0, ty0 + by0),
                   min(float(W), tx0 + bx1), min(float(H), ty0 + by1))
            raw.append({"box": box, "dino": scores, "source": source})
    return raw


def template_fill(detector, arr, gray, mask, openings, existing, memo):
    """Propose windows the DINO passes missed: NCC of the ACCEPT windows against horizontal bands around their
    rows, then a DINO re-check on a TEMPLATE_CONTEXT crop of each peak. -> raw dets tagged source 'template'."""
    import cv2
    H, W = gray.shape
    wins = [o for o in openings if o["type"] == "window" and o["decision"] == "ACCEPT"]
    if not wins:
        return []
    cy_of = lambda o: (o["box_px"][1] + o["box_px"][3]) / 2
    hmed = float(np.median([o["box_px"][3] - o["box_px"][1] for o in wins]))
    rows = []
    for o in sorted(wins, key=cy_of):
        if rows and abs(cy_of(o) - float(np.mean([cy_of(i) for i in rows[-1]]))) < 0.6 * hmed:
            rows[-1].append(o)
        else:
            rows.append([o])
    mx0, my0, mx1, my1 = mask_bbox(mask)

    cands = []
    for row in rows:
        by0 = max(my0, int(min(o["box_px"][1] for o in row) - TEMPLATE_BAND_PAD * hmed))
        by1 = min(my1, int(max(o["box_px"][3] for o in row) + TEMPLATE_BAND_PAD * hmed))
        band = gray[by0:by1, mx0:mx1]
        for o in sorted(row, key=lambda o: -o["score"])[:TEMPLATE_MAX_PER_ROW]:
            x0, y0, x1, y1 = (int(round(v)) for v in o["box_px"])
            tpl = gray[y0:y1, x0:x1]
            for s in TEMPLATE_SCALES:
                for sq in TEMPLATE_SQUEEZE:
                    tw, th = int(round((x1 - x0) * s * sq)), int(round((y1 - y0) * s))
                    if tw < MIN_BOX_PX or th < MIN_BOX_PX or th >= band.shape[0] or tw >= band.shape[1]:
                        continue
                    t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
                    if t.std() < 3:                                  # flat template: NCC is meaningless
                        continue
                    res = np.nan_to_num(cv2.matchTemplate(band, t, cv2.TM_CCOEFF_NORMED), nan=0.0, posinf=0.0, neginf=0.0)
                    win = np.ones((max(3, th // 2) | 1, max(3, tw // 2) | 1), np.uint8)
                    for py, px in zip(*np.nonzero((res >= TEMPLATE_NCC_MIN) & (res == cv2.dilate(res, win)))):
                        box = (mx0 + px, by0 + py, mx0 + px + tw, by0 + py + th)
                        sub = mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        ccx, ccy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        if sub.size == 0 or sub.mean() < MIN_IN_MASK:
                            continue
                        if any(iou(box, e) > 0.15 or (e[0] <= ccx <= e[2] and e[1] <= ccy <= e[3])
                               or (box[0] <= (e[0] + e[2]) / 2 <= box[2] and box[1] <= (e[1] + e[3]) / 2 <= box[3])
                               for e in existing):
                            continue
                        cands.append((float(res[py, px]), box))
    picked = []
    for ncc, box in sorted(cands, key=lambda c: -c[0]):
        if all(iou(box, p[1]) < 0.3 for p in picked):
            picked.append((ncc, box))
        if len(picked) >= TEMPLATE_MAX_CANDIDATES:
            break

    widx, new = ALL_CLASSES.index("window"), []
    for ncc, (x0, y0, x1, y1) in picked:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = TEMPLATE_CONTEXT * (x1 - x0) / 2, TEMPLATE_CONTEXT * (y1 - y0) / 2
        crop = (max(0, int(cx - hw)), max(0, int(cy - hh)), min(W, int(cx + hw)), min(H, int(cy + hh)))
        if crop[2] - crop[0] < 32 or crop[3] - crop[1] < 32:
            continue
        best = None
        for (bx0, by0, bx1, by1), scores in _dino(detector, arr, crop, memo):
            box = (crop[0] + bx0, crop[1] + by0, crop[0] + bx1, crop[1] + by1)
            w = float(scores[widx])
            if iou(box, (x0, y0, x1, y1)) > 0.3 and w >= BOX_THRESHOLD and (best is None or w > best[0]):
                best = (w, box, scores)
        if best:
            new.append({"box": (max(0.0, best[1][0]), max(0.0, best[1][1]), min(float(W), best[1][2]), min(float(H), best[1][3])),
                        "dino": best[2], "source": "template", "template_ncc": round(ncc, 3)})
    return new


def _relabel(o, label, reason):
    """Turn an opening into type "other" (group "other"), keeping what it was and why it changed."""
    o.setdefault("relabeled_from", o["type"])
    o["type"], o["group"], o["other_label"], o["relabel_reason"] = "other", "other", label, reason


def _inside(inner, outer):
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    outer_area = (outer[2] - outer[0]) * (outer[3] - outer[1])
    return (inner_area > 0 and ix * iy / inner_area > CONTAIN_INNER_IN_OUTER
            and inner_area < CONTAIN_AREA_RATIO * outer_area)


def apply_containment(openings):
    """A box that contains another detection is not an opening: it is a recess (if what it contains is passable)
    or a facade section (a band of windows, a bay). Judged on the types before any relabelling, so nested
    containers all see the openings inside them. Only real openings count as contents; boxes already "other"
    (a distractor won) neither count as contents nor are relabelled."""
    types = [o["type"] for o in openings]
    for i, o in enumerate(openings):
        if types[i] == "other":
            continue
        # "other" boxes (a distractor won, or a weak garage) are not contents: a column inside an entrance is not
        # a reason to call the entrance a facade section.
        inner = [j for j in range(len(openings))
                 if j != i and types[j] != "other" and _inside(openings[j]["box_px"], o["box_px"])]
        if inner:
            passable = any(group_of(types[j]) == "passable" for j in inner)
            _relabel(o, "recess" if passable else "facade_section",
                     f"contains {len(inner)} detection(s): " + ", ".join(sorted({types[j] for j in inner})))
    return openings


def classify(dets, mask, gray, geom):
    """Priors + edge factors + the three-group decision, for every box. -> list of opening dicts (unsorted)."""
    mx0, my0, mw, mh = geom
    openings = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        bw, bh = x1 - x0, y1 - y0
        ground = local_ground_y(mask, x0, x1)
        gap_px = (ground - y1) if ground is not None else None
        touches = gap_px is not None and gap_px < GROUND_TOL_BOX * bh          # relative to the opening itself
        gap = gap_px / mh if gap_px is not None else 1.0                       # legacy: relative to the facade
        peers = row_peers(d, dets)
        pri = priors(touches, bh / bw, bw / mw, peers)
        feat = edge_features(gray, mask, d["box"])
        fac = edge_factors(feat)
        dino = {c: float(d["dino"][i]) for i, c in enumerate(ALL_CLASSES)}
        comb = {c: dino[c] * pri[c] * fac[c] for c in ALL_CLASSES}

        order = sorted(ALL_CLASSES, key=comb.get, reverse=True)
        top, s1, s2 = order[0], comb[order[0]], comb[order[1]]
        g_top = group_of(top)
        rival = max((c for c in ALL_CLASSES if group_of(c) != g_top), key=comb.get)
        group_margin = (s1 - comb[rival]) / s1 if s1 > 0 else 0.0
        type_margin = (s1 - s2) / s1 if s1 > 0 else 0.0

        reasons = []
        if s1 < ACCEPT_SCORE:
            reasons.append(f"weak detection: combined score {s1:.2f} < {ACCEPT_SCORE}")
        if group_margin < GROUP_MARGIN_ACCEPT:
            names = {g_top, group_of(rival)}
            label = "door-vs-window" if names == {"passable", "window"} else f"{g_top}-vs-{group_of(rival)}"
            reasons.append(f"{label} ambiguous: margin {group_margin:.2f} < {GROUP_MARGIN_ACCEPT}")
        flags = []
        if not reasons and top in PASSABLE and type_margin < TYPE_MARGIN_FLAG:
            flags.append(f"subtype_uncertain ({order[0]} vs {order[1]})")

        o = {
            "type": "other" if g_top == "other" else top,
            "group": g_top,
            "decision": "REVIEW" if reasons else "ACCEPT",
            "reasons": reasons,
            "flags": flags,
            "score": round(s1, 4),
            "group_margin": round(group_margin, 4),
            "type_margin": round(type_margin, 4),
            "dino_scores": {k: round(v, 4) for k, v in dino.items()},
            "priors": {k: round(v, 3) for k, v in pri.items()},
            "box_px": [round(v, 1) for v in (x0, y0, x1, y1)],
            # normalised to the building-mask bbox: u right, v down, 0-1
            "box_uv": [round((x0 - mx0) / mw, 4), round((y0 - my0) / mh, 4),
                       round((x1 - mx0) / mw, 4), round((y1 - my0) / mh, 4)],
            "center_uv": [round(((x0 + x1) / 2 - mx0) / mw, 4),
                          round(((y0 + y1) / 2 - my0) / mh, 4)],
            "bottom_above_ground_frac": round(gap, 4),
            "touches_ground": bool(touches),
            "aspect_h_over_w": round(bh / bw, 3),
            "row_peers": peers,
            # added fields
            "source": d.get("source", "dino"),
            "other_label": PHRASES[top] if g_top == "other" else None,
            "gap_over_box_h": round(gap_px / bh, 4) if gap_px is not None else None,
            "touches_ground_facade": bool(gap < GROUND_TOL),
            "edge_features": {**feat, "factor": {k: round(v, 3) for k, v in fac.items()}},
        }
        if "template_ncc" in d:
            o["template_ncc"] = d["template_ncc"]
        if top == "garage_door" and not (dino["garage_door"] >= GARAGE_MIN_SCORE
                                         and dino["garage_door"] - dino["door"] >= GARAGE_MIN_LEAD):
            _relabel(o, "recess", f"garage evidence too weak: garage {dino['garage_door']:.2f}, door {dino['door']:.2f}")
        openings.append(o)
    return apply_containment(openings)


def detect_openings(photo, detector, *, fine=True, template=True, memo=None):
    photo = Path(photo)
    mask, _meta, _png = seg.segment_file(photo)
    mask = np.asarray(mask).astype(bool)
    img = load_aligned(photo, mask)
    H, W = mask.shape

    mx0, my0, mx1, my1 = mask_bbox(mask)
    mw, mh = mx1 - mx0, my1 - my0
    pad_x, pad_y = int(CROP_MARGIN * mw), int(CROP_MARGIN * mh)
    crop = (max(0, mx0 - pad_x), max(0, my0 - pad_y), min(W, mx1 + pad_x), min(H, my1 + pad_y))

    gray = np.array(img.convert("L"))
    arr = np.array(img)
    arr[~mask] = GREY                       # hide cars, trees, neighbours
    memo = {} if memo is None else memo

    raw = _run_tiles(detector, arr, tiles_for(*crop), crop, (W, H), memo, "dino")
    if fine:
        raw += _run_tiles(detector, arr, fine_tiles_for(*crop), crop, (W, H), memo, "dino",
                          max_tile_area_frac=FINE_MAX_TILE_AREA_FRAC)
    kept = nms(_filter_dets(raw, mask, mw, mh))
    geom = (mx0, my0, mw, mh)
    openings = classify(kept, mask, gray, geom)

    n_template = 0
    if template:
        new = template_fill(detector, arr, gray, mask, openings, [d["box"] for d in kept], memo)
        new = nms(_filter_dets(new, mask, mw, mh))
        new = [d for d in new if all(iou(d["box"], k["box"]) < NMS_IOU for k in kept)]
        if new:
            n_template = len(new)
            kept = kept + new
            openings = classify(kept, mask, gray, geom)   # row peers changed: re-decide everything

    openings.sort(key=lambda o: (o["box_px"][1] // max(1, mh * 0.08), o["box_px"][0]))
    for i, o in enumerate(openings):
        o["id"] = f"{o['type']}_{i}"

    return {
        "photo": photo.name,
        "image_sha256_16": seg.image_sha256(photo)[:16],
        "image_size": [W, H],
        "mask_bbox_px": [mx0, my0, mx1, my1],
        "model": detector.model_id,
        "prompt": PROMPT,
        "prompt_core": PROMPT_CORE,
        "thresholds": {"box": BOX_THRESHOLD, "accept_score": ACCEPT_SCORE,
                       "group_margin": GROUP_MARGIN_ACCEPT, "ground_tol": GROUND_TOL,
                       "ground_tol_box": GROUND_TOL_BOX},
        "passes": {"fine": bool(fine), "template": bool(template), "template_boxes": n_template},
        "openings": openings,
    }


def memo_path(photo, model_id):
    """Where this photo's DINO outputs are cached. The key covers everything the outputs depend on: the image,
    the model, both prompts, the thresholds, and the mask settings (the mask greys out the background)."""
    tag = hashlib.sha256(json.dumps([model_id, PROMPT_CORE, PROMPT, BOX_THRESHOLD, DISTRACTOR_MIN, GREY,
                                     seg.config_fingerprint()]).encode()).hexdigest()[:8]
    return seg.CACHE_DIR.parent / "openings" / f"{seg.image_sha256(photo)[:16]}_{tag}.pkl"


# --------------------------------------------------------------------- result cache
# Final results are cached like masks: .cache/perception/openings/<sha16>_<config_fingerprint>.json. A cache hit
# needs neither torch nor OpenCV nor a model (a laptop without them can load what another machine computed);
# only computing does. The pickled DINO outputs in the same directory are a separate, larger, machine-local cache.
OPENINGS_CACHE_DIR = seg.CACHE_DIR.parent / "openings"
CACHE_SCHEMA = 2
_LOGIC = ("priors", "edge_features", "edge_factors", "classify", "apply_containment", "template_fill", "nms")


def config_fingerprint(model_id=None, fine=True, template=True):
    """Everything a result depends on: the image is keyed separately, this is the rest. Covers the model, both
    prompts, every tuning constant, the mask settings (the mask defines the search area), the pass flags, and
    the source of the decision functions, so editing a prior or a factor invalidates old results too."""
    settings = [CACHE_SCHEMA, model_id or seg.DINO_MODEL, PROMPT_CORE, PROMPT, bool(fine), bool(template),
                seg.config_fingerprint(),
                BOX_THRESHOLD, DISTRACTOR_MIN, NMS_IOU, TILE_ASPECT, TILE_OVERLAP, FINE_TILE_SCALE,
                FINE_MAX_TILE_AREA_FRAC, EDGE_PX, CROP_MARGIN, GREY, MIN_IN_MASK, MAX_BOX_AREA_FRAC, MIN_BOX_PX,
                GROUND_TOL, GROUND_TOL_BOX, ACCEPT_SCORE, GROUP_MARGIN_ACCEPT, TYPE_MARGIN_FLAG, OTHER_PRIOR,
                CONTAIN_INNER_IN_OUTER, CONTAIN_AREA_RATIO, GARAGE_MIN_SCORE, GARAGE_MIN_LEAD,
                list(TEMPLATE_SCALES), list(TEMPLATE_SQUEEZE), TEMPLATE_MAX_PER_ROW, TEMPLATE_NCC_MIN,
                TEMPLATE_MAX_CANDIDATES, TEMPLATE_CONTEXT, TEMPLATE_BAND_PAD,
                [inspect.getsource(globals()[f]) for f in _LOGIC]]
    return hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:8]


def cache_path(photo, model_id=None, fine=True, template=True):
    return OPENINGS_CACHE_DIR / f"{seg.image_sha256(photo)[:16]}_{config_fingerprint(model_id, fine, template)}.json"


def load_cached(photo, *, model_id=None, fine=True, template=True):
    """The cached result for these exact bytes and this exact configuration, else None. No torch, no OpenCV."""
    path = cache_path(photo, model_id, fine, template)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def save_cached(photo, res, *, model_id=None, fine=True, template=True):
    path = cache_path(photo, model_id, fine, template)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return path


def import_openings(src):
    """Copy another machine's <sha16>_<config>.json results into the cache (mirrors check_photos.import_masks).
    A configuration mismatch shows up as "no cached openings" and means the two checkouts differ."""
    OPENINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in Path(src).iterdir():
        if f.suffix.lower() == ".json":
            shutil.copy2(f, OPENINGS_CACHE_DIR / f.name)
            n += 1
    print(f"imported {n} files into {OPENINGS_CACHE_DIR}")
    return n


def export_openings(dst):
    """Copy the cached results out, to hand to a machine without torch. The pickled DINO outputs stay behind."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(OPENINGS_CACHE_DIR.glob("*.json")) if OPENINGS_CACHE_DIR.is_dir() else []:
        shutil.copy2(f, dst / f.name)
        n += 1
    print(f"exported {n} files from {OPENINGS_CACHE_DIR} to {dst}")
    return n


def openings_for(photo, detector=None, *, model_id=None, fine=True, template=True, use_cache=True, memo=None):
    """-> (result, from_cache). Cached result if there is one, else compute (needs torch) and cache it."""
    if use_cache:
        res = load_cached(photo, model_id=model_id, fine=fine, template=template)
        if res is not None:
            return res, True
    if detector is None:
        try:
            detector = Detector(model_id or seg.DINO_MODEL)
        except ImportError as exc:
            raise RuntimeError(
                f"no cached openings for {Path(photo).name} (key {cache_path(photo, model_id, fine, template).name}) "
                f"and the detector cannot run here ({exc}). Import them from a machine that has them: "
                f"python scripts/check_photos.py --import-openings DIR") from exc
    res = detect_openings(photo, detector, fine=fine, template=template, memo=memo)
    if use_cache:
        save_cached(photo, res, model_id=model_id, fine=fine, template=template)
    return res, False


def counts(res):
    by = {}
    for o in res["openings"]:
        key = f"{o['type']}/{o['decision']}"
        by[key] = by.get(key, 0) + 1
    return {"total": len(res["openings"]), "by_type_decision": dict(sorted(by.items()))}


def draw_overlay(photo, result, out_png, max_side=2000):
    mask_shape = (result["image_size"][1], result["image_size"][0])
    img = load_aligned(photo, np.zeros(mask_shape, dtype=bool))
    s = min(1.0, max_side / max(img.size))
    img = img.resize((round(img.width * s), round(img.height * s)))
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(14, int(18 * s * 2)))
    except TypeError:
        font = ImageFont.load_default()
    mx0, my0, mx1, my1 = (v * s for v in result["mask_bbox_px"])
    dr.rectangle([mx0, my0, mx1, my1], outline=(255, 255, 255), width=1)
    for o in result["openings"]:
        x0, y0, x1, y1 = (v * s for v in o["box_px"])
        accept = o["decision"] == "ACCEPT"
        if o["type"] == "other":
            col = COLORS["other"]
        else:
            col = COLORS[o["type"]] if accept else (230, 40, 40)
        dr.rectangle([x0, y0, x1, y1], outline=col, width=3 if accept else 2)
        label = f"{o['id']} {o['score']:.2f}"
        if o.get("other_label"):
            label += f" ({o['other_label']})"
        label += ("" if accept else " ?") + (" T" if o.get("source") == "template" else "")
        dr.text((x0 + 2, max(0, y0 - 20)), label, fill=col, font=font)
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser(description="Detect and classify facade openings.")
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--model", default=seg.DINO_MODEL,
                    help="Grounding DINO checkpoint (try IDEA-Research/grounding-dino-base)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-fine", action="store_true", help="skip the half-size tiling pass")
    ap.add_argument("--no-template", action="store_true", help="skip template gap-fill")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the cached result")
    ap.add_argument("--no-dino-cache", action="store_true", help="ignore and do not write the on-disk DINO cache")
    ap.add_argument("--ablate", action="store_true",
                    help="also report counts with and without the fine pass and the template gap-fill")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det = None  # built on first need: a fully cached run never imports torch
    want = (not args.no_fine, not args.no_template)
    use_cache = not args.no_cache
    for p in args.photos:
        p = Path(p)
        from_cache = False
        cache, memo, n_memo = None, {}, 0
        if args.ablate:  # the variants are computed, whatever is cached; only the chosen one is cached
            det = det or Detector(args.model)
            cache = None if args.no_dino_cache else memo_path(p, args.model)
            memo = pickle.loads(cache.read_bytes()) if cache is not None and cache.is_file() else {}
            n_memo = len(memo)
            variants = {"base": (False, False), "+fine": (True, False),
                        "+template": (False, True), "+fine+template": (True, True)}
            results = {n: detect_openings(p, det, fine=f, template=t, memo=memo) for n, (f, t) in variants.items()}
            res = next(r for n, r in results.items() if variants[n] == want)
            if use_cache:
                save_cached(p, res, model_id=args.model, fine=want[0], template=want[1])
            res = {**res, "ablation": {n: counts(r) for n, r in results.items()}}
        else:
            hit = use_cache and load_cached(p, model_id=args.model, fine=want[0], template=want[1]) is not None
            if not hit:  # only computing needs the detector and the DINO memo
                det = det or Detector(args.model)
                if not args.no_dino_cache:
                    cache = memo_path(p, args.model)
                    memo = pickle.loads(cache.read_bytes()) if cache.is_file() else {}
                    n_memo = len(memo)
            res, from_cache = openings_for(p, det, model_id=args.model, fine=want[0], template=want[1],
                                           use_cache=use_cache, memo=memo)
        if cache is not None and len(memo) > n_memo:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(memo))
        (out_dir / f"{p.stem}.json").write_text(json.dumps(res, indent=2))
        draw_overlay(p, res, out_dir / f"{p.stem}.png")
        c = counts(res)
        print(f"{p.name}: {c['total']} openings  " +
              "  ".join(f"{k}={v}" for k, v in c["by_type_decision"].items()) +
              f"  (template boxes: {res['passes']['template_boxes']}{', cached' if from_cache else ''})"
              f"  -> {out_dir / (p.stem + '.png')}")
        for n, cc in res.get("ablation", {}).items():
            print(f"    {n:<15} {cc['total']:>3}  " + "  ".join(f"{k}={v}" for k, v in cc["by_type_decision"].items()))


if __name__ == "__main__":
    main()
