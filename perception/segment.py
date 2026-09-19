#!/usr/bin/env python
"""Building mask from a photo (ML_ADDENDUM A.3, stage 1).

    Grounding DINO ("building. house. facade.") -> box -> SAM 2 -> clean_components() -> PNG

Pipeline API (contracts.SegmentBuilding; what pipeline.py imports):
    segment_building(path) -> (mask HxW bool, mask_area_frac, occluded)
    clean(mask, close_px=5) -> mask            largest + big-enough components, close, fill holes
    assess_mask(mask)       -> (area_frac, occluded, note)     A.4 failure detection

segment_building runs the REAL models by default (PROCEDURA_PERCEPTION=real). Set PROCEDURA_PERCEPTION=stub
(or pass backend="stub") for the non-ML central-box stand-in, which needs no GPU and no torch; see
perception/backend.py. Real segmentation raises SegmentationError rather than falling back to the stand-in.

Command line:
    python perception/segment.py PHOTO.jpg [-o OUT.png] [--force]

Output is a single-channel PNG the size of the *EXIF-upright* photo (EXIF
orientation is applied before anything else, so the mask lines up with what a
person sees): 255 = building, 0 = background. Whoever consumes the mask must
use the same convention (PIL.ImageOps.exif_transpose).

clean_components() keeps the largest component plus big-enough components inside the detection box.

Masks are cached in cache/masks/ keyed by the SHA-256 of the image bytes (plus a
short hash of the settings below, so changing the prompt/models/clean_components() params
never returns a stale mask). A sidecar .json records every intermediate choice.
A cache hit does not import torch or load any model.

Failures are loud (exit 1): no detection, or a mask that still touches >3 image
edges / covers >90% of the frame after one tighter-box re-prompt (A.4).
"""
import argparse
import gc
import hashlib
import json
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `python perception/segment.py` puts perception/ first, not the repo root
    sys.path.insert(0, str(ROOT))

from perception.backend import STUB, resolve_backend  # noqa: E402

CACHE_DIR = ROOT / ".cache" / "perception" / "masks"  # .cache/ is gitignored
DEFAULT_OUT_DIR = ROOT / "outputs" / "masks"

PROMPT = "building. house. facade."
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL = "facebook/sam2.1-hiera-base-plus"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

CLOSE_KERNEL_PX = 5  # A.3: "~5 px kernel to seal window gaps"
MAX_EDGES_TOUCHED = 3  # A.4: mask touching >3 image edges is the wrong thing
MAX_FRAME_COVERAGE = 0.90  # A.4: mask covering >90% of the frame is the wrong thing
CENTRE_BOX_FRACTION = 0.80  # A.2/A.4 re-prompt: central 80% box + centre click
MERGE_MIN_FRACTION = 0.10  # clean(): also keep components >= 10% of the largest (if inside the DINO box)
DISCARDED_WARN_FRACTION = 0.15  # diagnostic only (not in the cache key): warn above this

# A.4 detection thresholds for assess_mask(), applied to a mask whatever produced it.
MIN_AREA_FRAC = 0.04  # tiny mask -> occluded or wrong object
STUB_CENTRAL_BOX_FRAC = 0.80  # the stand-in's box (A.2 fallback)


class SegmentationError(Exception):
    """No usable mask. Reported loudly; never defaulted."""


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------- caching


def config_fingerprint():
    settings = [PROMPT, DINO_MODEL, SAM_MODEL, BOX_THRESHOLD, TEXT_THRESHOLD, CLOSE_KERNEL_PX,
                MAX_EDGES_TOUCHED, MAX_FRAME_COVERAGE, CENTRE_BOX_FRACTION, MERGE_MIN_FRACTION]
    return hashlib.sha256(json.dumps(settings).encode()).hexdigest()[:8]


def image_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_paths(sha):
    stem = f"{sha[:16]}_{config_fingerprint()}"
    return CACHE_DIR / f"{stem}.png", CACHE_DIR / f"{stem}.json"


def cache_write(png_path, json_path, mask, meta):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = png_path.with_suffix(".tmp.png")
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(tmp)
    tmp.replace(png_path)
    tmp_json = json_path.with_suffix(".tmp")
    tmp_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp_json.replace(json_path)  # the sidecar lands last: png+json together mean complete


# ------------------------------------------------------------------ clean()


def clean_components(mask, dino_box=None, close_px=CLOSE_KERNEL_PX, merge_min_fraction=MERGE_MIN_FRACTION):
    """A.3 clean(), extended to survive occluders.

    Keeps the largest connected component plus every component that is at least
    `merge_min_fraction` of the largest AND whose bounding box overlaps `dino_box`
    (the Grounding DINO detection; None = the whole image, for an already-cleaned mask), so parts of one building split by a tree or
    pole are kept while neighbouring structures outside the detection are not.
    Then a morphological close over the kept set and interior hole fill.

    The close is `close_px` wide: it seals window gaps and hairline splits, but
    parts separated by more than that stay separate blobs in the mask.

    Returns (bool mask, stats); stats has components_total, components_merged
    (kept in addition to the largest) and discarded_area_frac (share of the input
    mask's area in discarded components). Empty in -> empty out with zero stats;
    the caller's mask_problems() rejects it.
    """
    import cv2
    from scipy import ndimage

    m = np.asarray(mask, dtype=np.uint8)
    n, labels, cc, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    stats = {"components_total": n - 1, "components_merged": 0, "discarded_area_frac": 0.0}
    if n < 2:
        return np.zeros(m.shape, dtype=bool), stats
    areas = cc[:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas[1:]))
    bx0, by0, bx1, by1 = (0, 0, m.shape[1], m.shape[0]) if dino_box is None else dino_box
    keep = [largest]
    for i in range(1, n):
        if i == largest:
            continue
        x, y, w, h = (int(cc[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        overlaps_box = x < bx1 and x + w > bx0 and y < by1 and y + h > by0
        if areas[i] >= merge_min_fraction * areas[largest] and overlaps_box:
            keep.append(i)
    stats["components_merged"] = len(keep) - 1
    stats["discarded_area_frac"] = float(1 - areas[keep].sum() / areas[1:].sum())
    m = np.isin(labels, keep).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)))
    # Holes reachable from the image border are background, not windows, and stay open.
    return ndimage.binary_fill_holes(m), stats


def clean(mask, close_px=CLOSE_KERNEL_PX):
    """The pipeline's clean(mask) -> mask. No detection box exists here, so this keeps every component that is at
    least MERGE_MIN_FRACTION of the largest. It is unchanged by a second application, which matters because
    pipeline.py calls it on segment_building()'s output, which is already clean_components()-ed: a
    largest-component-only clean here would throw the merged occlusion-split parts away again."""
    return clean_components(mask, None, close_px)[0]


def assess_mask(mask):
    """A.4 failure detection on any mask, whatever produced it. -> (area_frac, occluded, note)."""
    mask = np.asarray(mask, dtype=bool)
    frac = float(mask.mean())
    if frac > MAX_FRAME_COVERAGE:
        return frac, True, "mask covers >90% of frame - segmented the streetscape"
    if frac < MIN_AREA_FRAC:
        return frac, True, "mask tiny relative to frame - occluded or wrong object"
    edges = int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())
    if edges > MAX_EDGES_TOUCHED:
        return frac, True, "mask touches >3 image edges"
    # High boundary complexity relative to area indicates tree/vehicle occlusion chewing holes in the silhouette.
    filled = ndimage.binary_fill_holes(mask).astype(int)
    perimeter = float(np.abs(np.diff(filled, axis=0)).sum() + np.abs(np.diff(filled, axis=1)).sum())
    complexity = perimeter / np.sqrt(max(float(filled.sum()), 1.0))
    if complexity > 12.0:
        return frac, True, f"boundary complexity {complexity:.1f} - likely occlusion"
    return frac, False, ""


def mask_problems(mask):
    """A.4 'SAM segments the wrong thing' checks. Returns a list of problems (empty = ok)."""
    if not mask.any():
        return ["mask is empty"]
    problems = []
    edges = int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())
    if edges > MAX_EDGES_TOUCHED:
        problems.append(f"mask touches {edges} image edges (limit {MAX_EDGES_TOUCHED})")
    if mask.mean() > MAX_FRAME_COVERAGE:
        problems.append(f"mask covers {mask.mean():.0%} of the frame (limit {MAX_FRAME_COVERAGE:.0%})")
    return problems


# --------------------------------------------------------------------- models


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _free_gpu():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def detect_boxes(image):
    """Grounding DINO. Returns [{'box': [x0,y0,x1,y1], 'score', 'label'}], loading and freeing the model."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = _device()
    processor = AutoProcessor.from_pretrained(DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(device).eval()
    try:
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]
    finally:
        del model
        _free_gpu()
    labels = result.get("text_labels", result.get("labels"))
    return [
        {"box": [float(v) for v in box], "score": float(score), "label": str(label)}
        for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), labels)
    ]


def select_box(detections, width, height):
    """A.4 'multiple buildings': largest area x centrality (both in [0,1])."""
    cx, cy, half_diag = width / 2, height / 2, (width**2 + height**2) ** 0.5 / 2

    def merit(d):
        x0, y0, x1, y1 = d["box"]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (width * height)
        dist = ((x0 + x1) / 2 - cx) ** 2 + ((y0 + y1) / 2 - cy) ** 2
        return area * max(0.0, 1.0 - dist**0.5 / half_diag)

    return max(detections, key=merit)


@contextmanager
def sam_session(image):
    """Load SAM 2 for one image; yields segment(box, point=None) -> bool mask. Frees the model on exit."""
    import torch
    from transformers import Sam2Model, Sam2Processor

    device = _device()
    processor = Sam2Processor.from_pretrained(SAM_MODEL)
    model = Sam2Model.from_pretrained(SAM_MODEL).to(device).eval()

    def segment(box, point=None):
        prompts = {"input_boxes": [[box]]}
        if point is not None:
            prompts.update(input_points=[[[list(point)]]], input_labels=[[[1]]])
        inputs = processor(images=image, return_tensors="pt", **prompts).to(device)
        with torch.inference_mode():
            outputs = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        return masks[0, 0].numpy().astype(bool)

    try:
        yield segment
    finally:
        del model
        _free_gpu()


# ------------------------------------------------------------------- pipeline


def segment_image(image):
    """Returns (cleaned bool mask, metadata dict). Raises SegmentationError."""
    width, height = image.size
    detections = detect_boxes(image)
    if not detections:
        raise SegmentationError(
            f"Grounding DINO found nothing for {PROMPT!r} (box threshold {BOX_THRESHOLD}, text threshold {TEXT_THRESHOLD})"
        )
    chosen = select_box(detections, width, height)
    log(f"  {len(detections)} detection(s); using {chosen['label']!r} score {chosen['score']:.2f} box {[round(v) for v in chosen['box']]}")

    attempts = []
    with sam_session(image) as segment:
        prompt_used, box, point = "dino_box", chosen["box"], None
        for retry in (False, True):
            if retry:
                m = (1 - CENTRE_BOX_FRACTION) / 2
                box = [width * m, height * m, width * (1 - m), height * (1 - m)]
                point, prompt_used = (width / 2, height / 2), "centre_box_retry"
            mask, clean_stats = clean_components(segment(box, point), chosen["box"])
            problems = mask_problems(mask)
            attempts.append(
                {"prompt": prompt_used, "box": [round(v, 1) for v in box], "problems": problems, **clean_stats}
            )
            if not problems:
                break
            log(f"  attempt '{prompt_used}' rejected: {'; '.join(problems)}")
    if problems:
        raise SegmentationError(
            "mask rejected after re-prompting with a tighter centre box: "
            + "; ".join(f"[{a['prompt']}] " + ", ".join(a["problems"]) for a in attempts)
        )

    x0, y0, x1, y1 = box
    # The merge-vs-discard tradeoff of clean(): a high discarded share means SAM's mask had large
    # pieces that were small, or outside the detection box, so the silhouette may be a fragment (A.4).
    if clean_stats["discarded_area_frac"] > DISCARDED_WARN_FRACTION:
        log(f"  WARNING clean_components() discarded {clean_stats['discarded_area_frac']:.0%} of SAM's mask "
            f"(kept {clean_stats['components_merged'] + 1} of {clean_stats['components_total']} components); inspect the mask")
    meta = {
        "prompt": PROMPT,
        "dino_model": DINO_MODEL,
        "sam_model": SAM_MODEL,
        "detections": detections,
        "chosen_detection": chosen,
        "attempts": attempts,
        "prompt_used": prompt_used,
        "image_size_wh": [width, height],
        "mask_area_px": int(mask.sum()),
        "mask_frame_fraction": float(mask.mean()),
        "mask_to_box_area_ratio": float(mask.sum() / max(1.0, (x1 - x0) * (y1 - y0))),  # A.4 occlusion cue
        "components_total": clean_stats["components_total"],
        "components_merged": clean_stats["components_merged"],
        "discarded_area_frac": clean_stats["discarded_area_frac"],
        "close_kernel_px": CLOSE_KERNEL_PX,
        "merge_min_fraction": MERGE_MIN_FRACTION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return mask, meta


def segment_file(photo, force=False):
    """Real segmentation of one file, cached by image hash. -> (bool mask, sidecar metadata, cached PNG path)."""
    photo = Path(photo)
    if not photo.is_file():
        raise SegmentationError(f"no such file: {photo}")
    sha = image_sha256(photo)
    png_path, json_path = cache_paths(sha)
    if not force and png_path.exists() and json_path.exists():
        log(f"cache hit {png_path.name}")
    else:
        try:
            image = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
        except OSError as exc:
            raise SegmentationError(f"cannot read {photo} as an image: {exc}")
        log(f"segmenting {photo.name} ({image.size[0]}x{image.size[1]}) on {_device()}")
        mask, meta = segment_image(image)
        meta["image_sha256"] = sha
        meta["source_name"] = photo.name
        cache_write(png_path, json_path, mask, meta)
    return np.array(Image.open(png_path)) > 0, json.loads(json_path.read_text(encoding="utf-8")), png_path


def run(photo, out_path, force=False):
    _, meta, png_path = segment_file(photo, force=force)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(png_path, out_path)
    return out_path, meta


def _segment_building_stub(image_path):
    """Non-ML stand-in: the photo is OF a building, so the central 80% box is the mask (A.2's own fallback)."""
    with Image.open(image_path) as im:
        w, h = im.size
    mask = np.zeros((h, w), dtype=bool)
    mx, my = int(w * (1.0 - STUB_CENTRAL_BOX_FRAC) / 2.0), int(h * (1.0 - STUB_CENTRAL_BOX_FRAC) / 2.0)
    mask[my : h - my, mx : w - mx] = True
    return mask, float(mask.mean()), False


def segment_building(image_path, *, backend=None):
    """contracts.SegmentBuilding: -> (binary HxW mask, mask_area_frac, occluded).

    Real (default): Grounding DINO + SAM 2 via segment_file(); raises SegmentationError on failure.
    `occluded` is True when clean_components() discarded more than DISCARDED_WARN_FRACTION of SAM's mask (the
    building was split by an occluder) or assess_mask() flags the result. NOTE pipeline.py currently discards this
    flag (`mask, _, _ = segment_building(...)`) and recomputes occlusion from the mask alone.
    Stub: the central-box mask; needs no GPU.
    """
    if resolve_backend(backend) == STUB:
        return _segment_building_stub(image_path)
    mask, meta, _ = segment_file(image_path)
    frac, occluded_by_mask, _ = assess_mask(mask)
    return mask, frac, bool(occluded_by_mask or meta["discarded_area_frac"] > DISCARDED_WARN_FRACTION)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("photo", help="path to a JPG")
    parser.add_argument("-o", "--out", help=f"output PNG (default {DEFAULT_OUT_DIR}/<photo stem>.png)")
    parser.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    args = parser.parse_args()
    out = args.out or DEFAULT_OUT_DIR / f"{Path(args.photo).stem}.png"
    try:
        out_path, meta = run(args.photo, out, force=args.force)
    except SegmentationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"{out_path}  mask {meta['mask_frame_fraction']:.1%} of frame, "
        f"{meta['mask_to_box_area_ratio']:.0%} of prompt box, via {meta['prompt_used']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
