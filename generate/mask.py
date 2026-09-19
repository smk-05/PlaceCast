"""
Cut the building out of the photo before generation. Plan phase 3.1.

Why: TRELLIS lifts EVERYTHING in the image. NCB's bushes and lamp posts came
back as a 12x15 m lobe of geometry beside the building (Hausdorff 14.25 m,
decision REJECT), and no amount of solver tuning can fix a mesh that contains
scenery. The segmentation mask (Owen's Grounding DINO + SAM 2, perception/)
already says which pixels are building, so generation only ever sees those.

    photo --exif_transpose--> RGB --mask (dilated, holes filled)--> white bg
          --crop to mask bbox + margin, pad to square--> masked_0.png

Conventions (perception/segment.py): the mask is in the EXIF-TRANSPOSED frame,
255 = building. The photo must be transposed the same way or the mask lands
on the wrong pixels of every portrait phone shot.

Only a REAL mask is used. The stub's central 80% box carries no shape, so
"masking" with it would only crop — pipeline.py checks segmentation_model.
The crop changes nothing downstream: EXIF, silhouette and the photographed
side all read the ORIGINAL photo and mask; this file only feeds FLUX/TRELLIS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

BACKGROUND = (255, 255, 255)
DILATE_FRAC = 0.003    # of the image diagonal: keeps cornices SAM shaved off;
                       # 0.006 left a visible sky halo on 24 MP phone photos
CROP_MARGIN = 0.12     # of the mask bbox's larger side, on every edge
MIN_MASK_FRAC = 0.02   # below this the mask is not a building; refuse
MAX_SIDE_PX = 2048     # FLUX returns ~1 MP and TRELLIS runs at 518 px; a
                       # 4032 px phone crop only costs upload time


class MaskError(ValueError):
    pass


def prepare_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bool HxW mask -> holes filled, slightly dilated. Raises on a useless mask."""
    m = np.asarray(mask).astype(bool)
    if m.shape != shape:
        raise MaskError(f"mask is {m.shape[::-1]} but the photo is {shape[::-1]} — "
                        "was the photo EXIF-transposed the same way as the mask?")
    if m.mean() < MIN_MASK_FRAC:
        raise MaskError(f"mask covers {m.mean():.1%} of the frame; not a building")
    m = ndimage.binary_fill_holes(m)
    r = max(1, int(round(DILATE_FRAC * float(np.hypot(*shape)))))
    # cv2, not scipy: a 43 px disk on a 24 MP phone mask took minutes in
    # ndimage.binary_dilation and is ~0.1 s here.
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), kernel) > 0


def square_crop_box(m: np.ndarray, margin: float = CROP_MARGIN) -> tuple[int, int, int, int]:
    """Bbox of the mask plus margin, grown to a square. May extend past the
    image; the caller pads with background."""
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    side = int(round(side * (1 + 2 * margin)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    return left, top, left + side, top + side


def mask_photo(photo_path: Path, mask: np.ndarray, out_path: Path) -> Path:
    """Write the building-only image generation should see. -> out_path."""
    with Image.open(photo_path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    m = prepare_mask(mask, rgb.shape[:2])

    out = np.empty_like(rgb)
    out[:] = BACKGROUND
    out[m] = rgb[m]

    left, top, right, bottom = square_crop_box(m)
    canvas = Image.new("RGB", (right - left, bottom - top), BACKGROUND)
    canvas.paste(Image.fromarray(out), (-left, -top))
    if canvas.width > MAX_SIDE_PX:
        canvas = canvas.resize((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


# --------------------------------------------------------------------------
# After the edit: FLUX may invent a background around a masked building
# --------------------------------------------------------------------------

BORDER_FRAC = 0.04        # outer band of the image checked for background
WHITE_MIN = 235           # a channel value this high counts as "white"
BORDER_WHITE_OK = 0.85    # below this share of white border, FLUX added a scene
CLOSE_FRAC = 0.03         # closing radius (of the side) that bridges mask notches
REGROW_FRAC = 0.012       # slack for FLUX shifting edges by a few pixels


def border_white_fraction(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    b = max(1, int(round(BORDER_FRAC * min(h, w))))
    band = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                           img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    return float((band.min(axis=1) >= WHITE_MIN).mean())


def enforce_background(edited_path: Path, masked_input_path: Path,
                       out_path: Path) -> tuple[Path, float]:
    """If the edit invented a background, cut the building back out.

    -> (image to lift, white share of the edited image's border). Returns
    `edited_path` untouched when the border is already white. Otherwise the
    building region is taken from the MASKED INPUT (its non-white pixels —
    Kontext keeps the layout), closed so notches left by occluders are
    bridged, grown slightly, and everything outside it is painted white.
    """
    import cv2

    with Image.open(edited_path) as im:
        edited = np.asarray(im.convert("RGB")).copy()
    frac = border_white_fraction(edited)
    if frac >= BORDER_WHITE_OK:
        return Path(edited_path), frac

    with Image.open(masked_input_path) as im:
        src = im.convert("RGB").resize((edited.shape[1], edited.shape[0]),
                                       Image.BILINEAR)
    building = np.asarray(src).min(axis=2) < WHITE_MIN
    side = min(building.shape)
    rc = max(1, int(round(CLOSE_FRAC * side)))
    rg = max(1, int(round(REGROW_FRAC * side)))
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(building.astype(np.uint8), cv2.MORPH_CLOSE, k(rc))
    m = ndimage.binary_fill_holes(m > 0)
    m = cv2.dilate(m.astype(np.uint8), k(rg)) > 0

    edited[~m] = BACKGROUND
    Image.fromarray(edited).save(out_path)
    return Path(out_path), frac
