"""
Scorched Nebraska facade edits for the photos whose camera fits are reliable: burruss_1, patton_1, ncb_5.

For each photo:
  1. generate/mask.py cuts the building out (the same masked, square, white-background input generation always
     sees), from the cached real segmentation mask.
  2. generate/edit.py (flux-kontext-pro, paced by generate/throttle.py) restyles the facade. The model's
     aspect_ratio defaults to match_input_image, so the output has the input's aspect ratio.
  3. generate/mask.enforce_background is the sky/ground re-mask check: if FLUX invented a background around the
     building, the building is cut back out using the masked input.
  4. The edit is put back in the ORIGINAL photo's frame: resized to the crop's side and pasted at the crop's
     position on a canvas of the photo's exact (EXIF-upright) pixel size, white elsewhere. It is NOT clipped to the
     original mask: the alignment check below must see the silhouette FLUX actually drew.
  5. Alignment check: the framed edit goes through the same building segmentation (a new image, so no cache hit)
     and its mask is compared with the original photo's mask. IoU below 0.85 -> one retry with a stronger
     "do not change the shape" prompt and the next seed. At most 2 tries per photo and 6 Replicate calls in all.
  6. Afterwards, openings (perception/openings.py) are detected on the original and the final edit and matched by
     position, to say whether the windows and doors stayed where they were.

    python scripts/make_scorched_edits.py                       # dry run: prepare inputs, check prompts, no API calls
    python scripts/make_scorched_edits.py --run                 # spends Replicate credit (at most 6 calls)
    python scripts/make_scorched_edits.py --run --only ncb_5

Writes outputs/scorched/<stem>_edit.png (the best try, photo-sized), <stem>_compare.png (original | edit),
<stem>_try<k>.png (every try, so a rejected one can be looked at), report.json, and raw/ (what FLUX returned).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # REPLICATE_API_TOKEN lives in .env (gitignored)

from generate import mask as gmask  # noqa: E402
from generate.edit import edit_image, validate_prompt  # noqa: E402
from perception import segment as seg  # noqa: E402

PHOTOS = ROOT / "photos"
OUT_DIR = ROOT / "outputs" / "scorched"
STEMS = ("burruss_1", "patton_1", "ncb_5")
SEEDS = (42, 43)              # try 1, try 2
IOU_MIN = 0.85                # below this the edit has moved the building: retry once
SHIFT_MEDIAN_MAX = 0.02       # openings kept in place: median centre shift, as a fraction of building width ...
SHIFT_P90_MAX = 0.05          # ... and its 90th percentile
MATCH_REACH = 1.0             # an edit box can only match an original box within this many box-lengths of its centre
MATCH_AREA = (1 / 3, 3.0)     # ... and only if its area is within this ratio of the original's
OPENING_TYPES = {"window", "door", "entrance", "garage_door"}
MAX_TRIES = 2
MAX_CALLS = 6                 # hard ceiling on Replicate predictions for the whole run

# edit.py appends its own silhouette-preserving suffix and rejects silhouette-mutating words (extend, expand,
# taller, rebuild, ...), so none of those may appear here. The first try still asks for small roof masts and dishes
# ("without changing the roofline"); the retry drops them entirely and keeps wall-surface features only, because the
# prism's roof is flat and they cannot appear on the model anyway.
PROMPT = (
    "Same building, same camera angle, same footprint, proportions and silhouette. Keep every window and door "
    "exactly where it is, with the same size and shape. Restyle the facade for a post-apocalyptic industrial "
    "world: riveted steel plating over the walls, rust and brass, reactor coolant pipes running along the walls, "
    "small lattice radio masts and dish antennas mounted on the roof without changing the roofline, scorch marks "
    "and soot, ash haze, sepia palette."
)
PROMPT_STRONG = (
    "Same building, same camera angle, same footprint, proportions and silhouette. Do not change the shape of the "
    "building. The outline of the walls, the roofline, the towers and the edge of every window and door must stay "
    "exactly where they are in the input image, with the same size and shape. Repaint the existing wall surfaces "
    "only, and leave the roof exactly as it is: nothing on it, nothing above it. Restyle the facade for a "
    "post-apocalyptic industrial world: riveted steel plating over the walls, rust and brass, reactor coolant "
    "pipes running flat along the walls, scorch marks and soot, ash haze, sepia palette."
)
PROMPT_BOLD = (
    "Same building, same camera angle, same footprint, proportions and silhouette. Do not change the shape of the "
    "building. The outline of the walls, the roofline and the edge of every window and door must stay exactly where "
    "they are in the input image, with the same size and shape. Repaint the existing wall surfaces only, and leave "
    "the roof exactly as it is: nothing on it, nothing above it. Restyle the facade for a post-apocalyptic "
    "industrial world, boldly: heavy riveted steel plates bolted over large parts of the stone, bright orange rust "
    "streaks, thick pipes along the walls, black scorch marks around every window, soot, ash haze, sepia palette."
)
ROOF_WORDS = ("mast", "antenna", "dish", "radio")  # the retry and bold prompts must not ask for any of these
BOLD_STEM = "ncb_5"
BOLD_SEED = 42                # the same seed as ncb_5's first try, so the prompt is the only thing that changed


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Counts Replicate predictions. A call is counted when it is attempted, so a failure never gets a free retry."""

    def __init__(self, cap):
        self.cap, self.used = min(cap, MAX_CALLS), 0

    def spend(self):
        if self.used >= self.cap:
            raise BudgetExceeded(f"{self.used} Replicate calls used; the cap is {self.cap}")
        self.used += 1


# ----------------------------------------------------------------------------- geometry


def mask_iou(a, b):
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"masks differ in size: {a.shape[::-1]} vs {b.shape[::-1]}")
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


def upright(photo):
    """The photo as a person sees it (EXIF applied): the frame every mask in this repo lives in."""
    with Image.open(photo) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def frame_edit(edited_png, masked_png, crop, size, out_png, work_dir):
    """FLUX's square edit -> the original photo's frame. -> (out_png, white share of the edit's border,
    FLUX's output size). White outside the crop; the re-mask check runs first."""
    checked, border_white = gmask.enforce_background(edited_png, masked_png, Path(work_dir) / "bg_enforced.png")
    left, top, right, bottom = crop
    side = right - left
    with Image.open(checked) as im:
        out_size = im.size
        edit = im.convert("RGB").resize((side, side), Image.LANCZOS)
    canvas = Image.new("RGB", size, gmask.BACKGROUND)
    canvas.paste(edit, (left, top))  # the crop may start outside the frame; PIL clips
    canvas.save(out_png)
    return Path(out_png), border_white, out_size


def compare_png(photo_rgb, edit_png, out_png, width=1400):
    h = round(width * photo_rgb.height / photo_rgb.width)
    canvas = Image.new("RGB", (2 * width + 24, h), (255, 255, 255))
    canvas.paste(photo_rgb.resize((width, h), Image.LANCZOS), (0, 0))
    with Image.open(edit_png) as im:
        canvas.paste(im.convert("RGB").resize((width, h), Image.LANCZOS), (width + 24, 0))
    canvas.save(out_png)
    return Path(out_png)


# ----------------------------------------------------------------------------- one photo


def compare_all_png(photo_rgb, edit_pngs, out_png, width=1000):
    """The original and any number of edits, side by side, in the original's aspect ratio."""
    h = round(width * photo_rgb.height / photo_rgb.width)
    panels = [photo_rgb] + [Image.open(p).convert("RGB") for p in edit_pngs]
    canvas = Image.new("RGB", (len(panels) * width + (len(panels) - 1) * 24, h), (255, 255, 255))
    for i, im in enumerate(panels):
        canvas.paste(im.resize((width, h), Image.LANCZOS), (i * (width + 24), 0))
    canvas.save(out_png)
    return Path(out_png)


def compare_all(stem=BOLD_STEM):
    """<stem>_compare_all.png: original | subtle (the standard edit) | bold. No Replicate call."""
    edits = [OUT_DIR / f"{stem}_edit.png", OUT_DIR / f"{stem}_edit_bold.png"]
    missing = [p.name for p in edits if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing}: run the edits first")
    return compare_all_png(upright(PHOTOS / f"{stem}.jpg"), edits, OUT_DIR / f"{stem}_compare_all.png")


def prepare(stem):
    photo = PHOTOS / f"{stem}.jpg"
    rgb = upright(photo)
    original, _meta, _png = seg.segment_file(photo)  # a cache hit for the demo photos
    original = np.asarray(original).astype(bool)
    work = OUT_DIR / "raw"
    work.mkdir(parents=True, exist_ok=True)
    masked = gmask.mask_photo(photo, original, work / f"{stem}_masked.png")
    crop = gmask.square_crop_box(gmask.prepare_mask(original, (rgb.height, rgb.width)))
    return {"photo": photo, "rgb": rgb, "mask": original, "masked": masked, "crop": crop, "work": work}


def try_edit(stem, k, prompt, seed, ctx, budget, name=None):
    """One Replicate call and its alignment check. -> a record of the try."""
    name = name or f"try{k}"
    raw = ctx["work"] / f"{stem}_{name}_raw.png"
    budget.spend()
    edit_image(ctx["masked"], prompt, raw, seed=seed, force=True)
    framed, border_white, out_size = frame_edit(raw, ctx["masked"], ctx["crop"], ctx["rgb"].size,
                                                OUT_DIR / f"{stem}_{name}.png", ctx["work"])
    rec = {"try": k, "seed": seed, "prompt": {PROMPT: "base", PROMPT_STRONG: "strong", PROMPT_BOLD: "bold"}[prompt],
           "file": framed.name,
           "flux_output_size": list(out_size), "border_white": round(border_white, 3),
           "background_re_masked": border_white < gmask.BORDER_WHITE_OK, "iou": None, "note": ""}
    try:
        edit_mask, _meta, _png = seg.segment_file(framed)  # a new image: no cache hit
        rec["iou"] = round(mask_iou(edit_mask, ctx["mask"]), 4)
    except seg.SegmentationError as exc:
        rec["note"] = f"segmentation failed: {exc}"
    return rec


def baseline_iou(stem, ctx):
    """The IoU an UNCHANGED image scores through this exact framing and segmentation: the noise floor. The masked
    input itself stands in for a perfect edit, so anything below this is the edit's doing, not the segmenter's.
    Costs no Replicate call, and the segmentation is a cache hit after the first time (the file bytes repeat)."""
    out = ctx["work"] / f"{stem}_identity.png"
    framed, _bw, _size = frame_edit(ctx["masked"], ctx["masked"], ctx["crop"], ctx["rgb"].size, out, ctx["work"])
    edit_mask, _meta, _png = seg.segment_file(framed)
    return round(mask_iou(edit_mask, ctx["mask"]), 4)


def relative(iou, baseline):
    """-> (share of the noise floor kept, points lost against it). None-safe."""
    if iou is None or not baseline:
        return None, None
    return round(iou / baseline, 3), round(baseline - iou, 4)


def run_photo(stem, budget):
    ctx = prepare(stem)
    tries = []
    for k in range(1, MAX_TRIES + 1):
        rec = try_edit(stem, k, PROMPT if k == 1 else PROMPT_STRONG, SEEDS[k - 1], ctx, budget)
        tries.append(rec)
        print(f"  {stem} try {k} (seed {rec['seed']}, {rec['prompt']} prompt): IoU {rec['iou']}  {rec['note']}")
        if rec["iou"] is not None and rec["iou"] >= IOU_MIN:
            break
    base = baseline_iou(stem, ctx)
    for t in tries:
        t["iou_of_baseline"], t["iou_points_lost"] = relative(t["iou"], base)
    best = max(tries, key=lambda t: -1 if t["iou"] is None else t["iou"])
    final = OUT_DIR / f"{stem}_edit.png"
    shutil.copyfile(OUT_DIR / best["file"], final)
    compare_png(ctx["rgb"], final, OUT_DIR / f"{stem}_compare.png")
    return {"stem": stem, "size": list(ctx["rgb"].size), "crop": list(ctx["crop"]), "tries": tries,
            "final_try": best["try"], "final_iou": best["iou"], "baseline_iou": base,
            "final_iou_of_baseline": best["iou_of_baseline"],
            "aligned": best["iou"] is not None and best["iou"] >= IOU_MIN}


# ----------------------------------------------------------------------------- did the openings stay put?


def openings_kept(photo, edit_png, detector, min_iou=0.5):
    """Detect openings on the original and on the edit and match them by box overlap. Only openings the original
    ACCEPTed are counted; "kept" means an edit opening overlaps it by more than `min_iou`, whatever its type."""
    from perception import openings as O

    before, _ = O.openings_for(photo, detector)
    after, _ = O.openings_for(edit_png, detector)
    summary = {}
    for kind, types in (("windows", {"window"}), ("doors", {"door", "entrance", "garage_door"})):
        mine = [o for o in before["openings"] if o["type"] in types and o["decision"] == "ACCEPT"]
        hits = []
        for o in mine:
            best = max(after["openings"], key=lambda n: O.iou(o["box_px"], n["box_px"]), default=None)
            if best is not None and O.iou(o["box_px"], best["box_px"]) > min_iou:
                hits.append((O.iou(o["box_px"], best["box_px"]), best["type"] in types))
        summary[kind] = {"original_accepted": len(mine), "position_kept": len(hits),
                         "same_kind_kept": sum(1 for _, same in hits if same),
                         "median_box_iou": round(float(np.median([h for h, _ in hits])), 3) if hits else None}
    summary["edit_total_openings"] = len(after["openings"])
    return summary


def shift_stats(before, after):
    """How far did the openings move? Match the original photo's ACCEPTed openings one-to-one (Hungarian, on centre
    distance) to the edit's boxes of ANY type (this is about position, not what the detector called the box), gated
    so a box can only match within MATCH_REACH of its own length and within a MATCH_AREA size ratio. The shift is
    the centre distance as a fraction of the original building's width (its mask bbox). Originals with no
    admissible partner are counted, not dropped: an opening that moved further than that shows up there."""
    from scipy.optimize import linear_sum_assignment

    width = before["mask_bbox_px"][2] - before["mask_bbox_px"][0]
    orig = [o for o in before["openings"] if o["type"] in OPENING_TYPES and o["decision"] == "ACCEPT"]
    cand = after["openings"]

    def centre(b):
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    def area(b):
        return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))

    BIG = 1e12
    cost = np.full((len(orig), len(cand)), BIG)
    for i, o in enumerate(orig):
        reach = MATCH_REACH * max(o["box_px"][2] - o["box_px"][0], o["box_px"][3] - o["box_px"][1])
        for j, n in enumerate(cand):
            d = float(np.linalg.norm(centre(o["box_px"]) - centre(n["box_px"])))
            if d <= reach and MATCH_AREA[0] <= area(n["box_px"]) / area(o["box_px"]) <= MATCH_AREA[1]:
                cost[i, j] = d
    pairs = []
    if len(orig) and len(cand):
        rows, cols = linear_sum_assignment(cost)
        pairs = [(i, j, cost[i, j]) for i, j in zip(rows, cols) if cost[i, j] < BIG]
    shifts = np.array([d / width for _, _, d in pairs])
    by_kind = {}
    for kind, types in (("windows", {"window"}), ("doors", {"door", "entrance", "garage_door"})):
        idx = [k for k, (i, _, _) in enumerate(pairs) if orig[i]["type"] in types]
        n_kind = sum(1 for o in orig if o["type"] in types)
        by_kind[kind] = {"originals": n_kind, "matched": len(idx),
                         "median": round(float(np.median(shifts[idx])), 4) if idx else None}
    out = {"building_width_px": width, "originals": len(orig), "matched": len(pairs), "unmatched": len(orig) - len(pairs),
           "same_kind": sum(1 for i, j, _ in pairs if cand[j]["type"] in (OPENING_TYPES if orig[i]["type"] in OPENING_TYPES else set())),
           "median_shift": round(float(np.median(shifts)), 4) if len(shifts) else None,
           "p90_shift": round(float(np.percentile(shifts, 90)), 4) if len(shifts) else None,
           "by_kind": by_kind}
    out["passes"] = bool(len(shifts) and out["median_shift"] < SHIFT_MEDIAN_MAX and out["p90_shift"] < SHIFT_P90_MAX)
    return out


def run_bold(budget):
    """ONE Replicate call: a bolder wall-only variant of the NCB edit, saved as ncb_5_edit_bold.png and checked the
    same way as the others (alignment IoU, IoU relative to the noise floor, opening shift). Refuses to run if the
    output already exists, so a repeat cannot quietly spend a second call."""
    final = OUT_DIR / f"{BOLD_STEM}_edit_bold.png"
    if final.exists():
        raise FileExistsError(f"{final} already exists; delete it deliberately to spend another call")
    from perception import openings as O

    ctx = prepare(BOLD_STEM)
    rec = try_edit(BOLD_STEM, 1, PROMPT_BOLD, BOLD_SEED, ctx, budget, name="edit_bold")
    base = baseline_iou(BOLD_STEM, ctx)
    rec["iou_of_baseline"], rec["iou_points_lost"] = relative(rec["iou"], base)
    compare_png(ctx["rgb"], final, OUT_DIR / f"{BOLD_STEM}_bold_compare.png")
    print(f"  {BOLD_STEM} bold (seed {BOLD_SEED}): IoU {rec['iou']} ({rec['iou_of_baseline']} of the {base} noise floor)")
    print("  detecting openings on the bold edit (no Replicate call)...")
    detector = O.Detector(seg.DINO_MODEL)
    before, _ = O.openings_for(PHOTOS / f"{BOLD_STEM}.jpg", detector)
    after, _ = O.openings_for(final, detector)
    rec["shift"] = shift_stats(before, after)
    rec["baseline_iou"] = base
    path = OUT_DIR / "report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    report["bold"] = rec
    report["replicate_calls_bold_extra"] = budget.used
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return rec


def shift_report():
    """No Replicate calls and no detection: read the cached openings of each photo and its final edit."""
    from perception import openings as O

    path = OUT_DIR / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"{'photo':<11}{'originals':>10}{'matched':>8}{'unmatched':>10}{'median':>8}{'p90':>8}   verdict "
          f"(need median < {SHIFT_MEDIAN_MAX:.0%}, p90 < {SHIFT_P90_MAX:.0%})")
    for r in report["results"]:
        photo, edit = PHOTOS / f"{r['stem']}.jpg", OUT_DIR / f"{r['stem']}_edit.png"
        before, after = O.load_cached(photo), O.load_cached(edit)
        if before is None or after is None:
            print(f"{r['stem']:<11} openings not cached for {'the edit' if after is None else 'the original'}; "
                  "run the full script (detection needs torch) first")
            continue
        r["shift"] = st = shift_stats(before, after)
        print(f"{r['stem']:<11}{st['originals']:>10}{st['matched']:>8}{st['unmatched']:>10}"
              f"{(st['median_shift'] or 0):>8.2%}{(st['p90_shift'] or 0):>8.2%}   "
              f"{'USE THE EDIT' if st['passes'] else 'FAILS'}")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def rescore():
    path = OUT_DIR / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"{'photo':<11}{'try':>4} {'IoU':>7} {'baseline':>9} {'of baseline':>12} {'points lost':>12}")
    for r in report["results"]:
        base = baseline_iou(r["stem"], prepare(r["stem"]))
        r["baseline_iou"] = base
        for t in r["tries"]:
            t["iou_of_baseline"], t["iou_points_lost"] = relative(t["iou"], base)
            print(f"{r['stem']:<11}{t['try']:>4} {t['iou']:>7} {base:>9} {t['iou_of_baseline']:>12} {t['iou_points_lost']:>12}")
        r["final_iou_of_baseline"] = relative(r["final_iou"], base)[0]
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


# ----------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="store_true", help="actually call Replicate (default: dry run, nothing is spent)")
    ap.add_argument("--only", choices=STEMS, action="append", help="restrict to these photos")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS, help=f"cap on Replicate calls (never above {MAX_CALLS})")
    ap.add_argument("--skip-openings", action="store_true", help="skip the windows-and-doors position check")
    ap.add_argument("--bold", action="store_true",
                    help=f"ONE extra call: a bolder wall-only variant for {BOLD_STEM}, saved as {BOLD_STEM}_edit_bold.png")
    ap.add_argument("--compare-all", action="store_true",
                    help=f"no Replicate calls: {BOLD_STEM}_compare_all.png (original | subtle | bold)")
    ap.add_argument("--shift", action="store_true",
                    help="no Replicate calls: median / p90 centre shift of the openings between photo and edit")
    ap.add_argument("--rescore", action="store_true",
                    help="no Replicate calls: add the identity-edit baseline (and IoU relative to it) to report.json")
    args = ap.parse_args(argv)

    stems = args.only or list(STEMS)
    for name, prompt in (("base", PROMPT), ("strong", PROMPT_STRONG), ("bold", PROMPT_BOLD)):
        ok, reason = validate_prompt(prompt)
        if not ok:
            print(f"error: the {name} prompt is rejected by generate/edit.py: {reason}", file=sys.stderr)
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.shift:  # the two no-API modes come first: neither may fall through to a run
        return shift_report()
    if args.rescore:
        return rescore()
    if args.compare_all:
        print(compare_all())
        return 0
    if args.bold:
        if not args.run:
            print(f"DRY RUN (add --run to spend ONE Replicate call). Bold prompt for {BOLD_STEM}:\n  {PROMPT_BOLD}")
            return 0
        rec = run_bold(Budget(1))
        sh = rec["shift"]
        print(f"bold: IoU {rec['iou']}; shift median {sh['median_shift']:.2%}, p90 {sh['p90_shift']:.2%}, "
              f"{sh['unmatched']} of {sh['originals']} originals unmatched; "
              f"{'USE THE EDIT' if sh['passes'] else 'FAILS'} the shift criterion")
        return 0
    if not args.run:
        print("DRY RUN: nothing is sent to Replicate. Add --run to spend credit "
              f"(at most {min(args.max_calls, MAX_CALLS)} calls: {MAX_TRIES} tries x {len(stems)} photos).\n")
        for stem in stems:
            ctx = prepare(stem)
            print(f"{stem}: photo {ctx['rgb'].size[0]}x{ctx['rgb'].size[1]}, mask {ctx['mask'].mean():.1%} of the frame, "
                  f"crop {ctx['crop']}, masked input {Path(ctx['masked']).name} ({Image.open(ctx['masked']).size[0]} px square)")
        print(f"\nbase prompt:\n  {PROMPT}\nretry prompt adds:\n  {PROMPT_STRONG[:len(PROMPT_STRONG) - len(PROMPT)]}")
        return 0

    budget = Budget(args.max_calls)
    results = []
    try:
        for stem in stems:
            print(f"{stem}:")
            results.append(run_photo(stem, budget))
    except BudgetExceeded as exc:
        print(f"stopped: {exc}", file=sys.stderr)
    finally:
        (OUT_DIR / "report.json").write_text(json.dumps({"replicate_calls": budget.used, "results": results}, indent=2),
                                             encoding="utf-8")

    if results and not args.skip_openings:
        from perception import openings as O

        print("\nchecking that windows and doors stayed in place (detection only, no Replicate calls)...")
        detector = O.Detector(seg.DINO_MODEL)
        for r in results:
            r["openings"] = openings_kept(PHOTOS / f"{r['stem']}.jpg", OUT_DIR / f"{r['stem']}_edit.png", detector)
        (OUT_DIR / "report.json").write_text(json.dumps({"replicate_calls": budget.used, "results": results}, indent=2),
                                             encoding="utf-8")

    print(f"\nReplicate calls used: {budget.used} (cap {budget.cap})")
    for r in results:
        line = (f"{r['stem']}: IoU {r['final_iou']} ({r['final_iou_of_baseline']:.0%} of the {r['baseline_iou']} noise floor) "
                f"after {len(r['tries'])} try(ies) -> {'aligned' if r['aligned'] else 'NOT aligned'}")
        if "openings" in r:
            w, d = r["openings"]["windows"], r["openings"]["doors"]
            line += (f"; windows kept {w['position_kept']}/{w['original_accepted']}, "
                     f"doors kept {d['position_kept']}/{d['original_accepted']}")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
