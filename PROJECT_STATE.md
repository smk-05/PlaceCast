# Project state — VTHacks 14, Procedura AI track
Owen Wikel (perception/ML) · Soumik Saha (geometry/math) · as of Saturday ~6 PM, day 2

## ⚠️ Top priority (sponsor directive, day 2)
Procedura wants **building features identified and oriented**: doors, windows, garage doors, entrances. In the game, doors must be separable from windows and face the right way so players can interact with them.

Each opening is its own labelled element carrying `type`, position (ENU + lat/lon/height), outward normal + compass bearing, width/height in metres, score, margins, source photo, decision (ACCEPT / REVIEW) with reasons.

## The project
Photo + address → AI-redesigned building → 3D model with labelled openings → placed on a live map at the real location with correct position, scale, rotation, ground alignment. Sponsor: Procedura AI ("Scorched Nebraska"). Prize: internship opportunities for up to two teams. **Submission 8 AM Sunday, judging 9 AM Sunday in NCB.**

Governing docs in repo: `SPEC.md`, `ML_ADDENDUM.md` (§A segmentation/render-compare, §B confidence gate, §C monocular depth, §F work split), `BENCHMARK.md`.

## Big change this evening: the demo shows footprint prisms
All six single-view TRELLIS meshes were **rejected by our own gate**. Single-view TRELLIS builds a facade with too little behind it: plans too squat and thin, so the ends overhang and Hausdorff explodes. Not a solver problem (benchmark recovers known pose at median IoU 0.996; outline height sweep caps NCB at 0.824).

| building | IoU | Hausdorff | mesh aspect vs real |
|---|---|---|---|
| NCB (ncb_5) | 0.80 | 9.9 m | 1.6 vs 2.2 |
| Burruss | 0.78 | 30.7 m | 1.2 vs 1.4 |
| Goodwin | 0.59 | 37.5 m | 3.1 vs 1.3 |
| Whittemore | 0.54 | 22.5 m | 1.0 vs 1.9 |
| War Memorial | 0.45 | 27.2 m | thin shell, plan fills 44% of box |
| Patton | 0.33 | 55.3 m | sliver, fills 18% (real 81%) |

3-view NCB: aspect 2.0 vs 2.2 but grew a tail, IoU 0.45.

**Demo = footprint prisms** (real footprint, real OSM height, pinned terrain) placed correctly, generated meshes shown with their honest numbers. **Openings therefore go onto the prism**, not the TRELLIS mesh (see Openings §3).

## Status snapshot
- ✅ All 23 masks delivered and imported (suffix `_88e9b8b2`); masking removed Patton/War Memorial cars and NCB bushes from generation.
- ✅ Gate rule: non-authoritative height → REVIEW (`40b15c0`, with `tests/test_height_gate.py`).
- ✅ `perception/openings.py` detection upgraded, committed `4351e6c` (push + one regression fix in progress).
- ✅ `perception/openings_3d.py` (mesh projection + glTF markers/extras) pushed; 22 tests, synthetic meshes. Now secondary — meshes rejected.
- 🔄 `perception/openings_prism.py` — EXIF camera → rays onto the real footprint prism. **Critical path.**
- ⏳ `"benchmark"` source in `load_rows()`; retrain gate with leave-one-building-out.

## Repo
`github.com/smk-05/project_for_procedura`, work on **`main`** (silhouette PR branch fully merged). Local: `C:\Users\wikel\dev\procedura`, venv `.venv` (Python 3.11.9, torch 2.6.0+cu124). GPU RTX 2000 Ada 8 GB.

- Use `.venv\Scripts\python`, not `uv run` (uv may pull CPU torch). Don't `pip install -r` requirements files (same risk); install single packages.
- Installed into venv today: `replicate`, `python-dotenv`, `opencv-python-headless` (if Claude Code added it), `rtree`, `pygltflib`.
- Two Claude Code sessions share one checkout: never let one `git pull --rebase` while the other has uncommitted edits to tracked files.
- Close overlay PNGs in Photos before re-running — they cause `Errno 22` file-lock errors.

## Central design rule
Geometry handles everything determined by the footprint (OMBB → Umeyama → IoU refinement). ML handles everything determined by the photograph (segmentation, openings, silhouette), plus accept/reject. Judging line: "We used pretrained models for image editing, mesh generation, facade segmentation and opening detection; the model we trained is the confidence gate, on our own benchmark."

## Demo set
**NCB, Burruss, Patton, War Memorial, Whittemore, Goodwin.** Asset ids in `demo_assets.json`.
- McBryde dropped: rectilinearity 0.249.
- NCB and Goodwin are levels-only heights → capped at REVIEW by the gate rule. Story: "the gate knows what it doesn't know".
- Mask notes (not fixing tonight): Goodwin has a bite where a tree stood; Whittemore includes a sliver of the parking deck.

## Photos
- 23 JPEGs across 6 buildings in `photos/`, GPS + compass + focal EXIF, converted from iPhone HEIC. All six demo SHA-256 prefixes verified.
- **Never re-save or re-compress.** Masks keyed to file bytes (`<sha256[:16]>_<settings hash>`).
- Generated photos: `ncb_5 burruss_1 patton_1 warmemorial_2 whittemore_1 goodwin_1`.
- Weak building detections (non-demo): `burruss_3` 0.39, `patton_2` 0.37, `ncb_6` 0.42.

## Generation (Soumik)
- `generate/mask.py` cuts the building out before FLUX + TRELLIS. FLUX sometimes paints sky/ground onto a masked input; pipeline detects it from the image border and re-masks (raw case produced a 100 m dirt slab).
- TRELLIS pinned `e8f6c452…`; its only required input is **`images` (array)** — do NOT change to `image` (the 422 came from an unpinned version). Comment in `lift.py`.
- `generate/throttle.py`: 10 s between Replicate calls, backoff on 429 (credit < $5 → 6/min, burst 1).
- Replicate token rolled; lives in `.env` (gitignored). **Roll again after judging.**
- `$env:HF_HUB_OFFLINE="1"` for the demo; pre-bake every asset (SPEC §15).

## Orientation
- **Every record has a `facade` block** (Soumik): `front_heading_deg`, `front_source`, `front_angle_canonical_rad`, convention = compass degrees CW from true north, outward normal. Computed from refined `FitResult.theta`. Null when front unknown.
- **Never use `FitResult.heading_rad`** — it's the mesh +X bearing, 90° from the glTF front.
- Cesium: viewer places the model with the full 4×4, both axis corrections disabled (`upAxis: Z`, `forwardAxis: X`) — no convention mismatch.
- `score_silhouettes` flat across `azimuth_k` is **by design** (k rotates the mesh against the footprint; the camera view doesn't change; tests assert it). Side evidence is `photographed_side`.
- `photographed_side` margins are all below 0.08 (NCB 0.029, Burruss 0.047, Whittemore 0.016, Goodwin 0.004, Patton 0.005, War Memorial 0.002): thin single-view meshes + corner shots → sides tie. `front_source = gltf_prior` everywhere. Not lowering the threshold to force a vote.
- EXIF heading carried orientation on all six buildings. `geo/exif.py` now validates NaN / out-of-range.
- Road-normal rungs still dead while `roads_enu` is empty.

## Confidence gate
- `ASPECT_RATIO` guessed-orientation rule, exempt when `rotation_margin_footprint >= 0.05`. `ARBITRARY_SYMMETRIC`, `ROAD_NORMAL`, `FACADE_DETAIL` guessed unconditionally.
- ✅ `hard_rules()`: `not fit.height_source_authoritative` → REVIEW.
- Benchmark: `scripts/run_benchmark.py` — 20 real OSM footprints × synthetic meshes with known pose × corruptions × 4 solver configs = 1,680 `LabelledFit` rows. `kind = "synthetic_gt:<corruption>"`; label = IoU ≥ 0.80 and rotation ≤ 10°. At `outputs/benchmark/labelled_fits.pkl` or `scripts.run_benchmark.benchmark_labelled_rows()`.
  - To do: `"benchmark"` source in `load_rows()` (`train.py`). Evaluate **leave-one-building-out** (84 correlated rows per building). Old LOO 0.917 (fully synthetic) stays off slides.
  - Slide wording: "synthetic corruptions of real footprints, scored against known truth".
- 12 / 1,680 wrong auto-accepts, all mirrored Cassell Coliseum (symmetric footprint: mirror = 180° turn; only photo evidence can catch it).
- Strata measured from footprint shape; 8 / 20 name-based guesses were wrong.

## Openings pipeline (Owen)
### 1. Detection — `perception/openings.py` (committed `4351e6c`)
- Building mask from `segment.py` (cache hit; segmentation defaults untouched). Outside-mask pixels greyed (no car doors). Coarse + fine tiling; template gap-fill from accepted windows (NCC on a ~1500 px downscale, DINO-verified on zoomed crops).
- **Two prompts:** `door . window . garage door . entrance .` for calibrated opening scores (per-class, from token logits); a separate distractor prompt (`vent . sign . lamp . column . chimney . balcony .`) only for `other:<label>`. One combined prompt shifted the class score scale (window ~0.30 → 0.15).
- Priors: ground contact relative to the opening's own height; row peers (≥2 same-size neighbours → window, door/entrance ×0.5); door needs ground contact and h/w ≥ 1.5.
- **Containment:** a box containing another detection → `other: recess` (if it contains a passable) or `other: facade_section`. `MAX_BOX_AREA_FRAC = 0.06`. Garage needs DINO ≥ 0.35 and a 0.1 lead over door.
- Edge verifier features (OpenCV): `frame_score`, `mullion_count`, `interior_contrast`, soft factor 0.8–1.1; stored under `edge_features` for training.
- Decision: REVIEW if weak or door-vs-window (group) margin low; door/entrance/garage subtype only flagged.
- Results: burruss_1 50 openings (window/A 42), patton_1 51, whittemore_1 40. door_20 (Burruss tower door) and Whittemore's door ACCEPT. False garage doors removed (overhang, recessed entrance).
- Known issues: Burruss bay window labelled `entrance`; one Patton window relabelled `facade_section`; distractor boxes counted as contents (fix in progress — restores Burruss left-wing entrance).
- Outputs: `outputs/openings/<stem>.json` + overlay `.png`. To do: cache openings like masks (`.cache/perception/openings/<sha16>_<config>.json`) so Soumik's torch-free laptop can use them.

### 2. Mesh projection — `perception/openings_3d.py` (pushed; secondary now)
Fine camera search (azimuth ±60° / 3°, elevation 0–20° / 5°) → ray-cast centre + corners into the mesh → snapped wall normal, metric size via `mesh_to_enu`, bearing from the full transform → glTF child marker nodes + `pygltflib` extras, written to `<mesh>_openings.glb`. Viewer hardcodes `mesh.glb` (`web/src/main.js:152`); HEAD-check fallback suggested.

### 3. Prism projection — `perception/openings_prism.py` (IN PROGRESS, critical path)
- Pinhole camera from EXIF: GPS → ENU, 1.5 m above ground, yaw = heading, pitch = EXIF or 0, focal from 35 mm equivalent.
- Refine against the building mask (prism silhouette IoU): yaw ±15° / 1°, position ±8 m / 2 m, pitch ±5°. IoU < 0.6 or best at grid edge → all openings REVIEW.
- Ray-cast into the extruded ENU footprint; wall hit → exact edge normal → `bearing_deg`. Roof/miss → REVIEW.
- **Door sanity:** passable opening with bottom > 1.0 m above ground → REVIEW.
- Consistency check: front-wall openings vs `facade.front_heading_deg` (±10°).
- Export: markers on the prism glb or Cesium entities from `record["openings"]`, whichever the viewer uses.
- Waiting on Soumik: where the record stores ENU footprint / height / ground, and how the viewer draws the prism.

## Heights
Microsoft heights unusable (mean −5.9 m, median |err| 4.7 m). Metre-tagged OSM = authoritative; levels-only / untagged → REVIEW (enforced). §C monocular depth only if time.

## Soumik — open items (re-check)
- ✅ Silhouette PR merged · ✅ `facade_heading` in record · ✅ Cesium convention · ✅ requirements (`replicate`, `requests`, `python-dotenv`, `pillow-heif`; `rtree`, `pygltflib` added by Owen) · ✅ EXIF NaN validation · ✅ throttle.
- Unknown: `roads_enu` empty; `from_osm_tags` "82 ft" → 82 m; port-backs (match flags, footprint id, Nominatim throttle); pipeline records requested vs actual confidence method; Hahn Hall split case; `check_microsoft_heights.py` overwrites fixture.

## Next steps (in order)
1. Session 1: containment fix (ignore distractor boxes as contents) → push. Detection frozen after this.
2. Session 2: pull → `openings_prism.py` → tests → run burruss_1, whittemore_1, patton_1 → push.
3. Viewer shows opening markers on the prism (minimal change, coordinate with Soumik).
4. Run openings (detect + prism) on all six demo photos; check doors face the right way on the map.
5. Cache openings JSON for Soumik's laptop; pre-bake assets.
6. `"benchmark"` source in `load_rows()`; retrain gate with leave-one-building-out.
7. Freeze assets, `HF_HUB_OFFLINE=1`, rehearse offline. Slides.

## Things worth saying at judging
- Built-vs-pretrained boundary, stated plainly.
- Doors vs windows as the product: separate, typed, oriented, interactable — ambiguous ones routed to REVIEW.
- **Openings are placed by casting rays from the photo's GPS/compass camera onto the real footprint, not onto the AI mesh.** Bearings come from real footprint edges.
- A door that isn't on the ground gets flagged.
- "We know what isn't a door": distractor classes (column, lamp, sign, balcony) and recess detection (Whittemore's recessed entrance vs a false garage door).
- Calibration finding: adding distractor words to one prompt halved window scores → split into two prompts.
- **Our gate rejected our own generator:** all six single-view meshes, with the numbers.
- Masking before generation: without it TRELLIS modelled NCB's bushes as a 12×15 m blob and Patton's parked cars.
- Burruss multipolygon bug: 29% footprint error → IoU 0.762 → 0.921. The tree that split Burruss into 14 components; component-merge cut discarded mask area 43% → 6.3%.
- Coverage ≠ accuracy: Microsoft heights covered 95% of the benchmark but underestimated tall buildings by up to 25 m.
- Benchmark: 1,680 solves, 12 wrong auto-accepts, all mirrored Cassell Coliseum — symmetric footprints need photo evidence. Strata from shape: 8 of 20 name guesses wrong.
- Wrong-but-plausible detection (window instead of facade) that passed every check.
- Why the photo alone can't orient a building (silhouette ties on corner shots; EXIF heading does the work).

## Working style
VS Code integrated terminal, PowerShell: no `&&`, one command per line, replace `<placeholders>` including brackets. Claude Code CLI available (two sessions in one checkout — sequence pulls). Prefers direct, concise output.
