# PlaceCast

**Address + photo → 3D building, placed on a live map at the real building's
location with correct position, scale, rotation and ground alignment.**

Procedura AI / Scorched Nebraska challenge · VTHacks 14.

The generative half (FLUX redesign → image-to-3D) is API orchestration. The
hard part is the placement, so most of this repository is placement: footprint
retrieval, a closed-form similarity fit, four-fold orientation disambiguation,
and a confidence gate that refuses a bad placement rather than shipping it.

---

## Results

**The placement is right; the generated models are the weak link.**

| claim | evidence |
|---|---|
| A correctly shaped model is placed exactly | 1,680 benchmark solves on 20 real campus footprints (7 corruptions × 3 poses × 4 solver configs), scored against known truth: **median IoU 0.996**, orientation correct in 100% of clean cases ([`BENCHMARK.md`](BENCHMARK.md)) |
| The model's own frame doesn't matter | `scripts/scramble_demo.py`: the same mesh rotated 37°, scaled ×1000 and moved 1.2 km lands in the same place (**IoU 0.998**, scale recovered exactly) |
| Size and position from an unknown model | `scripts/blind_placement_test.py`: with the azimuth right, corners land **0.25 m** from truth, scale within **0.1%** |
| Bad placements are caught | **12 wrong auto-accepts in 1,680** (0.7%), all one case: a mirrored symmetric building, which is invisible in plan |

**Six demo buildings** (ids in [`demo_assets.json`](demo_assets.json)):

| building | IoU | Hausdorff | conformed IoU | conformed Hausdorff |
|---|---|---|---|---|
| New Classroom Building | 0.823 | 10.1 m | 0.842 | 12.2 m |
| Burruss Hall | 0.820 | 13.6 m | 0.854 | 10.7 m |
| Patton Hall | 0.623 | 9.8 m | **0.881** | **2.7 m** |
| Goodwin Hall | 0.633 | 19.7 m | 0.661 | 26.5 m |
| Whittemore Hall | 0.566 | 20.9 m | 0.772 | 10.0 m |
| War Memorial Hall | 0.462 | 29.9 m | 0.469 | 28.8 m |

All six generated meshes are **REJECT**, deliberately. They fail on Hausdorff
(worst-case boundary error), not position. Single-photo 3D generation returns
shallow shells with the wrong plan aspect (NCB 1.6:1 vs the real 2.2:1), and a
uniform scale cannot fix a wrong aspect. Orientation came from the EXIF compass
heading on all six.

The **footprint prisms** (the real OSM footprint extruded to its real height,
no generation) are correct on the map for all six, checked against satellite
imagery.

---

## Quick start

Windows / PowerShell. Every command is `uv run ...`: on Windows a Microsoft
Store `python.exe` stub intercepts a bare `python`.

```powershell
# One-time
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id OpenJS.NodeJS.LTS -e
# reopen the terminal, then:
uv python install 3.11

# Environment (geometry + generation)
uv venv --python 3.11
uv pip install -r requirements-geo.txt
Copy-Item .env.example .env     # REPLICATE_API_TOKEN, CESIUM_ION_TOKEN; Google key optional

# Verify
uv run pytest -q                # perception is forced to the stub in tests
uv run python scripts/prefetch_footprints.py
uv run python pipeline.py --address "Burruss Hall, Blacksburg, VA" --dry-run

# Viewer (two terminals) -> http://localhost:5173
uv run uvicorn server.app:app --reload
cd web; npm install; npm run dev
```

Real perception (`--perception real`: Grounding DINO + SAM 2) needs a CUDA
machine and the separate ML environment described at the top of
`requirements-ml.txt`. Weights are 10+ GB, so set `HF_HOME` first.

### Common runs

```bash
# footprint prism only: no photo, no generation, free
uv run python pipeline.py --address "Patton Hall, Blacksburg, VA" --dry-run

# full path: photo -> FLUX Kontext -> TRELLIS -> placed (billed on Replicate)
uv run python pipeline.py --address "Patton Hall, Blacksburg, VA" \
    --photo photos/patton_1.jpg --perception real

# re-solve an existing asset (cached edit + mesh, free); --conform stretches it onto the footprint
uv run python pipeline.py --address "..." --photo photos/patton_1.jpg \
    --perception real --asset-id <id> [--conform]

# place a model you already have (no generation)
uv run python pipeline.py --address "..." --mesh path/to/model.glb

# the three proofs
uv run python scripts/run_benchmark.py --reps 3          # writes BENCHMARK.md
uv run python scripts/scramble_demo.py <asset-id>
uv run python scripts/blind_placement_test.py --reps 3

# check photos (EXIF GPS + heading) before spending anything
uv run python scripts/check_photos.py
```

Other `pipeline.py` flags: `--lifter trellis|hunyuan`, `--confidence
threshold_table|learned`, `--force-candidate 0-3`, `--no-mask`, `--seed`.

Generation costs money (`scripts/generate_demo.py --yes`); everything else is
free. Replicate calls are spaced 10 s apart and retry on 429
(`generate/throttle.py`).

### Output

Each run writes `data/runs/<asset-id>/` (gitignored):

| file | what |
|---|---|
| `record.json` | the placement record: transform, metrics, decision, provenance |
| `overlay.png` | **the most useful file in the repo.** Footprint vs placement, all metrics, all four candidate orientations, and the reasons for the decision |
| `run.log` | every stage, in order |
| `mask.npy` | the cached segmentation mask (computed once at ingest) |

Read the overlay before the numbers. An IoU of 0.91 on a building rotated 90°
is the failure this project is about, and no scalar catches it.

### Viewer

CesiumJS on Cesium World Terrain. The dropdown labels each run
`address · mesh|prism · decision · IoU`.

- **prism**: the OSM footprint extruded to its resolved height. Views:
  **Photo** (default; the photo baked onto the walls through the fitted
  camera), **Schematic** (flat colour), **Scorched** (experimental; the
  Scorched Nebraska FLUX edit baked the same way). An "AI-filled" overlay marks
  wall regions inpainted by FLUX Fill rather than seen by the camera. Photo and
  Scorched textures exist only after a bake:
  `uv run python perception/prism_texture.py <run_dir> <photo> <mask> [--scorched <edit>] [--ai]`
  (camera from `perception/openings_prism.py`; `--ai` is billed and capped).
- **mesh**: the generated model placed by the solver. Conformed runs sit beside
  their originals.
- **View from photo** puts the camera where the photo was taken.
- **keep on map (compare)** draws the next selection alongside the current one;
  **clear** empties the map.

---

## How it works

```
address ──geocode──> OSM footprint (Overpass, cached) ──┐
photo ──EXIF──> GPS + compass heading                   │
photo ──Grounding DINO + SAM 2──> building mask         │
      ──FLUX Kontext──> redesign ──TRELLIS──> mesh ──> ground outline
                                                        │
                       OMBB init → Umeyama similarity → IoU refinement
                       → 4-fold orientation disambiguation
                       → height chain → terrain
                       → confidence gate (accept / review / reject)
                                                        │
                                       record.json + overlay.png → viewer
```

### Layout

```
contracts.py         frozen types every stage passes (v1.0, tag contracts-v1). Start here.
pipeline.py          stage wiring, one direction of flow

geo/                 the placement solver
  coords.py          ENU / ECEF / geoid / the three angle conventions
  footprint.py       geocode, Overpass, selection rule, rectilinearity, OMBB
  outline.py         up-axis, grounding, ground-outline extraction, bas-relief rejection
  fit.py             OMBB init, Umeyama similarity Procrustes, IoU refinement
  disambiguate.py    the four-fold azimuth ambiguity
  validate.py        metrics + the threshold-table gate
  height.py          height priority chain
  terrain.py         USGS 3DEP elevation, sloped sites
  placement.py       composes the fit into one mesh → ENU matrix
  exif.py            GPS + GPSImgDirection at ingest
  overlay.py         the debug overlay

generate/            Replicate: FLUX Kontext (edit), TRELLIS / Hunyuan3D (lift),
                     photo masking before generation, request throttling
perception/          segmentation, silhouette render-and-compare, orientation ladder,
                     facade openings (doors/windows), prism texturing + FLUX Fill
confidence/          learned gate: features, training, leave-one-building-out eval

web/                 CesiumJS viewer (Vite)
server/app.py        FastAPI: runs, assets, review queue, job submission
fixtures/            fake fits / photo evidence, committed OSM footprint cache
scripts/             prefetch, benchmark, proofs, photo checks, demo generation
tests/               synthetic + adversarial cases
```

### Orientation

An oriented bounding box fits a footprint four ways, 90° apart, and IoU often
can't tell them apart. Aspect ratio first excludes the 90° candidates where the
plan is elongated enough. `geo/disambiguate.py` then decides in priority order:

1. EXIF compass heading, if present and plausible
2. silhouette score, if its margin is ≥ 0.08
3. facade detail agreeing with the road normal
4. road normal alone (always flagged for review)
5. best footprint IoU (flagged if the margin is low)

The road prior is filtered to real streets, since campus "roads" are mostly
footpaths. A confident EXIF heading is never overridden by render-and-compare.

### Camera pose (prism texturing and openings)

`perception/openings_prism.py` fits a pinhole camera to each photo so the photo
can be projected onto the footprint prism.

1. **Initial pose from EXIF.** Position is the photo's GPS converted to local
   ENU, 1.5 m above the ground. Yaw is the EXIF compass heading; pitch and roll
   start at 0 (iPhone photos carry no pitch). Focal length in pixels comes from
   the 35 mm-equivalent focal length over the frame diagonal:
   `f_px = f35 × diag_px / 43.27`.
2. **Grid search.** The prism is projected into the image and its silhouette
   scored against the building mask by plain IoU, over yaw ±15° (1° steps),
   east/north ±24 m (4 m steps) and pitch 0–15° (2.5° steps): about 37k
   silhouettes, scored in parallel. Focal length is held fixed, because it
   trades off almost exactly against camera distance and drifted to the grid
   edge when searched.
3. **Refinement.** The best candidate is refined in position at 1 m steps.
4. **Reliability check.** The fit is trusted only if IoU ≥ 0.6 and no axis
   lands on the edge of the grid (an edge value means the optimum lies outside
   it). Unreliable fits send that photo's openings to review and are not baked
   unless forced.

The fitted camera is stored in `record.json` and reused unchanged by
`perception/prism_texture.py`, which projects each wall texel through it to
sample the photo. Door and window boxes are placed by casting rays from the
same camera onto the prism walls.

### Confidence gate

Every placement gets a decision: **accept**, **review** (sent to a human) or
**reject**.

- **Threshold table** (`geo/validate.py`): hand-set limits on IoU, Hausdorff,
  area ratio, rotation margin, anisotropy and neighbour overlap. The default.
- **Learned gate** (`confidence/`): L2 logistic regression on ten fit
  features (`confidence/features.py`: footprint IoU, Hausdorff, area ratio,
  rotation margins, anisotropy, neighbour overlap, rectilinearity, geocode
  precision, height source). Three are constant on the benchmark and are
  dropped there, leaving seven. It outputs a probability that the
  placement is acceptable; thresholds for accept and reject are chosen for 95%
  precision, and everything between goes to review. Hard rules (e.g. a
  non-authoritative height) still floor the decision in both methods.

The learned gate is trained and evaluated on the benchmark: 1,680 labelled
solves from 20 buildings, a solve labelled good when IoU vs truth ≥ 0.8 and
rotation error < 10°. Evaluation is **leave-one-building-out**, since the 84
rows of one building share a footprint and row-level CV would leak. The
regularisation strength and both thresholds are chosen inside each fold.

```bash
uv run python confidence/train.py --source benchmark --markdown BENCHMARK.md
```

Result: pooled out-of-fold **AUC 0.921**, accuracy 0.892 (majority baseline
0.665). At the same coverage as the threshold table it makes 15 wrong
auto-accepts to the table's 12, so it is not safer, but it wrongly rejects far
fewer good placements (24 vs 397). That is why the table stays the default and
the learned gate is opt-in (`--confidence learned` or
`PROCEDURA_LEARNED_GATE`). Model files are written to `outputs/confidence/`,
which is gitignored. Full tables are in [`BENCHMARK.md`](BENCHMARK.md).

### Perception backends

`--perception stub` (the default in `pipeline.py`) uses a central-box mask and
an abstaining silhouette scorer: no GPU. `--perception real` runs Grounding
DINO + SAM 2 and render-and-compare. The record states which backend ran.
Every ML component has a non-ML fallback, so nothing on the critical path
depends on a GPU.

---

## Findings

Measured, not estimated.

1. **Single-photo 3D generation gets depth wrong.** Models come back as shallow
   shells: Patton's plan fills 18% of its bounding box against the real 81%.
2. **Hunyuan3D-2.1 is worse here.** On NCB it gave a 4.8:1 sliver. It's kept as
   `--lifter hunyuan`; TRELLIS stays the default.
3. **A mirrored symmetric building is undetectable in plan.** All 12 wrong
   benchmark auto-accepts are Cassell Coliseum mirrored. Only photo evidence
   catches it.
4. **Microsoft GlobalMLBuildingFootprints heights are 44–67% low** against the
   five campus buildings with explicit OSM heights (all 20 values fall in a
   7–16 m band). Microsoft is therefore not treated as authoritative.
5. **OSM height coverage is 15/22 (68%)**, and only a third give metres
   directly, so `building:levels × 3 m` and the proportional fallback are the
   common path.
6. **Three silent footprint bugs**, each a confident wrong answer rather than an
   error, now regression-tested in `tests/test_footprint_selection.py`:
   - Multi-way outer rings were truncated (Burruss read 4384 m² instead of
     6136). Members must be stitched into rings before assembly.
   - Multi-part relations collapsed to one part (Lane Stadium read 8930 m²
     instead of 28042).
   - The selection rule guessed when nothing matched (a demolished Randolph
     Hall selected a wind tunnel 46 m away). It now fails loudly, except for
     a narrow `unnamed_sole_candidate` case that is always routed to review.
7. **Bought models break assumptions generated ones don't.** Outlines are now
   read from sampled surfaces, not vertices (a wall with four vertices produced
   no outline), and the up-axis sign comes from the base on the origin plane,
   not vertex mass (a detailed roof inverted it).

## Design decisions

**Geometry is closed-form; ML fills the perception gaps.** OMBB, Umeyama and
IoU refinement are exact. We didn't train a network to approximate a solved
problem.

**Store the transform, never the transformed mesh.** The transform is ten
floats. Model versions, seeds and the footprint's OSM *version* are pinned,
because OSM footprints get edited.

**Reflections are rejected.** The fit optimises over SO(2), not O(2); Umeyama's
`S` correction stops the SVD returning a mirror, and
`tests/test_fit_synthetic.py` asserts it.

**The redesign must not change the silhouette.** `generate/edit.py` rejects
silhouette-mutating prompts, and `geo/outline.py` fits only the lowest 15% of
the mesh.

**No OpenGL anywhere.** Silhouettes are rasterised in numpy and the overlay is
matplotlib, because offscreen GL on Windows has no OSMesa and EGL is Linux-only.

**One angle conversion per side.** The solver uses mathematical θ (CCW from
East); Cesium uses compass heading (`π/2 − θ`). The viewer disables both of
Cesium's glTF axis corrections because the model matrix carries the full
rotation, and all viewer polygons are synchronous primitives (deleting an
async one mid-build kills Cesium's renderer).

---

## Known gaps

- **Silhouette scoring doesn't vote.** `score_silhouettes` is flat across the
  four azimuth candidates by construction (`tests/test_azimuth_contract.py`),
  so orientation relies on EXIF and the priors.
- **The photographed side rarely clears its 0.08 margin** (measured
  0.002–0.047 on the demo set), so the mesh front usually falls back to the
  glTF "+Z is front" convention. For facade bearings, use
  `record["facade"]["front_heading_deg"]`, not `FitResult.heading_rad`.
- **Monocular depth (UniDepth) isn't built**, although the 68% height coverage
  says it should be.
- **The learned gate is evaluated on synthetic corruptions** of real
  footprints, not on human-judged placements of generated meshes.
- `pipeline.py` discards `segment_building`'s `occluded` flag and recomputes
  occlusion from the mask.
- Benchmark strata in `scripts/demo_addresses.py` are guesses from building
  names. Verify against measured aspect and rectilinearity before relying on
  them.
- Not built, in cut order: certified global solve, C2PA signing, skirt geometry
  for sloped sites, turning-function screening.

See [`HANDOVER.md`](HANDOVER.md) for asset ids and run state,
[`BENCHMARK.md`](BENCHMARK.md) for full tables, and [`SPEC.md`](SPEC.md) /
[`ML_ADDENDUM.md`](ML_ADDENDUM.md) for the original design.
