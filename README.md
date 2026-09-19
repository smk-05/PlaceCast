# Photo → Map-Ready 3D Building

Address + photograph → AI redesign → 3D mesh → placed on a live map at the real
building's location with correct position, scale, rotation and ground alignment,
automatically.

Procedura AI / Scorched Nebraska challenge · VTHacks 14.

**The generative half is API orchestration and works on day one. The entire
engineering difficulty is in the placement**, so that is where this repository
puts its weight.

---

## Quick start

```powershell
# One-time (winget)
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id OpenJS.NodeJS.LTS -e
# reopen the terminal, then:
uv python install 3.11

# Environment
uv venv --python 3.11
uv pip install -r requirements-geo.txt
Copy-Item .env.example .env     # fill in the tokens

# Verify
uv run pytest -q
uv run python scripts/prefetch_footprints.py
uv run python pipeline.py --address "Burruss Hall, Blacksburg, VA" --dry-run

# Viewer
cd web; npm install; npm run dev
# and in a second terminal:
uv run uvicorn server.app:app --reload
```

Every command is `uv run ...`. There is a Microsoft Store `python.exe` alias
stub on Windows that intercepts a bare `python`; `uv run` always resolves the
venv interpreter and sidesteps it.

---

## What runs today

`pipeline.py --dry-run` executes the whole chain end to end, using a plain box
as the mesh. Output lands in `data/runs/<uuid>/`:

| file | what |
|---|---|
| `record.json` | the spec 12.1 placement record — transform, metrics, provenance |
| `overlay.png` | **the most useful file in the repo.** Footprint vs placement, all metrics, all four candidate orientations, and the reasons for the decision |
| `run.log` | every stage, in order |
| `mask.npy` | the cached segmentation mask (ingest-time, never recomputed at fit time) |

Read the overlay before you read the numbers. An IoU of 0.91 on a building
rotated 90° is the failure this project is about, and no scalar catches it.

---

## Layout

```
contracts.py         the frozen types every stage passes. Start here.
pipeline.py          stage wiring, one direction of flow

geo/                 the placement solver — spec 2-9
  coords.py          ENU / ECEF / geoid / the three angle conventions
  footprint.py       geocode, Overpass, selection rule, rectilinearity, OMBB
  outline.py         up-axis, grounding, M_0 extraction, bas-relief rejection
  fit.py             OMBB init, Umeyama similarity Procrustes, IoU refinement
  disambiguate.py    the four-fold azimuth ambiguity — NO ML required
  validate.py        metrics + the threshold-table gate (the PRIMARY gate)
  height.py          spec 7's four-tier priority chain
  terrain.py         elevation sampling, sloped sites
  overlay.py         the debug overlay
  exif.py            EXIF at ingest — free, and it beats every other cue

generate/            spec 5 — FLUX Kontext + TRELLIS, hosted on Replicate
perception/          addendum A/C — segmentation, silhouette, depth  [see below]
confidence/          addendum B — the learned gate                   [see below]

web/                 CesiumJS viewer (Vite)
server/app.py        FastAPI: records, assets, review queue
fixtures/            fake_fits, fake_photo_evidence, the committed OSM cache
scripts/             prefetch, height-coverage check, benchmark
tests/               spec 14's synthetic + adversarial cases
```

## The division of labour

The ML half — addendum sections A (segmentation + silhouette scoring), B (the
learned confidence gate) and C (monocular depth) — is developed separately and
dropped in. **Nothing on the critical path blocks on it.** Every ML component
ships with a working non-ML stand-in that is a permanent fallback, not a
placeholder:

| component | stand-in now | upgrade later |
|---|---|---|
| segmentation | central-box mask | Grounded SAM 2 |
| silhouette scoring | abstains (margin 0) | render-and-compare |
| confidence gate | spec 9.1 threshold table | logistic regression |
| height | OSM tags → proportional | UniDepth |

Orientation is the case that matters. Addendum A.3 leans on silhouette
render-and-compare as "the only cue that is evidence rather than prior", but
`geo/disambiguate.py` resolves the four-fold ambiguity with **no ML at all**:
aspect-ratio arithmetic, EXIF `GPSImgDirection`, the road-normal prior, and
facade-detail scoring from vertex density and normal variance. When the
silhouette score arrives it becomes a *second, independent* signal — which is
what the addendum actually wants, since disagreement between the two margins is
itself a confidence feature.

### Handoff to the perception contributor

`contracts.py` is **frozen at v1.0** (git tag `contracts-v1`). Changing a field
name, type or meaning needs both contributors, a `CONTRACT_VERSION` bump and a
new tag.

Perception provides three functions; geometry provides one back:

```python
# perception -> geometry
perception/segment.py        segment_building(path)  -> (mask, area_frac, occluded)
perception/render_compare.py score_silhouettes(mesh, mask, candidates) -> {cid: iou}
confidence/gate.py           score_confidence(fit, photo, fp) -> (p, Decision)

# geometry -> perception
geo/disambiguate.py          facade_heading(fp, mo, candidate) -> compass bearing, degrees
```

Where the confidence features live — everything `score_confidence` needs is on
the three objects it receives:

| addendum B.3 feature | read it from |
|---|---|
| `footprint_iou`, `hausdorff_m`, `anisotropy_log_ratio`, `max_neighbor_overlap` | `fit.*` |
| `area_ratio_log` | `log(fit.area_ratio)` |
| `rotation_margin_footprint` | `fit.rotation_margin_footprint` |
| `rotation_margin_silhouette` | `photo.silhouette_margin` |
| `exif_silhouette_disagree` | `fit.exif_silhouette_disagree` — `None` when a cue was absent |
| `height_source_authoritative` | `fit.height_source_authoritative` |
| `rectilinearity` | `fp.rectilinearity` |
| `geocode_rooftop` | `fp.geocode_rooftop` |
| footprint match quality | `fp.match_quality`, `fp.is_weak_match` |

`exif_silhouette_disagree` lives on `FitResult`, not `PhotoEvidence`, because
it can only be computed during disambiguation: mapping an EXIF bearing to a
candidate needs `facade_heading`, which depends on the fitted mesh outline, and
`PhotoEvidence` is frozen before geometry runs.

Two warnings worth passing on:

1. **No pyrender, no OpenGL.** Offscreen GL on Windows has no OSMesa and EGL is
   Linux-only. Silhouette renders are plain triangle rasterisation in numpy.
2. **Ordering is load-bearing.** `MeshOutline.is_bas_relief` (spec 6.4) is set
   before any silhouette comparison. A flat relief matches well from its one
   good view and would otherwise win the orientation vote while being
   catastrophically wrong in 3D.

---

## Measured findings

Real numbers from live OSM on 2026-09-19, not estimates.

**Demo set** (`scripts/prefetch_footprints.py`):

| building | OSM | area m² | R | aspect | height tag |
|---|---|---|---|---|---|
| Burruss Hall | relation | 6136 | 1.000 | 1.44 | 20.7 m |
| Torgersen Hall | relation | 5353 | 0.919 | 2.42 | 6 levels |
| Goodwin Hall | way | 4068 | 0.999 | 1.26 | 4 levels |
| New Classroom Building | way | 2272 | 0.993 | 2.22 | 3 levels |
| Moss Arts Center | way | 7897 | 1.000 | 1.08 | **none** |

Moss Arts is the interesting one: aspect 1.08 is *below* spec 6.6 Filter 1's 1.1
cutoff, so the 90° candidates cannot be excluded on scale grounds and the whole
disambiguation chain has to carry it. It also has no height tag, so it exercises
spec 7's fallback. Torgersen has the lowest R at 0.919 — a genuinely complex
plan, with the bridge over Alumni Mall.

**Height coverage** (`scripts/check_height_coverage.py`) — **15/22 = 68%**.

Below the addendum's 80% threshold, so its conditional flips: monocular depth
(or the cheaper single-view metrology of C.4) should be built, not skipped. The
breakdown matters more than the headline — only a third of buildings give metres
directly, and `building:levels × 3 m` is load-bearing for the rest, so the
fallback chain is the common path rather than the exception.

**Microsoft GlobalMLBuildingFootprints** (`scripts/check_microsoft_heights.py`)
covers 6 of the 7 OSM-untagged buildings (not Hutcheson) — but its heights are
not trustworthy here. Against the five buildings with an explicit OSM `height`
in metres, it underestimated every one:

| building | OSM `height` | Microsoft | error |
|---|---|---|---|
| Burruss | 20.7 | 11.3 | −45% |
| McBryde | 23.9 | 11.6 | −51% |
| Whittemore | 37.4 | 12.2 | −67% |
| Patton | 16.4 | 9.2 | −44% |
| War Memorial | 18.8 | 9.0 | −52% |

All 20 Microsoft values fall in a 7–16 m band. So Microsoft is **not**
authoritative in this codebase (deviating from spec §7.1) and ranks below
monocular depth in the height chain. Section C is still needed. Those five
buildings are also the natural ground truth for validating it.

---

**Three silent footprint bugs**, found on live data and now regression-tested in
`tests/test_footprint_selection.py`. Each produced a confident, plausible, wrong
answer rather than an error — the worst failure mode this pipeline has:

1. **Multi-way outer rings were truncated.** OSM routinely splits one outer ring
   across several member ways that must be stitched end-to-end. Taking the first
   member shrank Burruss Hall — the primary demo building — from 6136 to 4384 m²
   and inverted its rectilinearity. Note that "build a MultiPolygon from the
   outer members" is *also* wrong here: those members are open segments, so it
   yields four degenerate slivers. Stitch into rings first, then assemble.
2. **Multi-part relations collapsed to one part.** Lane Stadium is four disjoint
   stands with the field as a genuine gap; it read 8930 m² instead of 28042.
   Spec §3.3 covers interior rings but never names this case.
3. **The §3.2 selection rule guessed instead of failing.** Step 2 ranks
   candidates, but the spec never says what to do when nothing matches — and
   ranking always returns something. A demolished Randolph Hall selected the
   "Stability Wind Tunnel", 360 m², 46 m away, no name match. Selection now
   fails loudly and names the candidates, with one narrow exception
   (`unnamed_sole_candidate`) that is tagged as weak evidence and routed to
   review however good the IoU looks.

`Footprint.match_quality` carries the outcome of (3) into the placement record
as a confidence feature.

## Design decisions worth knowing

**Geometry is closed-form; ML fills the perception gaps.** OMBB initialisation,
Umeyama's similarity Procrustes and IoU refinement are exact. We did not train a
network to approximate a solved problem. The learned component is the confidence
gate, and the threshold table stays in the codebase as the fallback path.

**Store the transform, never the transformed mesh.** The mesh is a large binary;
the transform is ten floats. Model versions, seeds and the footprint's OSM
*version* are all pinned, because OSM footprints get edited and a re-fetch six
months later may silently return a different polygon.

**The silhouette-mutation tension** (spec 10.1) is real and is named rather than
hidden: a redesign that changes the silhouette no longer matches the real
footprint. `generate/edit.py` rejects silhouette-mutating prompts up front, and
`geo/outline.py` fits the base only — the lowest 15% of the mesh, the part most
likely to preserve the original plan.

**Reflections are rejected, never accepted.** We optimise over SO(2), not O(2).
Umeyama's `S` correction is what stops the SVD silently returning a mirror, and
`tests/test_fit_synthetic.py` asserts it on deliberately reflected data.

**No OpenGL anywhere.** Spec 6.3's ground-outline extraction is an occupancy
grid in numpy, and the overlay is matplotlib.

---

## Known gaps

- The learned gate exists (`confidence/features.py`, `train.py`, `gate.py`) but is trained on
  SYNTHETIC rows only (`fixtures/fake_fits.py`); `model.json` says so and `gate.py` refuses to
  use it. It needs the real spec 11 benchmark rows. Until a real model exists, `--confidence
  learned` falls back to the threshold table and records that in the review reasons.
- Perception runs the REAL models by default (Grounding DINO + SAM 2). Set
  `PROCEDURA_PERCEPTION=stub` for the non-ML stand-ins on a machine without a GPU; the test
  suite does this in `tests/conftest.py`. `pipeline.py`'s own `--perception` flag does not reach
  `perception/` yet, so the environment variable is the switch.
- `score_silhouettes` is flat across `azimuth_k` on purpose (see `tests/test_azimuth_contract.py`):
  a silhouette of the mesh front cannot depend on which quarter-turn the placed mesh gets.
- `geo/terrain.py` uses USGS 3DEP (orthometric, needs the geoid correction).
  The viewer's Cesium World Terrain path is already ellipsoidal and is primary.
- Benchmark strata in `scripts/demo_addresses.py` are guesses from building
  names. Two such guesses were already wrong in the demo set — verify each
  against measured aspect and R before the benchmark run.
- Not built, in deliberate cut order: branch-and-bound certified global solve
  (spec 6.10), C2PA signing (12.2), skirt geometry for sloped sites (8),
  turning-function screening (6.7).
