# Handover: the geometry half → Owen

Written 2026-09-19, ~21:30. Soumik is done; everything below is the state of
the geometry side, what is proven, what is not, and what to do next.

Everything is pushed to `main`. `uv run pytest -q` is green (550 tests).

---

## 1. What works, with numbers

**The placement maths is the solid part.**

| claim | evidence |
|---|---|
| A correct-shaped model is placed exactly | Benchmark, 1,680 solves against known truth: **median IoU 0.996**, rotation correct in 100% of clean cases (`BENCHMARK.md`) |
| The model's own frame does not matter | `scripts/scramble_demo.py`: the same mesh rotated 37°, scaled ×1000, moved 1.2 km off-origin lands in the same place — **placed outlines agree to IoU 0.998**, recovered scale exactly 1/1000 |
| Size and position from an unknown model | `scripts/blind_placement_test.py`: when the azimuth is right, corners land **0.25 m** from truth, scale within **0.1%** |
| Bad placements are caught, not shipped | Benchmark: **12 wrong auto-accepts in 1,680** (0.7%), all one case — a mirrored symmetric building |

**Six demo buildings, current numbers** (after tonight's extraction fixes; also
in `demo_assets.json`):

| building | IoU | Hausdorff | conformed IoU | conformed Hausdorff | stretch |
|---|---|---|---|---|---|
| NCB | 0.823 | 10.1 m | 0.842 | 12.2 m | 1.17× |
| Burruss | 0.820 | 13.6 m | 0.854 | 10.7 m | 1.15× |
| Patton | 0.623 | 9.8 m | **0.881** | **2.7 m** | 1.60× |
| Goodwin | 0.633 | 19.7 m | 0.661 | 26.5 m | 1.62× |
| Whittemore | 0.566 | 20.9 m | 0.772 | 10.0 m | 2.17× |
| War Memorial | 0.462 | 29.9 m | 0.469 | 28.8 m | 1.03× |

All six are REJECT — deliberately. They fail on Hausdorff (worst-case boundary
error), not on position. **This is the honest headline: the placement is right,
the generated models are the weak link.** Patton conformed (IoU 0.881,
Hausdorff 2.7 m) is the closest to acceptable.

**Orientation** came from the EXIF compass heading on all six — evidence, not a
guess.

---

## 2. Run it

```bash
uv run uvicorn server.app:app --reload     # terminal 1
cd web && npm run dev                      # terminal 2 -> localhost:5173
```

The dropdown labels every run `address · mesh|prism · decision · IoU`.

- **prism** = the real OSM footprint extruded to its real height. No AI. All six
  are correct on the map and were checked against satellite imagery.
- **mesh** = the generated model placed by the solver.
- **conform** runs sit beside their originals (per-axis stretch onto the
  footprint; the stretch is recorded, and the gate still flags it).
- **"keep on map (compare)"** draws the next selection alongside the current one
  instead of replacing it. **clear** empties the map.

Useful commands:

```bash
# re-solve an existing asset (free: cached edit + mesh, no model runs)
uv run python pipeline.py --address "Patton Hall, Blacksburg, VA" \
    --photo photos/patton_1.jpg --perception real --asset-id d354f660-... [--conform]

# place a model you already have (no generation at all)
uv run python pipeline.py --address "..." --mesh path/to/model.glb

# clone a run's generated assets so the same mesh can be solved a second way
uv run python scripts/clone_run.py SRC_ID --then-conform

# the three proofs
uv run python scripts/run_benchmark.py --reps 3      # writes BENCHMARK.md
uv run python scripts/scramble_demo.py 3289ae77-ac15-44b2-85c7-91e1d7568845
uv run python scripts/blind_placement_test.py --reps 3

# check photos before spending anything
uv run python scripts/check_photos.py
```

**Generation costs money** (`scripts/generate_demo.py --yes`). Everything above
except that is free. Replicate calls are spaced 10 s apart and retry on 429
(`generate/throttle.py`) because the account is under $5 of credit.

---

## 3. Assets

`demo_assets.json` holds every id. The important ones:

- **NCB mesh** `3289ae77-ac15-44b2-85c7-91e1d7568845` — best generated result,
  and the judging venue. Its conformed twin is `0052eb8c-…`.
- **Patton conformed** `db4632e3-…` — the best fit overall (IoU 0.881).
- **Prisms** (correct on the map, use these for the "it works" shot):
  NCB `6518a90a-…`, Burruss `6dbfe23d-…`, Patton `5f6a9b13-…`,
  War Memorial `c24b872f-…`, Whittemore `1fd209f8-…`, Goodwin `b1068a03-…`.
- **`nyc-model-demo`** — a downloaded NYC block placed on the Undergraduate
  Science Laboratory Building, across the street from NCB: IoU 0.675, area
  ratio 1.057, scaled 1.72× from the model's own units, no photo and no
  generation. **The .glb is NOT in git** (61 MB, someone else's asset). To
  rebuild it, get `building-buildify-nyc/source/Untitled.glb` from Soumik and:
  ```bash
  uv run python pipeline.py --address "Undergraduate Science Laboratory Building, Blacksburg, VA" \
      --mesh building-buildify-nyc/source/Untitled.glb --asset-id nyc-model-demo
  ```
  `data/` is gitignored, so this run does not exist on a fresh clone.

---

## 4. What is left

**Presentation (all of it).** Nothing has been built for the pitch beyond the
viewer. Suggested story, in the order the evidence supports:

1. **The map.** Six buildings placed correctly from an address alone (prisms).
2. **The AI path.** Photo → FLUX redesign → TRELLIS → placed (NCB mesh).
3. **The proof it generalises.** Either the scramble demo (a model handed over
   in a hostile frame lands identically) or the NYC model placed across the
   street from NCB. Both are live commands; the scramble prints its own table.
4. **The benchmark table** (`BENCHMARK.md`) and the honest findings below.

**Freeze checklist** — do this before sleeping, it is the insurance:

- [ ] Re-run every demo asset (commands above) so records match the shipped code
- [ ] **Record a 60-second screen capture of the working viewer.** If the venue
      network dies, Cesium terrain will not load and the live demo is gone.
      Cached OSM footprints work offline; terrain does not.
- [ ] `git push` and tag (`git tag demo-v1 && git push --tags`)
- [ ] Note the asset ids you plan to show, in order, on a card

---

## 5. Honest findings worth saying out loud

These are measured, and they are more interesting than a clean demo:

1. **Single-photo 3D generation gets depth wrong.** The models come back as
   shallow shells: NCB's plan 1.6:1 where the building is 2.2:1, Patton's plan
   filling 18% of its bounding box against the real 81%. A uniform scale cannot
   fix a wrong aspect, which is why Hausdorff stays high while IoU looks fine.
2. **Hunyuan3D-2.1 was tried and is worse here.** On NCB it gave a 4.8:1 sliver
   filling 0.39, with background removal on and off. Kept as `--lifter hunyuan`;
   TRELLIS remains the default. Decided on the measurement, not the render.
3. **A mirrored symmetric building is undetectable in plan.** All 12 wrong
   auto-accepts in the benchmark are Cassell Coliseum mirrored: a mirror of a
   symmetric plan looks exactly like the building turned around. Only photo
   evidence catches it.
4. **Microsoft building heights are 44–67% low** against explicit OSM heights on
   this campus (`scripts/check_microsoft_heights.py`).
5. **Campus "roads" are mostly footpaths.** Near NCB, 67 of 90 highway ways were
   paths, footways or steps; near Burruss, 138 of 139. The street-facing prior
   is filtered to real streets for this reason.
6. **Three silent footprint bugs**, found and fixed: truncated multi-way
   relations (Burruss read 29% small), MultiPolygon collapse, and a selection
   rule that guessed when it should have refused.
7. **Bought models break assumptions generated ones do not** (found tonight with
   the NYC asset): outlines were read from the vertex cloud, so a wall with four
   vertices produced nothing (a 48×27 m model gave a 39 m² "plan"); and the
   up-axis sign came from vertex mass, which a detailed roof inverts. Both
   fixed — surfaces are sampled now, and a base sitting on the origin plane
   decides the sign.

---

## 6. Open items on your side

Flagged earlier, still open as far as I can see:

1. **Silhouette scoring is flat across azimuth.** Every run logs
   `u2a0=… u2a1=… u2a2=… u2a3=…` identical, margin 0.000, so the cue never
   votes and `has_silhouette_evidence` is never true.
2. **`photographed_side` never clears the 0.08 margin** (measured: NCB 0.029,
   Burruss 0.047, Whittemore 0.016, Goodwin 0.004, Patton 0.005, War Memorial
   0.002), so `record["facade"]["front_source"]` is `gltf_prior` everywhere and
   your openings inherit a front from the file convention rather than the photo.
3. **Door/window bearings**: use `record["facade"]["front_heading_deg"]`, never
   `FitResult.heading_rad` — the latter is the mesh's +X axis, 90° away from the
   glTF front.
4. **`confidence/train.py`** still only reads the fake source; the benchmark
   emits real rows to `outputs/benchmark/labelled_fits.pkl`
   (`scripts.run_benchmark.benchmark_labelled_rows()`).

---

## 7. Where things live

| area | files | owner |
|---|---|---|
| geometry, solver, placement | `geo/`, `pipeline.py` | Soumik (now yours) |
| generation | `generate/` (`edit`, `lift`, `mask`, `throttle`) | Soumik |
| perception, confidence | `perception/`, `confidence/` | Owen |
| viewer, API | `web/src/main.js`, `server/app.py` | shared |
| proofs and tools | `scripts/` (benchmark, scramble, blind test, clone_run, check_photos, generate_demo) | Soumik |
| frozen contract | `contracts.py` | do not change |

Conventions that will bite if forgotten: the solver works in mathematical θ
(CCW from East) and Cesium wants a compass heading (`heading = π/2 − θ`),
converted in exactly one place on each side; the viewer disables **both** of
Cesium's glTF axis corrections (`upAxis: Axis.Z`, `forwardAxis: Axis.X`) because
the full rotation is already in the model matrix; and every polygon in the
viewer is a **synchronous** primitive — entity polygons and terrain-draped
polygons are built on a worker, and deleting one mid-build kills Cesium's
renderer permanently.
