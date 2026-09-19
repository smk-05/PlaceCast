# Addendum — Learned Components
Photo → Map-Ready 3D Building · Procedura AI / Scorched Nebraska · VTHacks 14

**Status of this document.** This is an addendum to the main technical specification, not a replacement. The geometric solve described in §6 of the main spec — OMBB initialisation, Umeyama similarity Procrustes, IoU refinement — stays exactly as written. It is closed-form, exact, and better than any learned approximation of it. Nothing here touches it.

This addendum covers the three places where the pipeline has a genuine **perception or judgement gap that geometry cannot close**, and specifies learned components to fill them:

| § | Component | Slots into | Type | Priority |
|---|---|---|---|---|
| A | Segmentation-driven render-and-compare | §6.1(b), §6.6 Filter 4 | Pretrained inference | **Build** |
| B | Learned confidence gate | §9 | **We train this** | **Build** |
| C | Monocular metric depth for height | §7.3 | Pretrained inference | Conditional |

## A.0 Why these three and not a learned placement solver

The temptation is to train a network to predict the placement transform directly. **Do not.** The fitting objective is IoU maximisation over a planar similarity, and §6.8's Umeyama solve is a closed-form global optimum of the correspondence-based version of that problem. A network trained on synthetic transforms would approximate, with more code and more failure modes, something already solved exactly — and would need a synthetic dataset whose corruption model you would have to invent, meaning it would be trained on your guesses about mesh defects rather than on real ones.

The correct rule for where ML belongs in this pipeline:

> **Geometry handles everything determined by the footprint. ML handles everything determined by the photograph, plus the accept/reject judgement.**

The footprint is two extents and an orientation modulo π/2. It cannot tell you which facade faces the street (§6.6), it cannot tell you the building's height (§7), and it cannot tell you whether a fit that scores IoU 0.72 should be trusted (§9). Those three gaps are exactly where the photograph and the benchmark carry information the polygon does not, and they are exactly the three components below.

State this boundary explicitly in the pitch. "We used pretrained models for image editing, mesh generation, and facade segmentation; the model we *trained* is the confidence gate, on our own benchmark." Judges reward a clean line here far more than they reward a network that shouldn't exist.

---

# A. Segmentation-driven render-and-compare
### Resolves: §6.1 up-axis, §6.6 four-fold azimuth ambiguity

## A.1 What problem this actually solves

§6.6 lists four filters for the four-fold rotation ambiguity. Filter 2 (EXIF heading) is arithmetic when present and absent whenever the photo was downloaded. Filter 3 (street-facing prior) is a prior, not evidence — it fails on corner lots, buildings set back from the road, and campus buildings whose "front" faces a quad rather than a street. Filter 1 (aspect ratio) excludes the 90° candidates only when the footprint is not near-square, which is precisely the case where the ambiguity matters least.

That leaves Filter 4 as the only cue that is *evidence* rather than prior, and the only one available on every input: **the photograph shows which facade was photographed.** The mesh was generated from that photograph, so the mesh's detailed side is the photographed side. Rendering the mesh from each candidate orientation and comparing its silhouette to the building's silhouette in the photo recovers the azimuth directly.

The segmentation model is what makes the comparison possible. Without a mask, you are comparing a clean render against a photo containing sky, trees, parked cars, and neighbouring buildings, and the IoU is meaningless.

**This is the single highest-value learned component in the project.** It is load-bearing on the main spec's hardest sub-problem, not a bolt-on.

## A.2 Model choice

**Primary: Grounded SAM 2 (Grounding DINO → SAM 2).**

Grounding DINO is a vision-language transformer that extends DINO with grounding through pre-training on image-text pairs, supporting zero-shot detection by accepting arbitrary class names as text input. You pass the text prompt `"building. house. facade."` and receive boxes with no training and no fixed class list.

SAM is an out-of-the-box segmentation model pretrained on a massive billion-image dataset, capable of segmenting most objects given point or bounding-box priors without further training. SAM 2 extends it to video while preserving image performance, and accepts clicks, boxes, or masks as prompts.

Chaining them is the standard pattern — Grounded SAM 2 exists specifically to combine Grounding DINO detection with SAM 2 segmentation in open-world scenarios, and both are supported in HuggingFace `transformers`. Grounding DINO supplies the box, SAM 2 refines it to a pixel-accurate mask.

**Fallback if the chain is slow to set up (~20 min budget):** SAM 2 alone with a **box prompt covering the central 80% of the image** plus a positive point click at the image centre. Your input photos are photos *of a building*, so the building is centred and dominant by construction. This drops the language model entirely and costs almost nothing in accuracy on this input distribution.

**Second fallback (no GPU headroom):** a semantic-segmentation model with a `building` class (SegFormer / DeepLabv3 trained on ADE20K or Cityscapes) — single forward pass, ~50 MB, no prompting. Lower boundary quality, entirely sufficient for silhouette IoU at the resolution this comparison needs.

**Do not fine-tune anything.** These models are used zero-shot. Fine-tuning a segmentation model on building facades in a 36-hour window is a guaranteed way to lose Saturday.

## A.3 Implementation

**Stage 1 — Segment the source photograph (once per asset, cached).**

```
masks = []
for photo in input_photos:
    box  = grounding_dino(photo, text="building. house. facade.")   # highest-score box
    mask = sam2(photo, box_prompt=box)                              # binary mask
    masks.append(clean(mask))
```

`clean()` = keep the largest connected component, morphological close with a ~5 px kernel to seal window gaps, fill interior holes. A facade mask with windows punched out of it will not match a render.

Store the mask as a PNG under the asset UUID (main spec §14, "log every intermediate artifact"). You will look at these masks more than any other debug artifact except the §9.3 overlay.

**Stage 2 — Render the mesh from candidate orientations.**

Use the main spec's own candidate set, not a fresh one:

- **Up-axis (§6.1):** 6 signed axis candidates, already narrowed by the prismatic-consistency score of §6.1(a). Take the top 2–3, not all 6 — the prismatic score is reliable enough to prune, and pruning cuts render count by half.
- **Azimuth (§6.6):** the 4 OMBB-derived candidates θ₀ + k·π/2.

Render orthographically, silhouette only (no texture, no lighting — you are comparing shape, and texture only adds noise). `pyrender` offscreen, or `trimesh`'s built-in rasteriser. ~8–24 renders at 256×256 is milliseconds of work.

**Critical: match the projection to the photo's geometry.** A street-level photograph is perspective, taken from roughly eye height, often tilted up. An orthographic render from horizontal is a different projection of the same object, and the silhouettes will not align even when the orientation is correct. Two options:

1. **Normalised-silhouette comparison (recommended, robust).** Do not compare raw masks. Extract each silhouette's **shape descriptor** — normalise by bounding box, then compare via IoU after centring and scaling to a common box. This discards absolute position and scale, which are exactly the quantities the projection mismatch corrupts, and keeps the aspect and profile, which are what discriminate orientation.
2. **Perspective render with an estimated camera.** If EXIF gives focal length, render perspective at that FOV from an elevation of ~1.6 m at the distance implied by the building's apparent size. Better in principle, more moving parts, more ways to be subtly wrong. Only do this if option 1 proves insufficient on your benchmark.

**Stage 3 — Score and select.**

```
for each (up_axis, azimuth) candidate:
    sil       = render_silhouette(mesh, candidate)
    score     = normalised_IoU(sil, photo_mask)
best, second  = top two by score
margin        = score(best) - score(second)
```

Feed `margin` into the confidence gate of §B as a feature, exactly as the main spec's §6.6 already specifies for its own rotation margin. Two margins now exist — the footprint-IoU margin across the 4 OMBB candidates, and this silhouette margin. **Keep both as separate features.** They are independent evidence, and disagreement between them is itself a strong signal that something is wrong.

**Fusion with the other filters.** Do not let render-and-compare silently override a confident EXIF heading. Precedence:

1. EXIF `GPSImgDirection` present and plausible → it decides; record `disambiguated_by: "exif_heading"`.
2. Otherwise silhouette score decides, *if* `margin ≥ 0.08`; record `"silhouette"`.
3. Otherwise fall back to the road-normal prior (§6.6 Filter 3); record `"road_normal"`, and **flag for review regardless of IoU** — you are now guessing.

## A.4 Where this fails, and what to do instead

| Failure | Why | Detection | Response |
|---|---|---|---|
| **Truly symmetric building** | A square building with four identical facades has no correct answer. The photo cannot disambiguate what is genuinely ambiguous. | silhouette margin < 0.05 **and** footprint aspect < 1.1 | Accept any candidate, record `"arbitrary_symmetric"`, route to review. This is correct behaviour, not a bug — say so at judging. |
| **Occlusion by trees/vehicles** | Mask captures the visible fragment; silhouette is not the building's. | Mask area ≪ box area, or mask has high boundary complexity relative to area | Prefer another photo. If none, fall back to road-normal and flag. Main spec §10.2 failure 11. |
| **SAM segments the wrong thing** | Prompt ambiguity — a neighbouring building or the whole streetscape. | Mask touches >3 image edges, or covers >90% of frame | Re-prompt with a tighter centre box; if it persists, fail loudly (main spec §14: no silent defaults). |
| **Multiple buildings in frame** | Row houses, campus blocks. | Grounding DINO returns several high-confidence boxes | Take the box with the largest area × centrality product. Main spec §10.2 failure 10. |
| **Bas-relief mesh** | A flat mesh's silhouette matches well from one view and is garbage from all others — it will *pass* this test while being catastrophically wrong. | §6.4 extent-ratio test | **Run §6.4 before render-and-compare, not after.** This is an ordering constraint: reject reliefs before you trust their silhouettes. |
| **Perspective mismatch** | §A.3 Stage 2 discussion. | Silhouette scores uniformly low (< 0.4) across *all* candidates | Switch to normalised-shape comparison if not already; if already, the mask is probably bad — inspect it. |
| **GPU memory pressure** | SAM 2 + TRELLIS + FLUX all resident on 8 GB. | CUDA OOM | Run segmentation **once, offline, at ingest**, cache the mask, free the model. Never hold two large models simultaneously. See §D. |

**The ordering constraint is the one to internalise:** §6.4 bas-relief rejection → segmentation → render-and-compare. Getting this backwards means a flat mesh can win the orientation vote.

---

# B. Learned confidence gate
### Replaces: §9.1's threshold table. **This is the component we train.**

## B.1 Why a model rather than thresholds

§9.1 specifies a table of hand-chosen cutoffs across seven metrics, with accept/review/reject bands. This works, and it is the right thing to ship first. Two things are wrong with it as a final answer:

1. **The thresholds are guesses.** 0.75 IoU for auto-accept is a reasonable number pulled from the air. Whether it is *the* number for this pipeline on these buildings is an empirical question, and §11 of the main spec already specifies building the benchmark that answers it.
2. **The metrics interact, and a table cannot express that.** IoU 0.72 on a highly rectilinear footprint with a large rotation margin is a good fit that the table sends to review. IoU 0.78 on a round building with a 0.02 rotation margin is a coin-flip the table auto-accepts. The rule that captures this is a weighted combination, which is exactly what a logistic regression *is*.

The main spec says: "Make the confidence model explicit and inspectable — do not hide it behind a single float." A logistic regression satisfies this **better** than the threshold table, because its coefficients are directly readable: you can state, with a number, how much rectilinearity matters relative to IoU. That is more inspectable than seven independently-chosen cutoffs, not less. Make this argument explicitly — it is the difference between "we replaced your thresholds with a black box" and "we replaced your thresholds with a fitted, inspectable rule."

## B.2 The data — which already exists in the plan

§11 of the main spec specifies a 20-building benchmark across four strata (rectangular, complex, near-square, sloped) with hand-annotated ground-truth placements, plus an ablation across four solver configurations. That is your training set, and it costs nothing extra:

- 20 buildings × 4 ablation configurations = **80 labelled placements**
- Plus every debug run you generate on Saturday — log all of them, labelled
- Plus deliberate corruptions: feed a known-wrong orientation, a mirrored mesh, a 2× scale error. These are free negatives, and they populate the "reject" class that your benchmark otherwise under-samples.

**Realistic target: 100–150 labelled rows.** That is small, and §B.4 addresses what that means.

**Labelling protocol.** Binary, from the §9.3 overlay PNG: *would a human accept this placement into the game world?* Two labels, not three — collapse "review" and "reject" into "not acceptable," because the three-way decision is recovered from the probability threshold, and a three-class model on 120 rows is over-parameterised. Label from the overlay image alone, before looking at the metrics, so the labels are not contaminated by the features.

## B.3 The model

**Features** (all already computed by the main spec's §9.1, plus §A):

| Feature | Source | Expected sign |
|---|---|---|
| `footprint_iou` | §9.1 | + |
| `hausdorff_m` | §9.1 | − |
| `area_ratio_log` — use `log(ratio)`, symmetric about 1 | §9.1 | ‖·‖ − |
| `rotation_margin_footprint` | §6.6 | + |
| `rotation_margin_silhouette` | §A.3 | + |
| `rectilinearity` | §4.2 | + |
| `anisotropy_log_ratio` | §6.9 | − |
| `max_neighbor_overlap` | §9.2 | − |
| `geocode_rooftop` (binary) | §3.1 | + |
| `height_source_authoritative` (binary) | §7 | + |

**Model: L2-regularised logistic regression**, `sklearn.linear_model.LogisticRegression(penalty='l2', C=...)`, on standardised features. Ten features on ~120 rows is already at the edge of what is supportable; regularisation is not optional, and `C` is chosen by cross-validation.

**Why not gradient-boosted trees:** with 120 rows they will overfit, they are less inspectable, and — critically — logistic regression is often already well-calibrated, whereas boosted trees have notoriously poor initial calibration and would need a separate calibration step you do not have data for. Regression is the right tool at this sample size.

**Calibration.** Probability calibration can lead to overfitting on small datasets, and Platt scaling is sensitive to the amount of calibration data, becoming unreliable when the calibration set is small. Isotonic regression is worse still at this scale — it is less constrained than Platt scaling and therefore easier to overfit when the calibration set is small (Niculescu-Mizil & Caruana find Platt superior below ~2000 cases; you have two orders of magnitude fewer).

**Therefore: fit logistic regression and use its probabilities directly. Do not add a calibration layer.** If a judge asks about calibration, the correct answer is that logistic regression trained under log-loss is calibrated by construction on its training distribution, and that a post-hoc calibration layer on ~120 samples would overfit — cite the small-calibration-set result. That answer demonstrates you understand calibration rather than that you bolted one on.

**Report a reliability diagram anyway** (predicted probability vs. observed frequency, ~4 bins) as a validation artifact. Four bins on 120 points is coarse but honest, and it is a strong thing to show.

**Thresholds.** Two operating points on the fitted probability, chosen from the leave-one-out ROC rather than by hand:

- `p ≥ p_accept` → auto-accept. Choose `p_accept` at **high precision** — the cost of a wrong auto-accept (a broken building permanently in the game world) is much higher than the cost of an unnecessary review.
- `p ≤ p_reject` → reject outright.
- Between → review queue.

State the asymmetry in the pitch. It shows you thought about the loss function, not just the accuracy.

**Validation:** leave-one-out CV at this sample size (not a held-out split — you cannot afford to spend 20 rows). Report LOO accuracy, precision at the accept threshold, and the coefficient table.

**Ship the coefficients as an artifact.** A table of standardised coefficients, sorted by magnitude, is the single most compelling slide in the confidence half of your pitch: *"rectilinearity and rotation margin dominate; raw IoU matters less than the thresholds implied."* That is an empirical finding about the problem, produced by your system.

## B.4 Where this fails, and what to do instead

| Failure | Why | Detection | Response |
|---|---|---|---|
| **Too few rows** | <60 labelled placements makes 10 features indefensible. | Count them. | Drop to 4 features (`iou`, `hausdorff`, `rectilinearity`, `rotation_margin`). A 4-feature model on 60 rows is defensible; a 10-feature model is not. |
| **Class imbalance** | If the solver works well, almost everything is labelled accept, and the model learns "always accept." | Check the label ratio. | Generate synthetic negatives by deliberate corruption (§B.2). Use `class_weight='balanced'`. Report the base rate honestly. |
| **Labels contaminated by features** | Labeller sees IoU 0.9 and labels "accept" because of the number, not the overlay. | Procedural — you cannot detect it after the fact. | Label from the overlay PNG only, features hidden. Enforce this from the first labelled row. |
| **Model disagrees with the spec's thresholds** | It will, somewhere. That is the point. | Compare decisions on the benchmark. | **Report the disagreement as a finding**, with examples. "Our fitted model accepts these 3 placements the threshold table rejects; here are the overlays — the model is right." This is the strongest possible use of the benchmark. |
| **Overfitting to Blacksburg** | 20 buildings from one campus is not a sample of buildings. | Inherent. | State it as a limitation. A model honest about its training distribution beats one that overclaims. |
| **No time to train it** | Saturday goes badly. | The clock. | **Ship §9.1's threshold table.** It works. The learned gate is an upgrade to a working system, never a dependency. Build the table first, always. |

**The non-negotiable:** §9.1's thresholds ship first and stay in the codebase as the fallback path. The learned gate is a second code path selected by a flag. If the model is not ready at hour 34, you flip the flag and demo the table.

---

# C. Monocular metric depth for height
### Slots into: §7.3. **Conditional — build only if §7.1 proves insufficient.**

## C.1 Decide whether to build this at all

§7 already specifies a four-tier priority for height: OSM `height` / `building:levels` → single-view metrology → monocular depth → proportional fallback. Microsoft's footprint dataset carries height estimates for over 174 million buildings, and OSM tags cover most well-mapped areas.

**Run this check in hour 1, before committing:** for each of your 5 demo addresses and 20 benchmark buildings, query OSM and Microsoft for a height. Count the hits.

- **≥ 80% coverage** → skip this section entirely. Build §A and §B properly instead. Spending four hours on depth estimation to serve 4 buildings out of 20 is a bad trade against a hackathon clock.
- **< 80%** → build it, as below.

Virginia Tech campus buildings are well-mapped in OSM. Expect high coverage and expect to skip this. **That is a finding worth reporting**, not a failure: "authoritative tags covered 19 of 20 benchmark buildings; we implemented monocular depth as the fallback path and exercised it on the remaining one."

## C.2 Model and method

**UniDepth (CVPR 2024) / UniDepthV2.** It predicts metric 3D points from a single image at inference time without any additional information, using a self-promptable camera module that predicts a dense camera representation to condition the depth features, and a pseudo-spherical output representation that disentangles camera and depth. **The no-intrinsics property is what makes it usable here** — your input photos are arbitrary uploads with no calibration.

Metric3D/Metric3Dv2 is the alternative, but it maps images into a canonical camera space and consumes intrinsics as an explicit input — worse for this use case.

**Method.** Reuse the §A mask; do not segment again.

```
depth   = unidepth(photo)                      # metric depth, metres, per-pixel
K       = unidepth.predicted_intrinsics        # dense camera representation
base_px = lowest point of building mask on the ground line
top_px  = highest point of building mask (roofline)
P_base  = backproject(base_px, depth[base_px], K)
P_top   = backproject(top_px,  depth[top_px],  K)
h       = |P_top.y - P_base.y|                 # in the gravity-aligned frame
```

**The gravity-alignment subtlety.** The depth model's camera frame is not gravity-aligned — a photo taken tilted up (which is nearly every photo of a building) puts "vertical" at an angle in camera coordinates. Taking a naive Y-difference in camera frame underestimates height by roughly `cos(tilt)`. Two fixes:

1. **EXIF pitch**, if present, rotates the camera frame to gravity-aligned. Cheapest correct answer.
2. **Fit the ground plane** from depth points inside the road/ground region below the building mask; its normal is gravity. More robust, ~20 lines with RANSAC.

Without one of these, expect a systematic 5–15% underestimate on typical street photos. Record which correction was applied.

## C.3 Where this fails, and what to do instead

| Failure | Why | Detection | Response |
|---|---|---|---|
| **Tall buildings saturate** | The main spec already flags this: fully supervised metric-depth models inherit the bounded range of their training sensors. UniDepth is trained largely on driving and indoor data; a 100 m tower is outside that range. | Predicted `h > 40 m`, or depth at the roofline pinned near the model's max | Do not trust it. Fall back to §7.4 proportional + slenderness sanity check, and flag. Depth estimation is for 2–10 storey buildings, which is what Scorched Nebraska needs anyway. |
| **Scale drift out of domain** | UniDepth occasionally struggles to capture specific scene scales in out-of-domain cases; UniDepthV2 was developed partly because V1 lacks accuracy in local fine-grained geometry. | Sanity band from §7.4: `2.5 m ≤ h ≤ 300 m`, slenderness `h/√area ∈ [0.1, 8]` | Reject out-of-band estimates and fall through the priority chain. Never let a depth estimate override an authoritative OSM tag. |
| **Roofline not visible** | Photo cropped, or shot from too close. | Building mask touches the top image edge | Unusable. Fall through to §7.4. Detect this *before* running the model and skip the inference entirely. |
| **Sky/featureless dominance** | Depth degrades in open scenes with large sky regions due to lack of features. | Mask area small relative to frame; large sky fraction | Prefer another photo; else fall through. |
| **Ground line occluded** | Cars, hedges, walls hide the base. | Bottom of mask is not adjacent to a ground-plane region | Estimate the base from the ground-plane fit rather than the mask, or fall through. |
| **Disagrees with OSM** | Both can be wrong; OSM is more often right. | `\|h_depth − h_osm\| / h_osm > 0.3` | **OSM wins.** The main spec's §7.1 priority is correct and this does not override it. Log the disagreement — a table of these is a good honest-results slide. |

## C.4 The cheaper alternative worth considering first

§7.2's single-view metrology needs a reference object of known height, and the main spec observes that a convenient one is always in frame: the ground floor, since standard storey heights are tightly distributed. Combined with a **floor-band count from the facade** — countable from window rows, which your segmentation already gives you a mask for — you get height as `n_floors × 3 m` with no depth model at all.

Criminisi's method also comes with a first-order error propagation analysis, meaning you can report an uncertainty rather than a bare number — which pairs naturally with the confidence gate of §B.

This is less accurate than metric depth but far cheaper to implement, needs no extra model in VRAM, and degrades gracefully. **If §C.1's coverage check says you need height estimation and the clock is tight, do this instead of UniDepth.**

---

# D. Integration, ordering, and resource budget

## D.1 Execution order (ordering constraints are load-bearing)

```
ingest photo
  └─ EXIF extraction (heading, pitch, focal)          — free, do first
  └─ §A segmentation → cache mask PNG                  — once per photo, then free the model
generate
  └─ image edit (FLUX)                                 — main spec §5.1
  └─ image → mesh (TRELLIS)                            — main spec §5.2
fit
  └─ §6.4 bas-relief rejection                         — MUST precede render-and-compare
  └─ §6.1(a) prismatic up-axis score → prune to 2–3
  └─ §A render-and-compare over remaining candidates   — uses cached mask
  └─ §6.5–6.9 geometric solve                          — UNCHANGED
height
  └─ §7.1 authoritative tags                           — try first, usually sufficient
  └─ §C only if tags missing
judge
  └─ §9.1 threshold table  [always available]
  └─ §B learned gate       [flag-selected upgrade]
```

**Two hard ordering constraints:**
1. **§6.4 before §A.** A bas-relief mesh will score well on silhouette comparison from its one good view. Reject reliefs first.
2. **Segmentation at ingest, not at fit time.** Both for VRAM (§D.2) and because the mask is an input to a step you will re-run many times while debugging the solver.

## D.2 VRAM budget — 8 GB

Three large models want to be resident: FLUX (edit), TRELLIS (lift), SAM 2 + Grounding DINO (segment), plus optionally UniDepth. **They will not co-reside on 8 GB.**

- **Never hold two at once.** Load → infer → `del model; torch.cuda.empty_cache()`.
- **Better: run edit and lift as hosted API calls** (Replicate, per the main spec's §5.2 notes) and keep only the segmentation model local. This is the recommended configuration — it removes the two largest models from your machine entirely and makes the local GPU load trivial.
- **Cache every intermediate to disk** under the asset UUID, per the main spec's §14. Masks, renders, depth maps. Re-running the solver should never re-run a model.
- **Pre-bake all demo assets** (main spec §15). At judging, nothing loads a model.

## D.3 Additions to the placement record (main spec §12.1)

```json
"perception": {
  "segmentation": {
    "model": "grounded-sam2", "version": "...",
    "prompt": "building. house. facade.",
    "mask_sha256": "...", "mask_area_frac": 0.41,
    "occlusion_flag": false
  },
  "orientation": {
    "silhouette_iou_best": 0.83,
    "silhouette_margin": 0.14,
    "footprint_margin": 0.21,
    "margins_agree": true,
    "disambiguated_by": "silhouette"
  },
  "height_ml": {
    "model": "unidepth-v2", "value_m": 13.9,
    "gravity_correction": "exif_pitch",
    "used": false, "reason": "osm_authoritative_available"
  }
},
"confidence": {
  "method": "logistic_regression",
  "model_sha256": "...", "n_train": 118,
  "p_accept": 0.81, "threshold_accept": 0.75, "threshold_reject": 0.25,
  "decision": "auto_accepted",
  "top_features": [["rectilinearity", 1.42], ["rotation_margin_silhouette", 1.18], ["footprint_iou", 0.94]]
}
```

`"used": false` with a reason is deliberate. Recording *why a component did not fire* is as valuable as recording that it did, and it demonstrates the priority chain is real rather than decorative.

## D.4 Revised build-plan insertions (main spec §15)

| Hours | Insertion | Notes |
|---|---|---|
| 0–2 | **§C.1 height-coverage check** | 20 minutes. Decides whether §C is built at all. Do it before anything else. |
| 6–12 | **§A segmentation + render-and-compare**, inside the "Real fitting" block | This is part of solving the four-fold ambiguity, not an addition to it. Budget ~3 h. |
| 18–24 | §C only if §C.1 said so | Otherwise this block shrinks and §B gets the time. |
| 24–30 | §9.1 threshold table (**first**), then §B learned gate | The table is the deliverable; the model is the upgrade. |
| 30–34 | §B training + LOO validation + coefficient table + reliability diagram | Uses the §11 benchmark already being built in this block. |

**The hour-6 milestone from the main spec is unchanged and still governs:** a plain cube landing correctly on a real building. None of this addendum runs before that milestone is met.

## D.5 What to say at judging

> "The geometry is closed-form — OMBB initialisation, Umeyama's similarity Procrustes, IoU refinement. We did not train a network to approximate a solved problem.
>
> We used learning in the three places the footprint carries no information. Which facade faces the street is determined by the photograph, so we segment the building with Grounded SAM 2 and rank candidate orientations by silhouette agreement — that is the only cue that resolves near-square footprints, and we report the margin. Absolute height isn't determined by the footprint either; authoritative OSM tags covered 19 of 20 benchmark buildings, and monocular metric depth is the fallback for the rest.
>
> The model we *trained* is the confidence gate. The spec called for hand-tuned thresholds; we fit a logistic regression on our 20-building benchmark instead, and the coefficients say rectilinearity and rotation margin matter more than raw IoU — which the thresholds couldn't have told us. We didn't add a calibration layer, because Platt scaling on 120 samples overfits.
>
> And we kept the threshold table as a fallback path, because a learned gate should be an upgrade to a working system, not a dependency."

---

# E. References (additional to the main spec)

**Segmentation and open-vocabulary detection**
- Kirillov, A. et al. *Segment Anything*. ICCV 2023. https://segment-anything.com
- Ravi, N. et al. *SAM 2: Segment Anything in Images and Videos*. arXiv:2408.00714. https://ai.meta.com/research/sam2/
- Liu, S. et al. *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*. arXiv:2303.05499
- Ren, T. et al. *Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks*. https://github.com/IDEA-Research/Grounded-Segment-Anything

**Metric depth**
- Piccinelli, L. et al. *UniDepth: Universal Monocular Metric Depth Estimation*. CVPR 2024. arXiv:2403.18913
- Piccinelli, L. et al. *UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler*. arXiv:2502.20110
- Yin, W. et al. *Metric3D / Metric3Dv2* — zero-shot metric depth via canonical camera space.
- Criminisi, A., Reid, I. & Zisserman, A. *Single View Metrology*. IJCV 40(2), 2000. (Already in main spec §17; relevant here for §C.4.)

**Calibration**
- Platt, J. *Probabilistic Outputs for Support Vector Machines*. 1999.
- Niculescu-Mizil, A. & Caruana, R. *Obtaining Calibrated Probabilities from Boosting*. UAI 2005. arXiv:1207.1403 — the small-calibration-set result cited in §B.3.

---

# F. Two-person work split and repository layout

Two contributors, split by domain: **ML/perception** (§A, §B, §C of this addendum) and **geometry/math** (§2–§9 of the main spec). The split is clean because the two halves meet at exactly three data structures. Freeze those, then never open each other's files.

## F.1 Hour 0–0.5 — the only file both people touch

Write `contracts.py` together, commit it, and **freeze it**. Every later change to it needs both people present, because it is the one file that can produce a real merge conflict.

```python
# contracts.py — frozen after hour 0.5. Changes require both contributors.
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Footprint:                 # produced by geometry
    pts_enu: np.ndarray          # (N,2) metres, local ENU frame
    rectilinearity: float        # §4.2
    ombb: tuple                  # (centre, (u,v), (a,b))  §4.4
    has_holes: bool

@dataclass(frozen=True)
class MeshOutline:               # produced by geometry
    pts_enu: np.ndarray          # (M,2) mesh ground outline, model units
    ombb: tuple
    is_bas_relief: bool          # §6.4 — set BEFORE perception runs

@dataclass(frozen=True)
class PhotoEvidence:             # produced by perception
    mask: np.ndarray             # binary building mask, HxW  §A.3
    mask_area_frac: float
    occlusion_flag: bool
    exif_heading_deg: float | None
    exif_pitch_deg: float | None
    silhouette_scores: dict      # {candidate_id: normalised_iou}  §A.3
    silhouette_margin: float

@dataclass(frozen=True)
class FitResult:                 # produced by geometry
    theta: float; scale_x: float; scale_y: float
    tx: float; ty: float; z_offset: float
    iou: float; hausdorff_m: float; area_ratio: float
    rotation_margin_footprint: float
    anisotropy_log_ratio: float
    max_neighbor_overlap: float
    disambiguated_by: str

# perception owns this signature
def score_confidence(fit: FitResult, photo: PhotoEvidence,
                     fp: Footprint) -> tuple[float, str]:
    """→ (probability, decision in {auto_accept, review, reject})"""
```

Also agree in this half hour on: the ENU frame convention (main spec §2.2), the candidate-ID scheme for orientations (`(up_axis_idx, azimuth_k)`), and which of the two of you owns `pipeline.py` — **recommend the geometry contributor**, since more stages are theirs.

## F.2 File ownership

Strict rule: **you do not open a file you do not own.** If you need a change in the other person's module, ask for it — do not make it.

**Perception / ML contributor**
```
perception/segment.py          §A.3 Stage 1 — Grounded SAM 2 → mask
perception/render_compare.py   §A.3 Stages 2–3 — silhouette scoring
perception/depth.py            §C — only if §C.1 coverage check says build it
confidence/features.py         §B.3 — (FitResult, PhotoEvidence, Footprint) → vector
confidence/train.py            §B.3 — logistic regression, LOO CV, coefficients
confidence/gate.py             §B — score_confidence(), with threshold-table fallback
```

**Geometry / math contributor**
```
geo/coords.py                  §2 — ENU, geoid correction
geo/footprint.py               §3–§4 — Overpass, conditioning, rectilinearity, OMBB
geo/outline.py                 §6.1–§6.4 — up-axis, grounding, outline, bas-relief
geo/fit.py                     §6.5–§6.9 — OMBB init, Umeyama, IoU refinement
geo/validate.py                §9.1–§9.2 — metrics, collision checks
geo/terrain.py                 §8 — elevation sampling, sloped sites
pipeline.py                    stage wiring (owned by geometry)
```

**Shared, write-once**
```
contracts.py                   frozen at hour 0.5
fixtures/                      test data — see §F.4
```

**Gitignored personal scratch:** `scratch_<name>.py`. Do all interactive debugging here. This is what keeps two people out of `pipeline.py` simultaneously.

## F.3 Hour-0 task list, per person

**Perception contributor — first three tasks, in order**

1. **§C.1 height-coverage check (20 min, do it first).** Query OSM and Microsoft footprints for heights across the 20 benchmark buildings. Count hits. This decides whether §C is built at all, and you want that answer before planning Saturday. Report the number to your partner — it also affects their §7 work.
2. **Segmentation on one fixed test photo.** A standalone script: JPG in, mask PNG out. No pipeline, no integration. Look at the mask. Confirm `clean()` (largest component, morphological close, hole fill) produces something sane.
3. **Render-and-compare against a dummy box.** You do not need a real generated mesh for this. Build a box in `trimesh`, render 4 azimuths, score normalised silhouette IoU against your test mask. This proves the whole §A machinery before either of you has real TRELLIS output.

**Geometry contributor — first three tasks, in order**

1. Pick the 5 demo addresses; fetch and cache their footprints; verify they exist (main spec §15, hours 0–2).
2. ENU frame + geoid correction, with the `|z| < 2500 m` assertion from §10.2 failure 13.
3. Drive toward the **hour-6 milestone**: a unit cube landing at the correct place, size, and rotation on a real building. This governs the whole project.

## F.4 Working around the one real dependency

**The confidence model cannot train until the solver produces benchmark runs.** That is a hard dependency from perception onto geometry, and it lands late (hour 30+).

Do not wait. Write `fixtures/fake_fits.py` — a generator producing synthetic `FitResult` objects with plausible metric distributions and hand-set labels. Build and test `features.py` and `train.py` entirely against these. When the real benchmark data arrives at hour 30, you swap the data source and the code is already correct and tested.

Same trick in reverse for the geometry side: `fixtures/fake_photo_evidence.py` returning a `PhotoEvidence` with a hardcoded mask and silhouette scores, so the fitting code can be exercised before segmentation is wired in.

**Fixtures are the merge-conflict insurance.** Each person develops against the other's fake data and integrates late.

## F.5 Sync points

Work independently between these; do not integrate continuously.

| Hour | Sync | Gate |
|---|---|---|
| 0.5 | `contracts.py` frozen | Both agree on the three dataclasses |
| 6 | **Cube milestone** (geometry) | A unit cube lands correctly on a real building. If this slips, both people work on it. |
| 12 | Perception checkpoint | Masks + silhouette scoring working on dummy meshes |
| 20 | **First real integration** | Real mesh → real outline → real fit → real confidence, end to end, one address |
| 30 | Feature freeze | No new features. Benchmark run + §B training only. |
| 34 | Demo freeze | All assets pre-baked, rehearsed twice |

**Escalation rule:** if the hour-6 cube milestone slips, the perception contributor stops §A work and helps. Placement is the project; everything in this addendum is an enhancement to it. The main spec is blunt about this and it is correct: *"Teams lose this project by spending Saturday tuning the redesign prompt while placement is still broken."*

## F.6 Git hygiene for two people

- **Branch per component**, not per person: `feat/segmentation`, `feat/fit-solver`. Merge to `main` at the sync points above, not continuously.
- **`.gitignore`**: venv, `__pycache__`, `*.ckpt/*.pth/*.safetensors` (model weights blow past GitHub's 100 MB limit), `.env`, `scratch_*.py`, generated meshes, cached masks.
- **Commit the footprint cache** (main spec §3.2 — Overpass will rate-limit you mid-demo). It is small and it is demo insurance.
- **Never commit model weights or generated `.glb` files.** Cache them to disk, gitignore them, and document how to regenerate.
