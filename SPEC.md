# Photo → Map-Ready 3D Building
Technical Specification, Procedura AI / Scorched Nebraska Challenge — VTHacks 14

Saha, Soumik · Wikel, Owen

## 0. Project brief

Build a system that takes a street address and one or more photographs of the building at that address, applies an AI-described redesign to those photographs, converts the redesigned imagery into a textured 3D mesh, and places that mesh back onto a live web map at the building's real-world location with correct position, scale, rotation, and ground alignment — automatically, without a human dragging it into place.

The sponsor context: Procedura is a spatial conversational development environment that organises work inside persistent geographic buckets and keeps generated state durable and replayable. Scorched Nebraska is a persistent geospatial game world built on Procedura that turns real places into explorable post-apocalyptic regions. A building you redesign becomes a permanent object in that world, so its mesh must match the real footprint while reflecting whatever the generative step produced.

The generative half of this problem is API orchestration and will work on day one. The entire engineering difficulty is in the placement. An image-to-3D model emits a mesh in arbitrary units, arbitrary orientation, with its origin in an arbitrary place, and with no knowledge of how big the real building is. Recovering the similarity transform that maps that mesh onto an authoritative building footprint on the WGS84 ellipsoid is the problem this document specifies.

Prize: internship opportunities for members of up to two teams. Deadline: 8 AM Sunday. Judging: 9 AM Sunday, NCB.

## 1. Problem formalisation

### 1.1 The pipeline as a composition of maps

Let:
- `a` — an address string
- `I = {I₁ … I_k}` — input photographs
- `π` — a natural-language redesign prompt

The system computes:

```
a  --geocode-->        (φ, λ)  ∈  WGS84 geodetic
(φ, λ) --footprint-->  F ⊂ ℝ²  authoritative building footprint polygon
(I, π) --edit-->       Ĩ       redesigned imagery
Ĩ --lift-->            M ⊂ ℝ³  mesh in arbitrary model units
(M, F) --fit-->        T ∈ Sim(3)  placement transform
```

and emits the pair `(M, T)` — never a pre-transformed mesh. Storing the transform separately is what makes the result reproducible, auditable, and adjustable, which the challenge explicitly requires.

### 1.2 The fitting problem, stated precisely

Work in a local East-North-Up (ENU) metric frame anchored at the footprint centroid (§2). Let:
- `F ⊂ ℝ²` — the real footprint, a simple polygon (possibly with holes), in metres
- `M₀ ⊂ ℝ²` — the mesh's ground-plane outline, in model units

Find the planar similarity `T(x) = s·R(θ)·x + t` with `s > 0`, `R(θ) ∈ SO(2)`, `t ∈ ℝ²` maximising the Jaccard index

```
J(s, θ, t) = area( T(M₀) ∩ F ) / area( T(M₀) ∪ F )
```

subject to a vertical lift `h` and a ground offset `z₀` determined separately (§7, §8).

This objective is non-convex, non-differentiable, and has a large basin structure. It is piecewise-smooth in `(s, θ, t)` with kinks at every vertex-edge incidence event. Gradient descent from a random start fails. The strategy in §6 is therefore: closed-form geometric initialisation → discrete disambiguation → local continuous refinement → optional certified global search.

Reflections are excluded. We optimise over SO(2), not O(2) — a building is a physical object, not a shape-matching exercise. However, generated meshes occasionally come out mirrored (a flipped source image, or a generator that emits a left-handed basis). Detect this with `det(R) < 0` after the estimation step in §6.5 and reject rather than accept the mirror.

## 2. Coordinate systems

Getting this layer wrong invalidates everything downstream, and the errors are silent — a building that is 40 cm off or 33 m underground looks like a bug in the fitting code.

### 2.1 Why not UTM

UTM is the obvious choice and it is wrong here for two reasons: the transverse Mercator scale factor is 0.9996 at the central meridian and rises above 1.0 toward the zone edges, introducing up to roughly 1 part in 2500 of linear distortion — about 4 cm on a 100 m building, tolerable — and, worse, zone boundaries create discontinuities that will bite you at scale. For a single building, a local tangent plane is exact enough and has no seams.

### 2.2 Geodetic → ECEF → ENU

WGS84 constants:
```
a  = 6378137.0 m                (semi-major axis)
f  = 1/298.257223563            (flattening)
e² = 2f − f² = 6.69437999014e-3 (first eccentricity squared)
```

Prime vertical radius of curvature at latitude φ:
```
N(φ) = a / sqrt(1 − e² sin²φ)
```

Geodetic (φ, λ, h) → ECEF:
```
X = (N(φ) + h) cos φ cos λ
Y = (N(φ) + h) cos φ sin λ
Z = (N(φ)(1 − e²) + h) sin φ
```

ENU rotation at origin (φ₀, λ₀):
```
        ⎡ −sin λ₀            cos λ₀           0      ⎤
R_ENU = ⎢ −sin φ₀ cos λ₀   −sin φ₀ sin λ₀   cos φ₀ ⎥
        ⎣  cos φ₀ cos λ₀    cos φ₀ sin λ₀   sin φ₀ ⎦

p_ENU = R_ENU · (p_ECEF − p₀_ECEF)
```

All footprint geometry, all mesh fitting, and all metric reasoning happen in this ENU frame. Convert back only at render time.

### 2.3 The height datum trap

Digital elevation models publish orthometric heights (height above the geoid: NAVD88 in the US, EGM96/EGM2008 globally). CesiumJS, glTF, and WGS84 all want ellipsoidal heights. The difference is the geoid undulation `N_geoid`:

```
h_ellipsoidal = H_orthometric + N_geoid
```

In Virginia `N_geoid` is roughly −30 m. Ignore it and every building in your demo is buried three storeys underground. Compute it properly with pyproj's geoid grids or accept Cesium's terrain provider, which already serves ellipsoidal heights — but never mix the two sources without converting.

### 2.4 Angle conventions

Three conventions will collide in this codebase. Write them down once and convert at the boundaries:

| Frame | Zero direction | Positive sense |
|---|---|---|
| Mathematical θ (our fitting) | +East | counter-clockwise |
| Compass / Cesium heading | North | clockwise |
| glTF model space | +Z forward, +Y up | right-handed |

```
heading = π/2 − θ      (radians, wrapped to [0, 2π))
```

## 3. Stage A — Address resolution and footprint retrieval

### 3.1 Geocoding

Nominatim (free, OSM-backed) or Google Geocoding (better). Both return a point, and that point is not the building. Geocoders commonly return a street-centreline interpolation, a parcel centroid, or a rooftop point depending on provider and address quality. Record which — Google returns `location_type ∈ {ROOFTOP, RANGE_INTERPOLATED, GEOMETRIC_CENTER, APPROXIMATE}` — and propagate it into the confidence model.

### 3.2 Footprint sources

OpenStreetMap via Overpass, which carries semantic tags you will want later:

```
[out:json][timeout:25];
(
  way["building"](around:60, {lat}, {lon});
  relation["building"](around:60, {lat}, {lon});
);
out geom;
```

Microsoft GlobalMLBuildingFootprints as the fallback. Roughly 1.4 billion footprints detected from Bing Maps imagery between 2014 and 2024 (Maxar, Airbus, Vexcel, IGN France), released under CDLA Permissive 2.0. Each feature carries a `height` property in metres (−1 when unknown) and a confidence score. Height estimates exist for over 174 million buildings — a free, authoritative answer to the hardest sub-problem in §7. Data is partitioned by country and quadkey, line-delimited GeoJSON, gzip-compressed with a misleading `.csv.gz` extension.

**Selection rule — do not blindly take the nearest polygon:**
1. Point-in-polygon test against the geocoded point. If exactly one hit, take it.
2. Otherwise, candidates within 50 m, ranked by: `name`/`addr:housenumber` tag match, then centroid distance, then area plausibility.
3. Zero candidates → fall back to the other source; still zero → explicit failure with a clear message. Do not synthesise a footprint.

**Operational warning:** the public Overpass endpoint is a shared free service and will hand you 429 and 504 under any load. For the demo, pre-fetch and cache every building you intend to show, keyed by quadkey, and commit the cache to the repo.

### 3.3 Polygons with holes

Courtyard buildings arrive as GeoJSON polygons with interior rings; OSM multipolygon relations encode them as outer/inner members. Every area computation, IoU, and boundary distance downstream must respect interior rings. Shapely does this correctly; hand-rolled shoelace code does not.

## 4. Stage B — Footprint conditioning

Raw footprints — especially ML-derived ones — carry jagged boundaries, redundant vertices, and near-but-not-quite right angles. Conditioning them before fitting improves the rotation estimate materially.

### 4.1 Signed area and centroid

Shoelace, with vertices in order:
```
A  = ½ Σᵢ (xᵢ y_{i+1} − x_{i+1} yᵢ)

Cₓ = (1/6A) Σᵢ (xᵢ + x_{i+1})(xᵢ y_{i+1} − x_{i+1} yᵢ)
C_y = (1/6A) Σᵢ (yᵢ + y_{i+1})(xᵢ y_{i+1} − x_{i+1} yᵢ)
```
Sign of A gives winding order. Normalise to counter-clockwise for the outer ring.

### 4.2 Principal orientation via circular statistics

This is the single most valuable number you extract from the footprint, and the naive approach (take the longest edge) is fragile. Buildings are approximately Manhattan: their edges cluster around two perpendicular directions. Because directions are equivalent modulo π/2, map each edge angle θᵢ into a quadruple-angle representation, which collapses the four-fold symmetry to a single circular mean:

```
Let edge i have direction θᵢ = atan2(Δyᵢ, Δxᵢ) and length Lᵢ.

C = Σᵢ Lᵢ cos(4θᵢ)
S = Σᵢ Lᵢ sin(4θᵢ)

θ* = ¼ · atan2(S, C)          (principal axis, modulo π/2)
```

The rectilinearity of the footprint falls out of the same computation as the resultant length:
```
R = sqrt(C² + S²) / Σᵢ Lᵢ   ∈ [0, 1]
```

`R ≈ 1` means a strongly rectilinear building and a well-conditioned rotation estimate. `R ≲ 0.6` means a curved, organic, or round building — the rotation is intrinsically ill-posed and the fit should be flagged for review regardless of its IoU. This scalar is your cheapest and best early-warning signal, and it costs one pass over the edges. It is the von Mises concentration parameter of the edge-direction distribution in disguise.

### 4.3 Simplification and regularisation

Douglas–Peucker with ε ≈ 0.3–0.5 m removes vertex noise without destroying real corners. For ML-derived footprints, a proper regularisation step — snapping edges to the principal directions found in §4.2, optionally allowing 45° — measurably improves both visual quality and downstream alignment metrics. `buildingregulariser` (PyPI) implements this against GeoPandas and is a reasonable drop-in. Recent work regresses the two primary orientation angles continuously and then inserts vertices so consecutive edges are perpendicular, at roughly 3 ms per building.

**Caution:** regularise a *copy* used for fitting. The authoritative footprint stays untouched in the provenance record.

### 4.4 Oriented minimum bounding box

The OMBB provides the closed-form initialisation in §6.2. Use rotating calipers, which exploits the theorem that the minimum-area rectangle enclosing a convex polygon has a side collinear with one of the polygon's edges (Freeman & Shapira 1975; Toussaint 1983). This reduces the continuous search over orientations to n candidates and yields an O(n) algorithm after the convex hull.

Output: centre `c_F`, orthonormal axes `(u_F, v_F)`, extents `a_F ≥ b_F`.

`cv2.minAreaRect` and `shapely.minimum_rotated_rectangle` both implement this. Note that minimum-area and minimum-perimeter bounding rectangles can differ in orientation by nearly 45° for some shapes — pick one and be consistent.

## 5. Stages C & D — Generation

### 5.1 Image editing

FLUX.1 Kontext (Black Forest Labs) is a rectified-flow transformer that unifies text-to-image synthesis and instruction-guided editing in one backbone, operating in the latent space of a learned autoencoder. It is designed for surgical edits — you state what to change, the rest stays put — and reports strong identity preservation across multi-turn edits, generating 1024 px images in roughly 3–5 seconds. Gemini's image models and SDXL img2img are alternatives.

Its documented failure modes matter to you: excessive multi-turn editing introduces artifacts, and the model sometimes ignores parts of the instruction.

**Prompt engineering for this task.** The redesign prompt is not free-form. You are about to fit the resulting mesh to a fixed footprint, so any edit that changes the silhouette breaks the fit (see §10.1). Constrain prompts to surface-level transformations:

- Good: "weathered concrete, broken windows, ivy overgrowth, scorched upper floors, rusted fixtures"
- Dangerous: "add a collapsed tower", "extend the west wing", "partially demolished"

Bake this into the UI as a hint, or post-hoc reject edits whose segmented silhouette IoU against the original falls below a threshold.

**Multi-view consistency.** If you accept several photographs, edit them jointly or with a shared seed and a reference-image conditioning path. Independently edited views will disagree, and the image-to-3D stage will either average them into mush or pick one and ignore the others.

### 5.2 Image → mesh

TRELLIS (Microsoft, CVPR 2025) is the strongest open option. It encodes assets into a Structured LATent representation `z = {(zᵢ, pᵢ)}` — local latent vectors anchored at sparse surface voxels on a 64³ grid — and generates in two rectified-flow stages: a sparse-structure stage producing the voxel scaffold, then a structured-latent stage filling in geometry and appearance. It decodes to Radiance Fields, 3D Gaussians, or meshes. The image-conditioned variant substantially outperforms the text-conditioned one, and it accepts multiple input images for multi-view conditioning without a separate model variant. TRELLIS.2 (4B parameters) supersedes it with an O-Voxel representation handling open surfaces and non-manifold geometry.

Practical knobs on the Replicate deployment: two-stage sampling at 12 steps each by default, independent guidance strengths (7.5 and 3), baked textures 512–2048 px, quadric mesh simplification retaining 90–98% of triangles, GLB export.

Alternatives: Hunyuan3D, Tripo (hosted API, fastest to integrate), InstantMesh.

Budget 10–60 seconds per generation on a GPU. **Make it asynchronous from the first commit.** A synchronous HTTP handler that blocks on this will destroy your demo.

## 6. Stage E–F — Canonicalisation and the placement solve

This is the core of the project. Sections 6.1–6.4 are prerequisites; 6.5–6.9 are the actual solve.

### 6.1 Up-axis recovery

glTF specifies a right-handed system with +Y up and the asset front facing +Z. Generators mostly respect this; not always. Three cues, in increasing order of reliability:

**(a) Prismatic-consistency score.** Buildings are, to first order, extruded footprints — their cross-section is nearly constant along the vertical. For each candidate axis u (the six signed principal axes from PCA, or the six signed coordinate axes), slice the mesh into k ≈ 20 slabs along u and compute the convex-hull area `A_j` of each slab's vertex projection. Score by the negative coefficient of variation:

```
score(u) = − std({A_j}) / mean({A_j})
```

The true vertical maximises this score for any prismatic object. Unlike the naive face-normal-area heuristic, it does not flip sign between tall and squat buildings.

**(b) Render-and-compare.** Render the mesh orthographically from each of the ~24 axis-aligned orientations and score the silhouette IoU against the building mask segmented from the source photograph. This is the most reliable cue because it uses information the geometry alone does not contain, and it simultaneously resolves the azimuth ambiguity of §6.6. Cost: 24 cheap offscreen renders.

**(c) Learned canonicalisation.** Fu et al. (SIGGRAPH 2008) established the modern framing: PCA alone does not resolve orientation, it merely reduces the candidate set to six; their method generates candidates from static-equilibrium poses on the convex hull and learns an assessment function over shape-to-function attributes, reaching roughly 90% accuracy. Later work (UprightRL, Upright-Net) replaces the classifier with learned point-cloud models. Almost certainly out of scope for 36 hours — cite it, don't build it.

### 6.2 Pivot normalisation and grounding

Translate so the mesh's ground-plane centroid sits at the origin and its base sits at y = 0.

Use a **robust minimum**, not `min()`. Generated meshes routinely carry a handful of stray vertices or thin spikes below the main body. Take the 1st percentile of the vertex heights, or the mode of a fine histogram of the bottom 10%. A single outlier vertex 2 m below the base will float the entire building 2 m into the air, and this failure is nearly invisible until someone looks at the model from the side.

### 6.3 Extracting the mesh's ground outline M₀

Two approaches; the second is more robust on generated geometry.

**Alpha shapes.** Select vertices with `y < y_min + δ` (δ ≈ 10–20% of height), project to the XZ plane, and compute the α-shape. Introduced by Edelsbrunner, Kirkpatrick & Seidel (1983), the α-shape generalises the convex hull: α → ∞ recovers the convex hull, α → 0 degenerates to the point set itself. The standard implementation Delaunay-triangulates the points and keeps triangles whose circumradius is below α. Initialise α ≈ 2–3× the median Delaunay edge length.

The α-shape's weakness is exactly your use case: a single global α handles uniformly-dense convex point sets well but degrades on non-uniform density and on concave corners — L-shaped and courtyard buildings, in other words.

**Orthographic rasterisation (recommended).** Render the mesh top-down into a binary occupancy grid at ~10 cm/pixel, apply a morphological close to seal small gaps, extract contours with marching squares, and simplify with Douglas–Peucker. This is indifferent to non-manifold geometry, self-intersections, and disconnected components — all of which generated meshes produce — and it naturally yields interior rings for courtyards. It is also trivially parallel and about 20 lines of code with trimesh + scipy + skimage.

Compute the OMBB of M₀ exactly as in §4.4: `c_M`, `(u_M, v_M)`, `a_M ≥ b_M`.

### 6.4 Bas-relief detection

Some image-to-3D pipelines, given a single frontal photograph, produce a flattened single-view relief rather than a closed volume. Detect it **before** fitting: compute the mesh's extent along its two horizontal principal axes and reject if

```
min(extent) / max(extent) < 0.25
```

for a building whose real footprint is not correspondingly elongated. A relief will "fit" the footprint with a plausible IoU while looking catastrophically wrong in 3D.

### 6.5 Closed-form initialisation via OMBB alignment

Align principal axes:
```
θ₀ = angle(u_F) − angle(u_M)

s_iso = ½ (a_F/a_M + b_F/b_M)          uniform scale
t₀    = c_F − s_iso · R(θ₀) · c_M
```

This is exact when both shapes are rectangles and near-optimal when both are approximately rectilinear — which §4.2's rectilinearity R tells you in advance.

Because the OMBB is defined up to a relabelling of its axes, this produces four candidates:
```
θ_k = θ₀ + k·π/2,   k ∈ {0, 1, 2, 3}
```
with k ∈ {1, 3} additionally swapping the extent pairing in s.

### 6.6 Resolving the four-fold ambiguity

This is the interesting sub-problem. IoU alone discriminates well for asymmetric footprints and fails completely for square ones.

**Filter 1 — aspect ratio.** If `a_F/b_F > 1 + τ` with τ ≈ 0.1, the 90° candidates are already excluded on scale grounds. Two candidates remain (front vs. back).

**Filter 2 — photo EXIF.** If the source photograph carries GPS coordinates and a compass heading (`GPSImgDirection`), the azimuth from camera to building centroid directly determines which facade was photographed, which fixes the remaining ambiguity outright. Check for this first: it converts a hard inference problem into arithmetic. Phone photos frequently have it; downloaded images never do.

**Filter 3 — street-facing prior.** Query nearby OSM ways with `highway=*`, find the nearest road segment to the footprint, compute the outward normal from the footprint boundary toward it. The primary facade faces the street with high prior probability. Score each candidate orientation by the alignment of the mesh's most detailed side with that normal.

Quantify "most detailed" per candidate facade: project the mesh onto each of the four side planes and measure vertex density, normal-direction variance, or texture entropy in the baked atlas. Windows, doors, and trim concentrate geometric and textural detail on the facade; the back and roof of a single-photo generation are hallucinated and smooth. This works surprisingly well and is cheap.

**Filter 4 — render-and-compare.** As in §6.1(b), the silhouette IoU against the segmented source photograph resolves this directly if you have already built that machinery.

**Ambiguity margin.** Whatever combination you use, record
```
margin = IoU(best) − IoU(second best)
```
and route anything with `margin < 0.05` to manual review. Near-square buildings will dominate this bucket, and that is the correct behaviour.

### 6.7 Cheap global ranking via turning functions

Before spending IoU evaluations, rank candidates with the turning function distance (Arkin, Chew, Huttenlocher, Kedem & Mitchell, TPAMI 1991). This metric is invariant under translation, rotation, and scale, is a true metric (satisfies the triangle inequality), handles convex and non-convex polygons, and computes in O(mn log mn).

For a polygon A, the turning function `Θ_A(s)` gives the cumulative turning angle at normalised arc length `s ∈ [0,1]`; for polygons it is a step function. The distance is

```
d(A,B) = min over θ∈ℝ, t∈[0,1] of  ( ∫₀¹ |Θ_A(s+t) − Θ_B(s) + θ|² ds )^(1/2)
```

For fixed shift t the optimal rotation θ* is closed-form:
```
θ*(t) = ∫₀¹ [Θ_B(s) − Θ_A(s+t)] ds
```
so the minimisation reduces to the O(mn) critical values of t where step discontinuities coincide. Because the turning function is a step function, the integral evaluates in closed form and the whole computation is fast.

Its limitation is worth knowing: the arc-length parameterisation is normalised to unit perimeter, so boundary noise — a small protrusion on one shape — redistributes arc length globally and perturbs the match. Simplify (§4.3) before computing it.

Use this as a screen, not as the final objective. The thing you actually care about is area overlap, which is what IoU measures.

### 6.8 Continuous refinement

**Correspondence-based (Kabsch–Umeyama).** Once roughly aligned, densely resample both boundaries, associate each resampled mesh-boundary point `xᵢ` with its nearest point `yᵢ` on `∂F`, and solve the similarity Procrustes problem in closed form (Umeyama, TPAMI 1991):

```
μₓ = (1/n) Σ xᵢ              μ_y = (1/n) Σ yᵢ
σₓ² = (1/n) Σ ‖xᵢ − μₓ‖²
Σ  = (1/n) Σ (yᵢ − μ_y)(xᵢ − μₓ)ᵀ

SVD:  Σ = U D Vᵀ,   D = diag(d₁ ≥ d₂ ≥ 0)

S = I                     if det(U)·det(V) ≥ 0
S = diag(1, −1)           otherwise

R = U S Vᵀ
s = tr(D S) / σₓ²
t = μ_y − s R μₓ
```

The S correction is Umeyama's specific contribution over Arun et al. and Horn: without it, the SVD solution silently returns a reflection instead of a rotation when the data is badly corrupted. For this pipeline that means a mirrored building that passes every numeric check. **Do not omit it.**

Alternate correspondence and solve — this is 2-D ICP. It converges locally and quickly, which is exactly why §6.5's initialisation matters.

**Direct IoU maximisation.** Polygon IoU is piecewise-smooth, so derivative-free local methods work: Nelder–Mead or Powell over `(θ, tₓ, t_y, log s)`, with Shapely evaluating the objective. Parameterise scale logarithmically — it makes the step size scale-invariant and keeps s > 0 without a constraint. Budget 200–500 evaluations; each is sub-millisecond for polygons under a few hundred vertices.

Run ICP first (fast, gets you close), then IoU refinement (slower, optimises the metric you are actually judged on).

### 6.9 Anisotropic scaling, properly regularised

The challenge specifies: prefer proportion-preserving uniform scaling, allow limited non-uniform scaling when necessary. Formalise "limited" rather than hand-tuning it.

Let `S = diag(sₓ, s_y)`. The natural penalty on anisotropy is the squared log-ratio — symmetric under swapping the axes, scale-invariant, and zero exactly at isotropy:

```
minimise   −J(sₓ, s_y, θ, t)  +  λ · ( log(sₓ/s_y) )²

subject to  |log(sₓ/s_y)| ≤ log(1 + ε),    ε ≈ 0.15
```

λ ≈ 0.5 and ε = 0.15 are reasonable starting points; calibrate against §11's benchmark. Report the realised anisotropy in the placement record so a reviewer can see when the solver leaned on it.

The vertical scale `s_z` is not free — see §7.

### 6.10 Certified global optimality (stretch goal)

If you want a genuinely differentiating technical result, replace the heuristic candidate enumeration with branch-and-bound over the transform space, adapting the Go-ICP construction (Yang, Li, Campbell & Jia, TPAMI 2016) from SE(3) to Sim(2). Go-ICP guarantees global optimality of the L₂ registration error by BnB over SE(3) with bounds derived from uncertainty radii, integrating local ICP to tighten the upper bound.

The 2-D specialisation is considerably simpler. For a rotation interval of half-width `σ_θ` centred at `θ₀`:
```
‖R(θ)x − R(θ₀)x‖ ≤ 2‖x‖ sin(σ_θ/2) ≤ ‖x‖ σ_θ   ≕ γ_θ(x)
```
For a translation square of half-side `σ_t`:
```
‖t − t₀‖ ≤ σ_t √2   ≕ γ_t
```
Giving the admissible lower bound over a box `B = [θ₀±σ_θ] × [t₀±σ_t]`:
```
L(B) = Σᵢ max( 0, d(R(θ₀)xᵢ + t₀, F) − γ_θ(xᵢ) − γ_t )²
```
with the upper bound `U(B) = Σᵢ d(R(θ₀)xᵢ + t₀, F)²` obtained by evaluating at the box centre, optionally refined by local ICP. Best-first search on a priority queue ordered by `L(B)`, terminating when `U* − L(B) < ε`. Precompute `d(·, F)` as a distance transform on a grid.

This gives you a certificate — a provable bound on how far your placement is from the global optimum of the chosen objective — which is exactly the kind of claim that distinguishes a hackathon project at judging. Second-order lower bounds for 2-D registration exist if the first-order one proves too loose.

**Budget this for hour 30+, not hour 10. Ship the heuristic first.**

## 7. Stage G — Height estimation

Image-to-3D produces no absolute scale. The footprint supplies two constraints — two in-plane extents — leaving the vertical unconstrained. A two-storey house and a twenty-storey tower with the same floor plan are indistinguishable to the footprint fit. Resolve in this priority order.

### 7.1 Authoritative tags

OSM `height` (metres) is definitive when present. Otherwise `building:levels` gives the number of above-ground non-roof storeys; the OSM convention is 3 m per level as the default for 3D rendering when explicit height tags are absent, and the wiki notes this is a good approximation for most buildings. Some renderers assume 4 m — pick 3 m and record the assumption.

Microsoft footprints carry a `height` property for over 174 million buildings, −1 when unknown. Check both sources; prefer OSM when they disagree and OSM has an explicit height.

### 7.2 Single-view metrology

When no tag exists and you have a street-level photograph, the classical result applies. Criminisi, Reid & Zisserman (ICCV 1999 / IJCV 2000) showed that 3-D affine measurements are recoverable from a single perspective view given only the vanishing line `l` of a reference plane and a vanishing point `v` for a direction not parallel to it — without knowing the camera's internal calibration or its pose relative to the world. Given those, one can compute distances between planes parallel to the reference plane up to a common scale, length and area ratios on any such plane, and the camera's location.

For a vertical object with base b and top t (homogeneous image coordinates), and a reference vertical of known height `Z_ref` with base `b_r`, top `t_r`:

```
Z          ‖b × t‖ · (l · b_r) · ‖v × t_r‖
──── = ───────────────────────────────────────
Z_ref      ‖b_r × t_r‖ · (l · b) · ‖v × t‖
```

The method comes with a first-order error propagation analysis, which is unusual and useful — you can report an uncertainty on the estimated height rather than a bare number.

Modern pipelines automate the geometric inputs: extract vanishing points, line segments, and semantic segmentation maps with deep networks, then apply single-view metrology. Reported approaches cover facade parsing to count floors, corner-based estimation, and semi-supervised regression from Mapillary street-view plus OSM morphometric features.

A convenient reference object is always in frame: the ground floor. Standard storey heights are tightly distributed, so detecting one floor band gives you `Z_ref`.

### 7.3 Monocular metric depth

The modern shortcut. UniDepth (CVPR 2024) predicts metric depth without requiring camera intrinsics at test time: it uses a self-promptable camera module that predicts a dense camera representation to condition the depth features, and a pseudo-spherical output representation (azimuth, elevation, log-depth) that disentangles camera parameters from depth. Metric3D / Metric3Dv2 achieve the same by transforming images into a canonical camera space, but consume intrinsics as an explicit input.

Given metric depth d at the building base and top pixels and the (predicted or known) intrinsics, back-project both to 3-D and take the vertical difference. Note the caveat from the literature: fully supervised metric-depth models inherit the bounded range of the sensors in their training data, so very tall buildings may saturate.

### 7.4 Fallback

Preserve the mesh's intrinsic proportions under the uniform footprint scale. Then sanity-check: `2.5 m ≤ h ≤ 300 m`, and the slenderness `h / sqrt(area(F))` should fall in [0.1, 8] for anything short of a supertall.

**Always record which of the four sources produced the height.**

## 8. Stage H — Terrain and ground alignment

Sample elevation at the footprint vertices plus its centroid:

- CesiumJS: `sampleTerrainMostDetailed(terrainProvider, positions)` against Cesium World Terrain — already ellipsoidal, no geoid correction needed.
- US, higher resolution: USGS 3DEP 1 m DEM. Orthometric (NAVD88) — apply §2.3.

Take a robust statistic across the footprint, not a single centroid sample. Recommended: the 25th percentile of the sampled elevations, which approximates the downhill side of a pad-on-grade building without being hostage to a single spurious sample.

**Sloped sites.** Do not tilt the building — real buildings are level, and a tilted one reads as broken instantly. Two correct options: (a) set the base to the low side and accept that the uphill side is partially buried, which is what real buildings do; (b) extend a skirt of untextured geometry from the base polygon down to the terrain surface. Option (a) is one line; option (b) looks better and takes an hour.

**Terrain LOD interaction.** Cesium streams terrain progressively, so a building placed against a coarse tile will visibly jump when the fine tile loads. Sample at maximum detail and pin the height; do not re-sample on camera movement.

## 9. Validation, confidence, and the review queue

The challenge explicitly requires validating footprint overlap and nearby collisions, and sending uncertain results to manual review. Make the confidence model explicit and inspectable — do not hide it behind a single float.

### 9.1 Metrics

| Metric | Definition | Auto-accept | Review | Reject |
|---|---|---|---|---|
| Footprint IoU | `\|T(M₀) ∩ F\| / \|T(M₀) ∪ F\|` | ≥ 0.75 | 0.50–0.75 | < 0.50 |
| Symmetric Hausdorff | `max(sup_{a∈∂A} d(a,∂B), sup_{b∈∂B} d(b,∂A))` | ≤ 2.0 m | 2–5 m | > 5 m |
| Area ratio | `\|T(M₀)\| / \|F\|` | 0.85–1.15 | 0.7–1.3 | outside |
| Rotation margin | IoU₁ − IoU₂ over the 4 candidates | ≥ 0.05 | < 0.05 | — |
| Rectilinearity | R from §4.2 | ≥ 0.75 | < 0.75 | — |
| Anisotropy | `\|log(sₓ/s_y)\|` | ≤ 0.05 | 0.05–0.14 | ≥ 0.15 |
| Neighbour overlap | `max_j \|T(M₀) ∩ F_j\| / \|T(M₀)\|` | ≤ 0.02 | 0.02–0.10 | > 0.10 |

IoU and Hausdorff are complementary and you need both. IoU is an area measure and is dominated by the bulk of the shape; a building whose main mass is correctly placed but whose 8-metre wing points the wrong way can still score IoU 0.82. Hausdorff is a worst-case boundary measure and catches exactly that. A fit passes only if both pass.

### 9.2 Collision checking

Fetch neighbouring footprints within 100 m in the same Overpass call. Reject placements that overlap a neighbour by more than a small tolerance. Also check against highway buffer polygons — a building encroaching into the road is an obvious visual failure in a game world.

### 9.3 Review payload

Every flagged placement emits a review record containing the full transform, every metric above, a top-down overlay PNG of F and T(M₀) in contrasting colours, and the four candidate orientations with their scores so a human can pick the right one in a single click rather than re-running the solver. **The overlay image is the highest-value artifact in your entire debugging workflow — build it in hour three, not hour thirty.**

## 10. Failure-mode catalogue

Enumerated because the sponsor's slide explicitly lists validation and manual review as deliverables — demonstrating that you know how your own system fails is worth more at judging than a demo that only works on one address.

### 10.1 The silhouette-mutation tension (design-level)

The fundamental contradiction in this challenge. The task asks you to redesign a building via a generative prompt, then fit the result to the real footprint. But a redesign that changes the silhouette necessarily no longer matches the real footprint. The two requirements are in tension.

Three resolutions, in increasing order of ambition:

1. **Constrain the prompt** to facade-level edits (§5.1). Simple, and covers the post-apocalyptic weathering aesthetic that Scorched Nebraska actually needs.
2. **Fit the base only.** Extract M₀ from the lowest 10% of the mesh, which is the part most likely to preserve the original plan, and let upper-storey additions extend freely.
3. **Two-tier transform.** Solve the similarity on the base, then record the silhouette divergence as a first-class quantity in the provenance record — "this asset deviates from the authoritative footprint by X" — rather than treating it as error. In a game world, that is a feature.

Raise this explicitly in the pitch. It shows you read past the slide.

### 10.2 Everything else

| # | Failure | Detection | Mitigation |
|---|---|---|---|
| 1 | Geocoder returns street centreline | `location_type ≠ ROOFTOP` | Widen candidate radius; require tag match |
| 2 | No OSM footprint | empty Overpass result | Fall back to Microsoft dataset |
| 3 | Campus complex returned as one polygon | footprint area ≫ mesh-implied area | Split on `building:part`; ask the user to pick |
| 4 | Courtyard building | polygon has interior rings | Ensure the whole stack is ring-aware |
| 5 | Mirrored mesh | `det(R) < 0` after §6.8 | Reject; re-run with the source image unflipped |
| 6 | Bas-relief output | §6.4 extent ratio test | Regenerate with more input views |
| 7 | Near-square footprint | `a_F/b_F < 1.1` | EXIF heading → road normal → review |
| 8 | Round or organic building | `R < 0.6` (§4.2) | Force review; rotation is meaningless |
| 9 | Sloped site | elevation spread > 1.5 m across F | Skirt geometry (§8) |
| 10 | Photo contains several buildings | segmentation yields multiple instances | Segment and crop to the largest central instance |
| 11 | Trees occlude the facade | low visible-facade fraction | Prefer another photo; warn |
| 12 | Overpass rate-limits mid-demo | HTTP 429/504 | Pre-warmed cache, committed to the repo |
| 13 | Geoid confusion | building 30 m underground | §2.3; assert `\|z\| < 2500 m` |
| 14 | Terrain LOD pop | building jumps on zoom | `sampleTerrainMostDetailed`, pin once |
| 15 | Stray vertex floats the mesh | visible gap in side view | Robust percentile min (§6.2) |

## 11. Evaluation protocol

Build a benchmark. It takes ninety minutes and it is the difference between "it worked on the one we tried" and a defensible claim.

**Dataset.** Twenty buildings across four strata — five rectangular, five L-shaped or complex, five near-square, five on sloped ground. Include Virginia Tech buildings the judges will recognise (Burruss, Goodwin, NCB, Torgersen), which are well-mapped in OSM. Hand-annotate the ground-truth placement for each by manually aligning a reference box.

**Metrics.** Report median and 90th-percentile IoU; 90th-percentile symmetric Hausdorff in metres; absolute rotation error in degrees, reported both modulo 90° (did the axis alignment work) and absolute (did disambiguation work) — the gap between those two numbers is precisely the cost of the four-fold ambiguity, and it is the most informative single figure in your evaluation; auto-accept rate; and end-to-end latency broken down by stage.

**Ablation.** OMBB-only → +ICP → +IoU refinement → +BnB. Show the marginal gain of each stage. If IoU refinement buys 0.03 IoU for 400 ms, say so; honest negative results read as rigour.

## 12. Provenance and reproducibility

The challenge requires recording the generated asset and its provenance, and storing a reproducible placement transform. Two layers.

### 12.1 The placement record

```json
{
  "asset_id": "uuid-v7",
  "created_at": "2026-09-20T07:12:33Z",
  "address": { "raw": "...", "normalized": "..." },
  "geocode": {
    "provider": "google", "lat": 37.2296, "lon": -80.4239,
    "location_type": "ROOFTOP"
  },
  "footprint": {
    "source": "osm", "osm_type": "way", "osm_id": 123456789,
    "osm_version": 7, "fetched_at": "...",
    "geometry": { "type": "Polygon", "coordinates": [[...]] },
    "rectilinearity": 0.93
  },
  "crs": {
    "type": "ENU", "origin_geodetic": [37.2296, -80.4239, 634.2],
    "ellipsoid": "WGS84", "geoid_undulation_m": -33.1
  },
  "inputs": [
    { "role": "photo", "sha256": "...", "exif_heading_deg": 142.5 }
  ],
  "prompt": "weathered concrete, ivy overgrowth, scorched upper floors",
  "models": [
    { "stage": "edit", "name": "flux-1-kontext-pro", "version": "...", "seed": 42 },
    { "stage": "lift", "name": "trellis-image-large", "version": "...", "seed": 42,
      "params": { "ss_steps": 12, "slat_steps": 12, "ss_guidance": 7.5 } }
  ],
  "transform": {
    "frame": "ENU",
    "translation_m": [0.14, -0.22, 0.0],
    "rotation_z_rad": 1.0472,
    "scale": [1.0043, 1.0043, 1.0],
    "ground_offset_m": 634.2
  },
  "height": { "value_m": 14.7, "source": "osm:building:levels", "levels": 5,
              "assumed_level_height_m": 3.0 },
  "fit": {
    "iou": 0.89, "hausdorff_m": 1.42, "area_ratio": 1.03,
    "rotation_margin": 0.21, "anisotropy_log_ratio": 0.0,
    "max_neighbor_overlap": 0.004,
    "disambiguated_by": "exif_heading",
    "status": "auto_accepted", "solver": "ombb+icp+nelder-mead"
  }
}
```

Store the transform, never the transformed mesh. The mesh is a large binary; the transform is 10 floats. Re-deriving the placement from the record must be bit-identical, which means pinning model versions and seeds and recording the footprint's OSM version — OSM footprints get edited, and a re-fetch six months later may silently return a different polygon.

### 12.2 Content Credentials (C2PA)

If you want the provenance layer to be genuinely defensible rather than a JSON blob, sign it. C2PA — now on an international-standards track as ISO/DIS 22144, with version 1 adopted into JPEG Trust as ISO/IEC 21617-1:2025 — defines a manifest: a cryptographically signed metadata structure containing assertions about an asset's origin and edit history, including which tools and AI models produced it. Manifests reference ingredient assets, each of which may carry its own manifest, forming a provenance graph rooted at the final asset. Assertions are grouped into a claim and signed with PKI keys; the only mandatory assertion is a cryptographic hash binding the manifest to the content.

Bindings come in two forms: hard bindings (cryptographic hashes over the asset bytes) and soft bindings (perceptual hashes or invisible watermarks). Note the operational catch: any re-encoding, recompression, or format conversion invalidates a hard binding even when the content is visually unchanged — which matters here, because your pipeline re-encodes the image at least twice. Plan where the hard binding is computed.

For a 36-hour build, use the `c2pa-python` library with a self-signed development certificate, attach a manifest to the edited image and the exported `.glb` declaring the model, prompt, seed, and source photograph as ingredients, and mention in the pitch that production would require a certificate from a CA on the C2PA Trust List. Almost no hackathon project does this, and the sponsor slide explicitly asks for provenance.

## 13. Rendering and integration

### 13.1 Coordinate conventions (the three-way collision)

glTF 2.0: right-handed, +Y up, asset front faces +Z. 3D Tiles: a generic right-handed Cartesian system whose Z-axis meaning depends on whether the tileset is global or local; glTF is strictly Y-up while the 3D Tiles bounding-volume hierarchy is Z-up, and 3D Tiles 1.0 requires a Y-up→Z-up transform applied to glTF content. This mismatch is a long-running source of confusion and it will cost you an hour if you meet it unprepared.

**The rule that avoids all of it:** keep the mesh in its native glTF Y-up local frame and put every scrap of georeferencing in the model matrix. Bake nothing into the vertices.

### 13.2 Cesium placement

```javascript
const origin = Cesium.Cartesian3.fromDegrees(lon, lat, groundHeightEllipsoidal);
const heading = Cesium.Math.PI_OVER_TWO - thetaENU;   // §2.4
const hpr = new Cesium.HeadingPitchRoll(heading, 0.0, 0.0);
const modelMatrix = Cesium.Transforms.headingPitchRollToFixedFrame(origin, hpr);
// then post-multiply a scale matrix for [sx, sy, sz]
```

`Transforms.eastNorthUpToFixedFrame(origin)` builds the ENU→ECEF matrix directly; `headingPitchRollToFixedFrame` composes it with the orientation, where heading is measured from local north with positive angles increasing eastward, pitch from the east-north plane, and roll about the local east axis.

### 13.3 Asset hygiene for a game world

- Decimate to a triangle budget (TRELLIS's quadric simplification at 0.95 retention is a starting point, but 5–20k triangles per building is the realistic target).
- Bake to a single texture atlas at 1024 px; 2048 only for hero assets.
- Generate at least one LOD, or a footprint-extruded box as LOD1.
- Validate with `gltf-validator` before shipping. Generated meshes fail it routinely.
- Compress with Draco or Meshopt. A 40 MB glb will not stream.

## 14. Engineering practices

**One Placement type, one direction of flow.** Every stage consumes and returns an immutable dataclass. No stage reaches backwards. This makes the whole pipeline replayable from any intermediate state, which is what the sponsor means by "durable and replayable."

**Synthetic unit tests, written before the real pipeline.** Generate a known box, apply a known (s, θ, t), run the solver, assert recovery to within 1e-6. Then deliberately test the adversarial cases: a perfect square (must report low margin), an L-shape (must resolve uniquely), a mirrored input (must be rejected), a mesh with one stray vertex 5 m below the base (must ground correctly). These tests take twenty minutes to write and will catch the bugs that otherwise surface at 4 AM.

**Golden-file regression on five real buildings.** Fixture the cached footprints, fixture a pre-generated mesh, assert the transform to 3 decimal places. This is how you refactor the solver at hour 28 without fear.

**Determinism everywhere.** Pin seeds for both generative models. Pin library versions. An irreproducible demo is a demo you cannot debug.

**Log every intermediate artifact** to disk under the asset UUID: the source photo, the edited image, the raw glb, M₀ as GeoJSON, the four candidate overlays, the final transform. Disk is free; a failure you cannot reproduce is not.

**Structured errors, never silent defaults.** If no footprint is found, fail loudly. A pipeline that quietly substitutes a 10×10 m default box will produce a demo that appears to work and is entirely fictitious.

**Async from commit one.** Generation takes 10–60 s. Enqueue, poll, stream progress. Retrofitting this at hour 30 is miserable.

## 15. Build plan (36 hours)

| Hours | Phase | Exit criterion |
|---|---|---|
| 0–2 | Scaffold & recon | Five demo addresses chosen; footprints fetched, cached, and verified to exist; Cesium renders an empty globe |
| 2–6 | Skeleton placement | A unit cube sits at the correct place, size, and rotation on a real building. No generative models involved. |
| 6–12 | Real fitting | M₀ extraction, OMBB, 4-way disambiguation, ICP, IoU refinement, overlay debug image |
| 12–18 | Generation | Image edit → TRELLIS → glb, async queue, disk cache |
| 18–24 | Height & terrain | OSM tags, geoid correction, `sampleTerrainMostDetailed`, sloped-site handling |
| 24–30 | Validation & provenance | Confidence metrics, review queue with overlay UI, placement record JSON, C2PA if time |
| 30–34 | Benchmark & polish | 20-building evaluation with numbers; BnB if it fits |
| 34–36 | Demo prep | Every demo asset pre-baked; rehearsed twice; offline fallback path |

**The hour-6 milestone is the one that matters.** If a plain cube lands correctly on a real building at hour 6, every hard problem is solved and the rest is substitution. Teams lose this project by spending Saturday tuning the redesign prompt while placement is still broken.

**Pre-bake every demo asset.** Never generate live at judging. Wi-Fi fails, APIs rate-limit, GPUs queue. Keep a live-generation path as the second half of the demo, after the pre-baked one has already landed the point.

## 16. What distinguishes a winning submission

The sponsor's placement slide is, read carefully, a specification for a robust geometric solver. Most teams will implement a greedy version — fit the bounding box, accept it, move on. The differentiators, roughly in order of impact per hour spent:

1. **Quantified confidence and a working review queue.** The slide asks for it and almost nobody will build it.
2. **Principled rotation disambiguation** — EXIF heading, road-normal prior, facade-detail scoring — with a reported margin, rather than "pick the best IoU."
3. **The rectilinearity statistic** (§4.2) as a principled predictor of when the problem is ill-posed, computed before you try to solve it.
4. **A real benchmark with real numbers**, including an ablation and honest failure cases.
5. **Signed provenance via C2PA**, not just a JSON blob.
6. **Naming the silhouette-mutation tension** (§10.1) and showing you thought about the resolution.
7. **A certified global solve via BnB**, if the clock allows.

## 17. References

**3D generation**
- Xiang, J. et al. *Structured 3D Latents for Scalable and Versatile 3D Generation* (TRELLIS). CVPR 2025 Spotlight. https://github.com/microsoft/TRELLIS
- *TRELLIS.2: Native and Compact Structured Latents for 3D Generation.* https://microsoft.github.io/TRELLIS.2/
- Batifol, S. et al. *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space.* arXiv:2506.15742

**Computational geometry**
- Toussaint, G. T. *Solving Geometric Problems with the Rotating Calipers.* Proc. MELECON '83, Athens.
- Freeman, H. & Shapira, R. *Determining the minimum-area encasing rectangle for an arbitrary closed curve.* CACM 18(7):409–413, 1975.
- Arkin, E. M., Chew, L. P., Huttenlocher, D. P., Kedem, K. & Mitchell, J. S. B. *An Efficiently Computable Metric for Comparing Polygonal Shapes.* IEEE TPAMI 13(3):209–216, 1991.
- Edelsbrunner, H., Kirkpatrick, D. & Seidel, R. *On the shape of a set of points in the plane.* IEEE Trans. Inf. Theory, 1983. (α-shapes)

**Registration and alignment**
- Umeyama, S. *Least-Squares Estimation of Transformation Parameters Between Two Point Patterns.* IEEE TPAMI 13(4):376–380, 1991.
- Yang, J., Li, H., Campbell, D. & Jia, Y. *Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration.* IEEE TPAMI 38(11), 2016. arXiv:1605.03344
- *A Second-Order Lower Bound for Globally Optimal 2D Registration.* arXiv:1901.09641
- Fu, H., Cohen-Or, D., Dror, G. & Sheffer, A. *Upright Orientation of Man-Made Objects.* ACM TOG 27(3):42, 2008.

**Metrology and height**
- Criminisi, A., Reid, I. & Zisserman, A. *Single View Metrology.* IJCV 40(2), 2000; ICCV 1999 pp. 434–442.
- Piccinelli, L. et al. *UniDepth: Universal Monocular Metric Depth Estimation.* CVPR 2024. arXiv:2403.18913 · UniDepthV2, arXiv:2502.20110
- Yin, W. et al. *Metric3D / Metric3Dv2* — zero-shot metric depth via canonical camera space.
- *Estimation of building height using a single street view image via deep neural networks.* ISPRS J. Photogramm. Remote Sens. 192:83, 2022.
- *Semi-supervised Learning from Street-View Images and OpenStreetMap for Automatic Building Height Estimation.* arXiv:2307.02574

**Geospatial data**
- Microsoft. *GlobalMLBuildingFootprints.* https://github.com/microsoft/GlobalMLBuildingFootprints (CDLA Permissive 2.0)
- OpenStreetMap Wiki. *Key:building:levels* and *Simple 3D Buildings.*
- *Rectilinear Building Footprint Regularization Using Deep Learning.* ISPRS Annals X-2-2024:217.
- *Building-Regulariser.* https://github.com/DPIRD-DMA/Building-Regulariser

**Standards and rendering**
- *C2PA Specifications 2.x — C2PA and Content Credentials Explainer.*
- Khronos Group. *glTF 2.0 Specification* — coordinate system and units.
- CesiumGS. *Transforms API documentation* — `eastNorthUpToFixedFrame`, `headingPitchRollToFixedFrame`.
- CesiumGS. *3d-tiles Issue #504* — glTF Y-up vs. 3D Tiles Z-up.
