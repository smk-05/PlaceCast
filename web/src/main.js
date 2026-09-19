/*
 * CesiumJS placement viewer. Spec 13.2.
 *
 * THE RULE THAT AVOIDS THE WHOLE COORDINATE MESS (spec 13.1): keep the mesh in
 * its native glTF Y-up local frame and put every scrap of georeferencing in the
 * model matrix. Bake nothing into the vertices.
 *
 * glTF 2.0 is right-handed, +Y up, asset front facing +Z. The 3D Tiles
 * bounding-volume hierarchy is Z-up. That mismatch is a long-running source of
 * confusion and it costs an hour if you meet it unprepared. We never meet it,
 * because we only ever place a single glTF via a model matrix.
 *
 * Angle convention (spec 2.4): the solver works in mathematical theta, CCW from
 * +East. Cesium wants heading, CW from North. heading = pi/2 - theta, and that
 * conversion happens HERE and nowhere else on the JS side.
 */

import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';

const TOKEN = import.meta.env.CESIUM_ION_TOKEN || import.meta.env.VITE_CESIUM_ION_TOKEN;
if (!TOKEN) console.warn('No CESIUM_ION_TOKEN in the root .env: World Terrain will not load.');
if (TOKEN) Cesium.Ion.defaultAccessToken = TOKEN;

const viewer = new Cesium.Viewer('cesium', {
  timeline: false,
  animation: false,
  baseLayerPicker: true,
  geocoder: false,
  sceneModePicker: false,
  homeButton: false,
  navigationHelpButton: false,
});

// Cesium World Terrain is ALREADY ELLIPSOIDAL. This is the reason the geoid
// correction of spec 2.3 is mostly sidestepped: no NAVD88 conversion, no PROJ
// grid download, no building buried three storeys underground.
//
// The provider loads asynchronously, and until it resolves viewer.terrainProvider
// is the flat ellipsoid. Sampling ground before then silently returned 0 and put
// the building ~600 m inside the hill. Everything that needs the ground awaits
// this promise instead.
const terrainReady = (async () => {
  try {
    const provider = await Cesium.CesiumTerrainProvider.fromIonAssetId(1);
    viewer.terrainProvider = provider;
    return provider;
  } catch (e) {
    console.warn('Cesium World Terrain unavailable; using the ellipsoid.', e);
    document.getElementById('err').textContent =
      'World Terrain failed to load (check CESIUM_ION_TOKEN) — showing the bare ellipsoid.';
    return null;
  }
})();

viewer.scene.globe.depthTestAgainstTerrain = true;

const runsEl = document.getElementById('runs');
const metaEl = document.getElementById('meta');
const errEl = document.getElementById('err');

let current = null;

async function loadRuns() {
  try {
    const res = await fetch('/api/runs');
    const runs = await res.json();
    if (!runs.length) {
      runsEl.innerHTML = '<option>no runs yet — run pipeline.py</option>';
      return;
    }
    runsEl.innerHTML = runs
      .map((r) => `<option value="${r.asset_id}">${r.label}</option>`)
      .join('');
    show(runs[0].asset_id);
  } catch (e) {
    errEl.textContent = 'Cannot reach the API. Start it with: uv run uvicorn server.app:app';
  }
}

// Switching records while one is still loading used to kill the whole scene:
// removeAll() DESTROYS what it removes, and the in-flight load then touched a
// destroyed object ("DeveloperError: This object was destroyed"), which stops
// Cesium rendering for good. Every show() takes a token and abandons itself as
// soon as a newer one starts, so only the newest load ever touches the scene.
let showToken = 0;

async function show(assetId) {
  const token = ++showToken;
  errEl.textContent = '';
  try {
    const rec = await (await fetch(`/api/runs/${assetId}`)).json();
    if (token !== showToken) return;
    current = rec;

    // Ground FIRST: both the footprint and the building are drawn at this
    // height, so neither needs terrain draping (see drawFootprint).
    const [lat, lon] = rec.enu_origin_geodetic;
    const ground = await sampleGround(Cesium.Cartographic.fromDegrees(lon, lat));
    if (token !== showToken) return;
    rec._ground = ground;

    viewer.entities.removeAll();          // nothing uses entities any more
    viewer.scene.primitives.removeAll();

    drawFootprint(rec, ground);
    await drawBuilding(rec, ground, () => token === showToken);
    if (token !== showToken) return;
    renderPanel(rec);
    flyTo(rec);
  } catch (e) {
    if (token !== showToken) return;
    console.error(e);
    errEl.textContent = `Could not show this run: ${e.message ?? e}`;
  }
}

/* EVERYTHING here is drawn as a SYNCHRONOUS Primitive, never as an entity and
 * never draped on terrain.
 *
 * Both of the easy ways to draw a polygon build their geometry on a worker:
 * classificationType TERRAIN makes a GroundPrimitive, and an entity polygon is
 * batched by the GeometryVisualizer. Switching records deletes them mid-build
 * and Cesium's render loop then throws "DeveloperError: This object was
 * destroyed" and STOPS — permanently, and no guard in this file can catch it,
 * because the failure is inside Cesium's own pending job. `asynchronous: false`
 * builds the geometry inline, so a removed primitive has no pending work left.
 */
function polygonHierarchy(rings) {
  return new Cesium.PolygonHierarchy(
    Cesium.Cartesian3.fromDegreesArray(rings[0].flatMap(([lon, lat]) => [lon, lat])),
    rings.slice(1).map(
      (r) => new Cesium.PolygonHierarchy(
        Cesium.Cartesian3.fromDegreesArray(r.flatMap(([lon, lat]) => [lon, lat])),
      ),
    ),
  );
}

function addPolygon(rings, { height, extrudedHeight, colour, outlineColour }) {
  const hierarchy = polygonHierarchy(rings);
  const common = { polygonHierarchy: hierarchy, height, extrudedHeight };
  viewer.scene.primitives.add(new Cesium.Primitive({
    geometryInstances: new Cesium.GeometryInstance({
      geometry: new Cesium.PolygonGeometry({
        ...common,
        vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
      }),
      attributes: {
        color: Cesium.ColorGeometryInstanceAttribute.fromColor(colour),
      },
    }),
    appearance: new Cesium.PerInstanceColorAppearance({ translucent: colour.alpha < 1 }),
    asynchronous: false,
  }));
  if (!outlineColour) return;
  viewer.scene.primitives.add(new Cesium.Primitive({
    geometryInstances: new Cesium.GeometryInstance({
      geometry: new Cesium.PolygonOutlineGeometry(common),
      attributes: {
        color: Cesium.ColorGeometryInstanceAttribute.fromColor(outlineColour),
      },
    }),
    // No renderState.lineWidth: WebGL on Windows allows 1 only, and anything
    // else is a hard DeveloperError that stops rendering outright.
    appearance: new Cesium.PerInstanceColorAppearance({ flat: true, translucent: false }),
    asynchronous: false,
  }));
}

/** The authoritative footprint, as ground truth to eyeball the placement against. */
function drawFootprint(rec, ground) {
  const geom = rec.footprint_geojson;
  if (!geom || !geom.coordinates) return;
  const polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;
  for (const rings of polys) {
    addPolygon(rings, {
      height: ground + 0.2,              // clear of the terrain, no z-fighting
      colour: Cesium.Color.fromCssColorString('#1b6ca8').withAlpha(0.35),
      outlineColour: Cesium.Color.fromCssColorString('#1b6ca8'),
    });
  }
}

/**
 * Place the mesh — or, on a dry run, the spec 15 hour-6 box.
 *
 * ALL of the placement comes from rec.mesh_to_enu, one 4x4 computed in Python
 * (geo/placement.py) from the same numbers the solver used, and tested there to
 * reproduce the solver's placement exactly. This function adds only the one
 * thing Python cannot know: where the ground is (spec 8), sampled once here.
 *
 *   modelMatrix = ENU->ECEF (footprint centroid, at ground) x mesh_to_enu
 */
async function drawBuilding(rec, ground, stillCurrent = () => true) {
  const m2e = rec.mesh_to_enu;
  if (!m2e) {
    errEl.textContent = 'This record predates mesh_to_enu — re-run pipeline.py for it '
      + '(add --asset-id to reuse the cached mesh; nothing is re-billed).';
    return;
  }

  const [lat, lon] = rec.enu_origin_geodetic;
  // Ground was sampled once in show() (spec 8: maximum detail, then PINNED).
  console.info(`ground at footprint centroid: ${ground.toFixed(1)} m (ellipsoidal)`);
  const enuToFixed = Cesium.Transforms.eastNorthUpToFixedFrame(
    Cesium.Cartesian3.fromDegrees(lon, lat, ground),
  );
  const modelMatrix = Cesium.Matrix4.multiply(
    enuToFixed, Cesium.Matrix4.fromColumnMajorArray(m2e.column_major), new Cesium.Matrix4(),
  );

  const glbUrl = `/assets-data/${rec.asset_id}/mesh.glb`;

  if (m2e.mesh_frame === 'gltf_scene') {
    // A mesh record's matrix is for the MESH. Never fall back to drawing the box
    // with it: a unit box through a mesh matrix becomes a large, wrong object
    // that looks like a placement (this happened when the server answered the
    // HEAD check with 405). If the mesh cannot load, say so.
    const head = await fetch(glbUrl, { method: 'HEAD' });
    if (!head.ok) {
      errEl.textContent = `mesh.glb unavailable (HTTP ${head.status}) — not drawing a substitute.`;
      return;
    }
    // Cesium by default applies TWO rotations to a glTF: Y-up -> Z-up, and a
    // second one turning glTF's +Z "forward" into +X (ModelUtility.
    // getAxisCorrectionMatrix). mesh_to_enu already contains the full rotation,
    // so both must be off: upAxis Z skips the first, forwardAxis X the second.
    // Leaving the default forwardAxis would twist every building 90 degrees.
    if (!stillCurrent()) return;
    const model = await Cesium.Model.fromGltfAsync({
      url: glbUrl,
      modelMatrix,
      upAxis: Cesium.Axis.Z,
      forwardAxis: Cesium.Axis.X,
    });
    if (!stillCurrent()) {
      model.destroy?.();
      return;
    }
    viewer.scene.primitives.add(model);
  } else if (m2e.mesh_frame === 'unit_box_centred') {
    // No generated mesh. Draw the AUTHORITATIVE footprint extruded to the
    // resolved height (spec 7) instead of the hour-6 unit box: the box is the
    // footprint's OMBB, so an L-shaped building came out rectangular, and the
    // box's own IoU (0.65-0.92) measured the rectangle approximation rather
    // than the placement. The prism is exact in plan by construction — what it
    // still shows is everything the placement chain contributes: which
    // building, its outline, its height, and the ground it stands on.
    drawFootprintPrism(rec, ground);
    drawOpenings(rec, enuToFixed);
  } else {
    errEl.textContent = `Unknown mesh_frame "${m2e.mesh_frame}" — not drawing anything.`;
  }
}

/** The footprint extruded to the resolved height, for records with no mesh. */
function drawFootprintPrism(rec, ground) {
  const geom = rec.footprint_geojson;
  const height = rec.height?.value_m;
  if (!geom || !height) {
    errEl.textContent = 'No mesh and no height — nothing to draw.';
    return;
  }
  // Every ring, so courtyards stay holes and Lane Stadium keeps its stands.
  const polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;
  for (const rings of polys) {
    addPolygon(rings, {
      height: ground,                        // pinned terrain, as for a mesh
      extrudedHeight: ground + height,
      colour: Cesium.Color.fromCssColorString('#c9a227').withAlpha(0.85),
      outlineColour: Cesium.Color.fromCssColorString('#5c4708'),
    });
  }
}

/**
 * Facade openings on the prism (perception/openings_prism.py writes rec.openings): one thin box per opening,
 * width x height x 0.1 m, on its wall. position_enu / normal_enu are in the same ENU frame as the prism, so the
 * frame is the drawBuilding one. Local Y is up and local Z the outward normal, so local X = up x normal. REVIEW is
 * red; the other colours are per type, as in perception/openings_3d.py.
 */
const OPENING_COLOURS = { door: '#2ea043', entrance: '#009696', garage_door: '#f58c14', window: '#286ee6' };

function drawOpenings(rec, enuToFixed) {
  const items = (rec.openings || []).filter(
    (o) => o.position_enu && o.normal_enu && o.width_m && o.height_m,
  );
  if (!items.length) return;
  viewer.scene.primitives.add(new Cesium.Primitive({
    geometryInstances: items.map((o) => {
      const [nx, ny] = o.normal_enu;
      const [e, n, u] = o.position_enu;
      const local = Cesium.Matrix4.fromColumnMajorArray([-ny, nx, 0, 0, 0, 0, 1, 0, nx, ny, 0, 0, e, n, u, 1]);
      const css = o.decision === 'REVIEW' ? '#dc2828' : (OPENING_COLOURS[o.type] || '#969696');
      return new Cesium.GeometryInstance({
        id: o.id,
        geometry: Cesium.BoxGeometry.fromDimensions({
          dimensions: new Cesium.Cartesian3(o.width_m, o.height_m, 0.1),
          vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
        }),
        modelMatrix: Cesium.Matrix4.multiply(enuToFixed, local, new Cesium.Matrix4()),
        attributes: { color: Cesium.ColorGeometryInstanceAttribute.fromColor(Cesium.Color.fromCssColorString(css)) },
      });
    }),
    appearance: new Cesium.PerInstanceColorAppearance({ translucent: false }),
    asynchronous: false,
  }));
}

/**
 * Spec 8: sample at maximum detail and PIN the height. Cesium streams terrain
 * progressively, so a building placed against a coarse tile visibly jumps when
 * the fine tile loads. Do not re-sample on camera movement.
 */
async function sampleGround(carto) {
  // Wait for the REAL terrain. With no terrain the globe itself is the
  // ellipsoid, so 0 is then the correct ground — consistent, not buried.
  const provider = await terrainReady;
  if (!provider) return 0;
  try {
    const [sampled] = await Cesium.sampleTerrainMostDetailed(provider, [
      Cesium.Cartographic.clone(carto),
    ]);
    return Number.isFinite(sampled.height) ? sampled.height : 0;
  } catch (e) {
    console.warn('terrain sample failed', e);
    return 0;
  }
}

function renderPanel(rec) {
  const fit = rec.fit || {};
  const d = rec.decision || 'review';
  const rows = [
    ['IoU', fmt(fit.iou, 3)],
    ['Hausdorff', fit.hausdorff_m != null ? `${fit.hausdorff_m.toFixed(2)} m` : '-'],
    ['area ratio', fmt(fit.area_ratio, 3)],
    ['rotation margin', fmt(fit.rotation_margin_footprint, 3)],
    ['rectilinearity', fmt(rec.rectilinearity, 3)],
    ['neighbour overlap', fmt(fit.max_neighbor_overlap, 4)],
    ['height', rec.height ? `${rec.height.value_m.toFixed(1)} m` : '-'],
    ['height source', rec.height ? rec.height.source : '-'],
    ['disambiguated by', fit.disambiguated_by || '-'],
    ['solver', fit.solver || '-'],
  ];

  // A prism record's fit metrics come from the solver's OMBB BOX proxy, not
  // from the footprint prism on screen: the prism is the footprint, so its plan
  // is exact by construction and an IoU of 0.65 would be read as a placement
  // error when it is the rectangle approximation being measured.
  const isPrism = rec.mesh_to_enu?.mesh_frame === 'unit_box_centred';
  const note = isPrism
    ? '<div class="flags"><b>drawn: footprint prism</b><br>No generated mesh for '
      + 'this run. The authoritative footprint is extruded to the resolved '
      + 'height, so its plan matches exactly. The numbers below are the '
      + "solver's rectangular-box proxy, not this prism.</div>"
    : '';

  metaEl.innerHTML = `
    <div style="margin:8px 0"><span class="badge ${d}">${d.replace('_', ' ')}</span></div>
    ${note}
    <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
    ${(rec.review_reasons || []).length
      ? `<div class="flags"><b>flags</b><ul>${rec.review_reasons
          .map((r) => `<li>${r}</li>`).join('')}</ul></div>`
      : ''}
  `;
}

const fmt = (v, n) => (typeof v === 'number' ? v.toFixed(n) : '-');

/**
 * Aim at the building rather than at a fixed altitude. The old version flew to
 * 260 m above the ELLIPSOID — but Blacksburg's terrain is ~600 m above it, so
 * the camera ended up ~340 m inside the hill looking at terrain from below
 * (the spec 2.3 datum trap, applied to the camera instead of the building).
 */
function flyTo(rec) {
  const [lat, lon] = rec.enu_origin_geodetic;
  const ground = rec._ground ?? 0;
  const target = new Cesium.BoundingSphere(
    Cesium.Cartesian3.fromDegrees(lon, lat, ground + 10), 60,
  );
  viewer.camera.flyToBoundingSphere(target, {
    offset: new Cesium.HeadingPitchRange(
      Cesium.Math.toRadians(20), Cesium.Math.toRadians(-35), 260,
    ),
    duration: 1.6,
  });
}

runsEl.addEventListener('change', (e) => show(e.target.value));
document.getElementById('fly').addEventListener('click', () => current && flyTo(current));

loadRuns();
