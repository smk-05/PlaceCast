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
const keepEl = document.getElementById('keep');
const clearEl = document.getElementById('clear');
const sceneEl = document.getElementById('scene');
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

// Compare mode: with "keep on map" ticked, selecting another run ADDS it to
// the scene instead of replacing it, so two placements stand side by side —
// a generated mesh next to a bought model, say. The line under the panel says
// what is currently drawn.
const drawn = [];

function renderScene() {
  sceneEl.innerHTML = drawn.length > 1
    ? `on the map: ${drawn.map((d) => `<b>${d}</b>`).join(' + ')}`
    : '';
}

async function show(assetId) {
  const token = ++showToken;
  errEl.textContent = '';
  const keep = keepEl.checked;
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

    if (!keep) {
      viewer.entities.removeAll();        // nothing uses entities any more
      viewer.scene.primitives.removeAll();
      texturedModels = {};
      openingPrims = [];
      openingsById = new Map();
      showOpeningInfo(undefined);
      drawn.length = 0;
    }
    const label = rec.address_raw.split(',')[0];
    if (!drawn.includes(label)) drawn.push(label);
    renderScene();

    drawFootprint(rec, ground);
    await drawBuilding(rec, ground, () => token === showToken);
    if (token !== showToken) return;
    renderPanel(rec);
    renderMarkerControls(rec);
    renderPhotoViewButton(rec);
    renderTextureToggle();
    if (!keep) flyTo(rec);     // comparing: leave the camera where it is
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
    // A record with baked textures (perception/prism_texture.py) draws those walls instead of the flat polygon;
    // the polygon stays as the fallback when there are none or they fail to load.
    const textured = await drawTexturedPrism(rec, enuToFixed, stillCurrent);
    if (!stillCurrent()) return;
    if (!textured) drawFootprintPrism(rec, ground);
    if (!cameraUnreliable(rec)) drawOpenings(rec, enuToFixed);  // an unreliable fit withholds its markers (panel says so)
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
 * Textured prism: rec.textured_glbs = { photo: 'prism_photo.glb', scorched: 'prism_scorched.glb' }, served from the
 * run directory. The glb is in ENU metres (x east, y north, z up), so like the generated-mesh path it is loaded with
 * upAxis Z / forwardAxis X (no axis correction) and modelMatrix = enuToFixed. Both variants are loaded and the
 * toggle flips `show`. The opening markers stand 7 cm proud of the wall plane, so they never z-fight with it.
 * Returns true when at least one model is on screen.
 */
let texturedModels = {};

async function drawTexturedPrism(rec, enuToFixed, stillCurrent) {
  texturedModels = {};
  for (const [key, file] of Object.entries(rec.textured_glbs || {})) {
    try {
      const model = await Cesium.Model.fromGltfAsync({
        url: `/assets-data/${rec.asset_id}/${file}`,
        modelMatrix: enuToFixed,
        upAxis: Cesium.Axis.Z,
        forwardAxis: Cesium.Axis.X,
      });
      if (!stillCurrent()) {
        model.destroy?.();
        return false;
      }
      model.show = false;
      viewer.scene.primitives.add(model);
      texturedModels[key] = model;
    } catch (e) {
      console.warn(`textured prism "${key}" failed to load`, e);
    }
  }
  const first = texturedModels.photo ? 'photo' : Object.keys(texturedModels)[0];
  if (!first) return false;
  texturedModels[first].show = true;
  return true;
}

/** Photo <-> Scorched switch at the top of the panel (call after renderPanel, which rewrites the panel). */
function renderTextureToggle() {
  if (!Object.keys(texturedModels).length) return;
  const shown = Object.keys(texturedModels).find((k) => texturedModels[k].show);
  const button = (key, label) => `<button data-tex="${key}"${texturedModels[key] ? '' : ' disabled'}
    style="margin-right:6px;font-weight:${key === shown ? 'bold' : 'normal'}">${label}</button>`;
  metaEl.insertAdjacentHTML('afterbegin',
    `<div class="flags"><b>texture</b><div>${button('photo', 'Photo')}${button('scorched', 'Scorched')}</div></div>`);
  metaEl.querySelectorAll('button[data-tex]').forEach((b) => b.addEventListener('click', () => {
    for (const [key, model] of Object.entries(texturedModels)) model.show = key === b.dataset.tex;
    metaEl.querySelectorAll('button[data-tex]').forEach((x) => {
      x.style.fontWeight = x === b ? 'bold' : 'normal';
    });
  }));
}

/**
 * Facade openings on the prism (perception/openings_prism.py writes rec.openings): one thin box per opening,
 * width x height x 0.1 m, on its wall. position_enu / normal_enu are in the same ENU frame as the prism, so the
 * frame is the drawBuilding one. Local Y is up and local Z the outward normal, so local X = up x normal.
 *
 * Drawn as an OUTLINE plus a ~25% alpha FILL in the type colour, so the painted windows of a textured wall show
 * through. REVIEW does not change the fill: its outline is red and thicker (WebGL on Windows draws 1 px lines, so
 * "thicker" is three concentric outlines, each a little larger). Every instance carries the opening's id, which
 * viewer.scene.pick returns, so a click can show that opening's details (showOpeningInfo).
 */
const OPENING_COLOURS = { door: '#2ea043', entrance: '#009696', garage_door: '#f58c14', window: '#286ee6' };
const OPENING_FALLBACK = '#969696';
const REVIEW_RED = '#dc2828';
const REVIEW_OUTLINE_GROW = [0, 0.05, 0.1]; // metres added to width, height and depth of each red outline
let openingPrims = [];
let openingsById = new Map();
const defaultFov = viewer.camera.frustum.fov;

function drawOpenings(rec, enuToFixed) {
  openingPrims = [];
  openingsById = new Map();
  const items = (rec.openings || []).filter(
    (o) => o.position_enu && o.normal_enu && o.width_m && o.height_m,
  );
  if (!items.length) return;
  const fills = [];
  const lines = [];
  for (const o of items) {
    openingsById.set(o.id, o);
    const [nx, ny] = o.normal_enu;
    const [e, n, u] = o.position_enu;
    const local = Cesium.Matrix4.fromColumnMajorArray([-ny, nx, 0, 0, 0, 0, 1, 0, nx, ny, 0, 0, e, n, u, 1]);
    const modelMatrix = Cesium.Matrix4.multiply(enuToFixed, local, new Cesium.Matrix4());
    const typeColour = Cesium.Color.fromCssColorString(OPENING_COLOURS[o.type] || OPENING_FALLBACK);
    const review = o.decision === 'REVIEW';
    fills.push(new Cesium.GeometryInstance({
      id: o.id,
      geometry: Cesium.BoxGeometry.fromDimensions({
        dimensions: new Cesium.Cartesian3(o.width_m, o.height_m, 0.1),
        vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
      }),
      modelMatrix,
      attributes: { color: Cesium.ColorGeometryInstanceAttribute.fromColor(typeColour.withAlpha(0.25)) },
    }));
    const outline = review ? Cesium.Color.fromCssColorString(REVIEW_RED) : typeColour;
    for (const grow of review ? REVIEW_OUTLINE_GROW : [0]) {
      lines.push(new Cesium.GeometryInstance({
        id: o.id,
        geometry: Cesium.BoxOutlineGeometry.fromDimensions({
          dimensions: new Cesium.Cartesian3(o.width_m + grow, o.height_m + grow, 0.1 + grow),
        }),
        modelMatrix,
        attributes: { color: Cesium.ColorGeometryInstanceAttribute.fromColor(outline) },
      }));
    }
  }
  const primitives = [
    new Cesium.Primitive({
      geometryInstances: fills,
      appearance: new Cesium.PerInstanceColorAppearance({ translucent: true }),
      asynchronous: false,
    }),
    new Cesium.Primitive({
      geometryInstances: lines,
      // No renderState.lineWidth: WebGL on Windows allows 1 only, and anything else is a hard DeveloperError.
      appearance: new Cesium.PerInstanceColorAppearance({ flat: true, translucent: false }),
      asynchronous: false,
    }),
  ];
  for (const p of primitives) {
    viewer.scene.primitives.add(p);
    openingPrims.push(p);
  }
}

/** Details of one opening (from a click on its marker), or hidden when `o` is undefined. */
function showOpeningInfo(o) {
  let box = document.getElementById('opening-info');
  if (!o) {
    if (box) box.style.display = 'none';
    return;
  }
  if (!box) {
    box = document.createElement('div');
    box.id = 'opening-info';
    box.style.cssText = 'position:fixed;left:12px;bottom:12px;max-width:340px;padding:8px 10px;z-index:10;'
      + 'background:rgba(20,20,20,.9);color:#eee;font:12px/1.45 sans-serif;border-radius:4px;';
    document.body.appendChild(box);
  }
  const esc = (v) => String(v).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const num = (v, d) => (v == null ? '-' : Number(v).toFixed(d));
  const reasons = (o.reasons || []).map((r) => `<li>${esc(r)}</li>`).join('');
  box.innerHTML = `<b>${esc(o.type)}</b> &middot; ${esc(o.decision)}<br>`
    + `score ${num(o.score, 3)} &middot; bearing ${num(o.bearing_deg, 0)}&deg;<br>`
    + `${num(o.width_m, 2)} &times; ${num(o.height_m, 2)} m`
    + (reasons ? `<ul style="margin:4px 0 0 16px;padding:0">${reasons}</ul>` : '');
  box.style.display = 'block';
}

const openingClicks = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
openingClicks.setInputAction((click) => {
  const picked = viewer.scene.pick(click.position);
  showOpeningInfo(Cesium.defined(picked) ? openingsById.get(picked.id) : undefined);
}, Cesium.ScreenSpaceEventType.LEFT_CLICK);

/** perception/openings_prism.py could not fit the photo's camera to the building (low IoU, or a best value on the edge of
 *  the search grid): the openings are still in the record, but where they would stand is not to be trusted. */
function cameraUnreliable(rec) {
  return rec.openings_camera?.reliable === false;
}

function withheldNotice(rec) {
  const cam = rec.openings_camera;
  const n = (rec.openings || []).length;
  if (!cameraUnreliable(rec) || !n) return '';
  const why = [];
  if (cam.camera_iou < 0.6) why.push('low IoU');
  if ((cam.at_grid_edge || []).length) why.push(`${cam.at_grid_edge.join('/')} at the search-grid edge`);
  return `<div class="flags">Camera fit unreliable (IoU ${fmt(cam.camera_iou, 2)}, ${why.join(' + ') || 'see record'}): `
    + `${n} openings withheld for review.</div>`;
}

/**
 * "View from photo": fly to where the photo was taken, looking the way it looked, through its field of view. All of it
 * comes from rec.openings_camera (perception/openings_prism.py): position in the record's ENU frame, yaw (compass
 * heading), pitch above the horizon, and the photo's horizontal / vertical field of view. Not offered when that fit is
 * unreliable, since the camera would then be wrong by construction.
 */
function renderPhotoViewButton(rec) {
  const c = rec.openings_camera;
  if (!c || cameraUnreliable(rec) || !c.camera_position_enu || !c.hfov_deg) return;
  metaEl.insertAdjacentHTML('afterbegin', '<div class="flags"><button id="view-from-photo">View from photo</button></div>');
  document.getElementById('view-from-photo').addEventListener('click', () => viewFromPhoto(rec));
}

function viewFromPhoto(rec) {
  const c = rec.openings_camera;
  const [lat, lon] = rec.enu_origin_geodetic;
  const enuToFixed = Cesium.Transforms.eastNorthUpToFixedFrame(
    Cesium.Cartesian3.fromDegrees(lon, lat, rec._ground ?? 0),
  );
  const [e, n, u] = c.camera_position_enu;
  const canvas = viewer.scene.canvas;
  // Cesium's fov is the HORIZONTAL angle on a canvas wider than tall, the vertical one otherwise.
  const wide = canvas.clientWidth >= canvas.clientHeight;
  viewer.camera.frustum.fov = Cesium.Math.toRadians(wide ? c.hfov_deg : c.vfov_deg);
  viewer.camera.flyTo({
    destination: Cesium.Matrix4.multiplyByPoint(enuToFixed, new Cesium.Cartesian3(e, n, u), new Cesium.Cartesian3()),
    orientation: {
      heading: Cesium.Math.toRadians(c.camera_yaw_deg),
      pitch: Cesium.Math.toRadians(c.camera_pitch_deg),
      roll: 0,
    },
    duration: 2,
  });
}

/** "Markers" on/off and the colour legend at the top of the panel (call after renderPanel, which rewrites it). */
function renderMarkerControls(rec) {
  const notice = withheldNotice(rec);
  if (notice) {
    metaEl.insertAdjacentHTML('afterbegin', notice);
    return;
  }
  if (!openingPrims.length) return;
  const swatch = (css, label, border = 2) => `<span style="display:inline-block;margin:0 10px 2px 0;white-space:nowrap">`
    + `<span style="display:inline-block;width:10px;height:10px;box-sizing:content-box;border:${border}px solid ${css};`
    + `background:${css}40;margin-right:4px;vertical-align:-2px"></span>${label}</span>`;
  const legend = [['door', 'door'], ['entrance', 'entrance'], ['garage_door', 'garage'], ['window', 'window']]
    .map(([type, label]) => swatch(OPENING_COLOURS[type], label)).join('')
    + swatch(REVIEW_RED, 'red outline = needs review', 3);
  metaEl.insertAdjacentHTML('afterbegin',
    `<div class="flags"><label><input type="checkbox" id="markers-on" checked> <b>Markers</b></label>`
    + `<div style="margin-top:4px;font-size:12px">${legend}</div></div>`);
  document.getElementById('markers-on').addEventListener('change', (e) => {
    for (const p of openingPrims) p.show = e.target.checked;
    if (!e.target.checked) showOpeningInfo(undefined);
  });
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
    ${orientationPanel(rec)}
    <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
    ${(rec.review_reasons || []).length
      ? `<div class="flags"><b>flags</b><ul>${rec.review_reasons
          .map((r) => `<li>${r}</li>`).join('')}</ul></div>`
      : ''}
  `;
  wireOrientationButtons(rec);
}

/* Spec 9.3, the review queue's point: when the four-fold azimuth ambiguity is
 * resolved wrongly, a person fixes it in one click. Each button re-solves the
 * SAME asset with that azimuth — cached mesh, no model run, nothing billed. */
function orientationPanel(rec) {
  const o = rec.orientation;
  if (!o || !o.candidates?.length) return '';
  const buttons = o.candidates
    .slice()
    .sort((a, b) => a.k - b.k)
    .map(({ k, iou }) => {
      const chosen = k === o.chosen_k;
      return `<button class="cand${chosen ? ' chosen' : ''}" data-k="${k}"
        title="re-solve facing candidate ${k}">k=${k}<br><small>IoU ${iou.toFixed(2)}</small>
        </button>`;
    }).join('');
  const how = o.forced_by_reviewer
    ? `set by a reviewer (solver said k=${o.auto_k} via ${o.auto_disambiguated_by})`
    : `chosen by ${o.auto_disambiguated_by}`;
  return `<div class="flags"><b>orientation</b> — ${how}
    <div class="cands">${buttons}</div>
    <small>Pick a facing to re-solve. Free: the mesh is reused.</small>
    <div id="choose-status"></div></div>`;
}

function wireOrientationButtons(rec) {
  const status = document.getElementById('choose-status');
  metaEl.querySelectorAll('button.cand').forEach((b) => {
    b.addEventListener('click', async () => {
      const k = Number(b.dataset.k);
      metaEl.querySelectorAll('button.cand').forEach((x) => { x.disabled = true; });
      status.textContent = `re-solving with k=${k}…`;
      try {
        const res = await fetch(`/api/runs/${rec.asset_id}/choose`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ k }),
        });
        if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
        const { job_id: jobId } = await res.json();
        const job = await waitForJob(jobId, status);
        if (job.status === 'failed') throw new Error(job.error);
        status.textContent = 'done — reloading';
        await show(rec.asset_id);          // redraws with the new placement
      } catch (e) {
        status.textContent = `failed: ${e.message ?? e}`;
        metaEl.querySelectorAll('button.cand').forEach((x) => { x.disabled = false; });
      }
    });
  });
}

async function waitForJob(jobId, status, tries = 120) {
  for (let i = 0; i < tries; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    const job = await (await fetch(`/api/jobs/${jobId}`)).json();
    if (job.status === 'done' || job.status === 'failed') return job;
    status.textContent = `re-solving… (${job.status})`;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error('timed out waiting for the re-solve');
}

const fmt = (v, n) => (typeof v === 'number' ? v.toFixed(n) : '-');

/**
 * Aim at the building rather than at a fixed altitude. The old version flew to
 * 260 m above the ELLIPSOID — but Blacksburg's terrain is ~600 m above it, so
 * the camera ended up ~340 m inside the hill looking at terrain from below
 * (the spec 2.3 datum trap, applied to the camera instead of the building).
 */
function flyTo(rec) {
  viewer.camera.frustum.fov = defaultFov;  // undo a "View from photo" field of view
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
clearEl.addEventListener('click', () => {
  viewer.entities.removeAll();
  viewer.scene.primitives.removeAll();
  texturedModels = {};
  openingPrims = [];
  openingsById = new Map();
  drawn.length = 0;
  renderScene();
});
document.getElementById('fly').addEventListener('click', () => current && flyTo(current));

loadRuns();
