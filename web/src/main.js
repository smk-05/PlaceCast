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

const TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN;
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
try {
  viewer.scene.setTerrain(
    new Cesium.Terrain(Cesium.CesiumTerrainProvider.fromIonAssetId(1)),
  );
} catch (e) {
  console.warn('Cesium World Terrain unavailable; using the ellipsoid.', e);
}

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

async function show(assetId) {
  errEl.textContent = '';
  const rec = await (await fetch(`/api/runs/${assetId}`)).json();
  current = rec;

  viewer.entities.removeAll();
  viewer.scene.primitives.removeAll();

  drawFootprint(rec);
  await drawBuilding(rec);
  renderPanel(rec);
  flyTo(rec);
}

/** The authoritative footprint, as ground truth to eyeball the placement against. */
function drawFootprint(rec) {
  const geom = rec.footprint_geojson;
  if (!geom || !geom.coordinates) return;

  const rings = geom.type === 'Polygon' ? geom.coordinates : geom.coordinates[0];
  const outer = rings[0].flatMap(([lon, lat]) => [lon, lat]);

  viewer.entities.add({
    name: 'authoritative footprint',
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(
        Cesium.Cartesian3.fromDegreesArray(outer),
        rings.slice(1).map(
          (r) => new Cesium.PolygonHierarchy(
            Cesium.Cartesian3.fromDegreesArray(r.flatMap(([lon, lat]) => [lon, lat])),
          ),
        ),
      ),
      material: Cesium.Color.fromCssColorString('#1b6ca8').withAlpha(0.35),
      outline: true,
      outlineColor: Cesium.Color.fromCssColorString('#1b6ca8'),
      classificationType: Cesium.ClassificationType.TERRAIN,
    },
  });
}

/**
 * Place the mesh — or, before generation exists, the spec 15 hour-6 unit cube.
 *
 * Everything georeferencing goes into the model matrix, per spec 13.1.
 */
async function drawBuilding(rec) {
  const fit = rec.fit;
  if (!fit) return;

  const [lat, lon] = [rec.enu_origin_geodetic[0], rec.enu_origin_geodetic[1]];
  const t = fit.transform.translation_m;
  const scale = fit.transform.scale;
  const height = rec.height ? rec.height.value_m : 10;

  // The solver's translation is in the ENU frame anchored at the footprint
  // centroid, so offset the origin by it rather than re-deriving a lat/lon.
  const enuOrigin = Cesium.Cartesian3.fromDegrees(lon, lat, 0);
  const enuToFixed = Cesium.Transforms.eastNorthUpToFixedFrame(enuOrigin);
  const offset = Cesium.Matrix4.multiplyByPoint(
    enuToFixed, new Cesium.Cartesian3(t[0], t[1], t[2]), new Cesium.Cartesian3(),
  );

  const carto = Cesium.Cartographic.fromCartesian(offset);
  const ground = await sampleGround(carto);

  const origin = Cesium.Cartesian3.fromRadians(
    carto.longitude, carto.latitude, ground,
  );

  // Spec 2.4. The one place this conversion happens on the JS side.
  const heading = fit.transform.heading_rad;
  const hpr = new Cesium.HeadingPitchRoll(heading, 0, 0);
  const modelMatrix = Cesium.Transforms.headingPitchRollToFixedFrame(origin, hpr);

  const glbUrl = `/assets-data/${rec.asset_id}/mesh.glb`;
  const hasMesh = (await fetch(glbUrl, { method: 'HEAD' })).ok;

  if (hasMesh) {
    const model = await Cesium.Model.fromGltfAsync({ url: glbUrl, modelMatrix });
    viewer.scene.primitives.add(model);
  } else {
    // The hour-6 milestone object. If this lands correctly on a real building,
    // every hard problem is solved and the rest is substitution.
    const ombb = rec.footprint_ombb_m || [20, 14];
    viewer.entities.add({
      name: 'placement box (hour-6 milestone)',
      position: origin,
      orientation: Cesium.Transforms.headingPitchRollQuaternion(origin, hpr),
      box: {
        dimensions: new Cesium.Cartesian3(
          ombb[0] * scale[0], ombb[1] * scale[1], height,
        ),
        material: Cesium.Color.fromCssColorString('#e8590c').withAlpha(0.7),
        outline: true,
        outlineColor: Cesium.Color.fromCssColorString('#7f2704'),
      },
    });
  }
}

/**
 * Spec 8: sample at maximum detail and PIN the height. Cesium streams terrain
 * progressively, so a building placed against a coarse tile visibly jumps when
 * the fine tile loads. Do not re-sample on camera movement.
 */
async function sampleGround(carto) {
  try {
    const provider = viewer.terrainProvider;
    if (!provider || !provider.availability) return 0;
    const [sampled] = await Cesium.sampleTerrainMostDetailed(provider, [
      Cesium.Cartographic.clone(carto),
    ]);
    return sampled.height || 0;
  } catch {
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

  metaEl.innerHTML = `
    <div style="margin:8px 0"><span class="badge ${d}">${d.replace('_', ' ')}</span></div>
    <table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>
    ${(rec.review_reasons || []).length
      ? `<div class="flags"><b>flags</b><ul>${rec.review_reasons
          .map((r) => `<li>${r}</li>`).join('')}</ul></div>`
      : ''}
  `;
}

const fmt = (v, n) => (typeof v === 'number' ? v.toFixed(n) : '-');

function flyTo(rec) {
  const [lat, lon] = rec.enu_origin_geodetic;
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon, lat - 0.0016, 260),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(-32), roll: 0 },
    duration: 1.6,
  });
}

runsEl.addEventListener('change', (e) => show(e.target.value));
document.getElementById('fly').addEventListener('click', () => current && flyTo(current));

loadRuns();
