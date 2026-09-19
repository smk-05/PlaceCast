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
 * Place the mesh — or, on a dry run, the spec 15 hour-6 box.
 *
 * ALL of the placement comes from rec.mesh_to_enu, one 4x4 computed in Python
 * (geo/placement.py) from the same numbers the solver used, and tested there to
 * reproduce the solver's placement exactly. This function adds only the one
 * thing Python cannot know: where the ground is (spec 8), sampled once here.
 *
 *   modelMatrix = ENU->ECEF (footprint centroid, at ground) x mesh_to_enu
 */
async function drawBuilding(rec) {
  const m2e = rec.mesh_to_enu;
  if (!m2e) {
    errEl.textContent = 'This record predates mesh_to_enu — re-run pipeline.py for it '
      + '(add --asset-id to reuse the cached mesh; nothing is re-billed).';
    return;
  }

  const [lat, lon] = rec.enu_origin_geodetic;
  // Spec 8: sample at maximum detail and PIN it; never re-sample on camera moves.
  const ground = await sampleGround(Cesium.Cartographic.fromDegrees(lon, lat));
  const enuToFixed = Cesium.Transforms.eastNorthUpToFixedFrame(
    Cesium.Cartesian3.fromDegrees(lon, lat, ground),
  );
  const modelMatrix = Cesium.Matrix4.multiply(
    enuToFixed, Cesium.Matrix4.fromColumnMajorArray(m2e.column_major), new Cesium.Matrix4(),
  );

  const glbUrl = `/assets-data/${rec.asset_id}/mesh.glb`;
  const hasMesh = m2e.mesh_frame === 'gltf_scene'
    && (await fetch(glbUrl, { method: 'HEAD' })).ok;

  if (hasMesh) {
    // Cesium by default applies TWO rotations to a glTF: Y-up -> Z-up, and a
    // second one turning glTF's +Z "forward" into +X (ModelUtility.
    // getAxisCorrectionMatrix). mesh_to_enu already contains the full rotation,
    // so both must be off: upAxis Z skips the first, forwardAxis X the second.
    // Leaving the default forwardAxis would twist every building 90 degrees.
    const model = await Cesium.Model.fromGltfAsync({
      url: glbUrl,
      modelMatrix,
      upAxis: Cesium.Axis.Z,
      forwardAxis: Cesium.Axis.X,
    });
    viewer.scene.primitives.add(model);
  } else {
    // The hour-6 milestone object: a unit box, placed by the same matrix.
    viewer.scene.primitives.add(new Cesium.Primitive({
      geometryInstances: new Cesium.GeometryInstance({
        geometry: Cesium.BoxGeometry.fromDimensions({
          dimensions: new Cesium.Cartesian3(1, 1, 1),
          vertexFormat: Cesium.PerInstanceColorAppearance.VERTEX_FORMAT,
        }),
        modelMatrix,
        attributes: {
          color: Cesium.ColorGeometryInstanceAttribute.fromColor(
            Cesium.Color.fromCssColorString('#e8590c').withAlpha(0.7),
          ),
        },
      }),
      appearance: new Cesium.PerInstanceColorAppearance({ translucent: true }),
    }));
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
