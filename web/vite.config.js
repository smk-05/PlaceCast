import { defineConfig } from 'vite';
import cesium from 'vite-plugin-cesium';

export default defineConfig({
  plugins: [cesium()],
  // Read the repo-root .env (one secrets file for the whole project), and expose
  // ONLY CESIUM_* variables to the browser. The Cesium ion token is a
  // client-side token by design; REPLICATE_API_TOKEN and the rest stay server-side
  // because they lack the prefix. Without this, Vite reads only web/.env and only
  // VITE_* names, so the token never arrived and terrain failed to load.
  envDir: '..',
  envPrefix: ['VITE_', 'CESIUM_'],
  server: {
    port: 5173,
    proxy: {
      // FastAPI serves the placement records and the generated .glb files.
      '/api': 'http://127.0.0.1:8000',
      '/assets-data': 'http://127.0.0.1:8000',
    },
  },
});
