import { defineConfig } from 'vite';
import cesium from 'vite-plugin-cesium';

export default defineConfig({
  plugins: [cesium()],
  server: {
    port: 5173,
    proxy: {
      // FastAPI serves the placement records and the generated .glb files.
      '/api': 'http://127.0.0.1:8000',
      '/assets-data': 'http://127.0.0.1:8000',
    },
  },
});
