import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: 'build',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'https://dmsocr-backend-new.lemonsand-685cd42d.eastus.azurecontainerapps.io',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('Proxy error:', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Sending request to backend:', req.method, req.url);
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            console.log('Received response from backend:', proxyRes.statusCode, req.url);
          });
        },
      },
    },
    cors: {
      origin: 'https://lemon-bush-06755d30f.3.azurestaticapps.net',
      methods: 'GET,POST,PUT,DELETE,OPTIONS',
      allowedHeaders: 'Content-Type,Authorization',
    },
  },
});
