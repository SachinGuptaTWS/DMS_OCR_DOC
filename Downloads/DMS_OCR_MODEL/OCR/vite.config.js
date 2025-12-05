import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // FIX: Always use root '/' for Azure Static Web Apps. 
  // Do NOT use '/static' unless your site is hosted in a subfolder.
  base: '/',
  build: {
    outDir: 'build',
    emptyOutDir: true,
  },
  server: {
    // Note: This proxy only works locally (npm run dev). 
    // In production, your VITE_API_BASE_URL env var handles this.
    proxy: {
      '/api': {
        target: 'https://dmsocr-backend-new.lemonsand-685cd42d.eastus.azurecontainerapps.io',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
