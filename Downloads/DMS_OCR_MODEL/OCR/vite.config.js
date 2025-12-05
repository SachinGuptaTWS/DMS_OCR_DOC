import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: process.env.NODE_ENV === 'production' ? '/static' : '/',
  build: {
    outDir: 'build',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'https://dmsocr-backend-new.lemonsand-685cd42d.eastus.azurecontainerapps.io',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
