import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Listen on all network interfaces
    port: 8000,
    // commented out proxy for now
    // proxy: {
    //   '/api': {
    //     target: 'http://127.0.0.1',
    //     changeOrigin: true,
    //   },
    // },
  },
  build: {
    chunkSizeWarningLimit: 1500,
    // target: 'esnext',
    // rollupOptions: {
    //   output: {
    //     manualChunks: {
    //       'react': ['react', 'react-dom'],
    //     },
    //   },
    // },
  }
})
