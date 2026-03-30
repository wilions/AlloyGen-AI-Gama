import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5178,
    proxy: {
      '/ws': { target: 'http://localhost:8005', ws: true },
      '/auth': { target: 'http://localhost:8005' },
      '/upload': { target: 'http://localhost:8005' },
      '/batch-predict': { target: 'http://localhost:8005' },
      '/download': { target: 'http://localhost:8005' },
      '/models': { target: 'http://localhost:8005' },
      '/sessions': { target: 'http://localhost:8005' },
      '/dataset-stats': { target: 'http://localhost:8005' },
    },
  },
})
