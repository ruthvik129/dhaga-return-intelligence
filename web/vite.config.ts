import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The API (FastAPI, api/server.py) runs on :8000 in development; Vite proxies
// /api to it so the browser only ever talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
