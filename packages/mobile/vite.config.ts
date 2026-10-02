import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Tauri serves the build from the app bundle and proxies the dev server to
// devices on the local network, so the host and port must stay fixed.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5173, strictPort: true },
  build: { target: ['es2022', 'safari16', 'chrome110'], outDir: 'dist' },
})
