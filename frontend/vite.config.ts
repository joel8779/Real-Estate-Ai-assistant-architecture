/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backendPort = process.env.BACKEND_PORT || '8000';
const backendTarget = `http://127.0.0.1:${backendPort}`;

export default defineConfig({
  plugins: [react()],
    server: {
      port: 5173,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/healthz': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/readyz': {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
    preview: {
      port: 5173,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/healthz': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/readyz': {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
});
