import rawPlugin from 'vite-raw-plugin';
import { defineConfig } from 'vite'

export default defineConfig({
    server: {
        open: true,
    },
    build: {
        target: 'esnext'
    },
    // Avoid TypeScript Node type requirement by accessing via globalThis
    base: (globalThis as any)?.process?.env?.GITHUB_ACTIONS_BASE || undefined,
    plugins: [
        rawPlugin({
            fileRegex: /\.wgsl$/,
        }),
    ],
})
