import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { viteStaticCopy } from "vite-plugin-static-copy"
import tailwindcss from '@tailwindcss/vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: "./",
        },
      ],
    }),
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  assetsInclude: ["**/*.onnx"],
  base: "/yolo-bytetrack-onnx-web/"
})