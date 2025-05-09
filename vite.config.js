import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import tailwindcss from '@tailwindcss/vite'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  assetsInclude: ["**/*.onnx"],
  base: "/YOLO-ByteTrack-ONNX-Web/"
})