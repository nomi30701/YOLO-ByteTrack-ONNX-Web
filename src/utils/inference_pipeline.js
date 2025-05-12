import * as ort from "onnxruntime-web/webgpu";
import { preProcess_img, applyNMS } from "./img_preprocess";

/**
 * Inference pipeline for YOLO model.
 * @param {cv.Mat} src_mat - Input image Mat.
 * @param {ort.InferenceSession} session - YOLO model session.
 * @param {BYTETracke} tracker - Object tracker instance.
 * @param {[Number, Number]} overlay_size - Overlay width and height. [width, height]
 * @param {String} imgsz_type - Image size type, either "dynamic" or "zeroPad".
 * @returns {[Array[Object], Number]} - Array of predictions and inference time.
 */
export async function inference_pipeline(
  src_mat,
  session,
  tracker,
  overlay_size,
  imgsz_type
) {
  try {
    const [input_tensor, xRatio, yRatio] = preProcess_img(
      src_mat,
      overlay_size,
      imgsz_type
    );
    src_mat.delete();

    const start = performance.now();
    const { output0 } = await session.run({
      images: input_tensor,
    });
    const end = performance.now();

    input_tensor.dispose();

    // Post process
    const NUM_PREDICTIONS = output0.dims[2];
    const NUM_BBOX_ATTRS = 4;
    const NUM_SCORES = 80;

    const predictions = output0.data;
    const bbox_data = predictions.subarray(0, NUM_PREDICTIONS * NUM_BBOX_ATTRS);
    const scores_data = predictions.subarray(NUM_PREDICTIONS * NUM_BBOX_ATTRS);

    const detections = new Array();
    let resultCount = 0;

    for (let i = 0; i < NUM_PREDICTIONS; i++) {
      let maxScore = 0;
      let cls_idx = -1;

      // get maximum score in 80 classes
      for (let c = 0; c < NUM_SCORES; c++) {
        const score = scores_data[i + c * NUM_PREDICTIONS];
        if (score > maxScore) {
          maxScore = score;
          cls_idx = c;
        }
      }
      // Filter low confidence for ByteTrack.
      if (maxScore <= 0.35) continue;

      // x_center, y_center, width, height
      const w = bbox_data[i + NUM_PREDICTIONS * 2] * xRatio;
      const h = bbox_data[i + NUM_PREDICTIONS * 3] * yRatio;
      const xc = bbox_data[i] * xRatio;
      const yc = bbox_data[i + NUM_PREDICTIONS] * yRatio;

      detections[resultCount++] = {
        xywh: [xc, yc, w, h],
        cls_idx,
        score: maxScore,
      };
    }
    output0.dispose();

    // nms
    const selected_indices = applyNMS(
      detections,
      detections.map((r) => r.score)
    );
    const nms_detections = selected_indices.map((i) => detections[i]);

    const tracked_objects = tracker.update(nms_detections);

    return [tracked_objects, (end - start).toFixed(2)];
  } catch (error) {
    console.error("Inference error:", error.name, error.message, error.stack);
    return [[], "0.00"];
  }
}
