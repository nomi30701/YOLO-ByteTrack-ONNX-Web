import * as ort from "onnxruntime-web/webgpu";
import { preProcess_dynamic, preProcess, applyNMS } from "./img_preprocess";

/**
 * Inference pipeline for YOLO model.
 * @param {cv.Mat} src_mat - Input image Mat.
 * @param {[Number, Number]} overlay_size - Overlay width and height. [width, height]
 * @param {ort.InferenceSession} session - YOLO model session.
 * @param {BYTETracke} tracker - Object tracker instance.
 * @returns {[Array[Object], Number]} - Array of predictions and inference time.
 */
export async function inference_pipeline(
  src_mat,
  overlay_size,
  session,
  tracker
) {
  try {
    // const [src_mat_preProcessed, xRatio, yRatio] = await preProcess(
    //   src_mat,
    //   640,
    //   640
    // );

    const [src_mat_preProcessed, div_width, div_height] =
      preProcess_dynamic(src_mat);
    const xRatio = overlay_size[0] / div_width; // scale factor for overlay
    const yRatio = overlay_size[1] / div_height;
    src_mat.delete();

    const input_tensor = new ort.Tensor(
      "float32",
      src_mat_preProcessed.data32F,
      [1, 3, div_height, div_width]
    );
    src_mat_preProcessed.delete();

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
      // Filter low confidence for ByteTrack
      if (maxScore <= 0.2) continue;

      // x_center, y_center, width, height
      const xc = bbox_data[i] * xRatio;
      const yc = bbox_data[i + NUM_PREDICTIONS] * yRatio;
      const w = bbox_data[i + NUM_PREDICTIONS * 2] * xRatio;
      const h = bbox_data[i + NUM_PREDICTIONS * 3] * yRatio;

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
