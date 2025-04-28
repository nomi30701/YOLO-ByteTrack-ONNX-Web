import * as ort from "onnxruntime-web/webgpu";
import { BYTETracker } from "./tracker";
import { preProcess_dynamic, preProcess, applyNMS } from "./img_preprocess";

const tracker = new BYTETracker();

export const inference_pipeline = async (input_el, session) => {
  let input_tensor = null;
  let output0 = null;

  try {
    const src_mat = cv.imread(input_el);

    // const [src_mat_preProcessed, xRatio, yRatio] = await preProcess(
    //   src_mat,
    //   sessionsConfig.input_shape[2],
    //   sessionsConfig.input_shape[3]
    // );

    const [src_mat_preProcessed, div_width, div_height] =
      preProcess_dynamic(src_mat);
    const xRatio = src_mat.cols / div_width;
    const yRatio = src_mat.rows / div_height;
    src_mat.delete();

    input_tensor = new ort.Tensor("float32", src_mat_preProcessed.data32F, [
      1,
      3,
      div_height,
      div_width,
    ]);
    src_mat_preProcessed.delete();

    const start = performance.now();
    const { output0 } = await session.run({
      images: input_tensor,
    });
    const end = performance.now();

    input_tensor.dispose();

    // post process
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
      // filter low confidence for ByteTrack
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
    console.error("Inference error:", error);
    return [[], "0.00"];
  } finally {
    if (input_tensor) input_tensor.dispose();
    if (output0) output0.dispose();
  }
};
