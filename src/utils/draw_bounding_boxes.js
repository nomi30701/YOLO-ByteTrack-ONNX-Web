import classes from "./yolo_classes.json";
import { Colors } from "./img_preprocess.js";

/**
 * Draw bounding boxes in overlay canvas based on task type.
 * @param {Array[Object]} predictions - Detection results
 * @param {HTMLCanvasElement} overlay_el - Show boxes in overlay canvas element
 */
export async function draw_bounding_boxes(predictions, overlayCtx) {
  if (!predictions) return;

  const predictionsByClass = {};

  // Calculate diagonal length of the canvas
  const diagonalLength = Math.sqrt(
    Math.pow(overlayCtx.canvas.width, 2) + Math.pow(overlayCtx.canvas.height, 2)
  );
  const lineWidth = diagonalLength / 250;

  predictions.forEach((predict) => {
    const classId = predict.cls_idx;
    if (!predictionsByClass[classId]) predictionsByClass[classId] = [];
    predictionsByClass[classId].push(predict);
  });

  Object.entries(predictionsByClass).forEach(([classId, items]) => {
    const color = Colors.getColor(Number(classId), 0.2);
    const borderColor = Colors.getColor(Number(classId), 0.8);
    const rgbaFillColor = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3]})`;
    const rgbaBorderColor = `rgba(${borderColor[0]}, ${borderColor[1]}, ${borderColor[2]}, ${borderColor[3]})`;

    overlayCtx.fillStyle = rgbaFillColor;
    items.forEach((predict) => {
      const [x1, y1, width, height] = predict.tlwh;
      overlayCtx.fillRect(x1, y1, width, height);
    });

    // draw bounding box
    overlayCtx.lineWidth = lineWidth;
    overlayCtx.strokeStyle = rgbaBorderColor;
    items.forEach((predict) => {
      const [x1, y1, width, height] = predict.tlwh;
      overlayCtx.strokeRect(x1, y1, width, height);
    });

    // draw score text
    overlayCtx.fillStyle = rgbaBorderColor;
    overlayCtx.font = "16px Arial";
    items.forEach((predict) => {
      const [x1, y1] = predict.tlwh;
      const text = `${predict.track_id} ${
        classes.class[predict.cls_idx]
      } ${predict.score.toFixed(2)}`;
      drawTextWithBackground(overlayCtx, text, x1, y1);
    });
  });
}

const fontCache = {
  font: "16px Arial",
  measurements: {},
};

function getMeasuredTextWidth(text, ctx) {
  if (!fontCache.measurements[text]) {
    fontCache.measurements[text] = ctx.measureText(text).width;
  }
  return fontCache.measurements[text];
}

/**
 * Helper function to draw text with background
 */
function drawTextWithBackground(ctx, text, x, y) {
  ctx.font = fontCache.font;
  const textWidth = getMeasuredTextWidth(text, ctx);
  const textHeight = 16;

  // Calculate the Y position for the text
  let textY = y - 5;
  let rectY = y - textHeight - 4;

  // Check if the text will be outside the canvas
  if (rectY < 0) {
    // Adjust the Y position to be inside the canvas
    textY = y + textHeight + 5;
    rectY = y + 1;
  }

  const currentFillStyle = ctx.fillStyle;
  ctx.fillRect(x - 1, rectY, textWidth + 4, textHeight + 4);
  ctx.fillStyle = "white";
  ctx.fillText(text, x, textY);
  ctx.fillStyle = currentFillStyle;
}
