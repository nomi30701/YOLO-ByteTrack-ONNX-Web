# YOLO Multi-Object Tracking Web App

<div align="center">
<img src="https://github.com/nomi30701/YOLO-ByteTrack-ONNX-Web/blob/main/preview.png" width="80%" alt="YOLO Multi-Task Preview">
<br><br>
<img src="https://github.com/nomi30701/YOLO-ByteTrack-ONNX-Web/blob/main/preview.gif" width="80%" alt="YOLO Multi-Task Demo">
</div>

## ✨ Features

This web application leverages ONNX Runtime Web and YOLO models for multi-object detection and tracking.

It supports both YOLO11 and YOLO12 models, with the ByteTrack algorithm for object tracking.

Users can upload videos or use their webcam for real-time tracking, all accelerated by WebGPU or WASM on CPU.

- 🔍 **Object Detection** - Accurately detect and locate multiple objects using YOLO11 and YOLO12 models
- 👀 **Object Tracking** - Track detected objects across frames with the ByteTrack algorithm
- 📹 **Video Processing** - Upload mp4 videos to process and track objects
- 🖥️ **Real-Time Tracking** - Use webcam for live object tracking
- ⚙️ **Custom Model Support** - Use custom YOLO models by updating class definitions

## 💻 Technical Support

- ⚡ **WebGPU Acceleration** - Utilize WebGPU for faster inference on supported devices
- 🧠 **WASM (CPU)** - Fallback to WASM for CPU-based inference

## 📊 Available Models

| Model                                                   | Input Size | Param. |                 Best For                  |
| :------------------------------------------------------ | :--------: | :----: | :---------------------------------------: |
| [YOLO11-N](https://docs.ultralytics.com/models/yolo11/) |    640     |  2.6M  | 📱 Mobile devices & real-time applications |
| [YOLO11-S](https://docs.ultralytics.com/models/yolo11/) |    640     |  9.4M  |      🖥️ Higher accuracy requirements       |
| [YOLO12-N](https://docs.ultralytics.com/models/yolo12/) |    640     |  2.6M  | 📱 Mobile devices & real-time applications |
| [YOLO12-S](https://docs.ultralytics.com/models/yolo12/) |    640     |  9.3M  |      🖥️ Higher accuracy requirements       |

## 🛠️ Installation Guide

1. Clone this repository
```bash
git clone https://github.com/nomi30701/YOLO-ByteTrack-ONNX-Web.git
```

2. cd to the project directory
```bash
cd YOLO-ByteTrack-ONNX-Web
```

Install dependencies
```bash
yarn install
```

## 🚀 Running the Project

Start development server
```bash
yarn dev
```

Build the project
```bash
yarn build
```

## 🔧 Using Custom YOLO Models

To use a custom YOLO model, follow these steps:

### Step 1: Convert your model to ONNX format

Use Ultralytics or your preferred method to export your YOLO model to ONNX format. Ensure to use `opset=12` for WebGPU compatibility.

```python
from ultralytics import YOLO

# Load your model
model = YOLO("path/to/your/model.pt")

# Export to ONNX
model.export(format="onnx", opset=12, dynamic=True)
```

### Step 2: Add the model to the project

You can either:

- 📁 Copy your ONNX model file to the `./public/models/` directory
- 🔄 Upload your model directly through the `**Add model**` button in the web interface 

#### 📁 Copy your ONNX model file to the `./public/models/` directory

In App.jsx

```jsx
<label htmlFor="model-selector">Model:</label>
<select name="model-selector">
  <option value="yolo12n">yolo12n-2.6M</option>
  <option value="yolo12s">yolo12s-9.3M</option>
  <option value="your-custom-model-name">Your Custom Model</option>
</select>
```

Replace `"your-custom-model-name"` with the filename of your ONNX model.


### Step 3: Update class definitions

Update the `src/utils/yolo_classes.json` file with the class names that your custom model uses. This file should contain a dict of strings representing the class labels.

For example:

```json
{"class": 
  {"0": "person", 
   "1": "bicycle",
   "2": "car",
   "3": "motorcycle",
   "4": "airplane"
  }
}
```

Make sure the classes match exactly with those used during training of your custom model.

### Step 4: Refresh and select your new model 🎉

## 💡 Advanced Configuration Tips
> 📏 **Dynamic Input Size**
> Dynamic input size support is enabled by default. For fixed size, modify `/utils/inference_pipeline.js`:
> 1. Uncomment this code:
> ```Javascript
> const [src_mat_preProcessed, xRatio, yRatio] = await preProcess(
>   src_mat,
>   640,
>   640
> );
> ```
> 
> 2. Remove the dynamic sizing code:
> ```Javascript
> const [src_mat_preProcessed, div_width, div_height] = preProcess_dynamic(src_mat);
> const xRatio = overlay_el.width / div_width;
> const yRatio = overlay_el.height / div_height;
> ```
>
> 3. Change Tensor size
> ```Javascript
> const input_tensor = new ort.Tensor("float32", src_mat_preProcessed.data32F, [
>   1,
>   3,
>   640,
>   640,
> ]);
>

> 🚀 WebGPU Support
>
> Ensure you set `opset=12` when exporting ONNX models, as this is required for WebGPU compatibility.
