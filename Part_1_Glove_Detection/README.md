Gloved vs Bare Hand Detection — Component Guide

This component contains the inference script and directories for annotated outputs and JSON logs. It pairs with the repository-level training configuration.

Contents
detection_script.py — CLI for running detection on a folder or a single image

output/ — annotated images written by the script

logs/ — per-image JSON detections

Installation (PowerShell)
From the repo root:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
Dataset and Labels
Directories (from repo root):

dataset/train/images, dataset/train/labels
dataset/valid/images, dataset/valid/labels
dataset/test/images,  dataset/test/labels
Labels follow YOLO format (class x_center y_center width height) normalized to. Class mapping:

0: gloved_hand
1: bare_hand

Training
The training is launched from the repo root using train_glove.yaml with strong augmentations:

python -m ultralytics cfg=train_glove.yaml

Key outputs:
runs/detect/train/weights/best.pt — best-performing checkpoint
Preview plots and metrics inside runs/detect/train/

Tips:
If previews show unrealistic warping, reduce mosaic to 0.3–0.4 and scale to 0.25–0.30.
For faster experiments, switch to yolov8n.pt in train_glove.yaml.

Inference
Run detection on a folder of images:

python .\Part_1_Glove_Detection\detection_script.py `
  --weights .\runs\detect\train\weights\best.pt `
  --input .\dataset\test\images `
  --output .\Part_1_Glove_Detection\output `
  --logs .\Part_1_Glove_Detection\logs `
  --confidence 0.35 `
  --iou 0.5 `
  --imgsz 640


Script arguments:

--weights: path to trained .pt weights
--input: image file or directory (recurses)
--output: folder for annotated images (default: Part_1_Glove_Detection/output)
--logs: folder for JSON logs (default: Part_1_Glove_Detection/logs)
--confidence: detection confidence threshold (default: 0.35)
--iou: NMS IoU threshold (default: 0.5)
--imgsz: inference image size (default: 640)
--device: '' for auto, 'cpu', or CUDA device id like '0'

Example JSON produced:

{
  "filename": "image.jpg",
  "detections": [
    {"label": "gloved_hand", "confidence": 0.91, "bbox": [120.3, 88.1, 310.9, 290.7]},
    {"label": "bare_hand",  "confidence": 0.77, "bbox": [422.5, 64.0, 590.2, 280.4]}
  ]
}

What Worked
Transfer learning from YOLOv8s converges quickly for two-class detection
Motion blur, brightness/contrast, and erasing augmentations improve robustness to camera shake and partial occlusions.
JSON logs streamline downstream analytics and compliance checks.

What Didn’t
Extreme occlusions and heavy backlighting remain difficult; consider domain-specific fine-tuning and larger imgsz.
Over-aggressive mosaic/scale can distort small hands; keep them moderate.

Exporting for Deployment
Export to ONNX (example):
python -m ultralytics export model=.\runs\detect\train\weights\best.pt format=onnx