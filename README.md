# Glove Compliance Detection (Gloved vs Bare Hands)

End-to-end project for detecting **gloved hands** and **bare hands** in factory images using **YOLOv8**.  
The repository includes:
- Training configuration with strong augmentations  
- A CLI inference script to annotate images and emit JSON logs  
- A reasoning write-up  

---

## Project Structure
```

.
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── Part\_1\_Glove\_Detection/
│   ├── detection\_script.py
│   ├── output/
│   ├── logs/
│   └── README.md
├── Part\_2\_Answers.pdf
├── dataset.yaml
├── train\_glove.yaml
├── requirements.txt
└── README.md

````

---

## Setup (Windows PowerShell)

### 1. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> If using NVIDIA GPU:
> Install a CUDA-matched PyTorch build first, then run the requirements install again.

---

## Dataset Configuration

* Place YOLO-formatted images and labels under:

  ```
  dataset/{train,valid,test}/{images,labels}
  ```
* Each image must have a same-named `.txt` in the corresponding labels folder with YOLO box lines:

```
class x_center y_center width height   # all normalized
```

### Example `dataset.yaml`

```yaml
path: .
train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images

names:
  0: gloved_hand
  1: bare_hand
```

---

## Training with Augmentations

Create **`train_glove.yaml`** at the repo root:

```yaml
task: detect
mode: train

model: yolov8s.pt
data: dataset.yaml
epochs: 120
imgsz: 640
batch: 16
workers: 8
optimizer: auto
cos_lr: true
patience: 30

# Augmentations tuned for factory environments
hsv_h: 0.015
hsv_s: 0.70
hsv_v: 0.40
degrees: 5.0
translate: 0.08
scale: 0.40
shear: 3.0
perspective: 0.0005
flipud: 0.0
fliplr: 0.5
mosaic: 0.6
mixup: 0.15
copy_paste: 0.0
erasing: 0.3

# Extra Albumentations
augment: true
albumentations: [
  {"type": "MotionBlur", "p": 0.25, "blur_limit": 5},
  {"type": "GaussianBlur", "p": 0.15, "blur_limit": 3},
  {"type": "GaussNoise", "p": 0.20, "var_limit": [5.0, 25.0]},
  {"type": "CLAHE", "p": 0.15, "clip_limit": 2.0, "tile_grid_size": [8, 8]},
  {"type": "RandomBrightnessContrast", "p": 0.25, "brightness_limit": 0.2, "contrast_limit": 0.2},
  {"type": "RGBShift", "p": 0.10, "r_shift_limit": 10, "g_shift_limit": 10, "b_shift_limit": 10},
  {"type": "ToGray", "p": 0.05}
]

val: true
save: true
plots: true

device: ''         # '' auto, 'cpu', or CUDA id like '0'
lr0: 0.003
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
```

### Start training

```powershell
python -m ultralytics cfg=train_glove.yaml
```

* Artifacts will appear under: `runs/detect/train/`
* Best weights: `runs/detect/train/weights/best.pt`
* Inspect `train_batch*.jpg` and `val_batch*.jpg` for augmentation realism.

>  If mosaics look too distorted, reduce:
>
> * `mosaic: 0.3–0.4`
> * `scale: 0.25–0.30`

---

## Inference

Run the inference script to annotate images and log detections.

### Example (PowerShell):

```powershell
python .\Part_1_Glove_Detection\detection_script.py `
  --weights .\runs\detect\train\weights\best.pt `
  --input .\dataset\test\images `
  --output .\Part_1_Glove_Detection\output `
  --logs .\Part_1_Glove_Detection\logs `
  --confidence 0.35 `
  --iou 0.5 `
  --imgsz 640
```

Each image produces:

1. An annotated output file
2. A JSON log like:

```json
{
  "filename": "image1.jpg",
  "detections": [
    {"label": "gloved_hand", "confidence": 0.92, "bbox": [x1, y1, x2, y2]},
    {"label": "bare_hand", "confidence": 0.85, "bbox": [x1, y1, x2, y2]}
  ]
}
```

---

## 🛠 Troubleshooting

* **`yolo is not recognized`**
  Use module form:

  ```powershell
  python -m ultralytics cfg=train_glove.yaml
  ```

* **Slow training on CPU**
  Reduce:

  ```yaml
  batch: 8
  imgsz: 512
  ```

* **Many false positives**

  * Lower `mixup: 0.05` or disable
  * Increase confidence threshold at inference

* **Misses on motion**

  * Keep `MotionBlur` and `GaussNoise` enabled
  * Try larger `imgsz` (768–960)
  * Resume training with more epochs

---

## References

* [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
* [Albumentations Augmentations](https://albumentations.ai/docs/)