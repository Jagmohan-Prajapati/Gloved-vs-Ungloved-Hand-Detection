import argparseimport json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

# Lazy import with error message for clarity
try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: Failed to import ultralytics. Install with: pip install ultralytics")
    raise

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_box(img, box, label, conf, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Background for text
    top_left = (x1, max(0, y1 - th - baseline))
    bottom_right = (x1 + tw + 2, max(0, y1))
    cv2.rectangle(img, top_left, bottom_right, color, -1)
    cv2.putText(img, text, (x1 + 1, max(0, y1 - baseline - 1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def collect_images(input_dir: Path):
    if input_dir.is_file() and input_dir.suffix.lower() in SUPPORTED_EXTS:
        return [input_dir]
    if not input_dir.exists():
        raise SystemExit(f"Input path not found: {input_dir}")
    files = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files

def run_inference(
    weights_path: str,
    input_dir: str,
    output_dir: str,
    logs_dir: str,
    confidence: float,
    imgsz: int,
    device: str,
    iou: float
):
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    logs_path = Path(logs_dir)

    ensure_dir(out_path)
    ensure_dir(logs_path)

    image_files = collect_images(input_path)
    if not image_files:
        raise SystemExit(f"No images found in {input_dir}. Supported: {', '.join(sorted(SUPPORTED_EXTS))}")

    # Load model
    model = YOLO(weights_path)

    # Class names from model (fallback provided)
    names = model.names if hasattr(model, "names") and model.names else {0: "gloved_hand", 1: "bare_hand"}
    # Colors per label
    colors = {
        "gloved_hand": (0, 200, 0),
        "bare_hand": (0, 0, 200)
    }

    # Predict per image to keep memory low; alternatively pass list to model for batch infer
    for img_path in tqdm(image_files, desc="Processing"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        results = model.predict(
            source=str(img_path),
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        detections = []
        annotated = img.copy()

        for r in results:
            if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
                continue

            boxes_xyxy = r.boxes.xyxy
            clss = r.boxes.cls
            confs = r.boxes.conf

            # Move to CPU numpy if tensors
            boxes = boxes_xyxy.cpu().numpy() if hasattr(boxes_xyxy, "cpu") else np.array(boxes_xyxy)
            cls_ids = clss.cpu().numpy().astype(int) if hasattr(clss, "cpu") else np.array(clss, dtype=int)
            conf_vals = confs.cpu().numpy() if hasattr(confs, "cpu") else np.array(confs)

            for box, cls_id, conf in zip(boxes, cls_ids, conf_vals):
                x1, y1, x2, y2 = box.tolist()
                label = names.get(int(cls_id), str(int(cls_id)))
                detections.append({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
                color = colors.get(label, (255, 165, 0))
                draw_box(annotated, [x1, y1, x2, y2], label, float(conf), color)

        # Save annotated image with same filename under output/
        out_img_path = out_path / img_path.name
        cv2.imwrite(str(out_img_path), annotated)

        # Save JSON log under logs/ with same stem
        log_obj = {
            "filename": img_path.name,
            "detections": detections
        }
        with open(logs_path / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(log_obj, f, indent=2, ensure_ascii=False)

def build_parser():
    parser = argparse.ArgumentParser(
        description="Detect gloved_hand and bare_hand with YOLOv8 and save annotated images + JSON logs."
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--input", type=str, required=True, help="Input image file or directory (recurses)")
    parser.add_argument("--output", type=str, default="Part_1_Glove_Detection/output", help="Output folder for annotated images")
    parser.add_argument("--logs", type=str, default="Part_1_Glove_Detection/logs", help="Logs folder for per-image JSON files")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (0-1)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="", help="'' (auto), 'cpu', or CUDA device id like '0'")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_inference(
            weights_path=args.weights,
            input_dir=args.input,
            output_dir=args.output,
            logs_dir=args.logs,
            confidence=args.confidence,
            imgsz=args.imgsz,
            device=args.device,
            iou=args.iou
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except SystemExit as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
