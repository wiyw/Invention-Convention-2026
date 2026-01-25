from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

# Config: input image path
IMAGE_PATH = filedialog.askopenfilename(title="Select an image for YOLO26 Nano inference") 

# 1. Load the pre-trained YOLO26 Nano model
model = YOLO("yolo26n.pt")

# 2. Run inference on the image
results = model(IMAGE_PATH)

# 3. Extract first result
result = results[0]

# Try to get the original image from the result; fall back to cv2.imread
if hasattr(result, 'orig_img') and result.orig_img is not None:
    img = result.orig_img.copy()
else:
    img = cv2.imread(IMAGE_PATH)

if img is None:
    raise RuntimeError(f"Failed to load image: {IMAGE_PATH}")

height, width = img.shape[:2]

# Helper to convert tensors/arrays to python lists
def to_list(x):
    try:
        return x.cpu().numpy().tolist()
    except Exception:
        try:
            return np.array(x).tolist()
        except Exception:
            return []

# Names map for classes (if available)
names = getattr(model, 'names', {}) or {}

# Build structured detections list
detections = []
boxes = getattr(result, 'boxes', None)
if boxes is not None:
    xyxy = to_list(getattr(boxes, 'xyxy', []))
    confs = to_list(getattr(boxes, 'conf', []))
    clss = to_list(getattr(boxes, 'cls', []))
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box
        conf = float(confs[i]) if i < len(confs) else None
        cls = int(clss[i]) if i < len(clss) else None
        label = names.get(cls, str(cls))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        detections.append({
            'id': i,
            'label': label,
            'class_id': cls,
            'confidence': conf,
            'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
            'bbox_xywh': {'cx': float(cx), 'cy': float(cy), 'w': float(bw), 'h': float(bh)},
            'bbox_norm': {'cx': float(cx / width), 'cy': float(cy / height), 'w': float(bw / width), 'h': float(bh / height)}
        })

# Annotate image (draw rectangles + labels)
annot = img.copy()
for det in detections:
    bb = det['bbox']
    x1, y1, x2, y2 = bb['x1'], bb['y1'], bb['x2'], bb['y2']
    label = det['label']
    conf = det['confidence']
    text = f"{label} {conf:.2f}" if conf is not None else label
    cv2.rectangle(annot, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annot, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Output paths: include original filename + "_result"
out_dir = os.path.dirname(IMAGE_PATH)
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
annotated_fname = f"{base_name}_result.jpg"
json_fname = f"{base_name}_result.json"
annotated_path = os.path.join(out_dir, annotated_fname)
json_path = os.path.join(out_dir, json_fname)

# Save annotated image and JSON
cv2.imwrite(annotated_path, annot)
output = {
    'source_image_path': os.path.abspath(IMAGE_PATH),
    'source_image_name': os.path.basename(IMAGE_PATH),
    'result_image_name': annotated_fname,
    'result_json_name': json_fname,
    'result_image_path': os.path.abspath(annotated_path),
    'result_json_path': os.path.abspath(json_path),
    'width': width,
    'height': height,
    'detections': detections
}
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

print(f"Saved annotated image: {annotated_path}")
print(f"Saved structured JSON: {json_path}")