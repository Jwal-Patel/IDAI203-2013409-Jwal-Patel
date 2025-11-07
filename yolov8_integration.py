"""
yolov8_integration.py (fixed)
- Uses ultralytics YOLOv8 for detection and a provided classifier model for per-crop classification.
- No dependency on grad_cam.py.
- detect_and_classify accepts either a PIL.Image or a file path.
"""
from PIL import Image
import numpy as np

def get_img_array(img, target_size=(224,224)):
    """
    Convert a PIL.Image to a normalized numpy array shape (1,H,W,3) in [0,1].
    """
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def detect_and_classify(image_path_or_pil, classifier_model, idx_to_class, conf_thresh=0.3, device="cpu"):
    """
    Runs YOLOv8 detection and classifies each crop using classifier_model.
    Returns a list of dicts: {box, yolo_score, yolo_class, predicted_class, pred_confidence}.
    - image_path_or_pil: path string or PIL.Image
    - classifier_model: tf.keras.Model or similar with .predict returning probs
    - idx_to_class: dict mapping index->class name
    - conf_thresh: min confidence for YOLO detections
    - device: "cpu" or "cuda"
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("ultralytics (YOLOv8) is required. Install with: pip install ultralytics") from e

    # Load the YOLO model (this uses pretrained weights by default)
    # For speed use yolov8n.pt; if you have a custom .pt put its path here.
    yolo = YOLO("yolov8n.pt")

    # accept PIL.Image or path
    if isinstance(image_path_or_pil, Image.Image):
        pil = image_path_or_pil.convert("RGB")
    else:
        pil = Image.open(image_path_or_pil).convert("RGB")

    # Run detection
    # device: 'cuda' or 'cpu' (ultralytics handles auto-detect too)
    results = yolo.predict(source=np.array(pil), device=device, verbose=False)
    if len(results) == 0:
        return []

    detections = results[0].boxes  # Boxes object
    outputs = []
    # iterate detections; skip low-confidence ones
    for box in detections:
        try:
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]  # x1,y1,x2,y2
            score = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
        except Exception:
            # fallback if box fields not present
            continue
        if score < conf_thresh:
            continue
        # ensure coords are within image bounds
        x1, y1, x2, y2 = xyxy
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(pil.width-1, x2); y2 = min(pil.height-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = pil.crop((x1, y1, x2, y2)).resize((224,224))
        arr = get_img_array(crop, target_size=(224,224))
        # classifier_model expected to be TF/Keras; adapt if different
        try:
            preds = classifier_model.predict(arr)[0]
            pred_idx = int(preds.argmax())
            pred_conf = float(preds[pred_idx])
            pred_name = idx_to_class.get(pred_idx, str(pred_idx))
        except Exception as e:
            # classifier failed, skip
            pred_name = "error"
            pred_conf = 0.0

        outputs.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "yolo_score": float(score),
            "yolo_class": int(cls),
            "predicted_class": pred_name,
            "pred_confidence": float(pred_conf)
        })
    return outputs

if __name__ == "__main__":
    print("yolov8_integration.py module. Use detect_and_classify(...)")
