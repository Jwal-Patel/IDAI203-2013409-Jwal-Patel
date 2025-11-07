# app_no_gradcam.py
"""
Streamlit app (no Grad-CAM)
Features:
 - Top-3 classification (upload or webcam)
 - Batch upload + CSV download
 - Metrics tab (shows plots/)
 - Smart Detection (YOLOv8) tab: detect objects and classify each crop
Note: grad_cam.py is NOT used here.
"""
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tensorflow as tf
import json, os, pathlib
import pandas as pd
from yolov8_integration import detect_and_classify  # optional: keep for YOLO tab

MODEL_PATH = "waste_model_improved.h5"
DATA_DIR = "dataset"
POSSIBLE_IDX2NAME = ["class_indices_idx2name.json", "class_indices.json"]
POSSIBLE_NAME2IDX = ["class_indices_name2idx.json", "class_indices.json"]

st.set_page_config(page_title="Waste Classifier (No Grad-CAM)", layout="wide")

@st.cache_resource
def load_model_and_mapping():
    model = tf.keras.models.load_model(MODEL_PATH)
    idx_to_class = None
    # try mapping files
    for fname in POSSIBLE_IDX2NAME + POSSIBLE_NAME2IDX:
        if os.path.exists(fname):
            with open(fname, "r") as f:
                data = json.load(f)
            if all(str(k).isdigit() for k in data.keys()):
                idx_to_class = {int(k): v for k, v in data.items()}
                break
            if all(isinstance(v, int) for v in data.values()):
                idx_to_class = {int(v): k for k, v in data.items()}
                break
    # fallback to dataset
    if idx_to_class is None:
        if os.path.exists(DATA_DIR):
            classes = sorted([d.name for d in pathlib.Path(DATA_DIR).iterdir() if d.is_dir()])
            idx_to_class = {i: cls for i, cls in enumerate(classes)}
        else:
            raise FileNotFoundError("No class mapping or dataset found.")
    return model, idx_to_class

model, idx_to_class = load_model_and_mapping()

# helper for classification
def get_img_array(img_pil, target_size=(224,224)):
    img = img_pil.resize(target_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_pil(img_pil):
    arr = get_img_array(img_pil, target_size=(224,224))
    preds = model.predict(arr)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    results = [(idx_to_class.get(int(i), str(i)), float(preds[int(i)])) for i in top3_idx]
    return results, preds

# UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Webcam", "Smart Detection (YOLOv8)", "Metrics", "About"])

# Home: upload images, show top-3 and CSV
if page == "Home":
    st.title("♻️ Waste Classifier — Upload Image(s)")
    uploaded_files = st.file_uploader("Upload one or multiple images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        results_table = []
        cols = st.columns(2)
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 2]:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption=uploaded_file.name, use_container_width=True)
                results, preds = predict_pil(img)
                st.subheader("Top Predictions")
                for rank, (cls, conf) in enumerate(results, start=1):
                    st.write(f"{rank}. {cls} — {conf*100:.2f}%")
                if preds.max() < 0.6:
                    st.warning("Low confidence — please verify manually.")
                results_table.append({"filename": uploaded_file.name, "predicted": results[0][0], "confidence": float(results[0][1])})
        df = pd.DataFrame(results_table)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

# Webcam
elif page == "Webcam":
    st.title("📸 Webcam Mode")
    if st.checkbox("Enable webcam capture"):
        img_file_buffer = st.camera_input("Take a photo")
        if img_file_buffer:
            img = Image.open(img_file_buffer).convert("RGB")
            st.image(img, caption="Captured", use_container_width=True)
            results, preds = predict_pil(img)
            st.subheader("Top Predictions")
            for rank, (cls, conf) in enumerate(results, start=1):
                st.write(f"{rank}. {cls} — {conf*100:.2f}%")
            if preds.max() < 0.6:
                st.warning("Low confidence — please verify manually.")

# Smart Detection (YOLO)
elif page == "Smart Detection (YOLOv8)":
    st.title("🧠 Smart Detection — YOLOv8 + Classifier")
    uploaded_file = st.file_uploader("Upload an image (for detection)", type=["jpg","jpeg","png"])
    use_gpu = st.checkbox("Run YOLO on GPU (cuda) if available", value=True)
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Original Image", use_container_width=True)
        st.info("Running detection + classification...")
        try:
            device = "cuda" if use_gpu else "cpu"
            results = detect_and_classify(img, model, idx_to_class, conf_thresh=0.35, device=device)
            if not results:
                st.warning("No objects detected.")
            else:
                # draw boxes and show results
                img_draw = img.copy()
                draw = ImageDraw.Draw(img_draw)
                font = ImageFont.load_default()
                for r in results:
                    x1,y1,x2,y2 = r["box"]
                    label = f"{r['predicted_class']} ({r['pred_confidence']*100:.1f}%)"
                    draw.rectangle([x1,y1,x2,y2], outline="lime", width=3)
                    draw.text((x1+3,y1+3), label, fill="white", font=font)
                st.image(img_draw, caption="Detected objects and labels", use_container_width=True)
                df = pd.DataFrame(results)
                st.subheader("Detection Results")
                st.dataframe(df)
        except Exception as e:
            st.error(f"YOLO detection failed: {e}")

# Metrics
elif page == "Metrics":
    st.title("📊 Metrics")
    plots_dir = "plots"
    for plot in ["accuracy.png","loss.png","confusion_matrix.png"]:
        path = os.path.join(plots_dir, plot)
        if os.path.exists(path):
            st.image(path, caption=plot, use_container_width=True)
    txt_path = os.path.join(plots_dir, "classification_report.txt")
    if os.path.exists(txt_path):
        st.subheader("Classification Report")
        st.text(open(txt_path).read())

# About
elif page == "About":
    st.title("About")
    st.markdown("""
    **Waste Classifier (no Grad-CAM)**  
    - Model: MobileNetV2 transfer learning  
    - Features: Top-3 predictions, batch CSV, webcam mode, YOLOv8 detection + per-crop classification  
    - Developer: Your Name
    """)
