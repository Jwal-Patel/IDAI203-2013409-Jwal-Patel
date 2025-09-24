import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

# Load model
model = tf.keras.models.load_model("waste_model.h5")

# Load class indices
with open("class_indices.json") as f:
    class_indices = json.load(f)

# Reverse mapping (index → class name)
idx_to_class = {v: k for k, v in class_indices.items()}

# Bin mapping
bin_mapping = {
    "biological": "Green Bin (Biodegradable)",
    "paper": "Blue Bin (Recyclable)",
    "cardboard": "Blue Bin (Recyclable)",
    "plastic": "Blue Bin (Recyclable)",
    "metal": "Blue Bin (Recyclable)",
    "glass": "Blue Bin (Recyclable)",
    "clothes": "Blue Bin (Recyclable)",
    "shoes": "Blue Bin (Recyclable)",
    "battery": "Red Bin (Hazardous)",
    "trash": "Red Bin (Hazardous)"
}

# Preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("♻️ SmartWasteAI - Waste Classification System")
st.write("Upload a waste image to classify and get correct bin recommendation.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Prediction
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    predicted_class = idx_to_class[class_idx]
    confidence = np.max(preds) * 100

    # Bin recommendation
    bin_rec = bin_mapping.get(predicted_class, "Unknown")

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Waste Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Recommended Bin:** 🗑️ {bin_rec}")

