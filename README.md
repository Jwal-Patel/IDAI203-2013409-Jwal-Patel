# ♻️ Smart Waste Detection and Classification App

## 🧠 Overview
This project is a **Deep Learning–based Smart Waste Classification System** that leverages **Convolutional Neural Networks (CNNs)** for waste classification and **YOLOv8** for object detection. The app provides an AI-driven interface where users can upload images, use a webcam, and visualize model metrics through an interactive **Streamlit web application**.

It aims to automate the process of waste classification by identifying different types of waste such as **cardboard, glass, metal, paper, plastic, and trash** using a trained MobileNetV2-based classifier.

---

## ⚙️ Technology Stack
- **Python 3.10+**
- **TensorFlow 2.15.0** – for training and inference of the classification model  
- **Keras** – high-level deep learning API for model building  
- **Streamlit** – for building the user-friendly web application  
- **YOLOv8 (Ultralytics)** – for real-time object detection  
- **NumPy, Pandas, Matplotlib, PIL** – for preprocessing and visualization  

---

## 🧩 Model Architecture and Workflow
1. **Feature Extraction Model:**  
   The classifier is built using **MobileNetV2**, a pre-trained CNN model fine-tuned for waste classification.
2. **Data Processing:**  
   Images are preprocessed and normalized to ensure efficient model convergence.
3. **Callbacks and Fine-Tuning:**  
   Model training uses early stopping, learning rate reduction, and fine-tuning for improved accuracy.
4. **Evaluation:**  
   The model generates accuracy, loss, and confusion matrix plots for analysis.
5. **Deployment:**  
   The final model is deployed in a Streamlit app that supports:
   - Top-3 prediction display  
   - Webcam-based prediction  
   - Batch image uploads with CSV export  
   - YOLOv8 hybrid detection mode (detect multiple waste objects)  

---

## 🚀 App Features

| Feature | Description |
|----------|-------------|
| 🧠 **Top-3 Predictions** | Displays top three predicted waste categories with confidence scores. |
| 📸 **Webcam Capture** | Capture an image directly from the webcam for instant classification. |
| 📂 **Batch Upload & CSV Export** | Upload multiple images simultaneously and export predictions to CSV. |
| 📊 **Metrics Tab** | View accuracy, loss curves, and confusion matrix of the trained model. |
| 🔍 **Smart Detection (YOLOv8)** | Detects multiple objects in a single image and classifies each crop using the trained CNN. |
| ⚡ **GPU Acceleration** | Automatically uses the NVIDIA RTX GPU (if available) for faster inference. |

---

## 🔬 Limitations

| Area | Limitation | Explanation |
|-------|-------------|-------------|
| **YOLOv8 Detection** | Not trained on waste categories | The YOLOv8 model (`yolov8n.pt`) is pre-trained on the COCO dataset (80 common objects). It cannot detect specific waste types like *cardboard, plastic,* etc. It is currently used only for region detection. |
| **Bounding Box Accuracy** | May detect irrelevant objects | Since the model is not waste-specific, it may miss or wrongly detect objects that don’t match COCO categories. |
| **Single-Object Classifier** | Requires clear object images | The classifier performs best when one object is clearly visible in the image. |
| **Limited Dataset** | Training dataset was small | Accuracy depends on dataset diversity. A larger dataset would yield better results. |
| **No Real-Time Video Support** | Only static image input | The app processes one image at a time for accuracy and stability. |

---

## 📦 Folder Structure

```
project_root/
│
├── app_no_gradcam.py              # Streamlit app (no Grad-CAM)
├── train_model_improved.py        # Model training script with callbacks & fine-tuning
├── yolov8_integration.py          # YOLOv8 detection + classifier hybrid module
├── waste_model_improved.h5        # Trained classification model
├── class_indices.json             # Class index mapping
├── requirements.txt               # Required libraries
├── plots/                         # Training plots (accuracy, loss, confusion matrix)
└── dataset/                       # Image dataset used for training
```

---

## 🧾 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Classifier (optional if model already exists)
```bash
python train_model_improved.py
```

### 3️⃣ Run the Streamlit App
```bash
streamlit run app_no_gradcam.py
```

### 4️⃣ Using the App
- Upload an image or capture using webcam.
- View predicted waste type and confidence.
- Open the *Metrics* tab to see model performance.
- Try *Smart Detection (YOLOv8)* to visualize bounding boxes and predictions for multiple objects.

---

## 💡 Future Improvements
- Train a custom **YOLOv8 model** on waste-specific datasets for accurate detection of *cardboard, plastic, metal,* etc.
- Integrate **real-time video streaming** for live classification.
- Add **Grad-CAM visualization** (currently removed due to instability in nested model layers).
- Optimize model size for deployment on **edge devices (Raspberry Pi / Jetson Nano)**.

---

## 🧑‍💻 Contributors
- **Jwal Patel**  
  *IBCP Year 2 — Udgam School for Children*

---

## 🏁 Conclusion
The Smart Waste Detection and Classification App demonstrates the integration of **deep learning, computer vision, and web deployment** to create a practical environmental sustainability solution. While the YOLOv8 component is currently limited by pretrained COCO classes, it serves as a strong foundation for future work involving custom waste datasets and real-time detection.
