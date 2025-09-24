# IDAI203-2013409-Jwal-Patel
## ♻️ SmartWasteAI – Waste Classification & Bin Recommendation System  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)]()  
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-success)]()  

---

### 📌 Project Overview  
SmartWasteAI is a **Computer Vision-based Waste Segregation System** designed for **smart cities**.  
The system:  
- Classifies waste images into categories like *plastic, glass, battery, biological, clothes, paper, etc.*  
- Groups them into **Biodegradable, Recyclable, or Hazardous**.  
- Recommends the correct **color-coded bin**:  
  - ✅ Green → Biodegradable  
  - ✅ Blue → Recyclable  
  - ✅ Red → Hazardous  
- Provides **confidence scores** for predictions.  
- Offers a clean, interactive **Streamlit web interface**.  

This project helps reduce landfill waste, improves recycling rates, and supports sustainable smart cities.  

---

### 🎯 Learning Outcomes  
- Applied **Machine Learning & Deep Learning** techniques to a real-world challenge.  
- Understood how **Computer Vision** can help with environmental issues.  
- Built and deployed an AI-powered **web app** using **Streamlit Cloud**.  

---

### 🗂️ Dataset  
The dataset used contains **10 classes** of waste items:  


- Each class has ~100+ images.  
- Images resized to **224x224 px**.  
- Data preprocessing included **augmentation** (rotation, flipping, brightness, zoom).  
- Dataset split: **70% Training, 15% Validation, 15% Testing**.  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **TensorFlow / Keras** (MobileNetV2 transfer learning)  
- **OpenCV** (image preprocessing)  
- **Streamlit** (web app & deployment)  
- **NumPy, Matplotlib, scikit-learn** (metrics & visualization)  

---

## 🧑‍💻 Model Development  
- Base Model: **MobileNetV2** (pretrained on ImageNet).  
- Fine-tuned for **10 waste categories**.  
- Optimizer: `Adam (lr=0.0001)`  
- Loss: `categorical_crossentropy`  
- Metrics: `accuracy`, `confusion matrix`.  

Final model: `waste_model.h5`  
Class index mapping: `class_indices.json`  

---

## 🔄 Bin Recommendation Logic  
```python
if class in ["biological"]:
    bin = "Green Bin (Biodegradable)"
elif class in ["paper","cardboard","plastic","metal","glass","clothes","shoes"]:
    bin = "Blue Bin (Recyclable)"
elif class in ["battery","trash"]:
    bin = "Red Bin (Hazardous)"


## 🌐 Streamlit App
 - The app (app.py) allows users to:

 - Upload an image.

 - View uploaded image preview.

 - See predicted waste type + confidence %.

 - Get the recommended bin color.

 - Example Output:
Waste Type: Plastic

Confidence: 92.5%

Recommended Bin: Blue Bin (Recyclable)

 ## 📸 Screenshots:
<img width="967" height="1344" alt="image" src="https://github.com/user-attachments/assets/f4d0823d-2a91-4add-83da-617ed9822207" />
<img width="902" height="1352" alt="image" src="https://github.com/user-attachments/assets/466996c0-6623-44e4-b7da-9209e1203b2d" />
<img width="937" height="1357" alt="image" src="https://github.com/user-attachments/assets/0accf960-44f9-4c7f-93c9-e87f7bc75c22" />
<img width="939" height="1345" alt="image" src="https://github.com/user-attachments/assets/f1220ca0-73b1-4a7f-a386-e12c84c6f314" />
<img width="894" height="1304" alt="image" src="https://github.com/user-attachments/assets/5739d515-dc29-464d-a7ea-c825f405e61f" />

#Live Demo (Published APP Link):
https://idai203-2013409-jwal-patel-wasteclassificationsystem.streamlit.app/
