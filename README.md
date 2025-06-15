# 🌾 Wheat Disease & Pest Classifier

A Streamlit-powered web app for detecting **wheat leaf diseases and pests** using a deep learning model. The application also provides **Grad-CAM** visualizations to interpret which areas of the image influenced the prediction the most.

---

## 🚀 Try it Live

You can try the app on [Hugging Face Spaces](https://huggingface.co/spaces/) or run it locally by following the instructions below.

---

## 🧠 Model Overview

- **Architecture**: Custom CNN / ResNet  
- **Number of Classes**: 15 wheat leaf diseases and pest types  
- **Training Dataset**: 14,155 high-resolution images  
- **Explainability**: Grad-CAM for visual interpretation of predictions  

---

## 🖼️ How to Use

1. **Upload** an image of a wheat leaf.
2. The app will:
   - Classify the image.
   - Display the **predicted class** and **confidence score**.
   - Show a **Grad-CAM heatmap** highlighting the regions the model focused on.

---

## 📁 Directory Structure
wheat-disease-app/

├── app/

│   ├── streamlit_app.py

│   ├── wheat_disease_model.h5

│   └── classes.txt

├── requirements.txt

├── README.md

---

## 🛠️ Setup Instructions

### 🔧 Install Requirements

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
---
## 🧪 Sample Input
---

## 📊 Output with Explanation
- Prediction: Yellow Rust (97.32%)

- Grad-CAM: Heatmap overlay shows key focus areas
---

## 📌 Notes
- Make sure you have Python 3.7 or later installed.

- Grad-CAM is automatically generated for every prediction.

- Supported image formats: .jpg, .png, .jpeg
---

## 📬 Contact
- For any questions or contributions, feel free to open an issue or pull request.
