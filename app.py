# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

# Load model
model = tf.keras.models.load_model("wheat_disease_model.h5")

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

st.title("🌾 Wheat Disease & Pest Classifier")
st.write("Upload a wheat leaf image and let the AI detect the disease or pest.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Grad-CAM function
def get_grad_cam(model, img_array, pred_index):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(index=-3).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_batch)
    pred_index = np.argmax(preds)
    predicted_class = class_names[pred_index]
    confidence = np.max(preds)

    st.subheader(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")

    # Grad-CAM visualization
    heatmap = get_grad_cam(model, img_batch, pred_index)
    heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    img_np = np.array(img)
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    superimposed_img = heatmap_colored * 0.4 + img_np
    st.image(np.uint8(superimposed_img), caption="Grad-CAM Explanation", use_column_width=True)
