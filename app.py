import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("pneumonia_mobilenetv2_streamlit.h5")

st.title("Chest X-Ray Pneumonia Detection")
st.write("Upload your chest X-ray image and get prediction + heatmap.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded X-ray", use_column_width=True)
    
    # Preprocess image
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Predict
    pred = model.predict(img_input)[0][0]
    if pred > 0.5:
        result = "PNEUMONIA"
        confidence = pred * 100
    else:
        result = "NORMAL"
        confidence = (1 - pred) * 100
    
    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence:** {confidence:.2f}%")

