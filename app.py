import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 Chest X-Ray Pneumonia Detection")
st.write(
    "Upload a chest X-ray image, and the model will predict whether it's Normal or Pneumonia. "
    "The Grad-CAM heatmap highlights areas the model focused on."
)

@st.cache_resource(show_spinner=True)
def load_trained_model():
    model = load_model("pneumonia_mobilenetv2_streamlit.h5")
    return model

model = load_trained_model()

def get_gradcam(model, img_array, last_conv_layer_name="Conv_1"):
    """
    Generates Grad-CAM heatmap for a given image and model.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap.numpy()

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display uploaded image
    st.subheader("Uploaded X-ray Image")
    st.image(img_rgb, use_column_width=True)

    img_resized = cv2.resize(img_rgb, (224, 224))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    pred = model.predict(img_input)[0][0]
    if pred > 0.5:
        result = "PNEUMONIA"
        confidence = pred * 100
    else:
        result = "NORMAL"
        confidence = (1 - pred) * 100

    st.subheader("Prediction Result")
    st.write(f"**Class:** {result}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Grad-CAM Heatmap")
    heatmap = get_gradcam(model, img_input, last_conv_layer_name="Conv_1")
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

    st.image(superimposed_img, use_column_width=True)

    st.info("⚠️ Disclaimer: This tool is for educational purposes only and is not a medical diagnosis.")

st.markdown("---")
st.markdown(
    "Developed by **Anass Ahmed** | "
    "[GitHub](https://github.com/yourusername) | "
    "Educational AI Healthcare Project"
)
