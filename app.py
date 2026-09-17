import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "crack_detection_model.keras"
IMG_SIZE = (227, 227)

st.set_page_config(
    page_title="Crack Detection",
    page_icon="🧱",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🧱 Crack Detection")
st.write("Upload an image to detect whether a crack is present.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_resized = image.resize(IMG_SIZE)

    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "Crack Detected"
        confidence = prediction
    else:
        result = "No Crack Detected"
        confidence = 1 - prediction

    st.subheader(f"Result: {result}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
