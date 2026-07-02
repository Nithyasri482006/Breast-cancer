import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "breast_cancer_model.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False
    )
    return model

model = load_model()

# ----------------------------
# Class Labels
# ----------------------------

class_names = [
    "Benign",
    "Malignant",
    "Normal"
]

# ----------------------------
# Title
# ----------------------------

st.title("🩺 Breast Cancer Detection using Deep Learning")
st.markdown("---")

st.write(
    """
Upload a **Breast Ultrasound Image** and the AI model
will classify it into one of the following classes:

- 🟢 Normal
- 🟡 Benign
- 🔴 Malignant
"""
)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.header("About")

st.sidebar.info(
    """
Model : MobileNetV2

Transfer Learning

Dataset : BUSI Dataset

Validation Accuracy : 96.79%
"""
)

# ----------------------------
# Upload Image
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload Breast Ultrasound Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# Prediction
# ----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    image = image.resize((224,224))

    image = np.array(image).astype(np.float32)

    # IMPORTANT
    # No preprocess_input()
    # because model already contains Lambda(preprocess_input)

    image = np.expand_dims(image, axis=0)

    with st.spinner("Analyzing..."):

        prediction = model.predict(
            image,
            verbose=0
        )

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction))*100

    # ----------------------------
    # Result
    # ----------------------------

    with col2:

        st.subheader("Prediction")

        if predicted_class == "Normal":

            st.success(f"✅ {predicted_class}")

        elif predicted_class == "Benign":

            st.warning(f"🟡 {predicted_class}")

        else:

            st.error(f"🔴 {predicted_class}")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.markdown("---")

        st.subheader("Prediction Probabilities")

        for cls, score in zip(class_names, prediction[0]):

            st.progress(float(score))

            st.write(
                f"**{cls} : {score*100:.2f}%**"
            )

# ----------------------------
# Footer
# ----------------------------

st.markdown("---")

st.caption(
    "Developed using TensorFlow • MobileNetV2 • Streamlit"
)