
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Potato CNN", page_icon="🥔", layout="wide")

ROOT = Path(__file__).resolve().parent
IMG_SIZE = (128, 128)

# UI Styling
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #4ade80;
}
[data-testid="stMetric"] {
    background: #172033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #334155;
}
.stButton > button {
    background-color: #16a34a;
    color: white;
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load model and class names
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(ROOT / "potato_cnn_model.keras")

@st.cache_data
def load_classes():
    return (ROOT / "classification_report.txt").read_text().splitlines()

# Sidebar
st.sidebar.title("🥔 Potato AI")
page = st.sidebar.radio("Navigation", ["Home", "About"])

if page == "Home":

    st.title("🥔 Potato Leaf Classification")
    st.caption("CNN Based Intelligent Disease Detection System")
    st.write("Upload a potato leaf image to predict its class.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "CNN")
    col2.metric("Image Size", "128 × 128")
    col3.metric("Classes", "3")

    st.divider()

    uploaded_file = st.file_uploader(
        "📤 Upload Potato Leaf Image",
        type=["jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, width="stretch")

        with col2:
            st.subheader("🔍 Prediction")

            st.write("File:", uploaded_file.name)
            st.write("Image Size:", image.size)

            if st.button("🚀 Predict Image", type="primary"):

                try:
                    model = load_model()
                    classes = load_classes()

                    image = image.resize(IMG_SIZE)
                    x = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

                    scores = model.predict(x, verbose=0)[0]
                    best = np.argmax(scores)

                    st.success("Prediction Completed!")

                    st.metric(
                        "Predicted Class",
                        classes[best].replace("potato__", "")
                    )

                    st.metric(
                        "Confidence",
                        f"{scores[best]*100:.2f}%"
                    )

                    st.divider()
                    st.subheader("📊 Class Probabilities")

                    for name, score in zip(classes, scores):
                        st.write(f"**{name}: {score*100:.2f}%**")
                        st.progress(float(score))

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

else:

    st.title("📘 About Project")
    st.caption("Potato Leaf Classification using Deep Learning")

    st.divider()

    st.subheader("Project Information")

    st.write("This project uses a Convolutional Neural Network (CNN) to classify potato leaf images.")

    st.info("The user uploads an image and the trained CNN model predicts its class.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Framework", "TensorFlow / Keras")
        st.metric("Frontend", "Streamlit")

    with col2:
        st.metric("Image Size", "128 × 128")
        st.metric("Output Classes", "3")

    st.divider()

    st.success("Developed using Python, CNN and Streamlit.")

st.divider()
st.caption("🥔 Potato AI | CNN Image Classification System")
