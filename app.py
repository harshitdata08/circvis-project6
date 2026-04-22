import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="CIRCVIS Demo", page_icon="♻️", layout="wide")

PREPROCESS = {
    "mobilenet": tf.keras.applications.mobilenet_v2.preprocess_input,
    "resnet": tf.keras.applications.resnet.preprocess_input,
    "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
}

MODEL_PATH = "models/best_model.keras"
LABELS_PATH = "models/class_names.json"
BACKBONE = "efficientnet"
IMG_SIZE = 224


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32)
    arr = PREPROCESS[BACKBONE](arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


st.title("♻️ CIRCVIS Waste Classification Demo")
st.caption("Context-Aware Waste Classification for Circular Cities")

st.markdown(
    """
This demo predicts waste categories from uploaded images using a transfer-learning image classifier.
Use this in your class presentation as a simple proof-of-concept interface.
"""
)

if not st.session_state.get("model_checked"):
    st.session_state.model_checked = True
    try:
        load_model()
        st.success("Model files loaded successfully.")
    except Exception:
        st.warning("Model files are not present yet. First run train.py and place best_model.keras + class_names.json in the models folder.")

uploaded = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])
if uploaded:
    image = Image.open(uploaded)
    col1.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        model, class_names = load_model()
        arr = preprocess_image(image)
        probs = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))

        col2.subheader("Prediction Result")
        col2.metric("Predicted Class", class_names[top_idx])
        col2.metric("Confidence", f"{probs[top_idx]*100:.2f}%")

        result_data = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
        col2.subheader("Class Probabilities")
        for label, prob in result_data:
            col2.progress(float(prob), text=f"{label}: {prob*100:.2f}%")
    except Exception as e:
        col2.error(f"Prediction unavailable: {e}")
else:
    st.info("Upload an image to test the demo.")

st.markdown("---")
st.subheader("Suggested Classes")
st.write("cardboard, glass, metal, paper, plastic, trash")
