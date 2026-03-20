import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF warnings

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


st.set_page_config(page_title="🏥 Medical AI Suite", layout="centered")

st.title("🏥 Medical Image AI Suite")
st.write("Upload an image and choose diagnosis mode")

mode = st.radio(
    "Select Mode",
    ["🤖 Auto Diagnosis (Recommended)", "🎯 Manual Model Selection"]
)

@st.cache_resource
def load_models():
    models = {}

    def safe_load(path):
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
        else:
            return None

    models["brain"] = safe_load("brain_tumor_model.h5")
    models["chest"] = safe_load("chest_1model.keras")
    models["bone"] = safe_load("fracture_model (1).h5")
    models["multi"] = safe_load("medical_3class_model.h5")

    return models

models = load_models()

def preprocess_brain(image):
    image = image.convert("RGB").resize((224, 224))
    img = np.array(image)
    return np.expand_dims(img, axis=0)

def preprocess_chest(image):
    image = image.convert("RGB").resize((64, 64))
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_bone(image):
    image = image.convert("RGB").resize((180, 180))
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_multi(image):
    image = image.convert("RGB").resize((128, 128))
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

if mode == "🎯 Manual Model Selection":
    model_choice = st.selectbox(
        "Choose Model",
        [
            "🧠 Brain Tumor",
            "🫁 Chest X-ray (Pneumonia)",
            "🦴 Bone Fracture",
            "🧪 Multi-Class (Basic)"
        ]
    )

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Diagnosis"):

        st.markdown("---")


        if mode == "🤖 Auto Diagnosis (Recommended)":

            st.subheader("🧪 Stage 1: Detect Image Type")

            if models["multi"] is None:
                st.error("❌ Multi-class model missing!")
                st.stop()

            img1 = preprocess_multi(image)
            preds1 = models["multi"].predict(img1)[0]

            classes1 = ['bone', 'brain', 'chest']
            stage1_class = classes1[np.argmax(preds1)]
            conf1 = np.max(preds1)

            st.success(f"Detected: {stage1_class.upper()}")
            st.info(f"Confidence: {conf1*100:.2f}%")

            st.markdown("---")
            st.subheader("Stage 2: Detailed Diagnosis")

            if stage1_class == "brain":
                if models["brain"] is None:
                    st.error("Brain model missing!")
                    st.stop()

                img = preprocess_brain(image)
                preds = models["brain"].predict(img)

                classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
                pred_class = classes[np.argmax(preds)]
                confidence = np.max(preds)

                st.success(f"🧠 Tumor: {pred_class.upper()}")
                st.info(f"Confidence: {confidence*100:.2f}%")

            elif stage1_class == "chest":
                if models["chest"] is None:
                    st.error("❌ Chest model missing!")
                    st.stop()

                img = preprocess_chest(image)
                pred = models["chest"].predict(img)[0][0]

                if pred > 0.5:
                    st.error("⚠️ PNEUMONIA DETECTED")
                    confidence = pred
                else:
                    st.success("✅ NORMAL")
                    confidence = 1 - pred

                st.info(f"Confidence: {confidence*100:.2f}%")

            elif stage1_class == "bone":
                if models["bone"] is None:
                    st.error("❌ Bone model missing!")
                    st.stop()

                img = preprocess_bone(image)
                pred = models["bone"].predict(img)[0][0]

                if pred > 0.5:
                    st.error("🟥 Fracture Detected")
                    confidence = pred
                else:
                    st.success("🟩 No Fracture")
                    confidence = 1 - pred

                st.info(f"Confidence: {confidence:.4f}")


        else:

            st.subheader("🔍 Prediction Result")

            if model_choice == "🧠 Brain Tumor":
                if models["brain"] is None:
                    st.error("❌ Brain model missing!")
                    st.stop()

                img = preprocess_brain(image)
                preds = models["brain"].predict(img)

                classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
                pred_class = classes[np.argmax(preds)]
                confidence = np.max(preds)

                st.success(f"🧠 Tumor: {pred_class.upper()}")
                st.info(f"Confidence: {confidence*100:.2f}%")

            elif model_choice == "🫁 Chest X-ray (Pneumonia)":
                if models["chest"] is None:
                    st.error("❌ Chest model missing!")
                    st.stop()

                img = preprocess_chest(image)
                pred = models["chest"].predict(img)[0][0]

                if pred > 0.5:
                    st.error(" PNEUMONIA DETECTED")
                    confidence = pred
                else:
                    st.success(" NORMAL")
                    confidence = 1 - pred

                st.info(f"Confidence: {confidence*100:.2f}%")

            elif model_choice == "🦴 Bone Fracture":
                if models["bone"] is None:
                    st.error("❌ Bone model missing!")
                    st.stop()

                img = preprocess_bone(image)
                pred = models["bone"].predict(img)[0][0]

                if pred > 0.5:
                    st.error("🟥 No Fracture ")
                    confidence = pred
                else:
                    st.success("🟩 Fracture Detected")
                    confidence = 1 - pred

                st.info(f"Confidence: {confidence:.4f}")

            else:
                if models["multi"] is None:
                    st.error("❌ Multi-class model missing!")
                    st.stop()

                img = preprocess_multi(image)
                preds = models["multi"].predict(img)[0]

                classes = ['bone', 'brain', 'chest']
                pred_class = classes[np.argmax(preds)]
                confidence = np.max(preds)

                st.success(f"Prediction: {pred_class.upper()}")
                st.info(f"Confidence: {confidence:.2f}")

else:
    st.warning("Please upload an image 👆")