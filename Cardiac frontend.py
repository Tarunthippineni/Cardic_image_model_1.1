import streamlit as st
import gdown
import os
import nibabel as nib
import numpy as np
import tensorflow as tf

# Google Drive link (replace with your link)
MODEL_URL = "https://drive.google.com/uc?export=download&id=11u17qmmYUYyaAvAyVt7wn6WA4_HHVjd6"  # Replace YOUR_FILE_ID
MODEL_PATH = "best_model (1).keras"

# Model ni download
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Model load
with st.spinner("Loading model..."):
    model = tf.keras.models.load_model(MODEL_PATH)
label_classes = [
    "Healthy",
    "Hypertrophy",
    "Heart Failure with Infarction",
    "Heart Failure without Infarction",
    "Unknown"
]


# Prediction function
def predict_heart_disease(image_path):
    try:
        if os.path.getsize(image_path) <= 0:
            return "Error: Empty file detected."
        img = nib.load(image_path).get_fdata()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        if img.shape[2] == 0:
            return "Error: Invalid image shape."
        img = img[:, :, img.shape[2] // 2]
        img = tf.image.resize(img[..., None], [128, 128]).numpy()
        img = img[None, ...]
        preds = model.predict(img)
        threshold = 0.5
        pred_labels = (preds > threshold).astype(int)
        predicted_diseases = []
        for i, label in enumerate(label_classes):
            if pred_labels[0][i] == 1:
                predicted_diseases.append(label)
        if not predicted_diseases:
            return "No disease detected (or model confidence too low)."
        return predicted_diseases
    except Exception as e:
        return f"Error processing image: {str(e)}"


# Streamlit UI
st.title("Heart Disease Prediction from MRI")
uploaded_file = st.file_uploader("Upload .nii File", type=["nii"])

if uploaded_file is not None:
    # File save cheyadam
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Predict cheyadam
    with st.spinner("Predicting..."):
        result = predict_heart_disease(temp_path)
    st.write("Predicted Heart Disease(s):", result)

    # Temporary file delete cheyadam
    if os.path.exists(temp_path):
        os.remove(temp_path)
