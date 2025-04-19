
import streamlit as st
import gdown
import os
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable

# Define custom loss function
@register_keras_serializable()
def weighted_combined_loss(y_true, y_pred):
    """
    Placeholder for weighted_combined_loss. Replace with the actual implementation.
    This function should compute the loss used during model training.
    Example: A combination of binary cross-entropy and another loss with weights.
    """
    # Placeholder: Using binary cross-entropy as a default
    binary_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return binary_loss  # REPLACE with actual loss function implementation

# Google Drive link for the model
MODEL_URL = "https://drive.google.com/uc?export=download&id=1KsC6QCRQYVYYh6z5YC3Pn8UDRgHe147J"
MODEL_PATH = "best_model (1).keras"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            st.stop()

# Load model
with st.spinner("Loading model..."):
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'weighted_combined_loss': weighted_combined_loss}
        )
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error(
            "The model requires a custom loss function 'weighted_combined_loss'. "
            "Please provide the correct implementation of this function."
        )
        st.stop()

# Define label classes
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
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.read())
    
    # Make prediction
    with st.spinner("Predicting..."):
        result = predict_heart_disease(temp_path)
    st.write("Predicted Heart Disease(s):", result)
    
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
