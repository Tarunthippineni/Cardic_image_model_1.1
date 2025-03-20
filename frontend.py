import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Title of the app
st.title("Heart Disease Prediction from MRI Images")

# Upload the image
uploaded_file = st.file_uploader("Upload an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])

# Load the trained .keras model
model_path = "model.keras"  # Replace with your .keras file path
model = tf.keras.models.load_model(model_path)

# Class labels (based on your training)
class_labels = ['HCM', 'DCM', 'MINF', 'ARV']

# Function to preprocess the image
def preprocess_image(image):
    # Convert to numpy array
    image = np.array(image)
    # Convert to RGB if needed (Streamlit uploads in RGB, but model expects RGB)
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Resize to match model input (299x299 for Inception v3)
    image = cv2.resize(image, (299, 299))
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match model input shape (1, 299, 299, 3)
    image = np.expand_dims(image, axis=0)
    return image

# Predict function
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100
    return class_labels[predicted_class], confidence

# If an image is uploaded, process and predict
if uploaded_file is not None:
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    # Predict
    with st.spinner("Predicting..."):
        predicted_class, confidence = predict(image)
    
    # Display the result
    st.success(f"Prediction: **{predicted_class}** (Confidence: {confidence:.2f}%)")
else:
    st.info("Please upload an MRI image to get a prediction.")

# Footer
st.markdown("---")
st.write("Built with ❤️ by Prendu using Streamlit and TensorFlow")