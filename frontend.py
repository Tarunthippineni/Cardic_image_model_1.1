import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from streamlit_option_menu import option_menu
import gdown

# Define custom loss function (placeholder - replace with actual implementation)
@tf.keras.utils.register_keras_serializable()
def weighted_combined_loss(y_true, y_pred):
    # Placeholder implementation: Replace with the actual loss function used during training
    categorical_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    # Add other loss components or weights as needed
    return categorical_loss

# Define custom metric function
@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Menu for selecting prediction type
selected = option_menu("Multiple Disease Prediction System",
                       ["❤️ Heart Disease Prediction",
                        "🧠 Brain Disease Prediction"],
                       default_index=0)

@st.cache_resource
def load_model():
    if selected == '❤️ Heart Disease Prediction':
        url = 'https://drive.google.com/uc?export=download&id=1KsC6QCRQYVYYh6z5YC3Pn8UDRgHe147J'
        output = 'best_model (1).keras'
    elif selected == '🧠 Brain Disease Prediction':
        url = 'https://drive.google.com/uc?export=download&id=1WVkjuSSXYCg8VZppR1s6lPm35tbMvbLI'
        output = 'inceptionv3_binary_model.keras'

    gdown.download(url, output, quiet=False)
    # Include custom loss and metric in custom_objects
    model = tf.keras.models.load_model(output, custom_objects={
        'weighted_combined_loss': weighted_combined_loss,
        'dice_coefficient': dice_coefficient
    })
    return model

# Preprocessing function with dynamic target size and channels
def preprocess_image(image, target_size, channels=3):
    # Resize to target size based on the model
    image = cv2.resize(image, target_size)
    # Convert to grayscale if 1 channel is required
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Add channel dimension for grayscale
        image = np.expand_dims(image, axis=-1)
    # Convert to RGB if 3 channels are required
    elif channels == 3 and image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Normalize to [0, 1]
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict(image, model, class_labels, target_size, channels=3):
    preprocessed_image = preprocess_image(image, target_size, channels=channels)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100
    return class_labels[predicted_class], confidence

# Load the model once
model = load_model()

# Define target sizes, channels, and class labels for each model
if selected == '❤️ Heart Disease Prediction':
    st.title("Heart Disease Prediction from MRI Images")
    uploaded_file = st.file_uploader("Upload an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    class_labels = [
        "Healthy",
        "Hypertrophy",
        "Heart Failure with Infarction",
        "Heart Failure without Infarction"
    ]
    target_size = (224, 224)  # Updated to match model input
    channels = 1  # Grayscale input

elif selected == '🧠 Brain Disease Prediction':
    st.title("Brain Tumor Prediction from MRI Images")
    uploaded_file = st.file_uploader("Upload a brain MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary_tumor']
    target_size = (299, 299)  # Matches InceptionV3 input size
    channels = 3  # RGB input

# Process uploaded image
if uploaded_file is not None:
    # Read the image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        st.error("Error reading the image. Please upload a valid image file.")
        st.stop()
    
    # Display the uploaded image
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    
    # Predict
    with st.spinner("Predicting..."):
        predicted_class, confidence = predict(image, model, class_labels, target_size, channels=channels)
    
    # Display the result
    st.success(f"**Prediction:** {predicted_class} \n**Confidence:** {confidence:.2f}%")
else:
    st.info("Please upload an MRI image to get a prediction.")

# Footer
st.markdown("---")
st.write("Built with ❤ by Prendu using Streamlit and TensorFlow")
