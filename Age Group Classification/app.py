import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

# Load the model
model = tf.keras.models.load_model('age_group_classifier.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# App title
st.title("Age Group Prediction from Face Image")
st.write("Upload a face image to predict the person's age group.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # Convert RGBA to RGB
        img_array = img_array[..., :3]

    img_array = img_array.reshape(1, 128, 128, 3)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = le.classes_[class_index]

    st.success(f"Predicted Age Group: **{class_label}**")
