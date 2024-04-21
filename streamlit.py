import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained models
inception_model = tf.keras.models.load_model("inception_model.h5")
vgg_model = tf.keras.models.load_model("vgg_model.h5")

# Define label categories
label_categories = {
    0: "Jerry",
    1: "Tom",
    2: "None",
    3: "Both"
}

# Function to preprocess the image
def preprocess_image(image):
    image = tf.image.resize(image, (75, 75))
    image = tf.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    image = preprocess_image(image)
    inception_prediction = inception_model.predict(image)
    vgg_prediction = vgg_model.predict(image)
    ensemble_prediction = (inception_prediction + vgg_prediction) / 2.0
    predicted_label = np.argmax(ensemble_prediction)
    return label_categories[predicted_label]

# Streamlit app
def main():
    st.title("Tom & Jerry Image Classifier")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            predicted_label = predict(image)
            st.success(f"The image contains: {predicted_label}")

if __name__ == "__main__":
    main()
