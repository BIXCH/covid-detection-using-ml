import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Define the path for the model and test dataset
model_path = 'D:/new/Covid19-dataset/covid_xray_model.h5'  # Make sure the path is correct
test_dir = 'D:/new/Covid19-dataset/test'  # Make sure the path is correct

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
    exit()

# Load the pre-trained model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except OSError as e:
    print(f"Error loading model: {e}")
    exit()

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Ensure it matches the input shape of the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Verify if the test directory exists
if not os.path.exists(test_dir):
    print(f"Test directory not found: {test_dir}")
    exit()

# Loop through images in the test folders
for category in ['covid', 'normal']:  # Make sure your categories are correct (covid and normal)
    category_path = os.path.join(test_dir, category)
    
    if not os.path.exists(category_path):
        print(f"Directory not found: {category_path}")
        continue

    print(f"\nPredicting images in category: {category}")
    
    # Loop through all images in the category
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
            img_array = load_and_preprocess_image(img_path)
            prediction = model.predict(img_array)

            # Output the prediction result
            if prediction[0][0] > 0.5:
                print(f"Image {img_name}: COVID-19 positive")
            else:
                print(f"Image {img_name}: Normal")
