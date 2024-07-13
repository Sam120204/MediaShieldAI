import os
import zipfile
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
data_dir = 'D:/桌面/CS Waterloo/MediaShieldAI/data'
real_images_dir = os.path.join(data_dir, 'real_images')
fake_images_dir = os.path.join(data_dir, 'fake_images')

# Ensure directories exist
os.makedirs(real_images_dir, exist_ok=True)
os.makedirs(fake_images_dir, exist_ok=True)

# Function to preprocess images
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

# Preprocess and load real and fake images
real_images = []
fake_images = []

for img_name in os.listdir(real_images_dir):
    img_path = os.path.join(real_images_dir, img_name)
    img_array = preprocess_image(img_path)
    real_images.append(img_array)

for img_name in os.listdir(fake_images_dir):
    img_path = os.path.join(fake_images_dir, img_name)
    img_array = preprocess_image(img_path)
    fake_images.append(img_array)

# Convert lists to numpy arrays
real_images = np.array(real_images)
fake_images = np.array(fake_images)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator to the real images (example)
datagen.fit(real_images)
