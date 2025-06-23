import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load model MobileNetV2 (tanpa classifier, hanya feature extractor)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def load_and_extract_features(image_paths):
    features = []
    for path in image_paths:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature = model.predict(img_array, verbose=0)
        features.append(feature.flatten())
    return np.array(features)
