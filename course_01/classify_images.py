import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained model
model = MobileNetV2(weights="imagenet")

# Folder with images to classify
image_folder = "images/"   # change to your path

# Loop over all image files
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(image_folder, filename)
        print(f"\n--- Processing: {filename} ---")

        # Load the image with the correct target size
        img = load_img(file_path, target_size=(224, 224))

        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]

        # Print top 3 predictions
        for wnid, label, confidence in decoded:
            print(f"\t{label:15s} --> {confidence:.4f}")
