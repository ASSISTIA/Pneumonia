import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('./Model/pneumonia.h5')  # Load pneumonia model

# Define the class labels 
class_labels = ["Normal", "Pneumonia"] # Assuming binary classification

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200)) # Input size used during training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET'])
def working():
    return "Pneumonia API is active"

# API endpoint to receive image and return prediction
@app.route('/predict/pneumonia', methods=['POST']) # Changed endpoint
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = np.round(predictions[0][0]) # Round for binary classification
        predicted_label = class_labels[int(predicted_class)] # Convert to int for indexing
        os.remove(file_path)
        return jsonify({"prediction": predicted_label})
    return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)