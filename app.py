from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
from werkzeug.utils import secure_filename
from extract import extract_features
from main import predict_raga
import joblib
import torch
from main import TDNN_LSTM

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_CSV = 'processed_features.csv'
MODEL_PATH = 'model.pth'
PREPROCESSING_PATH = 'preprocessing.pkl'
STATIC_FOLDER = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and preprocessing objects
le, scaler, selector = joblib.load(PREPROCESSING_PATH)
model = TDNN_LSTM(input_dim=24, num_classes=len(le.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # Ensure compatibility with CPU
model.eval()

@app.route('/')
def index():
    """Serve the front-end HTML file."""
    return send_from_directory(STATIC_FOLDER, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and predict raga."""
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audioFile']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Extract features and save to CSV
        features_df = extract_features(filepath)
        features_df.to_csv(PROCESSED_CSV, index=False)

        # Predict raga
        final_raga, predictions = predict_raga(model, PROCESSED_CSV, le, scaler, selector)
        return jsonify({'finalRaga': final_raga, 'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)