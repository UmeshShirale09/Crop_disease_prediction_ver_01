from flask import Flask, request, jsonify, render_template
import joblib
import cv2
import numpy as np

# Constants
IMAGE_SIZE = (128, 128)
MODEL_PATH = "svm_crop_disease_model.pkl"

# Load the model and label encoder
svm_model = joblib.load(MODEL_PATH)

# Load Label Encoder
label_encoder_path = "label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)

# Create Flask app
app = Flask(__name__)

# Set the maximum allowed payload to 16MB (adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template("index.html")  # Create an HTML file for UI


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Preprocess the image
        img_resized = cv2.resize(img, IMAGE_SIZE)
        img_normalized = img_resized / 255.0
        img_flat = img_normalized.flatten().reshape(1, -1)

        # Predict the label
        predicted_label = svm_model.predict(img_flat)[0]
        confidence = svm_model.predict_proba(img_flat).max()

        # Map label index to class name
        class_name = label_encoder.inverse_transform([predicted_label])[0]

        # Response
        return jsonify({
            "predicted_label": class_name,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
