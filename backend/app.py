from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "..", "results", "models", "global_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "results", "models", "scaler.pkl")

# load model
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model loaded successfully")

# flask setup
app = Flask(__name__)
CORS(app)

@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", []) if isinstance(data, dict) else []

        if len(features) != 13:
            return jsonify({
                "error": "Invalid input. 13 features required."
            }), 400

        feature_array = np.array(features, dtype=float).reshape(1, -1)
        feature_array = scaler.transform(feature_array)

        prediction = int(model.predict(feature_array)[0])
        probabilities = model.predict_proba(feature_array)[0]

        predicted_index = int(np.where(model.classes_ == prediction)[0][0])
        confidence = round(float(probabilities[predicted_index]) * 100, 2)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "error": "Server error: " + str(e)
        }), 500



if __name__ == "__main__":
    app.run(debug=True)
