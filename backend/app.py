from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "global_model.pkl")

# load model
model = joblib.load(MODEL_PATH)

# flask setup
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

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

        print("Received:", data)

        features = data.get("features", [])

        # FIX message and validation
        if len(features) != 13:
            return jsonify({
                "prediction": "Invalid input. 13 features required."
            })

        # convert to float
        features = [float(x) for x in features]

        columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]

        df = pd.DataFrame([features], columns=columns)

        print("DataFrame:")
        print(df)

        # GET PREDICTION
        prediction = model.predict(df)[0]

        # GET PROBABILITY
        probabilities = model.predict_proba(df)[0]

        print("Probabilities:", probabilities)

        confidence = round(max(probabilities) * 100, 2)

        result = "Disease Detected" if prediction == 1 else "No Disease"

        print("Prediction:", result)
        print("Confidence:", confidence)

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "prediction": "Server error: " + str(e)
        }), 500



if __name__ == "__main__":
    app.run(debug=True)
