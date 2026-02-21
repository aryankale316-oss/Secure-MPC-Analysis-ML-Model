from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

# ------------------------------
# PROJECT ROOT PATH FIX
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "results", "models", "global_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "results", "models", "scaler.pkl")

# ------------------------------
# LOAD MODEL & SCALER
# ------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run main.py first.")

model = joblib.load(MODEL_PATH)

scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully.")
else:
    print("⚠️ WARNING: scaler.pkl not found.")

# ------------------------------
# FLASK SETUP
# ------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(FRONTEND_DIR, path)

# ------------------------------
# FEATURE ORDER (VERY IMPORTANT)
# ------------------------------
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# ------------------------------
# PREDICT ROUTE
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", [])

        if len(features) != 13:
            return jsonify({"error": "Exactly 13 features required"}), 400

        # Convert to float safely
        features = [float(x) for x in features]

        df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        # Apply scaler if available
        input_data = scaler.transform(df) if scaler else df

        # Model prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # ✅ Correct disease probability
        disease_probability = round(probabilities[1] * 100, 2)

        result = "Disease Detected" if prediction == 1 else "No Disease"

        print("Input:", features)
        print("Prediction:", result)
        print("Disease Probability:", disease_probability)

        return jsonify({
            "prediction": result,
            "confidence": disease_probability
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ------------------------------
# DEBUG ROUTE
# ------------------------------
@app.route("/test_healthy")
def test_healthy():
    healthy = [30,1,0,110,150,0,0,150,0,0,0,0,2]
    df = pd.DataFrame([healthy], columns=FEATURE_COLUMNS)
    df_scaled = scaler.transform(df) if scaler else df
    prob = model.predict_proba(df_scaled)[0][1] * 100
    return f"Healthy risk: {prob:.2f}%"

# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)