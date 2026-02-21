import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("📥 Loading hospital datasets...")

# Load datasets
h1 = pd.read_csv("data/processed/hospital1.csv")
h2 = pd.read_csv("data/processed/hospital2.csv")
h3 = pd.read_csv("data/processed/hospital3.csv")

# Combine datasets
global_data = pd.concat([h1, h2, h3])

# Shuffle
global_data = global_data.sample(frac=1, random_state=42)

# Split features and target
X = global_data.drop("target", axis=1)
y = global_data["target"]

print("🔎 Features used:", list(X.columns))

# ------------------------------
# SCALE FEATURES
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# TRAIN MODEL
# ------------------------------
print("🧠 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_scaled, y)

print("✅ Training complete.")

# ------------------------------
# SAVE MODEL
# ------------------------------
os.makedirs("results/models", exist_ok=True)
joblib.dump(model, "results/models/global_model.pkl")
joblib.dump(scaler, "results/models/scaler.pkl")

print("💾 Model and scaler saved successfully.")