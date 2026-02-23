import pandas as pd
import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from backend.federated.aggregator import federated_round

print("Loading hospital datasets...")

h1 = pd.read_csv("data/processed/hospital1.csv")
h2 = pd.read_csv("data/processed/hospital2.csv")
h3 = pd.read_csv("data/processed/hospital3.csv")

# combine data for scaling, dummy initialization and evaluation
global_data = pd.concat([h1, h2, h3], ignore_index=True)

# shuffle
global_data = global_data.sample(frac=1, random_state=42)

# split features and target
X = global_data.drop("target", axis=1)
y = global_data["target"]

if y.nunique() < 2:
    raise ValueError("Global dataset must contain at least 2 target classes.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Running federated round with secure aggregation...")

(aggregated_coef, aggregated_intercept), round_accuracies = federated_round([h1, h2, h3], rounds=5)

# Convert aggregated parameters (raw feature space) to global scaled space.
aggregated_coef = np.asarray(aggregated_coef, dtype=float)
aggregated_intercept = np.asarray(aggregated_intercept, dtype=float)
coef_scaled = aggregated_coef * scaler.scale_.reshape(1, -1)
intercept_scaled = aggregated_intercept + np.sum(
    aggregated_coef * scaler.mean_.reshape(1, -1), axis=1
)

print("Initializing global model with dummy fit...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# dummy fit initializes model internals/shapes before parameter injection
model.fit(X_scaled, y)

# inject federated aggregated parameters
model.coef_ = coef_scaled
model.intercept_ = intercept_scaled

print("Parameter injection complete")

# save model
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

joblib.dump(model, "results/models/global_model.pkl")
joblib.dump(scaler, "results/models/scaler.pkl")

# compute and save accuracy
preds = model.predict(X_scaled)
accuracy = accuracy_score(y, preds)

with open("results/metrics/accuracy.txt", "w", encoding="utf-8") as f:
    f.write(f"{accuracy:.6f}")

print("Global model and accuracy saved successfully")
print("Round accuracies:", round_accuracies)
