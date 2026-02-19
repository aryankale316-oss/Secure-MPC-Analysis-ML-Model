import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

print("Loading hospital datasets...")

h1 = pd.read_csv("data/processed/hospital1.csv")
h2 = pd.read_csv("data/processed/hospital2.csv")
h3 = pd.read_csv("data/processed/hospital3.csv")

print("Simulating Secure MPC aggregation...")

# simulate secure aggregation by combining datasets
global_data = pd.concat([h1, h2, h3])

# shuffle
global_data = global_data.sample(frac=1, random_state=42)

# split features and target
X = global_data.drop("target", axis=1)
y = global_data["target"]

print("Training global model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X, y)

print("Training complete")

# save model
os.makedirs("results/models", exist_ok=True)

joblib.dump(model, "results/models/global_model.pkl")

print("Global model saved successfully")
