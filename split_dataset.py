import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ensure processed folder exists
os.makedirs("data/processed", exist_ok=True)

# load dataset
data = pd.read_csv("data/raw/medical_dataset.csv")

# shuffle dataset
data = data.sample(frac=1, random_state=42)

# split into 3 hospitals
h1, temp = train_test_split(data, test_size=0.66, random_state=42)
h2, h3 = train_test_split(temp, test_size=0.5, random_state=42)

# save datasets
h1.to_csv("data/processed/hospital1.csv", index=False)
h2.to_csv("data/processed/hospital2.csv", index=False)
h3.to_csv("data/processed/hospital3.csv", index=False)

print("Hospital datasets created successfully")
