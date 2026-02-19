from src.federated.client import Client
from src.federated.aggregation import aggregate_models
from config.config import PROCESSED_DATA_PATHS, MODEL_PATH, METRICS_PATH

from src.model.evaluate import evaluate_model

import joblib

def start_server():

    print("Starting Secure MPC Federated Training...")

    clients = []

    for path in PROCESSED_DATA_PATHS:
        client = Client(path)
        clients.append(client)

    models = []

    for client in clients:
        model = client.train()
        models.append(model)

    print("All clients trained successfully.")

    global_model = aggregate_models(models)

    print("Secure aggregation completed.")

    joblib.dump(global_model, MODEL_PATH)

    accuracy = evaluate_model(global_model, PROCESSED_DATA_PATHS[0])

    with open(METRICS_PATH, "w") as f:
        f.write(f"Global Model Accuracy: {accuracy}")

    print("Global Model Accuracy:", accuracy)

    print("Model saved successfully.")
