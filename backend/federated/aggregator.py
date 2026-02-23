import numpy as np
import pandas as pd

from .local_train import train_local_model
from .security import mask_weights, secure_average


def _accuracy_from_params(X, y, weights, intercept):
    """Compute binary classification accuracy from linear model parameters."""
    logits = np.dot(X, weights.T) + intercept
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int).ravel()
    return float(np.mean(preds == y))


def federated_round(datasets, rounds=5):
    """
    Run multiple federated training rounds across hospital datasets.

    Args:
        datasets: List of pandas DataFrames (each must include 'target').
        rounds: Number of federated rounds to execute.

    Returns:
        Tuple of ((final_weights, final_intercept), round_accuracies).
    """
    if not datasets:
        raise ValueError("datasets must contain at least one dataframe.")
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")

    combined = pd.concat(datasets, ignore_index=True)
    X_global = combined.drop(columns=["target"]).to_numpy(dtype=float)
    y_global = combined["target"].to_numpy(dtype=int)

    round_accuracies = []
    global_weights = None
    global_intercept = None

    for _ in range(rounds):
        masked_weight_list = []
        masked_intercept_list = []

        for df in datasets:
            scaler, weights_scaled, intercept_scaled = train_local_model(df)

            # Convert local parameters from scaled space back to raw feature space
            # so parameters are comparable across hospitals before aggregation.
            scale = scaler.scale_.reshape(1, -1)
            mean = scaler.mean_.reshape(1, -1)

            weights = np.asarray(weights_scaled, dtype=float) / scale
            intercept = np.asarray(intercept_scaled, dtype=float) - np.sum(weights * mean, axis=1)

            masked_weight_list.append(mask_weights(weights))
            masked_intercept_list.append(mask_weights(intercept))

        global_weights = secure_average(masked_weight_list)
        global_intercept = secure_average(masked_intercept_list)

        round_accuracy = _accuracy_from_params(
            X_global, y_global, global_weights, global_intercept
        )
        round_accuracies.append(round_accuracy)

    return (global_weights, global_intercept), round_accuracies
