import numpy as np


def mask_weights(weights, noise_scale=0.01, dp_enabled=True):
    """
    Simulate differential privacy by adding Gaussian noise to model weights.

    Args:
        weights: Input weights/intercepts as array-like.
        noise_scale: Standard deviation of Gaussian noise.
        dp_enabled: If False, no noise is added and original weights are returned.

    Returns:
        Numpy array of masked (or original) weights.
    """
    weights_array = np.asarray(weights, dtype=float)
    if not dp_enabled:
        return weights_array

    noise = np.random.normal(loc=0.0, scale=noise_scale, size=weights_array.shape)
    return weights_array + noise


def secure_average(weight_list):
    """Compute the element-wise average of a list of weight arrays."""
    if not weight_list:
        raise ValueError("weight_list must contain at least one array.")

    arrays = [np.asarray(w, dtype=float) for w in weight_list]
    return np.mean(arrays, axis=0)
