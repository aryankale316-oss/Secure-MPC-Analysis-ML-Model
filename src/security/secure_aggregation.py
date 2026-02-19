
import numpy as np
from src.security.encryption import encrypt_weights, decrypt_weights

def secure_average(models):

    encrypted_weights = []

    for model in models:
        weights = model.coef_
        encrypted = encrypt_weights(weights)
        encrypted_weights.append(encrypted)

    avg_encrypted = np.mean(encrypted_weights, axis=0)

    decrypted = decrypt_weights(avg_encrypted)

    return decrypted
