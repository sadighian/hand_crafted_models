import numpy as np


def sigmoid(
        logits: np.ndarray
) -> np.ndarray:
    return 1 / (1 + np.exp(-logits))


def softmax(
        logits: np.ndarray
) -> np.ndarray:
    exp = np.exp(logits)
    probabilities = exp / exp.sum()
    return probabilities
