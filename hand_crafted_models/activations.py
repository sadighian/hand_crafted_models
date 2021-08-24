import numpy as np


def sigmoid(
        logits: np.ndarray
) -> np.ndarray:
    """
    Sigmoid activation function (I.e., scales to [0, 1]).
    
    :param logits: Output from dot-product
    :return: Prediction probability
    """
    return 1 / (1 + np.exp(-logits))


def softmax(
        logits: np.ndarray
) -> np.ndarray:
    """
    Softmax activation function (I.e., scales to SUM to 1).
    
    :param logits: Output from dot-product
    :return: Prediction probabilities
    """
    exp = np.exp(logits)
    probabilities = exp / exp.sum()
    return probabilities
