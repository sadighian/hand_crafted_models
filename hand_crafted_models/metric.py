import numpy as np

from hand_crafted_models.utils import convert_probability_to_boolean, ensure_dims


def accuracy(
        y_hat: np.ndarray,
        y: np.ndarray
) -> float:
    """
    Accuracy of predictions.
    
    :param y_hat: Model predictions
    :param y: Label or ground-truth
    :return: percentage correct
    """
    y_hat = ensure_dims(y_hat)
    y = ensure_dims(y)
    assert y_hat.shape == y.shape, \
        'Error: Labels and Predictions must have the same dimensions!'
    batch_size = y_hat.shape[0]
    binary_predictions = convert_probability_to_boolean(y_hat)
    pct_correct = (binary_predictions == y).astype(int).sum() / batch_size
    return pct_correct
