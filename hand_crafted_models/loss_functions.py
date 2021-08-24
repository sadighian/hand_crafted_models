import numpy as np

EPS = np.finfo(np.float32).eps


def log_loss(
        y_hat: np.ndarray,
        y: np.ndarray
) -> float:
    """
    Log-loss cost function.
    
    :param y_hat: Predicted value (scaled [0, 1])
    :param y: Label value (i.e., ground-truth)
    :return: loss value
    """
    is_one = y * np.log(y_hat)  # I.e., loss when Y is 1
    is_zero = (1.0 - y) * np.log(1.0 - y_hat)  # I.e., loss when Y is 0
    loss = -(is_one + is_zero).mean()
    return loss


def mean_squared_error(
        y_hat: np.ndarray,
        y: np.ndarray
) -> float:
    """
    Mean-squared-error cost function.
    
    :param y_hat: Predicted value (scaled [0, 1])
    :param y: Label value (i.e., ground-truth)
    :return: loss value
    """
    error = y - y_hat
    squared_error = error ** 2
    mse = squared_error.mean()
    return mse
