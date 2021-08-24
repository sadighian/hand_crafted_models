import numpy as np

EPS = np.finfo(np.float32).eps


def log_loss(
        y_hat: np.ndarray,
        y: np.ndarray
) -> float:
    is_one = y * np.log(y_hat)  # I.e., loss when y-hat is 1
    is_zero = (1.0 - y) * np.log(1.0 - y_hat)  # I.e., loss when y-hat is 0
    loss = -(is_one + is_zero).mean()
    return loss


def mean_squared_error(
        y_hat: np.ndarray,
        y: np.ndarray
) -> float:
    error = y - y_hat
    squared_error = error ** 2
    mse = squared_error.mean()
    return mse