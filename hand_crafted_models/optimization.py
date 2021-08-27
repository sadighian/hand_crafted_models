from typing import Tuple, Optional, Callable

import numpy as np

from hand_crafted_models.utils import ensure_dims

GradientStep = Tuple[float, np.ndarray, np.ndarray]
LossFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                        GradientStep]
WeightsAndBias = Tuple[np.ndarray, np.ndarray]


def gradient_descent(
        x: np.ndarray,
        y: np.ndarray,
        fn: LossFunction,
        lr: float = 0.001,
        tol: float = 1e-6,
        max_grad: float = 10.0,
        max_loops: int = 10000
) -> WeightsAndBias:
    """
    Fit parameters using gradient descent.
    
    :param x: Input data [Batch, Features]
    :param y: Label data [Batch, 1]
    :param fn: Gradient descent gradient calculation function
    :param lr: Learning rate (i.e., optimizer step size)
    :param tol: Tolerance for early-stopping
    :param max_grad: (Optional) Max size of gradient
    :param max_loops: Maximum number of steps to take
    :return: weight gradients, bias gradient
    """
    # Make sure vectors are converted to matrices
    x = ensure_dims(x)
    y = ensure_dims(y)
    # Dataset dimensions
    batch_size, num_of_features = x.shape
    # Feature params to learn
    weights = np.ones(shape=(1, num_of_features,), dtype=x.dtype)
    # Bias parameter to learn
    bias = np.zeros(shape=(1, 1,), dtype=x.dtype)
    # Ones vector for summing bias
    one = np.ones(shape=(batch_size, 1,), dtype=x.dtype)
    # Loss cache is for early stopping (see below)
    loss_cache: Optional[float] = None
    # Perform gradient descent 'n' times
    for i in range(max_loops):
        # Calculate gradients
        loss, d_w, d_b = fn(x, y, weights, bias, one)
        # Adjust gradients by the learning rate
        d_w *= lr
        d_b *= lr
        # (Optional) Clip the gradient values by a threshold
        if isinstance(max_grad, (int, float)):
            d_w = np.clip(d_w, -max_grad, max_grad)
            d_b = np.clip(d_b, -max_grad, max_grad)
        # Update parameters with gradient values
        weights += d_w
        bias += d_b
        # Check for early exit from training loop
        if loss_cache is None:
            loss_cache = loss
        else:
            loss_change = abs(loss - loss_cache)
            loss_cache = loss
            if loss_change < tol:
                print(f'Solved on Step #{i:,} | loss: {loss:,.3f}')
                break
    return weights, bias


def closed_form_linear_algebra(
        x: np.ndarray,
        y: np.ndarray,
        add_bias: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit parameters using matrices and linear algebra.
    
    :param x: Input data [Batch, Features]
    :param y: Label data [Batch, 1]
    :param add_bias: If 'true', append a column of ones to use for the bias
    :return: weight gradients, bias gradient
    """
    if add_bias:
        x_1 = np.hstack((x, np.ones(shape=x.shape, dtype=x.dtype)))
    else:
        x_1 = x
    try:
        betas = np.linalg.inv(x_1.T @ x_1) @ x_1.T @ y
    except np.linalg.LinAlgError:
        raise ValueError('Ran out of memory!!! Reduce matrix dims (i.e., rows OR columns)')
    weights = np.expand_dims(betas[:-1], 0)
    bias = np.expand_dims(betas[-1:], 0)
    return weights, bias
